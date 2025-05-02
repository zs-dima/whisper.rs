use crate::{
    config::service_params::ServiceParams, service::shared_state::shared_state::Transcript,
};

use axum::{
    extract::{
        Query as AxumQuery, State as AxumState,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};

use serde_json::json;
use std::{
    collections::{HashMap, VecDeque},
    ffi::{CStr, c_void},
    pin::Pin,
    sync::{Arc, atomic::Ordering},
};
use tokio::{
    sync::{
        Semaphore,
        mpsc::{self, error::TrySendError},
    },
    time::{Instant, Sleep, sleep_until},
};
use whisper_rs::whisper_rs_sys;

use super::shared_state::shared_state::{CppCallbackData, SharedState};

const SAMPLE_RATE: usize = 16_000; // Hz
const LOOKBACK_MS: usize = 200; // look at last 200 ms
const LOOKBACK_SAMPLES: usize = SAMPLE_RATE * LOOKBACK_MS / 1_000; // 4 800
const VAD_THOLD: f32 = 0.35; // energy_last < 0.35 × energy_all ⇒ silence
const MAX_BUFFER_MS: usize = 8_000 * 3; // hard cap to avoid RAM blow‑up
const MAX_BUFFER_SAMPLES: usize = SAMPLE_RATE * MAX_BUFFER_MS / 1_000; // 384 000
const MAX_PAR_THREADS: usize = 4;

pub struct WhisperService {
    params: Arc<ServiceParams>,
    semaphore: Arc<Semaphore>,
}
impl WhisperService {
    pub fn new(params: Arc<ServiceParams>) -> Self {
        Self {
            params,
            semaphore: Arc::new(Semaphore::new(MAX_PAR_THREADS)),
        }
    }

    pub async fn ws_handler(
        ws: WebSocketUpgrade,
        AxumQuery(q): AxumQuery<HashMap<String, String>>,
        AxumState(this): AxumState<Arc<Self>>,
    ) -> impl IntoResponse {
        if this.params.api_key.is_empty()
            || q.get("api-key").map(String::as_str) != Some(&this.params.api_key)
        {
            return (axum::http::StatusCode::UNAUTHORIZED, "Bad API key").into_response();
        }
        ws.on_upgrade(move |sock| this.handle_socket(sock))
    }

    async fn handle_socket(self: Arc<Self>, socket: WebSocket) {
        tracing::info!("ws connected");
        let (mut sender, mut receiver) = socket.split();

        let shared = Arc::new(SharedState::new());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Transcript>(512);

        // writer – single task
        let writer = tokio::spawn(async move {
            while let Some(t) = rx.recv().await {
                if sender
                    .send(Message::Text(
                        json!({"text":t.text,"seq":t.seq}).to_string().into(),
                    ))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });

        // reader
        let params_r = self.params.clone();
        let sem_r = self.semaphore.clone();
        let shared_r = shared.clone();
        let reader = tokio::spawn(async move {
            let mut ring: VecDeque<f32> = VecDeque::with_capacity(MAX_BUFFER_SAMPLES);
            // running energy metrics
            let mut energy_total: f64 = 0.0; // Σ|x|
            let mut energy_tail: f64 = 0.0; // Σ|x| over trailing window
            let threshold = params_r.sample_threshold; // 64 000 (4s) default 

            // ── idle-flush timer ──────────────────────────────────────────────────────
            let mut deadline = Instant::now() + params_r.idle_flush;
            let mut sleep: Pin<Box<Sleep>> = Box::pin(sleep_until(deadline));

            // while let Some(Ok(msg)) = receiver.next().await {
            loop {
                tokio::select! {
                            biased;
                            msg = receiver.next() => {
                    match msg {
                        Some(Ok(Message::Binary(buf))) => { // Message::Binary(buf) => {
                            for chunk in buf.chunks_exact(4) {
                                let s = f32::from_le_bytes(chunk.try_into().unwrap());
                                let abs = s.abs() as f64;
                                ring.push_back(s);
                                energy_total += abs;
                                energy_tail += abs;

                                // maintain the LOOKBACK window sum
                                if ring.len() > LOOKBACK_SAMPLES {
                                    let idx = ring.len() - LOOKBACK_SAMPLES - 1;
                                    energy_tail -= ring[idx].abs() as f64;
                                }
                            }

                            // energy‑ratio VAD
                            if ring.len() >= LOOKBACK_SAMPLES {
                                let energy_all = (energy_total / ring.len() as f64) as f32;
                                let energy_last = (energy_tail / LOOKBACK_SAMPLES as f64) as f32;
                                if energy_last < VAD_THOLD * energy_all {
                                    shared_r
                                        .last_sentence_end
                                        .fetch_max(ring.len() as u64, Ordering::Release);
                                }
                            }

                            // decide if we should flush
                            let cb_idx = shared_r.last_sentence_end.swap(0, Ordering::Acquire) as usize;
                            let boundary = if cb_idx >= threshold { cb_idx } else { 0 };
                            if boundary > 0 {
                                Self::flush_chunk(
                                    boundary, &mut ring, &tx, &sem_r, &params_r, &shared_r,
                                )
                                .await;
                            }

                            // TODO env var
                            // overflow guard
                            let ring_len = ring.len();
                            if ring_len >= MAX_BUFFER_SAMPLES {
                                tracing::warn!("audio spill-over {} - force flush", ring_len);
                                Self::flush_chunk(
                                    ring_len,
                                    &mut ring,
                                    &tx,
                                    &sem_r,
                                    &params_r,
                                    &shared_r,
                                )
                                .await;
                            }

                            // reset idle timer
                            deadline = Instant::now() + params_r.idle_flush;
                            sleep.as_mut().reset(deadline);
                        }
                        Some(Ok(Message::Close(_))) | None => break, // Message::Close(_) => break,
                        _ => {}
                    }

                }, _ = &mut sleep => {
                          // idle flush (no size threshold)
                          if !ring.is_empty() {
                              Self::flush(ring.len(), &mut ring, &tx, &sem_r, &params_r, &shared_r).await;
                          }
                          deadline = Instant::now() + params_r.idle_flush;
                          sleep.as_mut().reset(deadline);
                      }}
            }

            // final flush
            if !ring.is_empty() {
                Self::flush_chunk(ring.len(), &mut ring, &tx, &sem_r, &params_r, &shared_r).await;
            }
        });

        let _ = tokio::join!(reader, writer);
        tracing::info!("session ended");
    }

    ///
    /// Flush + Whisper helpers --------------------------------------------------------
    ///

    async fn flush(
        count: usize,
        ring: &mut VecDeque<f32>,
        tx: &mpsc::Sender<Transcript>,
        sem: &Arc<Semaphore>,
        params: &Arc<ServiceParams>,
        shared: &Arc<SharedState>,
    ) {
        let samples: Vec<f32> = ring.drain(..count).collect();
        shared.active_jobs.fetch_add(1, Ordering::AcqRel);
        let tx_cl = tx.clone();
        let params_cl = params.clone();
        let shared_cl = shared.clone();
        match sem.clone().acquire_owned().await {
            Ok(permit) => {
                tokio::spawn(async move {
                    let _g = permit;
                    Self::run_whisper(samples, params_cl, shared_cl, tx_cl).await;
                });
            }
            Err(_) => {
                shared.active_jobs.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    async fn flush_chunk(
        count: usize,
        ring: &mut VecDeque<f32>,
        tx: &mpsc::Sender<Transcript>,
        sem: &Arc<Semaphore>,
        params: &Arc<ServiceParams>,
        shared: &Arc<SharedState>,
    ) {
        let samples: Vec<f32> = ring.drain(..count).collect();
        let tx_cl = tx.clone();
        let params_cl = params.clone();
        let shared_cl = shared.clone();
        match sem.clone().acquire_owned().await {
            Ok(permit) => {
                tokio::spawn(async move {
                    let _g = permit;
                    Self::run_whisper(samples, params_cl, shared_cl, tx_cl).await;
                });
            }
            Err(_) => tracing::warn!("semaphore closed – drop chunk"),
        }
    }

    async fn run_whisper(
        samples: Vec<f32>,
        params: Arc<ServiceParams>,
        shared: Arc<SharedState>,
        out: mpsc::Sender<Transcript>,
    ) {
        tracing::info!(">>> transcribing {} samples", samples.len());
        tokio::task::spawn_blocking(move || {
            let mut fp = params.whisper_cfg.to_full_params();
            let data = Box::new(CppCallbackData {
                boundary: shared,
                out,
            });
            let ptr = Box::into_raw(data) as *mut c_void;
            unsafe {
                fp.set_new_segment_callback(Some(Self::callback));
                fp.set_new_segment_callback_user_data(ptr);
            }
            let mut st = params.whisper_ctx.create_state().expect("state alloc");
            st.full(fp, &samples).ok();
            unsafe {
                let data: Box<CppCallbackData> = Box::from_raw(ptr as *mut CppCallbackData);
                data.boundary.active_jobs.fetch_sub(1, Ordering::AcqRel);
            }
        })
        .await
        .ok();
    }

    ///
    /// C callback – unchanged from old version but with global seq
    ///
    unsafe extern "C" fn callback(
        _: *mut whisper_rs_sys::whisper_context,
        st: *mut whisper_rs_sys::whisper_state,
        _: i32,
        user: *mut c_void,
    ) {
        if user.is_null() {
            return;
        }
        let data = unsafe { &*(user as *const CppCallbackData) };
        let n = unsafe { whisper_rs_sys::whisper_full_n_segments_from_state(st) } - 1;
        let txt_ptr = unsafe { whisper_rs_sys::whisper_full_get_segment_text_from_state(st, n) };
        if txt_ptr.is_null() {
            return;
        }
        if let Ok(txt) = unsafe { CStr::from_ptr(txt_ptr) }.to_str() {
            let seq = data.boundary.next_seq.fetch_add(1, Ordering::Relaxed);
            loop {
                match data.out.try_send(Transcript {
                    seq,
                    text: txt.to_owned(),
                }) {
                    Ok(()) => break,
                    Err(TrySendError::Full(t)) => {
                        // Drop oldest instead of spinning forever:
                        let _ = data.out.try_send(t); // overwrite
                        break;
                    }
                    Err(TrySendError::Closed(_)) => return, // client is gone
                }
            }
            if txt.trim_end().ends_with(['.', '!', '?']) {
                let t1 =
                    unsafe { whisper_rs_sys::whisper_full_get_segment_t1_from_state(st, n) } as u64;
                data.boundary
                    .last_sentence_end
                    .store(t1 * 16, Ordering::Release); // 1 ms = 16 samples
            }
        }
    }
}
