use crate::{
    config::service_params::ServiceParams,
    service::shared_state::shared_state::Transcript,
    vad::energy_vad::{EnergyVad, LOOKBACK_SAMPLES, SAMPLE_RATE},
    whisper::{whisper_callback::WhisperCallback, whisper_helper::WhisperHelper},
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
    pin::Pin,
    sync::{Arc, atomic::Ordering},
};
use tokio::{
    sync::{
        Semaphore,
        mpsc::{self},
    },
    time::{Instant, Sleep, sleep_until},
};

use super::shared_state::shared_state::SharedState;

const MAX_BUFFER_MS: usize = 8_000 * 3; // hard cap to avoid RAM blow‑up
const MAX_BUFFER_SAMPLES: usize = SAMPLE_RATE * MAX_BUFFER_MS / 1_000; // 384 000
const MAX_SERVICE_THREADS: usize = 4;

pub struct WhisperService {
    params: Arc<ServiceParams>,
    semaphore: Arc<Semaphore>,
}
impl WhisperService {
    pub fn new(params: Arc<ServiceParams>) -> Self {
        Self {
            params,
            semaphore: Arc::new(Semaphore::new(MAX_SERVICE_THREADS)),
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
            let mut vad = EnergyVad::new(shared_r.clone());
            let threshold = params_r.sample_threshold; // 64 000 (4s) default 

            // idle-flush timer
            let mut deadline = Instant::now() + params_r.idle_flush;
            let mut sleep: Pin<Box<Sleep>> = Box::pin(sleep_until(deadline));

            // while let Some(Ok(msg)) = receiver.next().await {
            loop {
                tokio::select! {
                            biased;
                            msg = receiver.next() => {
                    match msg {
                        Some(Ok(Message::Binary(buf))) => { // Message::Binary(buf) => {
                            // Check and exttend the buffer size in advance
                            let chunk_count = buf.len() / 4;
                            if ring.capacity() < ring.len() + chunk_count {
                                ring.reserve(chunk_count);
                            }

                            for chunk in buf.chunks_exact(4) {
                                let s = f32::from_le_bytes(chunk.try_into().unwrap_or([0.0f32.to_le_bytes()[0]; 4]));

                                // Get old sample energy for VAD
                                let old_sample_energy = if ring.len() >= LOOKBACK_SAMPLES {
                                    let idx = ring.len() - LOOKBACK_SAMPLES;
                                    Some(ring[idx].abs() as f64)
                                } else {
                                    None
                                };

                                ring.push_back(s);

                                // Update energy VAD
                                vad.process_sample(s, ring.len(), old_sample_energy);
                            }

                            // Check VAD silence boundary
                            let cb_idx = shared_r.last_sentence_end.swap(0, Ordering::Acquire) as usize;
                            let boundary = if cb_idx >= threshold { cb_idx } else { 0 };
                            if boundary > 0 {
                                Self::flush(
                                    boundary, &mut ring, &tx, &sem_r, &params_r, &shared_r,false,
                                )
                                .await;
                            }

                            // TODO env var
                            // overflow guard
                            let ring_len = ring.len();
                            if ring_len >= MAX_BUFFER_SAMPLES {
                                tracing::warn!("audio spill-over {} - force flush", ring_len);
                                Self::flush(
                                    ring_len,
                                    &mut ring,
                                    &tx,
                                    &sem_r,
                                    &params_r,
                                    &shared_r,
                                    false,
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
                              Self::flush(ring.len(), &mut ring, &tx, &sem_r, &params_r, &shared_r, true).await;
                          }
                          deadline = Instant::now() + params_r.idle_flush;
                          sleep.as_mut().reset(deadline);
                      }}
            }

            // final flush
            if !ring.is_empty() {
                Self::flush(
                    ring.len(),
                    &mut ring,
                    &tx,
                    &sem_r,
                    &params_r,
                    &shared_r,
                    false,
                )
                .await;
            }
        });

        let _ = tokio::join!(reader, writer);
        tracing::info!("session ended");
    }

    /// Process audio samples from the ring buffer
    ///
    /// * `count` - Number of samples to process
    /// * `ring` - Ring buffer with audio samples
    /// * `track_jobs` - Whether to track active jobs count (for graceful shutdown)
    async fn flush(
        count: usize,
        ring: &mut VecDeque<f32>,
        tx: &mpsc::Sender<Transcript>,
        sem: &Arc<Semaphore>,
        params: &Arc<ServiceParams>,
        shared: &Arc<SharedState>,
        track_jobs: bool,
    ) {
        // Prepare ring buffer for flushing
        let mut samples = Vec::with_capacity(count);
        if count <= ring.len() {
            samples.extend(ring.drain(..count));
        } else {
            tracing::warn!("Attempted to flush more samples than available");
            samples.extend(ring.drain(..));
        }

        // Track active jobs if requested
        if track_jobs {
            shared.active_jobs.fetch_add(1, Ordering::AcqRel);
        }
        let flush_seq = shared.flush_seq.fetch_add(1, Ordering::AcqRel);

        let tx_cl = tx.clone();
        let params_cl = params.clone();
        let shared_cl = shared.clone();

        match sem.clone().acquire_owned().await {
            Ok(permit) => {
                tokio::spawn(async move {
                    let _g = permit;
                    Self::run_whisper(samples, params_cl, shared_cl, tx_cl, flush_seq).await;
                });
            }
            Err(_) => {
                if track_jobs {
                    shared.active_jobs.fetch_sub(1, Ordering::AcqRel);
                } else {
                    tracing::warn!("semaphore closed – drop chunk");
                }
            }
        }
    }

    /// Run the Whisper transcription
    async fn run_whisper(
        samples: Vec<f32>,
        params: Arc<ServiceParams>,
        shared: Arc<SharedState>,
        out: mpsc::Sender<Transcript>,
        flush_seq: usize,
    ) {
        tracing::info!(">>> transcribing {} samples", samples.len());
        tokio::task::spawn_blocking(move || {
            let mut fp = params.whisper_cfg.to_full_params();

            let callback = WhisperCallback::new(shared.clone(), out, flush_seq);
            callback.setup_callback(&mut fp, WhisperHelper::whisper_callback);

            match params.whisper_ctx.create_state() {
                Ok(mut st) => {
                    // Execute transcription and check result
                    if let Err(e) = st.full(fp, &samples) {
                        tracing::error!("Whisper processing error: {:?}", e);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to create Whisper state: {:?}", e);
                }
            }
        })
        .await
        .ok();
    }
}
