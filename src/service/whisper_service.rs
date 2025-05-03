use crate::{
    service::config::service_params::ServiceParams, service::engine::shared_state::Transcript,
    vad::energy_vad::EnergyVad, whisper::whisper_engine::WhisperEngine,
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
use tokio::sync::Semaphore;
use tokio::time::{Instant, Sleep, sleep_until};

use super::engine::shared_state::SharedState;

pub const SAMPLE_RATE: usize = 16_000; // Hz

pub struct WhisperService {
    params: Arc<ServiceParams>,
    max_buffer_samples: usize,
    min_buffer_samples: usize,
    lookback_samples: usize,
    vad_lookback_samples: usize,
    semaphore: Arc<Semaphore>,
}
impl WhisperService {
    pub fn new(params: Arc<ServiceParams>) -> Self {
        let connection_threads = params.connection_threads;
        let min_buffer_samples = SAMPLE_RATE * params.whisper_min_buffer_ms / 1_000;
        let max_buffer_samples = SAMPLE_RATE * params.whisper_max_buffer_ms / 1_000;
        let lookback_samples = SAMPLE_RATE * params.whisper_lookback_ms / 1_000;
        let vad_lookback_samples = SAMPLE_RATE * params.vad_lookback_ms / 1_000;

        Self {
            params: params.clone(),
            semaphore: Arc::new(Semaphore::new(connection_threads)),
            min_buffer_samples,
            max_buffer_samples,
            lookback_samples,
            vad_lookback_samples,
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

    /// Main handler for a websocket connection.
    /// Handles audio streaming, VAD, and transcription.
    async fn handle_socket(self: Arc<Self>, socket: WebSocket) {
        tracing::info!("WebSocket: client connected");
        let (mut sender, mut receiver) = socket.split();

        let shared = Arc::new(SharedState::new());
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Transcript>(512); // Bounded for backpressure

        let vad_thold = self.params.vad_thold;
        let vad_lookback_samples = self.vad_lookback_samples;
        let max_buffer_samples = self.max_buffer_samples;
        let min_buffer_samples = self.min_buffer_samples;
        let lookback_samples = self.lookback_samples;

        // Writer: sends transcriptions to client, robust to errors
        let writer = tokio::spawn(async move {
            while let Some(t) = rx.recv().await {
                let msg = Message::Text(
                    json!({"text": t.text.trim_start(), "seq": t.seq})
                        .to_string()
                        .into(),
                );
                if let Err(e) = sender.send(msg).await {
                    tracing::warn!("WebSocket send error: {e}");
                    break;
                }
            }
        });

        // Reader: receives audio, decodes, runs VAD, manages buffer, triggers flush
        let params_r = self.params.clone();
        let sem_r = self.semaphore.clone();
        let shared_r = shared.clone();
        let reader = tokio::spawn(async move {
            // Preallocate buffers for efficiency
            let mut ring: VecDeque<f32> = VecDeque::with_capacity(max_buffer_samples);
            let mut vad = EnergyVad::new(shared_r.clone(), vad_lookback_samples, vad_thold);
            let mut lookback_buffer: Vec<f32> = Vec::with_capacity(lookback_samples.max(1));

            // idle-flush timer
            let mut deadline = Instant::now() + params_r.idle_flush;
            let mut sleep: Pin<Box<Sleep>> = Box::pin(sleep_until(deadline));

            loop {
                tokio::select! {
                    biased;
                    msg = receiver.next() => {
                        match msg {
                            Some(Ok(Message::Binary(buf))) => {
                                // --- Audio Decoding Safety ---
                                // Assumes input is 16kHz mono, 32-bit float, little-endian PCM.
                                // If the input is not a multiple of 4 bytes, ignore the trailing bytes and log a warning.
                                // let mut incomplete = false;
                                for chunk in buf.chunks_exact(4) {
                                    // Safe conversion: skip malformed chunks
                                    let arr: [u8; 4] = match chunk.try_into() {
                                        Ok(arr) => arr,
                                        Err(_) => {
                                            tracing::warn!("Malformed audio chunk (expected 4 bytes)");
                                            continue;
                                        }
                                    };
                                    let s = f32::from_le_bytes(arr);
                                    let old_sample_energy = if ring.len() >= vad_lookback_samples {
                                        let idx = ring.len() - vad_lookback_samples;
                                        Some(ring[idx].abs() as f64)
                                    } else {
                                        None
                                    };
                                    ring.push_back(s);
                                    vad.process_sample(s, ring.len(), old_sample_energy);
                                }
                                if buf.len() % 4 != 0 {
                                    // incomplete = true;
                                    tracing::warn!("Received audio buffer with incomplete trailing bytes ({} bytes)", buf.len() % 4);
                                }
                                // --- VAD and Buffer Management ---
                                let cb_idx = shared_r.last_sentence_end.swap(0, Ordering::Acquire) as usize;
                                let boundary = if cb_idx >= min_buffer_samples { cb_idx } else { 0 };
                                if boundary > 0 {
                                    if let Err(e) = WhisperEngine::flush_with_lookback(&mut ring, &mut lookback_buffer, lookback_samples, boundary, &tx, &sem_r, &params_r, &shared_r, false).await {
                                        tracing::error!("Flush error: {e}");
                                    }
                                }
                                let ring_len = ring.len();
                                if ring_len >= max_buffer_samples {
                                    tracing::warn!("audio spill-over {} - force flush", ring_len);
                                    if let Err(e) = WhisperEngine::flush_with_lookback(&mut ring, &mut lookback_buffer, lookback_samples, ring_len, &tx, &sem_r, &params_r, &shared_r, false).await {
                                        tracing::error!("Flush error: {e}");
                                    }
                                }
                                deadline = Instant::now() + params_r.idle_flush;
                                sleep.as_mut().reset(deadline);
                            }
                            Some(Ok(Message::Close(_))) | None => {
                                tracing::info!("WebSocket closed by client");
                                break;
                            },
                            Some(Ok(_)) => {}, // Ignore non-binary messages
                            Some(Err(e)) => {
                                tracing::warn!("WebSocket receive error: {e}");
                                break;
                            }
                        }
                    },
                    _ = &mut sleep => {
                        if !ring.is_empty() {
                            let flush_count = ring.len();
                            if let Err(e) = WhisperEngine::flush_with_lookback(&mut ring, &mut lookback_buffer, lookback_samples, flush_count, &tx, &sem_r, &params_r, &shared_r, true).await {
                                tracing::error!("Idle flush error: {e}");
                            }
                        }
                        deadline = Instant::now() + params_r.idle_flush;
                        sleep.as_mut().reset(deadline);
                    }
                }
            }
            // Final flush
            if !ring.is_empty() {
                let flush_count = ring.len();
                if let Err(e) = WhisperEngine::flush_with_lookback(
                    &mut ring,
                    &mut lookback_buffer,
                    lookback_samples,
                    flush_count,
                    &tx,
                    &sem_r,
                    &params_r,
                    &shared_r,
                    false,
                )
                .await
                {
                    tracing::error!("Final flush error: {e}");
                }
            }
        });

        // Wait for both tasks to finish
        let _ = tokio::join!(reader, writer);
        tracing::info!("WebSocket: session ended");
    }
}
