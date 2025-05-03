use crate::{
    service::config::service_params::ServiceParams,
    service::engine::shared_state::{SharedState, Transcript},
    whisper::{whisper_callback::WhisperCallback, whisper_helper::WhisperHelper},
};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc};

pub struct WhisperEngine;

impl WhisperEngine {
    /// Like flush, but takes explicit samples (with lookback prepended)
    /// Returns Result for error propagation and testability.
    pub async fn flush_samples(
        samples: Vec<f32>,
        tx: &mpsc::Sender<Transcript>,
        sem: &Arc<Semaphore>,
        params: &Arc<ServiceParams>,
        shared: &Arc<SharedState>,
        track_jobs: bool,
    ) -> Result<(), String> {
        if track_jobs {
            shared
                .active_jobs
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        let flush_seq = shared
            .flush_seq
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let tx_cl = tx.clone();
        let params_cl = params.clone();
        let shared_cl = shared.clone();
        match sem.clone().acquire_owned().await {
            Ok(permit) => {
                tokio::spawn(async move {
                    let _g = permit; // Permit released when dropped
                    if let Err(e) =
                        WhisperEngine::run_whisper(samples, params_cl, shared_cl, tx_cl, flush_seq)
                            .await
                    {
                        tracing::error!("Whisper task error: {e}");
                    }
                });
                Ok(())
            }
            Err(_) => {
                if track_jobs {
                    shared
                        .active_jobs
                        .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                } else {
                    tracing::warn!("semaphore closed – drop chunk");
                }
                Err("Semaphore closed".to_string())
            }
        }
    }

    /// Run the Whisper transcription
    /// Returns Result for error propagation and testability.
    pub async fn run_whisper(
        samples: Vec<f32>,
        params: Arc<ServiceParams>,
        shared: Arc<SharedState>,
        out: mpsc::Sender<Transcript>,
        flush_seq: usize,
    ) -> Result<(), String> {
        tracing::info!(">>> transcribing {} samples", samples.len());
        let res = tokio::task::spawn_blocking(move || {
            let mut fp = params.whisper_cfg.to_full_params();
            let callback = WhisperCallback::new(shared.clone(), out, flush_seq);
            callback.setup_callback(&mut fp, WhisperHelper::whisper_callback);
            match params.whisper_ctx.create_state() {
                Ok(mut st) => {
                    if let Err(e) = st.full(fp, &samples) {
                        tracing::error!("Whisper processing error: {:?}", e);
                        return Err(format!("Whisper processing error: {:?}", e));
                    }
                    Ok(())
                }
                Err(e) => {
                    tracing::error!("Failed to create Whisper state: {:?}", e);
                    Err(format!("Failed to create Whisper state: {:?}", e))
                }
            }
        })
        .await;
        match res {
            Ok(inner) => inner,
            Err(e) => Err(format!("Whisper thread join error: {e}")),
        }
    }

    /// Helper: flushes samples with lookback, updates lookback buffer, and triggers transcription.
    /// Returns Result for error propagation and testability.
    pub async fn flush_with_lookback(
        ring: &mut VecDeque<f32>,
        lookback_buffer: &mut Vec<f32>,
        lookback_samples: usize,
        count: usize,
        tx: &mpsc::Sender<Transcript>,
        sem: &Arc<Semaphore>,
        params: &Arc<ServiceParams>,
        shared: &Arc<SharedState>,
        track_jobs: bool,
    ) -> Result<(), String> {
        // Prepare flush buffer
        let mut flush_samples = Vec::with_capacity(lookback_buffer.len() + count);
        flush_samples.extend_from_slice(&lookback_buffer);
        flush_samples.extend(ring.drain(..count));
        // Update lookback buffer with last lookback_samples from flushed region
        if lookback_samples > 0 {
            let total = flush_samples.len();
            let start = total.saturating_sub(lookback_samples);
            lookback_buffer.clear();
            lookback_buffer.extend_from_slice(&flush_samples[start..]);
        }
        WhisperEngine::flush_samples(flush_samples, tx, sem, params, shared, track_jobs).await
    }
}
