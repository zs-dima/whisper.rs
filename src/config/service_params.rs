use std::time::Duration;

use whisper_rs::WhisperContext;

use crate::whisper::whisper_config::WhisperConfig;

pub struct ServiceParams {
    pub whisper_ctx: WhisperContext,
    pub whisper_cfg: WhisperConfig,
    pub sample_threshold: usize,
    pub api_key: String,
    pub idle_flush: Duration,
    pub max_buffer_ms: usize,
    pub max_service_threads: usize,
    pub lookback_ms: usize,
    pub vad_thold: f32,
}
