use std::time::Duration;

use whisper_rs::WhisperContext;

use crate::whisper::whisper_config::WhisperConfig;

pub struct ServiceParams {
    pub api_key: String,
    pub connection_threads: usize,
    pub idle_flush: Duration,

    pub whisper_ctx: WhisperContext,
    pub whisper_cfg: WhisperConfig,
    pub whisper_min_buffer_ms: usize,
    pub whisper_max_buffer_ms: usize,
    pub whisper_lookback_ms: usize,

    pub vad_lookback_ms: usize,
    pub vad_thold: f32,
}
