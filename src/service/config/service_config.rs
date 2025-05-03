const DEFAULT_WS_PORT: u16 = 3030;
const DEFAULT_API_KEY: &str = "2e6c9193-d904-4fd1-9f4e-48c3d20fcdcc";
const DEFAULT_CONNECTION_THREADS: usize = 4;
const DEFAULT_IDLE_FLUSH_MS: usize = 5000; // 5 seconds

const DEFAULT_WHISPER_MODEL_NAME: &str = "ggml-base.en-q5_1.bin";
const DEFAULT_WHISPER_MIN_BUFFER_MS: usize = 4000; // 4 seconds
const DEFAULT_WHISPER_MAX_BUFFER_MS: usize = 24_000; // 8_000 * 3 (was 24_000 ms)
const DEFAULT_WHISPER_LOOKBACK_MS: usize = 350;

const DEFAULT_VAD_LOOKBACK_MS: usize = 200;
const DEFAULT_VAD_THOLD: f32 = 0.35;

#[derive(Debug)]
pub struct ServiceConfig {
    pub port: u16,
    pub api_key: String,
    pub connection_threads: usize,

    /// Flush the buffer after this many milliseconds of silence
    pub idle_flush_ms: usize,

    pub whisper_model_name: String,
    pub whisper_min_buffer_ms: usize,
    pub whisper_max_buffer_ms: usize,
    pub whisper_lookback_ms: usize,
    pub whisper_threads: Option<usize>,

    pub vad_lookback_ms: usize,
    pub vad_thold: f32,
}

impl ServiceConfig {
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("WS_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(DEFAULT_WS_PORT),

            api_key: std::env::var("API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.into()),

            connection_threads: std::env::var("CONNECTION_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_CONNECTION_THREADS),

            idle_flush_ms: std::env::var("IDLE_FLUSH_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_IDLE_FLUSH_MS),

            // Whisper settings
            whisper_model_name: std::env::var("WHISPER_MODEL_NAME")
                .unwrap_or_else(|_| DEFAULT_WHISPER_MODEL_NAME.into()),

            whisper_min_buffer_ms: std::env::var("WHISPER_MIN_BUFFER_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_WHISPER_MIN_BUFFER_MS),

            whisper_max_buffer_ms: std::env::var("WHISPER_MAX_BUFFER_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_WHISPER_MAX_BUFFER_MS),

            whisper_lookback_ms: std::env::var("WHISPER_LOOKBACK_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_WHISPER_LOOKBACK_MS),

            whisper_threads: std::env::var("WHISPER_THREADS")
                .ok()
                .and_then(|s| s.parse().ok()),

            // VAD settings
            vad_lookback_ms: std::env::var("VAD_LOOKBACK_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_VAD_LOOKBACK_MS),

            vad_thold: std::env::var("VAD_THOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_VAD_THOLD),
        }
    }

    pub fn check(&self) -> Result<(), String> {
        let available_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1); // fallback to 1 if error

        if let Some(whisper_threads) = self.whisper_threads {
            // Check if the number of threads is greater than the number of available CPUs
            if whisper_threads > available_threads {
                panic!(
                    "Requested more Whisper threads ({whisper_threads}) than available ({available_threads})"
                );
            }
            tracing::info!("Using {} threads for Whisper", whisper_threads);
        }

        if self.connection_threads < 1 {
            return Err("CONNECTION_THREADS must be > 0".into());
        } else if self.connection_threads > available_threads {
            panic!(
                "Requested more connection threads ({}) than available ({})",
                self.connection_threads, available_threads
            );
        }

        if self.whisper_min_buffer_ms < 1000 {
            return Err(
                "WHISPER_MIN_BUFFER_MS must be > 1000, 1 sec audio is minimum for Whisper".into(),
            );
        }

        if self.whisper_max_buffer_ms <= self.whisper_min_buffer_ms {
            return Err("WHISPER_MAX_BUFFER_MS must be > WHISPER_MIN_BUFFER_MS".into());
        }

        // if self.idle_flush_ms < 0 {
        //     return Err("IDLE_FLUSH_MS must be > 0".into());
        // }

        if !(0.0..=1.0).contains(&self.vad_thold) {
            return Err("VAD_THOLD must be between 0.0 and 1.0".into());
        }

        Ok(())
    }
}
