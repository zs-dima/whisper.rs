const DEFAULT_PORT: u16 = 3030;
const DEFAULT_API_KEY: &str = "2e6c9193-d904-4fd1-9f4e-48c3d20fcdcc";
const DEFAULT_MODEL_NAME: &str = "ggml-base.en-q5_1.bin";
const DEFAULT_SAMPLE_THRESHOLD: usize = 16000 * 4; // 4 seconds at 16 kHz
const DEFAULT_MAX_BUFFER_MS: usize = 24_000; // 8_000 * 3 (was 24_000 ms)
const DEFAULT_MAX_SERVICE_THREADS: usize = 4;
const DEFAULT_LOOKBACK_MS: usize = 200;
const DEFAULT_VAD_THOLD: f32 = 0.35;

#[derive(Debug)]
pub struct ServiceConfig {
    pub port: u16,
    pub api_key: String,
    pub model_name: String,
    pub threads: Option<usize>,
    pub sample_threshold: usize,
    pub max_buffer_ms: usize,
    pub max_service_threads: usize,
    pub lookback_ms: usize,
    pub vad_thold: f32,
}

impl ServiceConfig {
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("SERVICE_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(DEFAULT_PORT),
            api_key: std::env::var("API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.into()),
            model_name: std::env::var("WHISPER_MODEL_NAME")
                .unwrap_or_else(|_| DEFAULT_MODEL_NAME.into()),
            threads: std::env::var("WHISPER_THREADS")
                .ok()
                .and_then(|s| s.parse().ok()),
            sample_threshold: std::env::var("SAMPLE_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_SAMPLE_THRESHOLD),
            max_buffer_ms: std::env::var("MAX_BUFFER_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_MAX_BUFFER_MS),
            max_service_threads: std::env::var("MAX_SERVICE_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_MAX_SERVICE_THREADS),
            lookback_ms: std::env::var("LOOKBACK_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_LOOKBACK_MS),
            vad_thold: std::env::var("VAD_THOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_VAD_THOLD),
        }
    }

    pub fn check(&self) -> Result<(), String> {
        if let Some(threads) = self.threads {
            // Check if the number of threads is greater than the number of available CPUs
            let available = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1); // fallback to 1 if error
            if threads > available {
                panic!("Requested more threads ({threads}) than available ({available})");
            }
            tracing::info!("Using {} threads for Whisper", threads);
        }
        if self.sample_threshold < 16000 {
            return Err(
                "SAMPLE_THRESHOLD must be > 16000, 1 sec audio is minimum for Whisper".into(),
            );
        }
        if self.max_service_threads < 1 {
            return Err("MAX_SERVICE_THREADS must be > 0".into());
        }
        if self.max_buffer_ms == 0 {
            return Err("MAX_BUFFER_MS must be > 0".into());
        }
        if self.lookback_ms == 0 {
            return Err("LOOKBACK_MS must be > 0".into());
        }
        if !(0.0..=1.0).contains(&self.vad_thold) {
            return Err("VAD_THOLD must be between 0.0 and 1.0".into());
        }
        Ok(())
    }
}
