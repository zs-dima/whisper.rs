const DEFAULT_PORT: u16 = 3030;
const DEFAULT_API_KEY: &str = "2e6c9193-d904-4fd1-9f4e-48c3d20fcdcc";
const DEFAULT_MODEL_PATH: &str = "models/ggml-base.en-q5_1.bin";
const DEFAULT_SAMPLE_THRESHOLD: usize = 16000 * 4; // 4 seconds at 16 kHz

#[derive(Debug)]
pub struct ServiceConfig {
    pub port: u16,
    pub api_key: String,
    pub model_path: String,
    pub threads: Option<usize>,
    pub sample_threshold: usize,
}

impl ServiceConfig {
    pub fn from_env() -> Self {
        Self {
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(DEFAULT_PORT),
            api_key: std::env::var("API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.into()),
            model_path: std::env::var("WHISPER_MODEL_PATH")
                .unwrap_or_else(|_| DEFAULT_MODEL_PATH.into()),
            threads: std::env::var("WHISPER_THREADS")
                .ok()
                .and_then(|s| s.parse().ok()),
            sample_threshold: std::env::var("SAMPLE_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_SAMPLE_THRESHOLD),
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

        // if self.model_path.is_empty() {
        //     return Err("Model path cannot be empty".into());
        // }

        // if self.api_key.is_empty() {
        //     tracing::warn!("API key is empty, this is not secure!");
        // }

        Ok(())
    }
}
