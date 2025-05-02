use whisper_rs::{FullParams, SamplingStrategy};

#[derive(Clone)]
pub struct WhisperConfig {
    pub sampling_strategy: SamplingStrategy,
    /// Defaults to min(4, std::thread::hardware_concurrency()).
    pub n_threads: Option<i32>,
    // Enable translation.
    pub translate: bool,
    // Set the language to translate to to English. set_language(Some("en"))
    pub language: Option<String>,
    // Disable anything that prints to stdout.
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
    // Enable token level timestamps
    pub token_timestamps: bool,
}

impl WhisperConfig {
    pub fn to_full_params(&self) -> FullParams {
        let mut params = FullParams::new(self.sampling_strategy.clone());
        if let Some(ref threads) = self.n_threads {
            params.set_n_threads(*threads);
        }
        if let Some(ref lang) = self.language {
            params.set_language(Some(&lang));
        }
        params.set_translate(self.translate);
        params.set_print_special(self.print_special);
        params.set_print_progress(self.print_progress);
        params.set_print_realtime(self.print_realtime);
        params.set_print_timestamps(self.print_timestamps);
        params.set_token_timestamps(self.token_timestamps);
        params
    }
}
