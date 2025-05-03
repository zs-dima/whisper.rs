pub mod service {
    pub mod config {
        pub mod service_config;
        pub mod service_params;
    }

    pub mod engine {
        pub mod shared_state;
    }

    pub mod whisper_service;
}

pub mod vad {
    pub mod energy_vad;
}

pub mod whisper {
    pub mod whisper_callback;
    pub mod whisper_config;
    pub mod whisper_engine;
    pub mod whisper_helper;
}
