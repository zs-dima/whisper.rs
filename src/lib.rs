pub mod config {
    pub mod service_config;
    pub mod service_params;
}

pub mod service {
    pub mod whisper_service;
    pub mod shared_state {
        pub mod shared_state;
    }
}

pub mod vad {
    pub mod energy_vad;
}

pub mod whisper {
    pub mod whisper_callback;
    pub mod whisper_config;
    pub mod whisper_helper;
}
