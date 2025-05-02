pub mod config {
    pub mod service_config;
    pub mod service_params;
    pub mod whisper_config;
}

pub mod service {
    pub mod whisper_service;
    pub mod shared_state {
        pub mod shared_state;
    }
}
