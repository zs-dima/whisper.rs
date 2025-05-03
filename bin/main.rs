use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Query, State};
use whisper::config::service_params::ServiceParams;
use whisper::whisper::whisper_config::WhisperConfig;
use whisper::{config::service_config::ServiceConfig, service::whisper_service::WhisperService};

use axum::{Router, routing::get, serve};
use std::collections::HashMap;
use std::time::Duration;
use std::{net::SocketAddr, sync::Arc};
use tokio::signal;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use tracing;
use whisper_rs::{SamplingStrategy, WhisperContext};

#[tokio::main]
async fn main() {
    // Enable tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!(
                    "{}=debug,tower_http=debug,axum=trace,whisper=info",
                    env!("CARGO_CRATE_NAME")
                )
                .into()
            }),
        )
        .with(tracing_subscriber::fmt::layer().without_time())
        .init();

    // Load configuration from environment variables
    let config = ServiceConfig::from_env();
    tracing::info!("Configuration: {:?}", config);
    config.check().expect("Configuration check failed");

    // Create Whisper context
    // TODO: Default::default() from config
    let model_path = format!("models/{}", config.model_name);
    let whisper_ctx = WhisperContext::new_with_params(&model_path, Default::default())
        .expect("Failed to create WhisperContext");
    tracing::info!("Whisper model {} loaded", config.model_name);

    // Create a params object for running the Whisper model.
    let whisper_cfg = WhisperConfig {
        // The number of past samples to consider defaults to 1.
        sampling_strategy: SamplingStrategy::Greedy { best_of: 1 },
        n_threads: config.threads.map(|t| t as i32),
        translate: false, // Enable translation.
        language: None,   // Set the language to translate to to English.
        // Disable anything that prints to stdout.
        print_special: false,
        print_progress: false,
        print_realtime: false,
        print_timestamps: false,
        token_timestamps: false,
    };

    let service_params = Arc::new(ServiceParams {
        whisper_ctx: whisper_ctx,
        whisper_cfg: whisper_cfg,
        sample_threshold: config.sample_threshold,
        api_key: config.api_key,
        idle_flush: Duration::from_secs(5),
        max_buffer_ms: config.max_buffer_ms,
        max_service_threads: config.max_service_threads,
        lookback_ms: config.lookback_ms,
        vad_thold: config.vad_thold,
    });

    let whisper_service = Arc::new(WhisperService::new(service_params));

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        // .route("/ws", get(WhisperService::ws_handler))
        // .route("/webrtc", axum::routing::post(webrtc_handler))
        .route(
            "/ws",
            get(
                move |ws: WebSocketUpgrade,
                      query: Query<HashMap<String, String>>,
                      state: State<Arc<WhisperService>>| {
                    // Explicitly call the handler with your parameters
                    WhisperService::ws_handler(ws, query, state)
                },
            ),
        )
        .layer((
            TraceLayer::new_for_http(),
            // Graceful shutdown will wait for outstanding requests to complete. Add a timeout so requests don't hang forever.
            TimeoutLayer::new(Duration::from_secs(10)),
        ))
        .with_state(whisper_service);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Failed to bind to address {}: {}", addr, e);
            std::process::exit(1);
        }
    };
    tracing::info!("Listening on {}", addr);

    serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    // Wait for Ctrl+C signal for graceful shutdown
    let ctrl_c = async {
        match signal::ctrl_c().await {
            Ok(()) => tracing::info!("Ctrl+C pressed, shutting down"),
            Err(err) => tracing::error!("Failed to listen for Ctrl+C: {}", err),
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
                tracing::info!("Received terminate signal");
            }
            Err(err) => tracing::error!("Failed to install signal handler: {}", err),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
