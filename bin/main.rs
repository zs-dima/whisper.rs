use axum::{
    Router,
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    serve,
};
use futures::{SinkExt, StreamExt};
use serde_json::json;
use std::time::Duration;
use std::{ffi::CStr, net::SocketAddr, os::raw::c_void, sync::Arc};
use tokio::signal;
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use tracing;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, whisper_rs_sys};

const DEFAULT_PORT: u16 = 3030;
const DEFAULT_MODEL_PATH: &str = "models/ggml-base.en-q5_1.bin";
const DEFAULT_SAMPLE_THRESHOLD: usize = 16000 * 4; // 4 seconds at 16 kHz

pub struct AppState {
    pub whisper_ctx: WhisperContext,
    pub whisper_params: WhisperConfig,
    pub sample_threshold: usize,
}

#[derive(Clone)]
pub struct WhisperConfig {
    sampling_strategy: SamplingStrategy,
    /// Defaults to min(4, std::thread::hardware_concurrency()).
    n_threads: Option<i32>,
    // Enable translation.
    translate: bool,
    // Set the language to translate to to English. set_language(Some("en"))
    language: Option<String>,
    // Disable anything that prints to stdout.
    print_special: bool,
    print_progress: bool,
    print_realtime: bool,
    print_timestamps: bool,
    // Enable token level timestamps
    token_timestamps: bool,
}

impl WhisperConfig {
    fn to_full_params(&self) -> FullParams {
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

struct Config {
    model_path: String,
    threads: Option<usize>,
    port: u16,
    sample_threshold: usize,
}

impl Config {
    fn from_env() -> Self {
        Self {
            model_path: std::env::var("WHISPER_MODEL_PATH")
                .unwrap_or_else(|_| DEFAULT_MODEL_PATH.into()),
            threads: std::env::var("WHISPER_THREADS")
                .ok()
                .and_then(|s| s.parse().ok()),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(DEFAULT_PORT),
            sample_threshold: std::env::var("SAMPLE_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_SAMPLE_THRESHOLD),
        }
    }
}

#[tokio::main]
async fn main() {
    // Enable tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!(
                    "{}=debug,tower_http=debug,axum=trace",
                    env!("CARGO_CRATE_NAME")
                )
                .into()
            }),
        )
        .with(tracing_subscriber::fmt::layer().without_time())
        .init();

    // Load configuration from environment variables
    let config = Config::from_env();

    // Create Whisper context
    // TODO: Default::default() from config
    let whisper_ctx = WhisperContext::new_with_params(&config.model_path, Default::default())
        .expect("Failed to create WhisperContext");
    tracing::info!("Whisper model {} loaded", config.model_path);

    if let Some(threads) = config.threads {
        // Check if the number of threads is greater than the number of available CPUs
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1); // fallback to 1 if error
        if threads > available {
            panic!("Requested more threads ({threads}) than available ({available})");
        }
        tracing::info!("Using {} threads for Whisper", threads);
    }

    // Create a params object for running the Whisper model.
    let whisper_params = WhisperConfig {
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

    let app_state = AppState {
        whisper_ctx: whisper_ctx,
        whisper_params: whisper_params,
        sample_threshold: config.sample_threshold,
    };

    let app = Router::new()
        .route("/health", get(|| async { "OK" }))
        .route("/ws", get(ws_handler))
        .layer((
            TraceLayer::new_for_http(),
            // Graceful shutdown will wait for outstanding requests to complete. Add a timeout so requests don't hang forever.
            TimeoutLayer::new(Duration::from_secs(10)),
        ))
        .with_state(Arc::new(app_state));

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

// Handler to upgrade HTTP to WebSocket
async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::State(app_state): axum::extract::State<Arc<AppState>>,
) -> impl IntoResponse {
    // On upgrade, call handle_socket
    ws.on_upgrade(move |socket| handle_socket(socket, app_state))
}

// Asynchronously handle a WebSocket connection
async fn handle_socket(socket: WebSocket, app_state_arc: Arc<AppState>) {
    tracing::info!("WebSocket connection established");

    // Split the socket into send and receive halves for concurrent handling
    let (mut sender, mut receiver) = socket.split();

    // Channel for sending partial transcripts from blocking thread to async writer
    let (tx, mut rx) = unbounded_channel::<String>();

    // Task to read incoming audio messages
    let read_task = {
        let tx = tx.clone();
        tokio::spawn(async move {
            // Audio buffers (samples as f32)
            let mut buffer: Vec<f32> = Vec::new();
            let sample_threshold: usize = app_state_arc.sample_threshold;

            // Receive loop
            while let Some(Ok(msg)) = receiver.next().await {
                // Process incoming message
                if process_websocket_message(
                    msg,
                    &mut buffer,
                    &tx,
                    &app_state_arc,
                    sample_threshold,
                )
                .await
                {
                    break; // Close connection if needed
                }
            }

            if !buffer.is_empty() {
                let to_process = std::mem::take(&mut buffer);
                let tx_clone = tx.clone();
                let app_state = app_state_arc.clone();
                spawn_transcription(to_process, tx_clone, app_state);
            }

            // Drop sender clone to close channel when done
            drop(tx);
        })
    };

    // Task to send transcripts back to client
    let write_task = tokio::spawn(async move {
        while let Some(partial) = rx.recv().await {
            let json_msg = json!({ "text": partial });
            if sender
                .send(Message::Text(json_msg.to_string().into()))
                .await
                .is_err()
            {
                break; // client disconnected or error
            } else {
                tracing::info!("Sent partial transcript to client");
            }
        }
        tracing::info!("No more transcripts to send, write_task ending");
    });

    // Await both tasks to complete
    let _ = tokio::join!(read_task, write_task);
    tracing::info!("WebSocket handler done");
}

async fn process_websocket_message(
    msg: Message,
    buffer: &mut Vec<f32>,
    tx: &UnboundedSender<String>,
    app_state_arc: &Arc<AppState>,
    sample_threshold: usize,
) -> bool {
    // returns true if connection should close
    match msg {
        Message::Binary(data) => {
            // Convert binary to f32 samples (little endian)
            for chunk in data.chunks_exact(4) {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                buffer.push(sample);
            }
            // If we've reached a full chunk, process it
            if buffer.len() >= sample_threshold {
                spawn_transcription(std::mem::take(buffer), tx.clone(), app_state_arc.clone());
            }
            false
        }
        Message::Close(_) => {
            tracing::info!("WebSocket connection closed by client");
            true
        }
        // Ignore other message types (e.g. ping/pong/text)
        _ => false,
    }
}

fn spawn_transcription(samples: Vec<f32>, tx: UnboundedSender<String>, app_state: Arc<AppState>) {
    tracing::debug!("Received {} samples, spawning transcription", samples.len());
    tokio::task::spawn_blocking(move || {
        transcribe_chunk(
            &app_state.whisper_ctx,
            app_state.whisper_params.to_full_params(),
            samples,
            tx,
        );
    });
}

// Transcribe a chunk of audio samples (in a blocking context)
// Uses whisper-rs to run transcription and sends partial results via `sender`.
fn transcribe_chunk(
    ctx: &WhisperContext,
    mut params: FullParams,
    samples: Vec<f32>,
    sender: UnboundedSender<String>,
) {
    let mut state = ctx.create_state().expect("Failed to create WhisperState");

    // Set up callback for new segments (unsafe FFI)
    // Prepare a raw pointer to the sender for user_data
    let sender_box = Box::new(sender);
    let sender_ptr = Box::into_raw(sender_box) as *mut c_void;
    unsafe {
        // Register callback function and user_data
        params.set_new_segment_callback(Some(whisper_callback));
        params.set_new_segment_callback_user_data(sender_ptr);
    }

    // Run transcription (blocking)
    match state.full(params, &samples) {
        Ok(_) => tracing::debug!("Transcription completed successfully"),
        Err(e) => {
            tracing::error!("Whisper transcription failed: {}", e);
            // Send error notification to client
            // TODO let _ = sender.send("Error during transcription".to_string());
        }
    }

    // After transcription, reclaim the Box and let it drop
    unsafe {
        let _ = Box::from_raw(sender_ptr as *mut UnboundedSender<String>);
    }
}

// Whisper callback: called on each new text segment (C-style)
// user_data is the pointer we passed (an UnboundedSender<String>)
unsafe extern "C" fn whisper_callback(
    _ctx: *mut whisper_rs_sys::whisper_context,
    state: *mut whisper_rs_sys::whisper_state,
    _segment: i32,
    user_data: *mut c_void,
) {
    // Check for null pointer
    if user_data.is_null() {
        tracing::error!("Null user_data pointer in whisper callback");
        return;
    }

    // Recover the sender from user_data pointer
    let sender = unsafe { &*(user_data as *mut UnboundedSender<String>) };
    // Determine the latest segment index
    let n = unsafe { whisper_rs_sys::whisper_full_n_segments_from_state(state) } - 1;
    // Get the segment text (as C string)
    let ptr = unsafe { whisper_rs_sys::whisper_full_get_segment_text_from_state(state, n) };
    if ptr.is_null() {
        return;
    }
    // Convert C string to Rust String
    let c_str = unsafe { CStr::from_ptr(ptr) };
    if let Ok(text) = c_str.to_str() {
        // Send the partial text via channel (ignore if receiver dropped)
        let _ = sender.send(text.to_string());
    }
}
