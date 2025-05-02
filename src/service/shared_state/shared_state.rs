use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize},
};

use tokio::sync::mpsc;

///  Shared data between async reader and Whisper callback

///
///  Shared sentence boundary flag
///
/// `SharedState` is visible from the Web-Socket reader task **and** from the Whisper
/// segment callback that runs on a background thread.  The callback records the sample
/// index of the most recent sentence-final segment in `last_sentence_end`.
#[derive(Debug)]
pub struct SharedState {
    pub last_sentence_end: AtomicU64, // absolute index in samples
    pub next_seq: AtomicUsize,
    pub active_jobs: AtomicUsize,
    pub flush_seq: AtomicUsize,
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            last_sentence_end: AtomicU64::new(0),
            next_seq: AtomicUsize::new(0),
            active_jobs: AtomicUsize::new(0),
            flush_seq: AtomicUsize::new(1),
        }
    }
}

///
///  Shared state for the callback
///
/// Data passed from the Whisper callback back to Rust.
pub struct CppCallbackData {
    pub boundary: Arc<SharedState>,
    pub out: mpsc::Sender<Transcript>,
    pub flush_seq: usize,
}

#[derive(Debug)]
pub struct Transcript {
    pub seq: usize,
    pub text: String,
}
