use std::{
    ffi::{CStr, c_void},
    sync::atomic::Ordering,
};
use tokio::sync::mpsc::{self, error::TrySendError};
use whisper_rs::whisper_rs_sys;

use crate::service::engine::shared_state::{CppCallbackData, Transcript};

/// Helper functions for safer C API interactions
pub struct WhisperHelper;

impl WhisperHelper {
    /// Safely extract text from whisper state
    pub unsafe fn extract_text(
        st: *mut whisper_rs_sys::whisper_state,
        segment_idx: i32,
    ) -> Option<String> {
        let txt_ptr =
            unsafe { whisper_rs_sys::whisper_full_get_segment_text_from_state(st, segment_idx) };
        if txt_ptr.is_null() {
            tracing::warn!("Null text pointer for segment {}", segment_idx);
            return None;
        }

        match unsafe { CStr::from_ptr(txt_ptr) }.to_str() {
            Ok(s) => Some(s.to_owned()),
            Err(err) => {
                tracing::error!("Failed to convert segment text to UTF-8: {}", err);
                None
            }
        }
    }

    /// Safely get segment end time
    /// Returns None if the time is negative or if the segment index is invalid
    pub unsafe fn get_segment_time(
        st: *mut whisper_rs_sys::whisper_state,
        segment_idx: i32,
    ) -> Option<u64> {
        let t1 = unsafe { whisper_rs_sys::whisper_full_get_segment_t1_from_state(st, segment_idx) };

        if t1 < 0 {
            tracing::warn!("Negative time for segment {}: {}", segment_idx, t1);
            return None;
        }

        Some(t1 as u64)
    }

    /// Send the transcript to the channel with a timeout
    /// Returns true if successful, false if the channel is closed or full after max attempts
    pub fn try_send_transcript(channel: &mpsc::Sender<Transcript>, seq: usize, text: &str) -> bool {
        const MAX_ATTEMPTS: usize = 3;
        let mut attempts = 0;

        loop {
            match channel.try_send(Transcript {
                seq,
                text: text.to_owned(),
            }) {
                Ok(()) => return true,
                Err(TrySendError::Full(t)) => {
                    attempts += 1;
                    if attempts >= MAX_ATTEMPTS {
                        tracing::warn!(
                            "Channel full, dropping transcript after {} attempts",
                            attempts
                        );
                        return false;
                    }
                    // Try to replace the existing message in the channel
                    let _ = channel.try_send(t);
                }
                Err(TrySendError::Closed(_)) => {
                    tracing::debug!("Channel closed, client disconnected");
                    return false;
                }
            }
        }
    }

    /// Main C callback for Whisper
    pub unsafe extern "C" fn whisper_callback(
        _: *mut whisper_rs_sys::whisper_context,
        st: *mut whisper_rs_sys::whisper_state,
        _: i32,
        user: *mut c_void,
    ) {
        if user.is_null() || st.is_null() {
            return;
        }

        let data = unsafe { &*(user as *const CppCallbackData) };

        // Get the index of the last segment
        let last_segment_idx =
            match unsafe { whisper_rs_sys::whisper_full_n_segments_from_state(st) } {
                0 => return, // No segments available
                n => n - 1,  // Last segment index
            };

        // Extract the text from the last segment
        if let Some(txt) = unsafe { Self::extract_text(st, last_segment_idx) } {
            let seq = data.flush_seq * 100 + data.boundary.next_seq.fetch_add(1, Ordering::Relaxed);

            // Send the transcript to the channel with a timeout
            if !Self::try_send_transcript(&data.out, seq, &txt) {
                return;
            }

            // Check if the last segment ends with a sentence boundary
            if txt.trim_end().ends_with(['.', '!', '?']) {
                if let Some(t1) = unsafe { Self::get_segment_time(st, last_segment_idx) } {
                    // 1 ms = 16 samples for 16 kHz audio
                    data.boundary
                        .last_sentence_end
                        .store(t1 * 16, Ordering::Release);
                }
            }
        }
    }
}
