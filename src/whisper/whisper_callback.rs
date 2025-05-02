use std::ffi::c_void;
use std::sync::Arc;

use tokio::sync::mpsc;
use whisper_rs::FullParams;
use whisper_rs::whisper_rs_sys;

use crate::service::shared_state::shared_state::{CppCallbackData, SharedState, Transcript};

/// Safe wrapper around Whisper C API callback data
pub struct WhisperCallback {
    ptr: *mut c_void,
    consumed: bool,
}

impl WhisperCallback {
    /// Create a new callback and convert it to a raw pointer for C API
    pub fn new(shared: Arc<SharedState>, out: mpsc::Sender<Transcript>, flush_seq: usize) -> Self {
        let data = Box::new(CppCallbackData {
            boundary: shared,
            out,
            flush_seq,
        });

        Self {
            ptr: Box::into_raw(data) as *mut c_void,
            consumed: false,
        }
    }

    /// Get the raw pointer for passing to C API
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Set up callback function in FullParams
    pub fn setup_callback(
        &self,
        fp: &mut FullParams,
        callback: unsafe extern "C" fn(
            *mut whisper_rs_sys::whisper_context,
            *mut whisper_rs_sys::whisper_state,
            i32,
            *mut c_void,
        ),
    ) {
        // Safety: The pointer is valid and owned by this struct
        unsafe {
            fp.set_new_segment_callback(Some(callback));
            fp.set_new_segment_callback_user_data(self.ptr);
        }
    }

    /// Consume the callback data and clean up resources
    pub fn consume(mut self, _: &SharedState) -> bool {
        if self.consumed || self.ptr.is_null() {
            return false;
        }

        // Safety: We're reclaiming ownership of the Box we created
        unsafe {
            let data = Box::from_raw(self.ptr as *mut CppCallbackData);
            data.boundary
                .active_jobs
                .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        }

        self.consumed = true;
        true
    }
}

impl Drop for WhisperCallback {
    fn drop(&mut self) {
        // Safety: Only clean up if not already consumed and pointer is valid
        if !self.consumed && !self.ptr.is_null() {
            unsafe {
                let data = Box::from_raw(self.ptr as *mut CppCallbackData);
                data.boundary
                    .active_jobs
                    .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            }
        }
    }
}
