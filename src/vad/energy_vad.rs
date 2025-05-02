use std::sync::{Arc, atomic::Ordering};

use crate::service::shared_state::shared_state::SharedState;

/// Constants for Voice Activity Detection
pub const SAMPLE_RATE: usize = 16_000; // Hz
pub const LOOKBACK_MS: usize = 200; // look at last 200 ms
pub const LOOKBACK_SAMPLES: usize = SAMPLE_RATE * LOOKBACK_MS / 1_000; // 4,800
pub const VAD_THOLD: f32 = 0.35; // energy_last < 0.35 × energy_all ⇒ silence

/// Voice Activity Detector using energy-based algorithm
pub struct EnergyVad {
    /// Running sum of all sample energy: Σ|x|
    energy_total: f64,
    /// Running sum of energy in the lookback window: Σ|x| over trailing window
    energy_tail: f64,
    /// Shared state for marking boundaries
    shared: Arc<SharedState>,
}

impl EnergyVad {
    pub fn new(shared: Arc<SharedState>) -> Self {
        Self {
            energy_total: 0.0,
            energy_tail: 0.0,
            shared,
        }
    }

    /// Process a new audio sample
    /// Returns true if the sample was determined to be part of a silence boundary
    pub fn process_sample(
        &mut self,
        sample: f32,
        buffer_len: usize,
        old_sample_energy: Option<f64>,
    ) -> bool {
        let abs = sample.abs() as f64;
        self.energy_total += abs;
        self.energy_tail += abs;

        // Subtract old sample energy from the lookback window if provided
        if let Some(old_energy) = old_sample_energy {
            self.energy_tail -= old_energy;
        }

        // Only perform VAD check if we have enough samples
        if buffer_len >= LOOKBACK_SAMPLES {
            let energy_all = (self.energy_total / buffer_len as f64) as f32;
            let energy_last = (self.energy_tail / LOOKBACK_SAMPLES as f64) as f32;

            // Detect silence boundary using energy ratio
            if energy_last < VAD_THOLD * energy_all {
                self.shared
                    .last_sentence_end
                    .fetch_max(buffer_len as u64, Ordering::Release);
                return true;
            }
        }

        false
    }

    /// Reset the VAD state
    pub fn reset(&mut self) {
        self.energy_total = 0.0;
        self.energy_tail = 0.0;
    }
}
