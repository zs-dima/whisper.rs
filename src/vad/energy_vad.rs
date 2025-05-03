use std::sync::{Arc, atomic::Ordering};

use crate::service::engine::shared_state::SharedState;

/// Voice Activity Detector using energy-based algorithm
pub struct EnergyVad {
    /// Running sum of all sample energy: Σ|x|
    energy_total: f64,
    /// Running sum of energy in the lookback window: Σ|x| over trailing window
    energy_tail: f64,
    /// Shared state for marking boundaries
    shared: Arc<SharedState>,
    vad_thold: f32,
    lookback_samples: usize,
}

impl EnergyVad {
    pub fn new(shared: Arc<SharedState>, lookback_samples: usize, vad_thold: f32) -> Self {
        Self {
            energy_total: 0.0,
            energy_tail: 0.0,
            shared,
            vad_thold,
            lookback_samples,
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
        if buffer_len >= self.lookback_samples {
            let energy_all = (self.energy_total / buffer_len as f64) as f32;
            let energy_last = (self.energy_tail / self.lookback_samples as f64) as f32;

            // Detect silence boundary using energy ratio
            if energy_last < self.vad_thold * energy_all {
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
