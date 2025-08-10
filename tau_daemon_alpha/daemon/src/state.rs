use tracing::info;

/// Represents the operational state of the daemon.
#[derive(Debug, PartialEq, Eq)]
pub enum State {
    /// The daemon is running normally.
    Operational,
    /// The daemon has encountered an error and is in a safe, non-trading mode.
    Quarantine,
}

/// Manages the daemon's state machine.
#[derive(Debug)]
pub struct DaemonState {
    current: State,
    quarantine_ticks_remaining: u32,
    quarantine_clear_ticks: u32,
    last_price: Option<f64>,
}

impl DaemonState {
    /// Creates a new `DaemonState`, starting in `Operational` mode.
    pub fn new(quarantine_clear_ticks: u32) -> Self {
        Self {
            current: State::Operational,
            quarantine_ticks_remaining: 0,
            quarantine_clear_ticks,
            last_price: None,
        }
    }

    /// Returns the current state.
    pub fn current(&self) -> &State {
        &self.current
    }

    /// Enters the `Quarantine` state.
    pub fn enter_quarantine(&mut self) {
        info!("Entering Quarantine state for {} ticks.", self.quarantine_clear_ticks);
        self.current = State::Quarantine;
        self.quarantine_ticks_remaining = self.quarantine_clear_ticks;
    }

    /// Processes a tick, potentially clearing quarantine if conditions are met.
    pub fn tick(&mut self) {
        if self.current == State::Quarantine {
            self.quarantine_ticks_remaining -= 1;
            info!("In Quarantine. Ticks remaining to clear: {}", self.quarantine_ticks_remaining);
            if self.quarantine_ticks_remaining == 0 {
                self.current = State::Operational;
                info!("Quarantine cleared. Returning to Operational state.");
            }
        }
    }

    /// Gets the last observed price if available.
    pub fn last_price(&self) -> Option<f64> {
        self.last_price
    }

    /// Updates the last observed price.
    pub fn update_last_price(&mut self, price: f64) {
        self.last_price = Some(price);
    }
}
