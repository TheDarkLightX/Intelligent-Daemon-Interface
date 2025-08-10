use crate::{
    actuator::Actuator,
    execution::ExecutionManager,
    kernel::KernelManager,
    oracle::Oracle,
    state::{DaemonState, State},
    guards::{GuardCoordinator, freshness::FreshnessWitnessImpl, profit::ProfitGuardImpl, failure::FailureEchoImpl, cooldown::CooldownGuardImpl, risk::RiskGuardImpl},
    monitors::MonitorManager,
    fsio::{FileIO, TauRunner},
    ledger::Ledger,
};
use anyhow::Result;
use tau_core::{Config, model::*};
use tokio::time::{self, Duration};
use tracing::{info, warn, error, debug};

/// The main operational loop for the daemon with comprehensive safety integration
pub async fn run(config: Config) -> Result<()> {
    info!("Starting enhanced daemon with supporting specifications integration");

    // Initialize components
    let mut state = DaemonState::new(config.daemon.quarantine_clear_ticks);
    let oracle = Oracle::new();
    let kernel_manager = KernelManager::new();
    let execution_manager = ExecutionManager::new(&config)?;
    let actuator = Actuator::new();

    // Initialize safety components
    let freshness_witness = Box::new(FreshnessWitnessImpl::new());
    let profit_guard = Box::new(ProfitGuardImpl::new());
    let failure_echo = Box::new(FailureEchoImpl::new(config.tick.fail_hold_ticks));
    let cooldown_guard = Box::new(CooldownGuardImpl::new());
    let risk_guard = Box::new(RiskGuardImpl::new());

    let mut guard_coordinator = GuardCoordinator::new(
        freshness_witness,
        profit_guard,
        failure_echo,
        cooldown_guard,
        risk_guard,
    );

    let monitor_manager = MonitorManager::new(config.clone());
    let tau_runner = TauRunner::new(
        config.paths.tau_bin.clone(),
        config.paths.kernel_inputs.clone(),
        config.paths.kernel_outputs.clone(),
        config.daemon.tick_period_ms.saturating_mul(60), // generous timeout for kernel run
    );
    let mut ledger = Ledger::new(&config.paths.ledger_dir)?;

    // Ensure directories exist
    FileIO::ensure_dir(&config.paths.kernel_inputs)?;
    FileIO::ensure_dir(&config.paths.kernel_outputs)?;
    FileIO::ensure_dir(&config.paths.specs_root)?;

    // Validate ledger integrity
    if !ledger.validate_integrity().await? {
        error!("Ledger integrity validation failed");
        return Err(anyhow::anyhow!("Ledger integrity validation failed"));
    }

    // Recover state if possible
    if let Some(recovery_state) = ledger.recover_state().await? {
        info!("Recovered from tick {} with health: {}", recovery_state.last_tick, recovery_state.health);
        if !recovery_state.health {
            warn!("Recovering from unhealthy state, entering quarantine");
            state.enter_quarantine();
        }
    }

    let mut interval = time::interval(Duration::from_millis(config.tick.interval_ms));
    let mut tick_counter = 0u64;

    loop {
        interval.tick().await;
        tick_counter += 1;
        info!("=== Tick {} ===", tick_counter);

        state.tick();

        match state.current() {
            State::Operational => {
                info!("State is Operational. Executing enhanced logic...");

                // 1. Gather external data
                let market_snapshot = match oracle.get_market_snapshot().await {
                    Ok(snapshot) => {
                        debug!("Fetched market snapshot: {:?}", snapshot);
                        snapshot
                    }
                    Err(e) => {
                        warn!("Failed to get market snapshot: {}. Entering Quarantine.", e);
                        state.enter_quarantine();
                        continue;
                    }
                };

                // Convert MarketSnapshot to OracleSnapshot for guards (ensure timestamp is in ms)
                let oracle_snapshot = OracleSnapshot {
                    median: Fx::from_float((market_snapshot.bid_price + market_snapshot.ask_price) / 2.0),
                    age_ok: true, // Placeholder; real check done in FreshnessWitness
                    quorum_ok: true, // Placeholder; real check done in FreshnessWitness
                    sources: vec![
                        OracleSource {
                            price: Fx::from_float(market_snapshot.bid_price),
                            age_ms: 0,
                            within_tol: true,
                        }
                    ],
                    timestamp: market_snapshot.timestamp.saturating_mul(1000),
                };

                // 2. Get venue state (mock for now)
                let venue_state = VenueState {
                    position: Fx::new(0),
                    balance: Fx::new(1000000000000), // 1000 USD in scaled units
                    last_trade_tick: None,
                    pending_orders: Vec::new(),
                };

                // 3. Compute guards
                let health = Health::new(); // Will be updated after monitors
                let guards = match guard_coordinator.compute_guards(
                    &oracle_snapshot,
                    &venue_state,
                    &health,
                    &config,
                    tick_counter,
                ).await {
                    Ok(guards) => {
                        debug!("Computed guards: {:?}", guards);
                        guards
                    }
                    Err(e) => {
                        warn!("Failed to compute guards: {}. Entering Quarantine.", e);
                        state.enter_quarantine();
                        continue;
                    }
                };

                // 4. Derive core kernel inputs (price/volume/trend) and write guard inputs expected by V35
                let current_price = (market_snapshot.bid_price + market_snapshot.ask_price) / 2.0;
                let previous_price = state.last_price().unwrap_or(current_price);
                let price_eps = 0.000_000_1f64; // negligible epsilon

                // price_bit: 1 = high, 0 = low
                let price_bit = current_price > (previous_price * (1.0 + price_eps));
                // trend_bit: 1 = bullish (up), 0 = bearish (down or flat)
                let trend_bit = current_price > previous_price;
                // volume_bit: placeholder true until real volume is wired
                let volume_bit = true;

                FileIO::write_bool_atomic(&config.paths.kernel_inputs.join("price.in"), price_bit)?;
                FileIO::write_bool_atomic(&config.paths.kernel_inputs.join("volume.in"), volume_bit)?;
                FileIO::write_bool_atomic(&config.paths.kernel_inputs.join("trend.in"), trend_bit)?;

                // Guard inputs
                FileIO::write_bool_atomic(&config.paths.kernel_inputs.join("profit_guard.in"), guards.profit_guard)?;
                FileIO::write_bool_atomic(&config.paths.kernel_inputs.join("failure_echo.in"), guards.failure_echo)?;

                // 5. Run kernel
                match tau_runner.run_kernel(&config.paths.kernel_spec).await {
                    Ok(()) => {
                        debug!("Kernel execution completed");
                    }
                    Err(e) => {
                        warn!("Kernel execution failed: {}. Entering Quarantine.", e);
                        state.enter_quarantine();
                        continue;
                    }
                }

                // 6. Read kernel outputs
                let kernel_outputs = match read_kernel_outputs(&config.paths.kernel_outputs).await {
                    Ok(outputs) => {
                        debug!("Read kernel outputs: {:?}", outputs);
                        outputs
                    }
                    Err(e) => {
                        warn!("Failed to read kernel outputs: {}. Entering Quarantine.", e);
                        state.enter_quarantine();
                        continue;
                    }
                };

                // 7. Run monitors
                match tau_runner.run_monitors(&config.paths.specs_root).await {
                    Ok(()) => {
                        debug!("Monitor execution completed");
                    }
                    Err(e) => {
                        warn!("Monitor execution failed: {}. Entering Quarantine.", e);
                        state.enter_quarantine();
                        continue;
                    }
                }

                // 8. Read monitor outputs and aggregate health
                let monitor_outputs = match monitor_manager.read_outputs(tick_counter).await {
                    Ok(outputs) => {
                        debug!("Read monitor outputs: {:?}", outputs);
                        outputs
                    }
                    Err(e) => {
                        warn!("Failed to read monitor outputs: {}. Entering Quarantine.", e);
                        state.enter_quarantine();
                        continue;
                    }
                };

                let health = monitor_manager.aggregate_health(&monitor_outputs);

                // 9. Check for stall condition
                if monitor_manager.check_stalled(&monitor_outputs).await? {
                    warn!("Monitor stall detected, entering quarantine");
                    state.enter_quarantine();
                    continue;
                }

                // 10. Process kernel outputs and take action
                if health.health_ok {
                    actuator.handle_outputs(&kernel_outputs);
                } else {
                    warn!("Health check failed: {:?}, entering quarantine", health.failed_bits);
                    state.enter_quarantine();
                    continue;
                }

                // 11. Commit ledger record
                let actions = Vec::new(); // Will be populated by actuator
                if let Err(e) = ledger.append(
                    tick_counter,
                    &oracle_snapshot,
                    &kernel_outputs,
                    &monitor_outputs,
                    &guards,
                    &health,
                    None, // economics
                    actions,
                ).await {
                    error!("Failed to append ledger record: {}", e);
                }

                // Update last observed price
                state.update_last_price(current_price);

                // 12. Create periodic snapshots
                if tick_counter % 1000 == 0 {
                    if let Err(e) = ledger.create_snapshot(tick_counter).await {
                        warn!("Failed to create snapshot: {}", e);
                    }
                }
            }
            State::Quarantine => {
                info!("State is Quarantine. Skipping main logic.");
            }
        }
    }
}

/// Read kernel outputs from files
async fn read_kernel_outputs(outputs_dir: &std::path::Path) -> Result<KernelOutputs> {
    let outputs = KernelOutputs {
        state: FileIO::read_last_bool(&outputs_dir.join("state.out"))?,
        holding: FileIO::read_last_bool(&outputs_dir.join("holding.out"))?,
        buy: FileIO::read_last_bool(&outputs_dir.join("buy_signal.out"))?,
        sell: FileIO::read_last_bool(&outputs_dir.join("sell_signal.out"))?,
        oracle_fresh: FileIO::read_last_bool(&outputs_dir.join("oracle_fresh.out"))?,
        timer_b0: FileIO::read_last_bool(&outputs_dir.join("timer_b0.out"))?,
        timer_b1: FileIO::read_last_bool(&outputs_dir.join("timer_b1.out"))?,
        nonce: FileIO::read_last_bool(&outputs_dir.join("nonce.out"))?,
        entry_price: FileIO::read_last_bool(&outputs_dir.join("entry_price.out"))?,
        profit: FileIO::read_last_bool(&outputs_dir.join("profit.out"))?,
        burn: FileIO::read_last_bool(&outputs_dir.join("burn_event.out"))?,
        has_burned: FileIO::read_last_bool(&outputs_dir.join("has_burned.out"))?,
        obs_action_excl: FileIO::read_last_bool(&outputs_dir.join("obs_action_excl.out"))?,
        obs_fresh_exec: FileIO::read_last_bool(&outputs_dir.join("obs_fresh_exec.out"))?,
        obs_burn_profit: FileIO::read_last_bool(&outputs_dir.join("obs_burn_profit.out"))?,
        obs_nonce_effect: FileIO::read_last_bool(&outputs_dir.join("obs_nonce_effect.out"))?,
        progress: FileIO::read_last_bool(&outputs_dir.join("progress_flag.out"))?,
    };

    Ok(outputs)
}