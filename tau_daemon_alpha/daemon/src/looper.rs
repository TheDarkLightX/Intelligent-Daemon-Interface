use crate::{
    actuator::Actuator,
    execution::ExecutionManager,
    fsio::{FileIO, TauRunner},
    guards::{
        cooldown::CooldownGuardImpl, failure::FailureEchoImpl, freshness::FreshnessWitnessImpl,
        profit::ProfitGuardImpl, risk::RiskGuardImpl, GuardCoordinator,
    },
    kernel::KernelManager,
    ledger::Ledger,
    monitors::MonitorManager,
    oracle::Oracle,
    state::{DaemonState, State},
};
use anyhow::Result;
use tau_core::{model::*, Config};
use tokio::time::{self, Duration};
use tracing::{debug, error, info, warn};

/// The main operational loop for the daemon with comprehensive safety integration
pub async fn run(config: Config) -> Result<()> {
    info!("Starting enhanced daemon with supporting specifications integration");

    // Initialize components
    let mut state = DaemonState::new(config.daemon.quarantine_clear_ticks);
    let oracle = Oracle::new(&config.paths.data_dir)?;
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
    FileIO::ensure_dir(&config.paths.data_dir)?;

    // Validate ledger integrity
    if !ledger.validate_integrity().await? {
        error!("Ledger integrity validation failed");
        return Err(anyhow::anyhow!("Ledger integrity validation failed"));
    }

    // Recover state if possible
    if let Some(recovery_state) = ledger.recover_state().await? {
        info!(
            "Recovered from tick {} with health: {}",
            recovery_state.last_tick, recovery_state.health
        );
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

                let mid_price = (market_snapshot.bid_price + market_snapshot.ask_price) / 2.0;

                // Convert MarketSnapshot to OracleSnapshot for guards (ensure timestamp is in ms)
                let mut oracle_snapshot = OracleSnapshot {
                    median: Fx::from_float(mid_price),
                    age_ok: false,
                    quorum_ok: false,
                    sources: market_snapshot.sources.clone().unwrap_or_else(|| {
                        vec![
                            OracleSource {
                                price: Fx::from_float(market_snapshot.bid_price),
                                age_ms: 0,
                                within_tol: true,
                            },
                            OracleSource {
                                price: Fx::from_float(market_snapshot.ask_price),
                                age_ms: 0,
                                within_tol: true,
                            },
                            OracleSource {
                                price: Fx::from_float(mid_price),
                                age_ms: 0,
                                within_tol: true,
                            },
                        ]
                    }),
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
                let guards = match guard_coordinator
                    .compute_guards(
                        &oracle_snapshot,
                        &venue_state,
                        &health,
                        &config,
                        tick_counter,
                    )
                    .await
                {
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

                // 4. Build kernel inputs from guards + market data
                let guard_diagnostics = guard_coordinator.get_diagnostics().await?;
                oracle_snapshot.age_ok = guard_diagnostics.age_ok;
                oracle_snapshot.quorum_ok = guard_diagnostics.quorum_ok;

                let kernel_inputs = match kernel_manager.prepare_inputs(
                    &market_snapshot,
                    state.last_price(),
                    &guards,
                    &config,
                ) {
                    Ok(inputs) => inputs,
                    Err(e) => {
                        warn!(
                            "Failed to prepare kernel inputs: {}. Entering Quarantine.",
                            e
                        );
                        state.enter_quarantine();
                        continue;
                    }
                };

                // 5. Run kernel and collect outputs
                let kernel_outputs = match execution_manager.execute_tick(&kernel_inputs).await {
                    Ok(outputs) => {
                        debug!("Kernel outputs: {:?}", outputs);
                        outputs
                    }
                    Err(e) => {
                        warn!("Kernel execution failed: {}. Entering Quarantine.", e);
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
                        warn!(
                            "Failed to read monitor outputs: {}. Entering Quarantine.",
                            e
                        );
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
                let actions = if health.health_ok {
                    actuator.handle_outputs(&kernel_outputs)
                } else {
                    warn!(
                        "Health check failed: {:?}, entering quarantine",
                        health.failed_bits
                    );
                    state.enter_quarantine();
                    continue;
                };

                // 11. Commit ledger record
                if let Err(e) = ledger
                    .append(
                        tick_counter,
                        &oracle_snapshot,
                        &kernel_outputs,
                        &monitor_outputs,
                        &guards,
                        &health,
                        None, // economics
                        actions,
                    )
                    .await
                {
                    error!("Failed to append ledger record: {}", e);
                }

                // Update last observed price
                state.update_last_price(mid_price);

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
