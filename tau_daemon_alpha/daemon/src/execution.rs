use anyhow::Result;
use tau_core::{model::*, Config};
use tracing::{debug, info};

use crate::fsio::{FileIO, TauRunner};

/// Manages the execution of the Tau kernel and related processes.
#[derive(Debug)]
pub struct ExecutionManager {
    config: Config,
    tau_runner: TauRunner,
}

impl ExecutionManager {
    /// Creates a new `ExecutionManager`.
    pub fn new(config: &Config) -> Result<Self> {
        info!("Initializing execution manager.");
        Ok(Self {
            config: config.clone(),
            tau_runner: TauRunner::new(
                config.paths.tau_bin.clone(),
                config.paths.kernel_inputs.clone(),
                config.paths.kernel_outputs.clone(),
                config.daemon.tick_period_ms.saturating_mul(2),
            ),
        })
    }

    /// Executes a single tick of the kernel.
    pub async fn execute_tick(&self, kernel_inputs: &KernelInputs) -> Result<KernelOutputs> {
        info!("Executing kernel tick...");

        self.write_inputs(kernel_inputs)?;
        self.tau_runner
            .run_kernel(&self.config.paths.kernel_spec)
            .await?;
        let outputs = self.read_kernel_outputs()?;

        info!(?outputs, "Kernel execution completed.");
        Ok(outputs)
    }

    fn write_inputs(&self, kernel_inputs: &KernelInputs) -> Result<()> {
        let inputs_dir = &self.config.paths.kernel_inputs;
        FileIO::write_bool_atomic(&inputs_dir.join("price.in"), kernel_inputs.price_bit)?;
        FileIO::write_bool_atomic(&inputs_dir.join("volume.in"), kernel_inputs.volume_bit)?;
        FileIO::write_bool_atomic(&inputs_dir.join("trend.in"), kernel_inputs.trend_bit)?;
        FileIO::write_bool_atomic(
            &inputs_dir.join("profit_guard.in"),
            kernel_inputs.profit_guard,
        )?;
        FileIO::write_bool_atomic(
            &inputs_dir.join("failure_echo.in"),
            kernel_inputs.failure_echo,
        )?;
        FileIO::write_bool_atomic(
            &inputs_dir.join("oracle_fresh.in"),
            kernel_inputs.oracle_fresh,
        )?;
        Ok(())
    }

    /// Reads kernel outputs from files
    fn read_kernel_outputs(&self) -> Result<KernelOutputs> {
        let outputs_dir = &self.config.paths.kernel_outputs;
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
}
