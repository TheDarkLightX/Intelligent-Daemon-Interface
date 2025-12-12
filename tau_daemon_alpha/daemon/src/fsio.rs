use anyhow::Result;
use std::fs;
#[cfg(target_family = "unix")]
use std::os::unix::fs as unix_fs;
use std::path::{Path, PathBuf};
use tokio::process::Command as TokioCommand;
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info};

/// Atomic file operations for Tau integration
pub struct FileIO;

impl FileIO {
    /// Atomically write a boolean value to a file
    pub fn write_bool_atomic(path: &Path, value: bool) -> Result<()> {
        let content = if value { "1\n" } else { "0\n" };
        Self::write_atomic(path, content)
    }

    /// Atomically write content to a file
    pub fn write_atomic(path: &Path, content: &str) -> Result<()> {
        let temp_path = path.with_extension("tmp");

        // Write to temporary file
        fs::write(&temp_path, content)?;

        // Sync to disk
        if let Ok(file) = fs::File::open(&temp_path) {
            file.sync_all()?;
        }

        // Atomic rename
        fs::rename(&temp_path, path)?;

        debug!("Atomically wrote to {:?}", path);
        Ok(())
    }

    /// Append a boolean value to a file
    pub fn append_bool(path: &Path, value: bool) -> Result<()> {
        let content = if value { "1\n" } else { "0\n" };
        Self::append(path, content)
    }

    /// Append content to a file
    pub fn append(path: &Path, content: &str) -> Result<()> {
        use std::io::Write;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        file.write_all(content.as_bytes())?;
        file.flush()?;

        debug!("Appended to {:?}", path);
        Ok(())
    }

    /// Read the last line of a file as a boolean
    pub fn read_last_bool(path: &Path) -> Result<bool> {
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();

        if let Some(last_line) = lines.last() {
            match last_line.trim() {
                "1" => Ok(true),
                "0" => Ok(false),
                _ => Err(anyhow::anyhow!("Invalid boolean value: {}", last_line)),
            }
        } else {
            Err(anyhow::anyhow!("Empty file: {:?}", path))
        }
    }

    /// Read the last line of a file
    pub fn read_last_line(path: &Path) -> Result<String> {
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();

        if let Some(last_line) = lines.last() {
            Ok(last_line.to_string())
        } else {
            Err(anyhow::anyhow!("Empty file: {:?}", path))
        }
    }

    /// Ensure directory exists
    pub fn ensure_dir(path: &Path) -> Result<()> {
        if !path.exists() {
            fs::create_dir_all(path)?;
            debug!("Created directory: {:?}", path);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn write_and_read_bool_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("flag.txt");

        FileIO::write_bool_atomic(&path, true).expect("write should succeed");
        assert!(FileIO::read_last_bool(&path).expect("read should succeed"));

        FileIO::write_bool_atomic(&path, false).expect("overwrite should succeed");
        assert!(!FileIO::read_last_bool(&path).expect("read should succeed"));
    }

    #[test]
    fn append_and_read_last_bool() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("append.txt");

        FileIO::append_bool(&path, true).expect("append should succeed");
        FileIO::append_bool(&path, false).expect("append should succeed");

        assert!(!FileIO::read_last_bool(&path).expect("read should succeed"));
    }

    #[test]
    fn read_last_bool_rejects_invalid_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid.txt");

        FileIO::write_atomic(&path, "not-a-bool\n").expect("write should succeed");
        let err = FileIO::read_last_bool(&path).expect_err("should fail on invalid content");
        assert!(err.to_string().contains("Invalid boolean value"));
    }
}

/// Tau process runner
pub struct TauRunner {
    tau_bin: PathBuf,
    inputs_dir: PathBuf,
    outputs_dir: PathBuf,
    timeout_ms: u64,
}

impl TauRunner {
    pub fn new(
        tau_bin: PathBuf,
        inputs_dir: PathBuf,
        outputs_dir: PathBuf,
        timeout_ms: u64,
    ) -> Self {
        // Resolve tau_bin to an absolute path to avoid dependence on current_dir
        let resolved_tau_bin = if tau_bin.is_absolute() {
            tau_bin
        } else {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            cwd.join(tau_bin)
        };

        Self {
            tau_bin: resolved_tau_bin,
            inputs_dir,
            outputs_dir,
            timeout_ms,
        }
    }

    /// Run a Tau specification
    pub async fn run_spec(&self, spec_path: &Path) -> Result<()> {
        debug!("Running Tau spec: {:?}", spec_path);

        // Ensure working directory is the spec's directory so relative ifile/ofile paths resolve
        let current_dir = spec_path.parent().unwrap_or(Path::new("."));

        // Ensure "inputs" and "outputs" are present next to the spec, pointing to configured dirs
        #[cfg(target_family = "unix")]
        {
            let link_inputs = current_dir.join("inputs");
            if !link_inputs.exists() {
                if let Err(e) = unix_fs::symlink(&self.inputs_dir, &link_inputs) {
                    error!(
                        "Failed to create inputs symlink {:?} -> {:?}: {}",
                        link_inputs, self.inputs_dir, e
                    );
                }
            }

            let link_outputs = current_dir.join("outputs");
            if !link_outputs.exists() {
                if let Err(e) = unix_fs::symlink(&self.outputs_dir, &link_outputs) {
                    error!(
                        "Failed to create outputs symlink {:?} -> {:?}: {}",
                        link_outputs, self.outputs_dir, e
                    );
                }
            }
        }

        let output = match timeout(
            Duration::from_millis(self.timeout_ms.saturating_mul(2)),
            TokioCommand::new(&self.tau_bin)
                // Pass host path to wrapper; wrapper mounts spec dir and invokes tau inside container.
                .arg(spec_path)
                .env("TAU_INPUTS_DIR", &self.inputs_dir)
                .env("TAU_OUTPUTS_DIR", &self.outputs_dir)
                .current_dir(current_dir)
                .output(),
        )
        .await
        {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                error!("Failed to spawn tau: {}", e);
                return Err(anyhow::anyhow!("Failed to spawn tau: {}", e));
            }
            Err(_) => {
                error!(
                    "Tau spec timed out after {}ms",
                    self.timeout_ms.saturating_mul(2)
                );
                return Err(anyhow::anyhow!("Tau spec timed out"));
            }
        };

        if output.status.success() {
            info!("Tau spec completed successfully: {:?}", spec_path);
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Tau spec failed: {:?}\nStderr: {}", spec_path, stderr);
            Err(anyhow::anyhow!("Tau spec failed: {}", stderr))
        }
    }

    /// Run kernel specification
    pub async fn run_kernel(&self, kernel_path: &Path) -> Result<()> {
        self.run_spec(kernel_path).await
    }

    /// Run monitor specifications
    pub async fn run_monitors(&self, specs_root: &Path) -> Result<()> {
        debug!("Running monitor specs from: {:?}", specs_root);

        // Run invariant monitor pack
        let inv_monitor_path = specs_root.join("monitors_auditors/04_invariant_monitor_pack.tau");
        if inv_monitor_path.exists() {
            self.run_spec(&inv_monitor_path).await?;
        }

        // Run liveness monitor
        let liveness_monitor_path = specs_root.join("monitors_auditors/05_liveness_monitor.tau");
        if liveness_monitor_path.exists() {
            self.run_spec(&liveness_monitor_path).await?;
        }

        // Run cooldown monitor
        let cooldown_monitor_path =
            specs_root.join("monitors_auditors/06_rate_limiter_cooldown.tau");
        if cooldown_monitor_path.exists() {
            self.run_spec(&cooldown_monitor_path).await?;
        }

        Ok(())
    }

    /// Run guard specifications
    pub async fn run_guards(&self, specs_root: &Path) -> Result<()> {
        debug!("Running guard specs from: {:?}", specs_root);

        // Run freshness witness
        let freshness_path = specs_root.join("guards_witnesses/02_freshness_witness.tau");
        if freshness_path.exists() {
            self.run_spec(&freshness_path).await?;
        }

        // Run profit witness
        let profit_path = specs_root.join("guards_witnesses/01_profit_witness.tau");
        if profit_path.exists() {
            self.run_spec(&profit_path).await?;
        }

        // Run failure echo
        let failure_path = specs_root.join("guards_witnesses/03_failure_echo_kill_switch.tau");
        if failure_path.exists() {
            self.run_spec(&failure_path).await?;
        }

        Ok(())
    }
}
