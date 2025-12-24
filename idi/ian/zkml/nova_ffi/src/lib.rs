//! Nova-IAN FFI bindings for Python.
//!
//! Provides PyO3 bindings to Nova IVC for IAN zkML training proofs.
//!
//! # Build
//! ```bash
//! maturin build --release
//! pip install target/wheels/nova_ian-*.whl
//! ```

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use sha2::{Sha256, Digest};

const NOVA_DOMAIN: &[u8] = b"IAN_NOVA_V1";

/// Supported curve cycles
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustCurveCycle {
    Pasta = 0,
    BnGrumpkin = 1,
    Bls12381 = 2,
}

#[pymethods]
impl RustCurveCycle {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "pasta" => Ok(RustCurveCycle::Pasta),
            "bn_grumpkin" => Ok(RustCurveCycle::BnGrumpkin),
            "bls12_381" => Ok(RustCurveCycle::Bls12381),
            _ => Err(PyValueError::new_err(format!("Unknown curve: {}", name))),
        }
    }
}

/// Nova proof for training verification
#[pyclass]
#[derive(Clone)]
pub struct RustNovaProof {
    #[pyo3(get)]
    pub version: u32,
    #[pyo3(get)]
    pub curve: u8,
    #[pyo3(get)]
    pub final_instance: Vec<u8>,
    #[pyo3(get)]
    pub snark_proof: Vec<u8>,
    #[pyo3(get)]
    pub num_steps: u64,
    #[pyo3(get)]
    pub model_dimension: u64,
}

#[pymethods]
impl RustNovaProof {
    #[new]
    fn new(
        version: u32,
        curve: u8,
        final_instance: Vec<u8>,
        snark_proof: Vec<u8>,
        num_steps: u64,
        model_dimension: u64,
    ) -> Self {
        Self { version, curve, final_instance, snark_proof, num_steps, model_dimension }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&self.version.to_le_bytes());
        b.push(self.curve);
        b.extend_from_slice(&(self.final_instance.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.final_instance);
        b.extend_from_slice(&(self.snark_proof.len() as u32).to_le_bytes());
        b.extend_from_slice(&self.snark_proof);
        b.extend_from_slice(&self.num_steps.to_le_bytes());
        b.extend_from_slice(&self.model_dimension.to_le_bytes());
        b
    }
}

/// Nova prover (Rust implementation)
#[pyclass]
pub struct RustNovaProver {
    model_dimension: u64,
    curve: u8,
    state_hash: Vec<u8>,
    steps: u64,
}

#[pymethods]
impl RustNovaProver {
    #[new]
    fn new(model_dimension: u64, curve: u8) -> PyResult<Self> {
        if model_dimension == 0 {
            return Err(PyValueError::new_err("model_dimension must be positive"));
        }
        Ok(Self {
            model_dimension,
            curve,
            state_hash: vec![0u8; 32],
            steps: 0,
        })
    }

    /// Execute training step
    fn step(
        &mut self,
        epoch: u64,
        prev_state: Vec<u8>,
        batch_hash: Vec<u8>,
        weights_hash: Vec<u8>,
        gradient_hash: Vec<u8>,
    ) -> PyResult<(u64, Vec<u8>, Vec<u8>)> {
        // Compute new state
        let mut h = Sha256::new();
        h.update(NOVA_DOMAIN);
        h.update(b"STATE");
        h.update(&epoch.to_le_bytes());
        h.update(&prev_state);
        h.update(&batch_hash);
        let new_state = h.finalize().to_vec();

        // Compute new weights
        let mut h = Sha256::new();
        h.update(NOVA_DOMAIN);
        h.update(b"WEIGHTS");
        h.update(&weights_hash);
        h.update(&gradient_hash);
        let new_weights = h.finalize().to_vec();

        self.state_hash = new_state.clone();
        self.steps += 1;

        Ok((epoch + 1, new_state, new_weights))
    }

    /// Finalize proof
    fn finalize(&self) -> PyResult<RustNovaProof> {
        if self.steps == 0 {
            return Err(PyValueError::new_err("No steps taken"));
        }

        let mut instance = Vec::new();
        instance.extend_from_slice(NOVA_DOMAIN);
        instance.extend_from_slice(&self.state_hash);
        instance.extend_from_slice(&[0u8; 96]); // commitment + cross_term

        let mut h = Sha256::new();
        h.update(NOVA_DOMAIN);
        h.update(b"SNARK");
        h.update(&self.state_hash);
        let snark = h.finalize().to_vec();

        Ok(RustNovaProof {
            version: 1,
            curve: self.curve,
            final_instance: instance,
            snark_proof: snark,
            num_steps: self.steps,
            model_dimension: self.model_dimension,
        })
    }

    #[getter]
    fn current_epoch(&self) -> u64 { self.steps }

    #[getter]
    fn current_state_hash(&self) -> Vec<u8> { self.state_hash.clone() }
}

/// Nova verifier (Rust implementation)  
#[pyclass]
pub struct RustNovaVerifier {
    expected_dim: u64,
    expected_curve: u8,
}

#[pymethods]
impl RustNovaVerifier {
    #[new]
    fn new(expected_model_dimension: u64, expected_curve: u8) -> PyResult<Self> {
        if expected_model_dimension == 0 {
            return Err(PyValueError::new_err("expected_model_dimension must be positive"));
        }
        Ok(Self { expected_dim: expected_model_dimension, expected_curve })
    }

    /// Verify proof
    fn verify(
        &self,
        proof: &RustNovaProof,
        expected_state: Option<Vec<u8>>,
        expected_steps: Option<u64>,
    ) -> PyResult<bool> {
        if proof.model_dimension != self.expected_dim {
            return Err(PyValueError::new_err("Model dimension mismatch"));
        }
        if proof.curve != self.expected_curve {
            return Err(PyValueError::new_err("Curve mismatch"));
        }
        if let Some(s) = expected_steps {
            if proof.num_steps != s {
                return Err(PyValueError::new_err("num_steps mismatch"));
            }
        }

        let min_len = NOVA_DOMAIN.len() + 32 + 96;
        if proof.final_instance.len() < min_len {
            return Err(PyValueError::new_err("Invalid instance length"));
        }
        if !proof.final_instance.starts_with(NOVA_DOMAIN) {
            return Err(PyValueError::new_err("Invalid domain prefix"));
        }

        if let Some(expected) = expected_state {
            let offset = NOVA_DOMAIN.len();
            let state = &proof.final_instance[offset..offset+32];
            if state != expected.as_slice() {
                return Err(PyValueError::new_err("State mismatch"));
            }
        }

        // TODO: Real SNARK verification
        Ok(true)
    }
}

#[pymodule]
fn nova_ian(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustCurveCycle>()?;
    m.add_class::<RustNovaProof>()?;
    m.add_class::<RustNovaProver>()?;
    m.add_class::<RustNovaVerifier>()?;
    Ok(())
}
