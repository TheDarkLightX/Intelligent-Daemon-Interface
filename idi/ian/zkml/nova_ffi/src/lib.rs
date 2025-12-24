//! Nova-IAN FFI bindings for Python.
//!
//! Provides PyO3 bindings to Nova IVC for IAN zkML training proofs.
//!
//! # SECURITY WARNING: SIMULATION MODE
//!
//! This implementation provides **structural validation only**.
//! It does NOT perform real SNARK proof verification.
//!
//! For production use, integrate the actual Nova library:
//! - <https://github.com/microsoft/Nova>
//!
//! Current implementation is suitable for:
//! - Development and testing
//! - Protocol integration
//! - API design validation
//!
//! NOT suitable for:
//! - Production deployment
//! - Adversarial environments
//! - Privacy-critical applications
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
const STATE_HASH_LEN: usize = 32;
const MAX_MODEL_DIM: u64 = 1_000_000_000;
const MAX_STEPS: u64 = 1_000_000;
const MIN_INSTANCE_LEN: usize = 11 + 32 + 96; // domain + state + commitment + cross_term

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
        if model_dimension > MAX_MODEL_DIM {
            return Err(PyValueError::new_err("model_dimension exceeds maximum"));
        }
        if curve > 2 {
            return Err(PyValueError::new_err("invalid curve (0=Pasta, 1=BnGrumpkin, 2=Bls12381)"));
        }
        Ok(Self {
            model_dimension,
            curve,
            state_hash: vec![0u8; STATE_HASH_LEN],
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
        // Validate input lengths
        if prev_state.len() != STATE_HASH_LEN {
            return Err(PyValueError::new_err("prev_state must be 32 bytes"));
        }
        if batch_hash.len() != STATE_HASH_LEN {
            return Err(PyValueError::new_err("batch_hash must be 32 bytes"));
        }
        if weights_hash.len() != STATE_HASH_LEN {
            return Err(PyValueError::new_err("weights_hash must be 32 bytes"));
        }
        if gradient_hash.len() != STATE_HASH_LEN {
            return Err(PyValueError::new_err("gradient_hash must be 32 bytes"));
        }
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
        if expected_model_dimension > MAX_MODEL_DIM {
            return Err(PyValueError::new_err("expected_model_dimension exceeds maximum"));
        }
        if expected_curve > 2 {
            return Err(PyValueError::new_err("invalid expected_curve"));
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
        // Validate proof bounds first
        if proof.num_steps == 0 || proof.num_steps > MAX_STEPS {
            return Err(PyValueError::new_err("num_steps out of bounds"));
        }
        if proof.model_dimension == 0 || proof.model_dimension > MAX_MODEL_DIM {
            return Err(PyValueError::new_err("model_dimension out of bounds"));
        }
        if proof.curve > 2 {
            return Err(PyValueError::new_err("invalid curve in proof"));
        }
        if proof.final_instance.len() < MIN_INSTANCE_LEN {
            return Err(PyValueError::new_err("final_instance too short"));
        }
        if !proof.final_instance.starts_with(NOVA_DOMAIN) {
            return Err(PyValueError::new_err("invalid domain prefix"));
        }

        // Check expected values
        if proof.model_dimension != self.expected_dim {
            return Err(PyValueError::new_err("model dimension mismatch"));
        }
        if proof.curve != self.expected_curve {
            return Err(PyValueError::new_err("curve mismatch"));
        }
        if let Some(s) = expected_steps {
            if proof.num_steps != s {
                return Err(PyValueError::new_err("num_steps mismatch"));
            }
        }
        if let Some(ref expected) = expected_state {
            if expected.len() != STATE_HASH_LEN {
                return Err(PyValueError::new_err("expected_state must be 32 bytes"));
            }
            let offset = NOVA_DOMAIN.len();
            let state = &proof.final_instance[offset..offset + STATE_HASH_LEN];
            if state != expected.as_slice() {
                return Err(PyValueError::new_err("state mismatch"));
            }
        }

        // TODO: Real SNARK verification with Nova library
        // Current implementation performs structural checks only
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
