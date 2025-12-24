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
use std::io::Cursor;
use std::marker::PhantomData;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use bellpepper_core::{ConstraintSystem, SynthesisError};
use bellpepper_core::num::AllocatedNum;
use ff::PrimeField;
use nova_snark::{PublicParams, RecursiveSNARK};
use nova_snark::provider::{PallasEngine, VestaEngine};
use nova_snark::traits::circuit::StepCircuit;
use nova_snark::traits::Engine;
use pasta_curves::pallas;
use rand::rngs::StdRng;
use rand::SeedableRng;

const NOVA_DOMAIN: &[u8] = b"IAN_NOVA_V1";
const STATE_HASH_LEN: usize = 32;
const MAX_MODEL_DIM: u64 = 1_000_000_000;
const MAX_STEPS: u64 = 1_000_000;
const INSTANCE_HEADER_LEN: usize = NOVA_DOMAIN.len() + 8;
const INSTANCE_PAYLOAD_LEN: usize = STATE_HASH_LEN * 4;
const MIN_INSTANCE_LEN: usize = INSTANCE_HEADER_LEN + INSTANCE_PAYLOAD_LEN;
const PARAMS_SEED_DOMAIN: &[u8] = b"IAN_NOVA_PARAMS_V1";

const PASTA_CURVE: u8 = 0;

type HashBytes = [u8; STATE_HASH_LEN];

type PrimaryEngine = PallasEngine;
type SecondaryEngine = VestaEngine;
type PrimaryScalar = <PrimaryEngine as Engine>::Scalar;
type SecondaryScalar = <SecondaryEngine as Engine>::Scalar;

type PrimaryCircuit = TrainingStepCircuit<PrimaryScalar>;
type SecondaryCircuit = IdentityCircuit<SecondaryScalar>;
type NovaRecursiveSNARK = RecursiveSNARK<PrimaryEngine, SecondaryEngine, PrimaryCircuit, SecondaryCircuit>;
type NovaPublicParams = PublicParams<PrimaryEngine, SecondaryEngine, PrimaryCircuit, SecondaryCircuit>;

#[derive(Clone)]
struct StepInputs {
    epoch: u64,
    batch_hash: HashBytes,
    gradient_hash: HashBytes,
}

struct InstanceData {
    model_dimension: u64,
    z0_state: HashBytes,
    z0_weights: HashBytes,
    zi_state: HashBytes,
    zi_weights: HashBytes,
}

#[derive(Clone)]
struct TrainingStepCircuit<F: PrimeField + From<u64>> {
    model_dimension: u64,
    epoch: u64,
    batch: F,
    gradient: F,
}

impl<F: PrimeField + From<u64>> TrainingStepCircuit<F> {
    fn blank(model_dimension: u64) -> Self {
        Self {
            model_dimension,
            epoch: 0,
            batch: F::from(0u64),
            gradient: F::from(0u64),
        }
    }

    fn new(model_dimension: u64, epoch: u64, batch: F, gradient: F) -> Self {
        Self { model_dimension, epoch, batch, gradient }
    }
}

impl<F: PrimeField + From<u64>> StepCircuit<F> for TrainingStepCircuit<F> {
    fn arity(&self) -> usize {
        2
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        if z.len() != 2 {
            return Err(SynthesisError::AssignmentMissing);
        }

        let prev_state_val = z[0].get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let prev_weights_val = z[1].get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let epoch_f = F::from(self.epoch);
        let model_f = F::from(self.model_dimension);

        let new_state_val =
            prev_state_val + prev_weights_val + self.batch + self.gradient + epoch_f + model_f;
        let new_weights_val = prev_weights_val + self.gradient;

        let new_state = AllocatedNum::alloc(cs.namespace(|| "new_state"), || Ok(new_state_val))?;
        let new_weights = AllocatedNum::alloc(cs.namespace(|| "new_weights"), || Ok(new_weights_val))?;

        cs.enforce(
            || "state update",
            |lc| {
                lc + z[0].get_variable()
                    + z[1].get_variable()
                    + (self.batch, CS::one())
                    + (self.gradient, CS::one())
                    + (epoch_f, CS::one())
                    + (model_f, CS::one())
            },
            |lc| lc + CS::one(),
            |lc| lc + new_state.get_variable(),
        );

        cs.enforce(
            || "weights update",
            |lc| lc + z[1].get_variable() + (self.gradient, CS::one()),
            |lc| lc + CS::one(),
            |lc| lc + new_weights.get_variable(),
        );

        Ok(vec![new_state, new_weights])
    }
}

#[derive(Clone, Default)]
struct IdentityCircuit<F: PrimeField> {
    _marker: PhantomData<F>,
}

impl<F: PrimeField> StepCircuit<F> for IdentityCircuit<F> {
    fn arity(&self) -> usize {
        1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        _cs: &mut CS,
        z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        if z.len() != 1 {
            return Err(SynthesisError::AssignmentMissing);
        }
        Ok(vec![z[0].clone()])
    }
}

fn bytes_to_pallas(bytes: &HashBytes) -> PyResult<pallas::Scalar> {
    let wide = {
        let mut arr = [0u8; 64];
        arr[..32].copy_from_slice(bytes);
        arr
    };
    Ok(pallas::Scalar::from_bytes_wide(&wide))
}

fn pallas_to_bytes(value: &pallas::Scalar) -> HashBytes {
    let repr = value.to_repr();
    let mut out = [0u8; STATE_HASH_LEN];
    out.copy_from_slice(repr.as_ref());
    out
}

fn slice_to_hash(bytes: &[u8], label: &str) -> PyResult<HashBytes> {
    if bytes.len() != STATE_HASH_LEN {
        return Err(PyValueError::new_err(format!("{label} must be 32 bytes")));
    }
    let mut out = [0u8; STATE_HASH_LEN];
    out.copy_from_slice(bytes);
    Ok(out)
}

fn encode_instance(instance: &InstanceData) -> Vec<u8> {
    let mut out = Vec::with_capacity(MIN_INSTANCE_LEN);
    out.extend_from_slice(NOVA_DOMAIN);
    out.extend_from_slice(&instance.model_dimension.to_le_bytes());
    out.extend_from_slice(&instance.z0_state);
    out.extend_from_slice(&instance.z0_weights);
    out.extend_from_slice(&instance.zi_state);
    out.extend_from_slice(&instance.zi_weights);
    out
}

fn decode_instance(bytes: &[u8]) -> PyResult<InstanceData> {
    if bytes.len() < MIN_INSTANCE_LEN {
        return Err(PyValueError::new_err("final_instance too short"));
    }
    if !bytes.starts_with(NOVA_DOMAIN) {
        return Err(PyValueError::new_err("invalid domain prefix"));
    }

    let mut offset = NOVA_DOMAIN.len();
    let model_dimension = u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| PyValueError::new_err("invalid model_dimension"))?,
    );
    offset += 8;

    let z0_state = slice_to_hash(&bytes[offset..offset + STATE_HASH_LEN], "z0_state")?;
    offset += STATE_HASH_LEN;
    let z0_weights = slice_to_hash(&bytes[offset..offset + STATE_HASH_LEN], "z0_weights")?;
    offset += STATE_HASH_LEN;
    let zi_state = slice_to_hash(&bytes[offset..offset + STATE_HASH_LEN], "zi_state")?;
    offset += STATE_HASH_LEN;
    let zi_weights = slice_to_hash(&bytes[offset..offset + STATE_HASH_LEN], "zi_weights")?;

    Ok(InstanceData {
        model_dimension,
        z0_state,
        z0_weights,
        zi_state,
        zi_weights,
    })
}

fn derive_params_seed(model_dimension: u64) -> [u8; 32] {
    let mut seed = [0u8; 32];
    let domain_len = PARAMS_SEED_DOMAIN.len().min(24);
    seed[..domain_len].copy_from_slice(&PARAMS_SEED_DOMAIN[..domain_len]);
    seed[domain_len..domain_len + 8].copy_from_slice(&model_dimension.to_le_bytes());
    seed
}

fn setup_public_params(model_dimension: u64) -> PyResult<(NovaPublicParams, PrimaryCircuit, SecondaryCircuit)> {
    let circuit_primary = PrimaryCircuit::blank(model_dimension);
    let circuit_secondary = SecondaryCircuit::default();
    let mut rng = StdRng::from_seed(derive_params_seed(model_dimension));
    let pp = NovaPublicParams::setup(&circuit_primary, &circuit_secondary, &mut rng)
        .map_err(|e| PyValueError::new_err(format!("public params setup failed: {e}")))?;
    Ok((pp, circuit_primary, circuit_secondary))
}

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
            _ => Err(PyValueError::new_err(format!("Unknown curve: {name}"))),
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
    state_hash: HashBytes,
    weights_hash: HashBytes,
    initial_state: Option<HashBytes>,
    initial_weights: Option<HashBytes>,
    steps: u64,
    step_inputs: Vec<StepInputs>,
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
        if curve != PASTA_CURVE {
            return Err(PyValueError::new_err("only Pasta curves are supported for Nova proofs"));
        }
        Ok(Self {
            model_dimension,
            curve,
            state_hash: [0u8; STATE_HASH_LEN],
            weights_hash: [0u8; STATE_HASH_LEN],
            initial_state: None,
            initial_weights: None,
            steps: 0,
            step_inputs: Vec::new(),
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
        if self.steps >= MAX_STEPS {
            return Err(PyValueError::new_err("maximum steps exceeded"));
        }

        let prev_state = slice_to_hash(&prev_state, "prev_state")?;
        let batch_hash = slice_to_hash(&batch_hash, "batch_hash")?;
        let weights_hash = slice_to_hash(&weights_hash, "weights_hash")?;
        let gradient_hash = slice_to_hash(&gradient_hash, "gradient_hash")?;

        if self.steps == 0 {
            self.initial_state = Some(prev_state);
            self.initial_weights = Some(weights_hash);
            self.state_hash = prev_state;
            self.weights_hash = weights_hash;
        } else {
            if prev_state != self.state_hash {
                return Err(PyValueError::new_err("prev_state mismatch"));
            }
            if weights_hash != self.weights_hash {
                return Err(PyValueError::new_err("weights_hash mismatch"));
            }
        }

        let prev_state_f = bytes_to_pallas(&prev_state)?;
        let prev_weights_f = bytes_to_pallas(&weights_hash)?;
        let batch_f = bytes_to_pallas(&batch_hash)?;
        let gradient_f = bytes_to_pallas(&gradient_hash)?;
        let epoch_f = pallas::Scalar::from(epoch);
        let model_f = pallas::Scalar::from(self.model_dimension);

        let new_state_f = prev_state_f + prev_weights_f + batch_f + gradient_f + epoch_f + model_f;
        let new_weights_f = prev_weights_f + gradient_f;

        let new_state = pallas_to_bytes(&new_state_f);
        let new_weights = pallas_to_bytes(&new_weights_f);

        self.state_hash = new_state;
        self.weights_hash = new_weights;
        self.steps += 1;
        self.step_inputs.push(StepInputs { epoch, batch_hash, gradient_hash });

        Ok((epoch + 1, new_state.to_vec(), new_weights.to_vec()))
    }

    /// Finalize proof
    fn finalize(&self) -> PyResult<RustNovaProof> {
        if self.steps == 0 {
            return Err(PyValueError::new_err("No steps taken"));
        }
        if self.curve != PASTA_CURVE {
            return Err(PyValueError::new_err("only Pasta curves are supported for Nova proofs"));
        }

        let z0_state = self.initial_state.ok_or_else(|| PyValueError::new_err("missing initial_state"))?;
        let z0_weights = self.initial_weights.ok_or_else(|| PyValueError::new_err("missing initial_weights"))?;
        let zi_state = self.state_hash;
        let zi_weights = self.weights_hash;

        let instance = InstanceData {
            model_dimension: self.model_dimension,
            z0_state,
            z0_weights,
            zi_state,
            zi_weights,
        };

        let z0_primary = vec![bytes_to_pallas(&z0_state)?, bytes_to_pallas(&z0_weights)?];
        let zi_primary = vec![bytes_to_pallas(&zi_state)?, bytes_to_pallas(&zi_weights)?];
        let z0_secondary = vec![SecondaryScalar::from(0u64)];
        let zi_secondary = vec![SecondaryScalar::from(0u64)];

        let (pp, base_primary, base_secondary) = setup_public_params(self.model_dimension)?;

        let mut recursive_snark = NovaRecursiveSNARK::new(
            &pp,
            &base_primary,
            &base_secondary,
            z0_primary.clone(),
            z0_secondary.clone(),
        )
        .map_err(|e| PyValueError::new_err(format!("snark init failed: {e}")))?;

        for input in &self.step_inputs {
            let batch_f = bytes_to_pallas(&input.batch_hash)?;
            let gradient_f = bytes_to_pallas(&input.gradient_hash)?;
            let circuit_primary = PrimaryCircuit::new(
                self.model_dimension,
                input.epoch,
                batch_f,
                gradient_f,
            );
            recursive_snark
                .prove_step(&pp, &circuit_primary, &base_secondary)
                .map_err(|e| PyValueError::new_err(format!("snark step failed: {e}")))?;
        }

        let num_steps: usize = self
            .steps
            .try_into()
            .map_err(|_| PyValueError::new_err("num_steps too large"))?;

        recursive_snark
            .verify(&pp, num_steps, &z0_primary, &z0_secondary, &zi_primary, &zi_secondary)
            .map_err(|e| PyValueError::new_err(format!("snark verification failed: {e}")))?;

        let mut snark = Vec::new();
        recursive_snark
            .serialize_compressed(&mut snark)
            .map_err(|e| PyValueError::new_err(format!("snark serialization failed: {e}")))?;

        Ok(RustNovaProof {
            version: 2,
            curve: self.curve,
            final_instance: encode_instance(&instance),
            snark_proof: snark,
            num_steps: self.steps,
            model_dimension: self.model_dimension,
        })
    }

    #[getter]
    fn current_epoch(&self) -> u64 { self.steps }

    #[getter]
    fn current_state_hash(&self) -> Vec<u8> { self.state_hash.to_vec() }
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
        if expected_curve != PASTA_CURVE {
            return Err(PyValueError::new_err("only Pasta curves are supported for Nova proofs"));
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
        if proof.num_steps == 0 || proof.num_steps > MAX_STEPS {
            return Err(PyValueError::new_err("num_steps out of bounds"));
        }
        if proof.model_dimension == 0 || proof.model_dimension > MAX_MODEL_DIM {
            return Err(PyValueError::new_err("model_dimension out of bounds"));
        }
        if proof.curve > 2 {
            return Err(PyValueError::new_err("invalid curve in proof"));
        }
        if proof.curve != PASTA_CURVE {
            return Err(PyValueError::new_err("only Pasta curves are supported for Nova proofs"));
        }
        if proof.final_instance.len() < MIN_INSTANCE_LEN {
            return Err(PyValueError::new_err("final_instance too short"));
        }
        if proof.snark_proof.is_empty() {
            return Err(PyValueError::new_err("snark_proof is empty"));
        }

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

        let instance = decode_instance(&proof.final_instance)?;
        if instance.model_dimension != proof.model_dimension {
            return Err(PyValueError::new_err("instance model_dimension mismatch"));
        }
        if instance.model_dimension != self.expected_dim {
            return Err(PyValueError::new_err("instance model_dimension mismatch"));
        }

        if let Some(expected) = expected_state {
            let expected = slice_to_hash(&expected, "expected_state")?;
            if expected != instance.zi_state {
                return Err(PyValueError::new_err("state mismatch"));
            }
        }

        let z0_primary = vec![bytes_to_pallas(&instance.z0_state)?, bytes_to_pallas(&instance.z0_weights)?];
        let zi_primary = vec![bytes_to_pallas(&instance.zi_state)?, bytes_to_pallas(&instance.zi_weights)?];
        let z0_secondary = vec![SecondaryScalar::from(0u64)];
        let zi_secondary = vec![SecondaryScalar::from(0u64)];

        let (pp, _base_primary, _base_secondary) = setup_public_params(instance.model_dimension)?;
        let mut cursor = Cursor::new(&proof.snark_proof);
        let recursive_snark = NovaRecursiveSNARK::deserialize_compressed(&mut cursor)
            .map_err(|e| PyValueError::new_err(format!("snark deserialization failed: {e}")))?;

        let num_steps: usize = proof
            .num_steps
            .try_into()
            .map_err(|_| PyValueError::new_err("num_steps too large"))?;

        match recursive_snark.verify(&pp, num_steps, &z0_primary, &z0_secondary, &zi_primary, &zi_secondary) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
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
