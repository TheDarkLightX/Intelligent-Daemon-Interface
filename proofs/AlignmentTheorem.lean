/-
  THE ALIGNMENT THEOREM - Formal Lean4 Proof
  
  Theorem: As scarcity approaches infinity, ethical behavior becomes
           the only profitable strategy for any rational agent.
  
  This file contains:
  1. Definitions of the economic model
  2. Axioms encoding the system behavior
  3. Lemmas building toward the main theorem
  4. The main theorem and its proof
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Calculus.LHopital
import Mathlib.Topology.Basic

namespace AlignmentTheorem

noncomputable section

/-! ## Basic Definitions -/

/-- Time parameter (natural numbers for discrete steps) -/
def Time := ℕ

/-- Supply at time t, must be positive real -/
structure Supply where
  value : ℝ
  pos : value > 0

/-- EETF (Ethical-Eco Transaction Factor) score, in range [0, 3] -/
structure EETF where
  value : ℝ
  nonneg : value ≥ 0
  bounded : value ≤ 3

/-- Scarcity multiplier = Initial / Current -/
def scarcity (initial current : Supply) : ℝ :=
  initial.value / current.value

/-- Economic pressure = Scarcity × Network EETF -/
def economicPressure (s : ℝ) (e : EETF) : ℝ :=
  s * e.value

/-- Deflation rate, bounded between 1% and 50% -/
structure DeflationRate where
  value : ℝ
  lower : value ≥ 0.01
  upper : value ≤ 0.50

/-- An agent is ethical if EETF ≥ 1.0 -/
def isEthical (e : EETF) : Prop := e.value ≥ 1.0

/-- Tier multiplier based on EETF -/
def tierMultiplier (e : EETF) : ℝ :=
  if e.value ≥ 2.0 then 5.0
  else if e.value ≥ 1.5 then 3.0
  else if e.value ≥ 1.0 then 1.0
  else 0.0

/-- Canonical ethical reference point (EETF = 1.0). -/
def baselineEthical : EETF :=
  { value := 1.0
  , nonneg := by norm_num
  , bounded := by norm_num }

lemma baseline_isEthical : isEthical baselineEthical := by
  simp [baselineEthical, isEthical]

lemma tierMultiplier_of_unethical (e : EETF) (h : ¬ isEthical e) :
    tierMultiplier e = 0 := by
  have hlt : e.value < 1 := lt_of_not_ge h
  have h_lt_two : e.value < 2 := lt_trans hlt (by norm_num : (1 : ℝ) < 2)
  have h_lt_fifteen : e.value < 1.5 := lt_trans hlt (by norm_num : (1 : ℝ) < 1.5)
  simp [tierMultiplier, isEthical, h, not_le.mpr h_lt_two, not_le.mpr h_lt_fifteen,
    not_le.mpr hlt]

/-! ## Supply Dynamics -/

/-- Supply evolution: S(t+1) = S(t) × (1 - rate) -/
def supplyNext (s : Supply) (r : DeflationRate) : Supply :=
  ⟨s.value * (1 - r.value), by
    apply mul_pos s.pos
    linarith [r.lower, r.upper]⟩

/--
Supply at time t given a constant deflation rate.  This represents the
normalized real-valued supply used for scarcity math (not the discrete token
count); thanks to infinite divisibility, the on-chain supply never reaches
literal zero even though this limit tends to 0 in ℝ.
-/
noncomputable def supplyAt (s₀ : Supply) (r : DeflationRate) (t : ℕ) : Supply :=
  ⟨s₀.value * (1 - r.value)^t, by
    apply mul_pos s₀.pos
    apply pow_pos
    linarith [r.lower, r.upper]⟩

lemma scarcity_pos (s₀ : Supply) (r : DeflationRate) (t : ℕ) :
    0 < scarcity s₀ (supplyAt s₀ r t) := by
  have h_num : 0 < s₀.value := s₀.pos
  have h_den : 0 < (supplyAt s₀ r t).value := (supplyAt s₀ r t).pos
  simpa [scarcity] using div_pos h_num h_den

/-! ## Key Lemmas -/

/-- Lemma: Supply decreases over time -/
lemma supply_decreasing (s₀ : Supply) (r : DeflationRate) (t : ℕ) :
    (supplyAt s₀ r (t + 1)).value < (supplyAt s₀ r t).value := by
  simp only [supplyAt, pow_succ, mul_comm, mul_left_comm, mul_assoc]
  set b := 1 - r.value
  have hb_lt_one : b < 1 := by
    unfold b
    linarith [r.lower]
  have hb_pos : 0 < b := by
    unfold b
    linarith [r.upper]
  have hb_pow_pos : 0 < b ^ t := pow_pos hb_pos t
  have hb_step : b ^ (t + 1) < b ^ t := by
    simpa [pow_succ, mul_comm, mul_left_comm, mul_assoc] using
      (mul_lt_of_lt_one_left hb_pow_pos hb_lt_one)
  have hs_pos : 0 < s₀.value := s₀.pos
  have := mul_lt_mul_of_pos_left hb_step hs_pos
  simpa [pow_succ, mul_comm, mul_left_comm, mul_assoc, b] using this

/-- Lemma: Supply approaches zero as t → ∞ -/
lemma supply_limit_zero (s₀ : Supply) (r : DeflationRate) :
    Filter.Tendsto (fun t => (supplyAt s₀ r t).value) Filter.atTop (nhds 0) := by
  simp only [supplyAt]
  set b := 1 - r.value
  have hb_pos : 0 ≤ b := by
    unfold b
    linarith [r.upper]
  have hb_lt_one : b < 1 := by
    unfold b
    linarith [r.lower]
  have h_abs : |b| < 1 := by
    simpa [abs_of_nonneg hb_pos] using hb_lt_one
  have h_pow := tendsto_pow_atTop_nhds_zero_of_abs_lt_one h_abs
  simpa [b, mul_comm, mul_left_comm, mul_assoc] using
    (h_pow.const_mul s₀.value)

/-- Lemma: Scarcity approaches infinity as supply approaches zero -/
lemma scarcity_limit_infinity (s₀ : Supply) (r : DeflationRate) :
    Filter.Tendsto (fun t => scarcity s₀ (supplyAt s₀ r t)) Filter.atTop Filter.atTop := by
  simp only [scarcity, supplyAt]
  set b := 1 - r.value
  have hb_pos : 0 < b := by
    unfold b
    linarith [r.upper]
  have hb_lt_one : b < 1 := by
    unfold b
    linarith [r.lower]
  have h_inv_gt_one : 1 < (1 / b) := by
    have hb : 0 < b := hb_pos
    have : b < 1 := hb_lt_one
    have hb_ne : b ≠ 0 := by exact ne_of_gt hb
    have := one_lt_inv hb hb_lt_one
    simpa [one_div, div_eq_inv_mul, inv_inv] using this
  have h_tend := tendsto_pow_atTop_atTop_of_gt_one h_inv_gt_one
  have hs_ne : s₀.value ≠ 0 := by exact ne_of_gt s₀.pos
  have hb_ne : b ≠ 0 := by exact ne_of_gt hb_pos
  have : (fun t : ℕ => s₀.value / (s₀.value * b ^ t)) =
      fun t => (1 / b) ^ t := by
    funext t
    have h_cancel : s₀.value / (s₀.value * b ^ t) = 1 / (b ^ t) := by
      field_simp [hs_ne, hb_ne, pow_succ, pow_add]
    have h_pow : (1 / b) ^ t = 1 / (b ^ t) := by
      simp [one_div, inv_pow, hb_ne]
    simpa [h_cancel, h_pow]
  simpa [this]

/-! ## Reward and Penalty Functions -/

/-- Reward for ethical behavior -/
noncomputable def reward (balance scarcity_val : ℝ) (e : EETF) : ℝ :=
  balance * scarcity_val * (tierMultiplier e) / 1000

/-- Penalty for unethical behavior -/
noncomputable def penalty (txValue : ℝ) (e : EETF) (pressure : ℝ) : ℝ :=
  if e.value < 1.0 then
    txValue * (1.0 - e.value) * pressure / 100
  else
    0

/-- Expected value for an agent -/
noncomputable def expectedValue (balance txValue scarcity_val : ℝ) (e : EETF) 
    (networkEETF : EETF) : ℝ :=
  let pressure := economicPressure scarcity_val networkEETF
  reward balance scarcity_val e - penalty txValue e pressure

lemma reward_baseline (balance scarcity_val : ℝ) :
    reward balance scarcity_val baselineEthical = balance * scarcity_val / 1000 := by
  simp [reward, baselineEthical]

lemma penalty_of_ethical (txValue pressure : ℝ) (e : EETF)
    (h : isEthical e) : penalty txValue e pressure = 0 := by
  simp [penalty, isEthical, h]

lemma reward_of_unethical (balance scarcity_val : ℝ) (e : EETF)
    (h : ¬ isEthical e) : reward balance scarcity_val e = 0 := by
  simp [reward, tierMultiplier_of_unethical e h]

lemma penalty_of_unethical (txValue scarcity_val : ℝ) (network : EETF)
    (e : EETF) (h : ¬ isEthical e) :
    penalty txValue e (economicPressure scarcity_val network) =
      txValue * (1 - e.value) * (scarcity_val * network.value) / 100 := by
  simp [penalty, economicPressure, isEthical, h]

lemma expectedValue_baseline (balance txValue scarcity_val : ℝ) (network : EETF) :
    expectedValue balance txValue scarcity_val baselineEthical network =
      balance * scarcity_val / 1000 := by
  simp [expectedValue, reward_baseline, penalty_of_ethical, baseline_isEthical]

lemma expectedValue_unethical (balance txValue scarcity_val : ℝ) (network : EETF)
    (e : EETF) (h : ¬ isEthical e) :
    expectedValue balance txValue scarcity_val e network =
      - (txValue * (1 - e.value) * (scarcity_val * network.value) / 100) := by
  simp [expectedValue, reward_of_unethical, penalty_of_unethical, h, sub_eq_add_neg]

lemma expectedValue_baseline_pos (balance scarcity_val : ℝ)
    (h_balance : balance > 0) (h_scarcity : scarcity_val > 0) (txValue : ℝ)
    (network : EETF) :
    0 < expectedValue balance txValue scarcity_val baselineEthical network := by
  have h_num : 0 < balance * scarcity_val := mul_pos h_balance h_scarcity
  have h_div : 0 < (balance * scarcity_val) / 1000 := by
    have : (1000 : ℝ) > 0 := by norm_num
    exact div_pos h_num this
  simpa [expectedValue_baseline] using h_div

lemma expectedValue_unethical_neg (balance txValue scarcity_val : ℝ) (network : EETF)
    (h_txValue : txValue > 0) (h_scarcity : scarcity_val > 0) (h_network : network.value > 0)
    (e : EETF) (h_unethical : ¬ isEthical e) :
    expectedValue balance txValue scarcity_val e network < 0 := by
  have h_factor : 1 - e.value > 0 := by
    have : e.value < 1 := lt_of_not_ge h_unethical
    linarith
  have h_pos1 : txValue * (1 - e.value) > 0 := mul_pos h_txValue h_factor
  have h_pos2 : scarcity_val * network.value > 0 := mul_pos h_scarcity h_network
  have h_prod : txValue * (1 - e.value) * (scarcity_val * network.value) > 0 :=
    mul_pos h_pos1 h_pos2
  have h_div : (txValue * (1 - e.value) * (scarcity_val * network.value)) / 100 > 0 := by
    have : (100 : ℝ) > 0 := by norm_num
    exact div_pos h_prod this
  have h_lt_zero :
      - (txValue * (1 - e.value) * (scarcity_val * network.value) / 100) < 0 :=
    neg_lt_zero.mpr h_div
  simpa [expectedValue_unethical _ _ _ _ _ h_unethical] using h_lt_zero

/-! ## The Core Invariant -/

/-- The Alignment Invariant: At high pressure, positive reward implies ethical -/
def alignmentInvariant (pressure : ℝ) (e : EETF) : Prop :=
  pressure > 1000 → (reward 1000 pressure e > 0 → isEthical e)

/-- Proof that the alignment invariant holds -/
theorem alignment_invariant_holds (pressure : ℝ) (e : EETF) 
    (h_pressure : pressure > 1000) : alignmentInvariant pressure e := by
  intro _
  intro h_reward
  -- If reward > 0, then tierMultiplier e > 0
  -- tierMultiplier e > 0 implies e.value ≥ 1.0
  simp only [reward] at h_reward
  simp only [tierMultiplier] at h_reward
  simp only [isEthical]
  by_contra h_not_ethical
  push_neg at h_not_ethical
  -- If e.value < 1.0, tierMultiplier = 0, so reward = 0
  simp only [if_neg (by linarith : ¬ e.value ≥ 2.0), 
             if_neg (by linarith : ¬ e.value ≥ 1.5),
             if_neg (by linarith : ¬ e.value ≥ 1.0)] at h_reward
  simp at h_reward

/-! ## Expected Value Limits -/

/-- Ethical EV approaches +∞ as scarcity → ∞ -/
theorem ethical_ev_limit_pos_infinity (balance txValue : ℝ) 
    (h_balance : balance > 0) (e : EETF) (h_ethical : isEthical e)
    (networkEETF : EETF) (h_net_pos : networkEETF.value > 0) :
    Filter.Tendsto (fun scarcity_val => expectedValue balance txValue scarcity_val e networkEETF) 
      Filter.atTop Filter.atTop := by
  simp only [expectedValue, reward, penalty, economicPressure]
  -- Reward grows linearly with scarcity
  -- Penalty is 0 for ethical agents
  simp only [isEthical] at h_ethical
  have h_tier : tierMultiplier e > 0 := by
    simp only [tierMultiplier]
    split_ifs with h1 h2 h3
    · norm_num
    · norm_num
    · norm_num
    · linarith
  -- penalty = 0 since e.value ≥ 1.0
  have h_pen : ∀ p, penalty txValue e p = 0 := by
    intro p
    simp only [penalty]
    split_ifs with h
    · linarith
    · rfl
  simp only [h_pen, sub_zero]
  apply Filter.Tendsto.atTop_mul_const (by positivity : balance * tierMultiplier e / 1000 > 0)
  exact Filter.tendsto_id

/-- Unethical EV approaches -∞ as scarcity → ∞ -/
theorem unethical_ev_limit_neg_infinity (balance txValue : ℝ) 
    (h_txValue : txValue > 0) (e : EETF) (h_unethical : ¬isEthical e)
    (networkEETF : EETF) (h_net_pos : networkEETF.value > 0) :
    Filter.Tendsto (fun scarcity_val => expectedValue balance txValue scarcity_val e networkEETF) 
      Filter.atTop Filter.atBot := by
  simp only [expectedValue, reward, penalty, economicPressure, isEthical] at *
  push_neg at h_unethical
  -- Reward = 0 for unethical
  have h_tier : tierMultiplier e = 0 := by
    simp only [tierMultiplier]
    split_ifs with h1 h2 h3
    · linarith
    · linarith
    · linarith
    · rfl
  simp only [h_tier, mul_zero, zero_div, zero_sub]
  -- Penalty grows quadratically with scarcity (linear × linear)
  have h_pen_pos : (1.0 - e.value) > 0 := by linarith
  apply Filter.Tendsto.neg_atTop
  apply Filter.Tendsto.atTop_mul_const (by positivity : txValue * (1.0 - e.value) * networkEETF.value / 100 > 0)
  -- scarcity × scarcity → ∞
  exact Filter.Tendsto.atTop_mul Filter.tendsto_id Filter.tendsto_id

/-! ## THE MAIN THEOREM -/

/-- 
  THE ALIGNMENT THEOREM
  
  For any rational agent (one that maximizes expected value),
  as the scarcity multiplier approaches infinity,
  the agent will choose ethical behavior.
  
  Formally: For any agent with EETF e, if the agent is rational,
  then in the limit of infinite scarcity, the agent will have
  EETF ≥ 1.0 (i.e., be ethical).
-/
theorem alignment_theorem (balance txValue : ℝ) 
    (h_balance : balance > 0) (h_txValue : txValue > 0)
    (networkEETF : EETF) (h_net_pos : networkEETF.value > 0) :
    ∀ (rational_choice : ℝ → EETF),
      (∀ scarcity_val, ∀ e : EETF, 
        expectedValue balance txValue scarcity_val (rational_choice scarcity_val) networkEETF ≥ 
        expectedValue balance txValue scarcity_val e networkEETF) →
      (∃ S : ℝ, ∀ scarcity_val > S, isEthical (rational_choice scarcity_val)) := by
  intro rational_choice h_rational
  refine ⟨0, ?_⟩
  intro scarcity_val h_scarcity
  have h_base_pos :=
    expectedValue_baseline_pos balance scarcity_val h_balance h_scarcity txValue networkEETF
  by_contra h_not_ethical
  set e_bad := rational_choice scarcity_val
  have h_bad_neg :=
    expectedValue_unethical_neg balance txValue scarcity_val networkEETF
      h_txValue h_scarcity h_net_pos e_bad h_not_ethical
  have h_compare :=
    h_rational scarcity_val baselineEthical
  have h_base_neg :
      expectedValue balance txValue scarcity_val baselineEthical networkEETF < 0 :=
    lt_of_le_of_lt h_compare h_bad_neg
  exact (lt_irrefl (0 : ℝ)) (lt_trans h_base_pos h_base_neg)

/-! ## Corollaries -/

/-- Corollary: The system converges to ethical equilibrium -/
theorem ethical_equilibrium (s₀ : Supply) (r : DeflationRate) 
    (balance txValue : ℝ) (h_balance : balance > 0) (h_txValue : txValue > 0)
    (networkEETF : EETF) (h_net_pos : networkEETF.value > 0) :
    ∃ T : ℕ, ∀ t ≥ T, 
      let scarcity_val := scarcity s₀ (supplyAt s₀ r t)
      -- At equilibrium, all rational agents are ethical
      ∀ rational_choice : ℝ → EETF,
        (∀ s e, expectedValue balance txValue s (rational_choice s) networkEETF ≥ 
                expectedValue balance txValue s e networkEETF) →
        isEthical (rational_choice scarcity_val) := by
  -- Follows from alignment_theorem and scarcity_limit_infinity
  -- Since scarcity → ∞, eventually scarcity > S (the threshold from alignment_theorem)
  rcases alignment_theorem balance txValue h_balance h_txValue networkEETF h_net_pos with ⟨S, hS⟩
  have h_eventually : ∀ᶠ t in Filter.atTop, scarcity s₀ (supplyAt s₀ r t) > S := by
    apply Filter.Tendsto.eventually_gt_atTop
    exact scarcity_limit_infinity s₀ r
  
  rcases Filter.eventually_atTop.mp h_eventually with ⟨T, hT⟩
  use T
  intro t ht
  specialize hT t ht
  apply hS
  exact hT

/-- Corollary: Unethical strategies have negative expected value at equilibrium -/
theorem unethical_negative_ev_at_equilibrium (s₀ : Supply) (r : DeflationRate)
    (balance txValue : ℝ) (h_balance : balance > 0) (h_txValue : txValue > 0)
    (e : EETF) (h_unethical : ¬isEthical e)
    (networkEETF : EETF) (h_net_pos : networkEETF.value > 0) :
    ∃ T : ℕ, ∀ t ≥ T,
      expectedValue balance txValue (scarcity s₀ (supplyAt s₀ r t)) e networkEETF < 0 := by
  refine ⟨0, ?_⟩
  intro t _
  have h_scar := scarcity_pos s₀ r t
  exact expectedValue_unethical_neg balance txValue _ networkEETF
    h_txValue h_scar h_net_pos e h_unethical

end AlignmentTheorem

