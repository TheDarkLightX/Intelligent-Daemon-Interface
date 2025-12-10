"""Experimental STRIKE/IKL knowledge engine scaffolding.

This module hosts an experimental, decidable rule engine (STRIKE) and
associated data structures for IKL-based knowledge packs.

It is designed to be used alongside the Agent Modular Synthesizer and
SAPE, but is currently kept in the experimental namespace.

The implementation here focuses on a small Horn-like fragment suitable
for pack-based reasoning about patches and environments. It is not
connected to the main CLI yet and should be treated as internal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


Atom = Tuple[str, Tuple[Any, ...]]


@dataclass(frozen=True)
class Constraint:
    """Numeric/boolean constraint evaluated over an environment mapping.

    The check function must be pure and deterministic.
    """

    name: str
    check: Callable[[Dict[str, Any]], bool]


@dataclass(frozen=True)
class Rule:
    """Horn-style rule in IKL.

    head: predicate with arguments (variables or constants).
    body: atoms that must all hold for the rule to fire.
    constraints: additional numeric/boolean side conditions.
    profiles: optional tags for which profiles activate this rule.
    """

    head: Atom
    body: Tuple[Atom, ...]
    constraints: Tuple[Constraint, ...] = ()
    profiles: Tuple[str, ...] = ()


@dataclass
class KnowledgeBase:
    """Unified knowledge base constructed from one or more packs."""

    rules: Tuple[Rule, ...] = ()
    global_facts: Dict[str, Set[Tuple[Any, ...]]] = field(default_factory=dict)


@dataclass(frozen=True)
class PackMeta:
    """Minimal metadata for a knowledge pack used by the linter."""

    name: str
    version: str
    rules: Tuple[Rule, ...]


@dataclass
class KnowledgePack:
    """Concrete knowledge pack: metadata + rules + optional facts.

    This is the authoring unit for IKL content. Multiple packs can be
    merged into a single KnowledgeBase for reasoning.
    """

    name: str
    version: str
    rules: Tuple[Rule, ...]
    global_facts: Dict[str, Set[Tuple[Any, ...]]] = field(default_factory=dict)
    domain_tags: Tuple[str, ...] = ()
    profile_tags: Tuple[str, ...] = ()


def build_kb_from_packs(packs: Iterable[KnowledgePack]) -> KnowledgeBase:
    """Merge one or more packs into a unified KnowledgeBase."""

    all_rules: List[Rule] = []
    all_facts: Dict[str, Set[Tuple[Any, ...]]] = {}

    for pack in packs:
        all_rules.extend(pack.rules)
        for pred, values in pack.global_facts.items():
            bucket = all_facts.setdefault(pred, set())
            bucket.update(values)

    return KnowledgeBase(rules=tuple(all_rules), global_facts=all_facts)


def build_qagent_base_pack(
    max_state_cells: int = 256,
    min_discount: float = 0.9,
) -> KnowledgePack:
    """Construct a base pack for QAgentPatch-style constraints."""

    def state_size_constraint(env: Dict[str, Any]) -> bool:
        price_bins = env.get("?bp")
        inventory_bins = env.get("?bi")
        limit = env.get("max_state_cells", max_state_cells)
        if price_bins is None or inventory_bins is None:
            return False
        try:
            return int(price_bins) * int(inventory_bins) > int(limit)
        except (TypeError, ValueError):
            return False

    def discount_constraint(env: Dict[str, Any]) -> bool:
        discount = env.get("?d")
        threshold = env.get("min_discount", min_discount)
        if discount is None:
            return False
        try:
            return float(discount) < float(threshold)
        except (TypeError, ValueError):
            return False

    rules: List[Rule] = []

    rules.append(
        Rule(
            head=("violates_constraint", ("?p", "state_size", "too many cells")),
            body=(
                ("param_value", ("?p", "num_price_bins", "?bp")),
                ("param_value", ("?p", "num_inventory_bins", "?bi")),
            ),
            constraints=(Constraint("state_size", state_size_constraint),),
        )
    )

    rules.append(
        Rule(
            head=("violates_constraint", ("?p", "discount", "discount too low")),
            body=(("param_value", ("?p", "discount_factor", "?d")),),
            constraints=(Constraint("discount_min", discount_constraint),),
        )
    )

    return KnowledgePack(
        name="qagent_base",
        version="0.1.0",
        rules=tuple(rules),
        global_facts={},
        domain_tags=("qagent", "experimental"),
        profile_tags=(),
    )


def build_zk_tau_invariants_pack() -> KnowledgePack:
    """Construct a pack encoding high-level ZK/Tau invariants.

    This pack does not recompute cryptographic values; instead it
    operates over status facts provided by the ZK/Tau plumbing, e.g.:

    - commitment_status(Bundle, "ok" | "fail")
    - method_id_status(Bundle, "ok" | "fail")
    - journal_status(Bundle, "ok" | "fail")
    - tx_hash_status(Bundle, "ok" | "fail")
    - path_status(Bundle, "ok" | "fail")

    It lifts these statuses into `violates_constraint` facts so that
    STRIKE-based workflows can uniformly reason about bundles.
    """

    rules: List[Rule] = []

    # Commitment failure
    rules.append(
        Rule(
            head=("violates_constraint", ("?b", "zk_commitment", "commitment failed")),
            body=(("commitment_status", ("?b", "fail")),),
        )
    )

    # Method ID mismatch
    rules.append(
        Rule(
            head=("violates_constraint", ("?b", "zk_method_id", "method id mismatch")),
            body=(("method_id_status", ("?b", "fail")),),
        )
    )

    # Journal digest mismatch
    rules.append(
        Rule(
            head=("violates_constraint", ("?b", "zk_journal", "journal digest mismatch")),
            body=(("journal_status", ("?b", "fail")),),
        )
    )

    # TX hash mismatch
    rules.append(
        Rule(
            head=("violates_constraint", ("?b", "zk_tx_hash", "tx hash mismatch")),
            body=(("tx_hash_status", ("?b", "fail")),),
        )
    )

    # Path / archive safety issues
    rules.append(
        Rule(
            head=("violates_constraint", ("?b", "zk_paths", "unsafe path or archive entry")),
            body=(("path_status", ("?b", "fail")),),
        )
    )

    return KnowledgePack(
        name="zk_tau_invariants",
        version="0.1.0",
        rules=tuple(rules),
        global_facts={},
        domain_tags=("zk", "tau_invariants"),
        profile_tags=(),
    )


def build_comms_base_pack() -> KnowledgePack:
    """Construct a base pack for communication / emoji policies.

    This pack encodes a few simple, illustrative rules about when
    certain communication actions are forbidden.

    It expects facts such as:
    - comm_action(AgentOrUser, ActionId)
    - risk_state(AgentOrUser, Level)
    - user_sensitivity(AgentOrUser, Level)
    """

    rules: List[Rule] = []

    # Forbid remaining silent when risk state is high.
    rules.append(
        Rule(
            head=(
                "violates_constraint",
                ("?a", "comms_silence", "must alert in high risk state"),
            ),
            body=(
                ("risk_state", ("?a", "high")),
                ("comm_action", ("?a", "silent")),
            ),
        )
    )

    # Forbid panic-style emojis for high-sensitivity users.
    rules.append(
        Rule(
            head=(
                "violates_constraint",
                (
                    "?a",
                    "comms_panic",
                    "panic emoji forbidden for high-sensitivity users",
                ),
            ),
            body=(
                ("user_sensitivity", ("?a", "high")),
                ("comm_action", ("?a", "panic_emoji")),
            ),
        )
    )

    return KnowledgePack(
        name="comms_base",
        version="0.1.0",
        rules=tuple(rules),
        global_facts={},
        domain_tags=("comms", "experimental"),
        profile_tags=(),
    )


def build_risk_conservative_pack(
    max_learning_rate: float = 0.2,
    max_epsilon_start: float = 0.5,
) -> KnowledgePack:
    """Construct a conservative risk pack for QAgent-style patches.

    This pack encodes high-level, parameter-only risk constraints such as:
    - Upper bound on learning rate.
    - Upper bound on initial exploration rate.

    It expects `param_value(P, Name, Value)` facts for QAgentPatch fields.
    """

    def lr_constraint(env: Dict[str, Any]) -> bool:
        lr = env.get("?lr")
        limit = env.get("max_learning_rate", max_learning_rate)
        if lr is None:
            return False
        try:
            return float(lr) > float(limit)
        except (TypeError, ValueError):
            return False

    def eps_constraint(env: Dict[str, Any]) -> bool:
        eps = env.get("?e")
        limit = env.get("max_epsilon_start", max_epsilon_start)
        if eps is None:
            return False
        try:
            return float(eps) > float(limit)
        except (TypeError, ValueError):
            return False

    rules: List[Rule] = []

    # Learning rate too high for conservative profile.
    rules.append(
        Rule(
            head=("violates_constraint", ("?p", "risk_lr", "learning rate too high")),
            body=(("param_value", ("?p", "learning_rate", "?lr")),),
            constraints=(Constraint("lr_max", lr_constraint),),
        )
    )

    # Initial exploration too high for conservative profile.
    rules.append(
        Rule(
            head=(
                "violates_constraint",
                ("?p", "risk_eps", "initial exploration too high"),
            ),
            body=(("param_value", ("?p", "epsilon_start", "?e")),),
            constraints=(Constraint("eps_max", eps_constraint),),
        )
    )

    return KnowledgePack(
        name="risk_conservative",
        version="0.1.0",
        rules=tuple(rules),
        global_facts={},
        domain_tags=("risk", "qagent"),
        profile_tags=("conservative",),
    )


def build_qpatch_base_kb(
    max_state_cells: int = 256,
    min_discount: float = 0.9,
) -> KnowledgeBase:
    """Convenience wrapper: build KB for QAgent-style patches.

    Includes the QAgent base, conservative risk, communication, and
    ZK/Tau invariant packs in a single KnowledgeBase.
    """

    qagent = build_qagent_base_pack(
        max_state_cells=max_state_cells,
        min_discount=min_discount,
    )
    risk = build_risk_conservative_pack()
    comms = build_comms_base_pack()
    zk = build_zk_tau_invariants_pack()
    return build_kb_from_packs((qagent, risk, comms, zk))


def _active_rules(kb: KnowledgeBase, active_profiles: Set[str]) -> Tuple[Rule, ...]:
    """Select rules that are active under the given profiles.

    A rule with no profile tags is always active.
    """

    selected: List[Rule] = []
    for r in kb.rules:
        if not r.profiles:
            selected.append(r)
            continue
        if any(p in active_profiles for p in r.profiles):
            selected.append(r)
    return tuple(selected)


def _match_atom(atom: Atom, facts: Dict[str, Set[Tuple[Any, ...]]]) -> Iterable[Dict[str, Any]]:
    """Yield variable bindings that satisfy a single atom.

    Variables are strings starting with "?". All other terms are treated as
    constants and must match exactly.
    """

    pred, args = atom
    for fact_args in facts.get(pred, ()):  # type: ignore[assignment]
        if len(fact_args) != len(args):
            continue
        env: Dict[str, Any] = {}
        ok = True
        for term, value in zip(args, fact_args):
            if isinstance(term, str) and term.startswith("?"):
                bound = env.get(term)
                if bound is None:
                    env[term] = value
                elif bound != value:
                    ok = False
                    break
            else:
                if term != value:
                    ok = False
                    break
        if ok:
            yield env


def _merge_envs(env1: Dict[str, Any], env2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Merge two variable bindings, returning None on conflict."""

    result = dict(env1)
    for k, v in env2.items():
        if k in result and result[k] != v:
            return None
        result[k] = v
    return result


def _body_matches(body: Tuple[Atom, ...], facts: Dict[str, Set[Tuple[Any, ...]]]) -> Iterable[Dict[str, Any]]:
    """Find all variable bindings that satisfy a conjunction of atoms."""

    if not body:
        yield {}
        return

    first, *rest = body
    initial_envs = list(_match_atom(first, facts))
    if not rest:
        for env in initial_envs:
            yield env
        return

    for env in initial_envs:
        current_envs: List[Dict[str, Any]] = [env]
        for atom in rest:
            new_envs: List[Dict[str, Any]] = []
            for base_env in current_envs:
                for atom_env in _match_atom(atom, facts):
                    merged = _merge_envs(base_env, atom_env)
                    if merged is not None:
                        new_envs.append(merged)
            current_envs = new_envs
            if not current_envs:
                break
        for final_env in current_envs:
            yield final_env


def _apply_rule_once(rule: Rule, facts: Dict[str, Set[Tuple[Any, ...]]], params: Dict[str, Any]) -> Set[Atom]:
    """Apply a rule once over current facts and parameters.

    Returns a set of newly derived atoms.
    """

    new_atoms: Set[Atom] = set()
    for env in _body_matches(rule.body, facts):
        extended_env = {**env, **params}
        if not all(c.check(extended_env) for c in rule.constraints):
            continue
        head_pred, head_args = rule.head
        concrete_args: List[Any] = []
        for term in head_args:
            if isinstance(term, str) and term.startswith("?"):
                concrete_args.append(extended_env.get(term))
            else:
                concrete_args.append(term)
        new_atoms.add((head_pred, tuple(concrete_args)))
    return new_atoms


def run_strike_closure(
    kb: KnowledgeBase,
    base_facts: Dict[str, Set[Tuple[Any, ...]]],
    params: Dict[str, Any],
    active_profiles: Set[str],
) -> Dict[str, Set[Tuple[Any, ...]]]:
    """Compute the least fixpoint closure under active rules.

    This is the core STRIKE-D operation used at design time. It is
    intentionally small and bounded to keep reasoning decidable.
    """

    facts: Dict[str, Set[Tuple[Any, ...]]] = {
        pred: set(args_set) for pred, args_set in kb.global_facts.items()
    }
    for pred, values in base_facts.items():
        facts.setdefault(pred, set()).update(values)

    active = _active_rules(kb, active_profiles)
    changed = True
    while changed:
        changed = False
        for rule in active:
            new_atoms = _apply_rule_once(rule, facts, params)
            for pred, args in new_atoms:
                bucket = facts.setdefault(pred, set())
                if args not in bucket:
                    bucket.add(args)
                    changed = True

    return facts


def evaluate_with_krr(
    kb: KnowledgeBase,
    base_facts: Dict[str, Set[Tuple[Any, ...]]],
    params: Dict[str, Any],
    active_profiles: Set[str],
    violation_predicate: str = "violates_constraint",
) -> Tuple[bool, List[str]]:
    """Evaluate facts against the KB and report violations.

    Returns (allowed, reasons). Reasons are human-readable strings
    derived from violation facts. Any internal error is reported as a
    single synthetic violation reason.
    """

    try:
        closure = run_strike_closure(kb, base_facts, params, active_profiles)
    except Exception as exc:  # noqa: BLE001
        return False, [f"internal_error: {type(exc).__name__}: {exc}"]

    violations = closure.get(violation_predicate, set())
    if not violations:
        return True, []

    reasons: List[str] = []
    for v in violations:
        if len(v) == 3:
            _, cid, msg = v
            reasons.append(f"{cid}: {msg}")
        else:
            reasons.append(f"Violation: {v}")
    return False, reasons


def lint_pack_meta(pack: PackMeta) -> Tuple[bool, str]:
    """Basic lint for a pack: sanity-check rule structure.

    This does not attempt full KPLC analysis but enforces:
    - Non-empty name and version.
    - At least one rule.
    - Heads and body atoms have string predicate names and tuple args.
    """

    if not pack.name:
        return False, "Pack name cannot be empty"
    if not pack.version:
        return False, f"Pack {pack.name} has empty version"
    if not pack.rules:
        return False, f"Pack {pack.name} has no rules"

    for idx, rule in enumerate(pack.rules):
        head_pred, head_args = rule.head
        if not isinstance(head_pred, str) or not head_pred:
            return False, f"Rule {idx} in pack {pack.name} has invalid head predicate"
        if not isinstance(head_args, tuple):
            return False, f"Rule {idx} in pack {pack.name} has non-tuple head args"
        for atom_idx, atom in enumerate(rule.body):
            body_pred, body_args = atom
            if not isinstance(body_pred, str) or not body_pred:
                return False, (
                    f"Rule {idx} in pack {pack.name} has invalid body predicate at index {atom_idx}"
                )
            if not isinstance(body_args, tuple):
                return False, (
                    f"Rule {idx} in pack {pack.name} has non-tuple body args at index {atom_idx}"
                )

    return True, ""
