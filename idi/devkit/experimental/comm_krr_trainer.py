from __future__ import annotations

from typing import Dict, Iterable

from idi_iann.config import TrainingConfig
from idi_iann.trainer import CommAction, QTrainer, StateKey

from idi.devkit.experimental.comms_krr_integration import (
    CommContext,
    evaluate_comm_action_with_krr,
)


class KRRWrappedQTrainer(QTrainer):
    def __init__(
        self,
        config: TrainingConfig,
        *,
        use_crypto_env: bool = False,
        seed: int | None = 0,
        subject_id: str = "agent",
        sensitivity: str = "high",
    ) -> None:
        super().__init__(
            config=config,
            use_crypto_env=use_crypto_env,
            seed=seed,
        )
        self._krr_subject_id = subject_id
        self._krr_sensitivity = sensitivity
        self._krr_vetoes = 0

    def _choose_comm_action(self, state: StateKey) -> CommAction:
        base_action = super()._choose_comm_action(state)
        risk_level = self._infer_risk_level(state)
        ctx = CommContext(
            subject_id=self._krr_subject_id,
            risk_level=risk_level,
            sensitivity=self._krr_sensitivity,
            action=base_action,
        )
        allowed, _ = evaluate_comm_action_with_krr(ctx)
        if allowed:
            return base_action
        self._krr_vetoes += 1
        for candidate in self._fallback_actions(base_action):
            ctx_candidate = CommContext(
                subject_id=self._krr_subject_id,
                risk_level=risk_level,
                sensitivity=self._krr_sensitivity,
                action=candidate,
            )
            allowed_candidate, _ = evaluate_comm_action_with_krr(ctx_candidate)
            if allowed_candidate:
                return candidate
        return base_action

    def _infer_risk_level(self, state: StateKey) -> str:
        if len(state) <= 2:
            return "low"
        try:
            return "high" if int(state[2]) > 0 else "low"
        except (TypeError, ValueError):
            return "low"

    def _fallback_actions(self, base_action: CommAction) -> Iterable[CommAction]:
        priority = ("alert", "positive", "persist", "silent")
        seen: set[CommAction] = set()
        for action in priority:
            if action == base_action:
                continue
            seen.add(action)
            yield action
        for action in self._communication.actions:
            if action == base_action or action in seen:
                continue
            yield action

    def stats(self) -> Dict[str, float]:
        base = super().stats()
        base["krr_vetoes"] = float(self._krr_vetoes)
        return base
