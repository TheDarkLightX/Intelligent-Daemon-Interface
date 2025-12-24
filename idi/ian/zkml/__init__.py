"""
zkML Decentralized Training Module for IAN.

Provides zero-configuration onboarding, gradient commitments,
secure aggregation, and Nova folding proof integration.

Modules:
- onboarding: Zero-config hardware detection and network discovery
- commitments: Gradient commitment protocol
- secagg: Secure aggregation for privacy-preserving training
"""

from .onboarding import (
    HardwareCapabilities,
    GPUInfo,
    TauBinaryInfo,
    TauNetInfo,
    TauCapabilities,
    NetworkEndpoint,
    NetworkDiscovery,
    TaskRequirements,
    TaskMatcher,
    OnboardingState,
    OnboardingOrchestrator,
    OnboardingError,
)

from .commitments import (
    GradientTensor,
    GradientCommitment,
    GradientCommitmentScheme,
    AggregatedCommitment,
    CommitmentAggregator,
    CommitmentError,
)

from .secagg import (
    SecAggError,
    SecAggPhase,
    ShamirSecretSharing,
    ParticipantKeys,
    PairwiseMasking,
    MaskedGradient,
    SecAggSession,
    SecAggParticipant,
)

__all__ = [
    # Onboarding
    "HardwareCapabilities",
    "GPUInfo",
    "TauBinaryInfo",
    "TauNetInfo",
    "TauCapabilities",
    "NetworkEndpoint",
    "NetworkDiscovery",
    "TaskRequirements",
    "TaskMatcher",
    "OnboardingState",
    "OnboardingOrchestrator",
    "OnboardingError",
    # Commitments
    "GradientTensor",
    "GradientCommitment",
    "GradientCommitmentScheme",
    "AggregatedCommitment",
    "CommitmentAggregator",
    "CommitmentError",
    # Secure Aggregation
    "SecAggError",
    "SecAggPhase",
    "ShamirSecretSharing",
    "ParticipantKeys",
    "PairwiseMasking",
    "MaskedGradient",
    "SecAggSession",
    "SecAggParticipant",
]
