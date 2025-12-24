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

__all__ = [
    # Onboarding
    "HardwareCapabilities",
    "GPUInfo",
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
]
