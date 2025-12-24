"""
Zero-Configuration Onboarding for IAN zkML Training.

Implements Poka Yoke (mistake-proofing) principles:
- Prevention: Invalid states are unrepresentable via types
- Detection: Immediate feedback on errors
- Contact method: Input validation via types
- Fixed-value method: Required fields enforced
- Motion-step method: State machine enforces sequence

Security:
- All data classes are frozen (immutable)
- Hardware capabilities validated at construction
- Network endpoints verified before use
- Task requirements checked before joining
- Endpoint caps prevent DoS from mDNS/DNS flooding
- Semaphore limits concurrent connection tests

IMPORTANT: Network Discovery Trust Model
----------------------------------------
This module discovers endpoints via mDNS/DNS-SD which are inherently
unauthenticated. The discovered endpoints are ONLY used to establish
initial connectivity. The application layer MUST:

1. Establish TLS with certificate validation (or pinned certs)
2. Perform authenticated handshake (e.g., BLS signature challenge)
3. Verify node identity against a trusted registry

Do NOT trust discovered endpoints until application-layer authentication
is complete. This module only handles discovery, not authentication.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import shutil
import socket
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

# Size constants
BYTES_PER_GB = 1024 * 1024 * 1024
BYTES_PER_MB = 1024 * 1024

# Timeouts
DISCOVERY_TIMEOUT_SECONDS = 10.0
CONNECTION_TEST_TIMEOUT_SECONDS = 5.0

# Security limits
MAX_ENDPOINTS_TO_TEST = 20  # Cap concurrent connection tests to prevent DoS
MAX_DISCOVERED_ENDPOINTS = 100  # Cap total discovered endpoints
MAX_TASK_NAME_LEN = 256
MAX_TASK_ID_LEN = 128
MAX_REWARD_VALUE = 1e12  # Prevent overflow/NaN issues

# Bootstrap nodes (fallback)
BOOTSTRAP_NODES = [
    ("bootstrap1.ian.network", 8443),
    ("bootstrap2.ian.network", 8443),
]


class OnboardingError(Exception):
    """Error during onboarding process."""
    pass


class HardwareDetectionError(OnboardingError):
    """Error detecting hardware capabilities."""
    pass


class NetworkDiscoveryError(OnboardingError):
    """Error discovering network endpoints."""
    pass


# =============================================================================
# Hardware Detection (Poka Yoke: Contact Method - Validate Attributes)
# =============================================================================


@dataclass(frozen=True)
class GPUInfo:
    """
    Detected GPU information (immutable).
    
    Poka Yoke: Frozen dataclass prevents modification after detection.
    All fields validated at construction time.
    """
    name: str
    vram_bytes: int
    compute_capability: Tuple[int, int]  # e.g., (8, 6) for Ampere
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    
    def __post_init__(self) -> None:
        if self.vram_bytes < 0:
            raise ValueError("vram_bytes cannot be negative")
        if len(self.compute_capability) != 2:
            raise ValueError("compute_capability must be (major, minor)")
    
    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / BYTES_PER_GB
    
    def supports_fp16(self) -> bool:
        """Check if GPU supports FP16 (compute capability >= 5.3)."""
        return self.compute_capability >= (5, 3)
    
    def supports_bf16(self) -> bool:
        """Check if GPU supports BF16 (compute capability >= 8.0)."""
        return self.compute_capability >= (8, 0)
    
    def supports_int8(self) -> bool:
        """Check if GPU supports INT8 tensor cores (>= 7.5)."""
        return self.compute_capability >= (7, 5)


@dataclass(frozen=True)
class HardwareCapabilities:
    """
    Auto-detected hardware capabilities (immutable).
    
    Poka Yoke principles:
    - Frozen: Cannot be modified after detection
    - Validated: All fields checked at construction
    - Complete: All required fields present (no Optional for required)
    """
    # CPU
    cpu_cores: int
    cpu_threads: int
    cpu_freq_mhz: int
    
    # Memory
    ram_bytes: int
    available_ram_bytes: int
    
    # Storage
    disk_total_bytes: int
    disk_available_bytes: int
    
    # GPU (optional)
    gpus: Tuple[GPUInfo, ...] = field(default_factory=tuple)
    
    # Platform
    os_name: str = field(default_factory=platform.system)
    os_version: str = field(default_factory=platform.release)
    python_version: str = field(default_factory=platform.python_version)
    
    def __post_init__(self) -> None:
        if self.cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        if self.ram_bytes <= 0:
            raise ValueError("ram_bytes must be positive")
        if self.disk_total_bytes < 0:
            raise ValueError("disk_total_bytes cannot be negative")
        # Validate available values are within bounds
        if self.available_ram_bytes < 0:
            raise ValueError("available_ram_bytes cannot be negative")
        if self.available_ram_bytes > self.ram_bytes:
            raise ValueError("available_ram_bytes cannot exceed ram_bytes")
        if self.disk_available_bytes < 0:
            raise ValueError("disk_available_bytes cannot be negative")
        if self.disk_available_bytes > self.disk_total_bytes:
            raise ValueError("disk_available_bytes cannot exceed disk_total_bytes")
    
    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / BYTES_PER_GB
    
    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / BYTES_PER_GB
    
    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0
    
    @property
    def total_vram_bytes(self) -> int:
        return sum(gpu.vram_bytes for gpu in self.gpus)
    
    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_bytes / BYTES_PER_GB
    
    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        """Return GPU with most VRAM."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.vram_bytes)
    
    @classmethod
    def auto_detect(cls) -> "HardwareCapabilities":
        """
        Zero-config hardware detection.
        
        Detects CPU, RAM, disk, and GPU automatically.
        Poka Yoke: Returns validated, immutable capabilities.
        """
        import psutil
        
        # CPU detection
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or 1
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_mhz = int(cpu_freq.max) if cpu_freq else 0
        except Exception:
            cpu_freq_mhz = 0
        
        # Memory detection
        mem = psutil.virtual_memory()
        ram_bytes = mem.total
        available_ram_bytes = mem.available
        
        # Disk detection (use home directory)
        home = os.path.expanduser("~")
        disk = shutil.disk_usage(home)
        disk_total_bytes = disk.total
        disk_available_bytes = disk.free
        
        # GPU detection
        gpus = cls._detect_gpus()
        
        return cls(
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            cpu_freq_mhz=cpu_freq_mhz,
            ram_bytes=ram_bytes,
            available_ram_bytes=available_ram_bytes,
            disk_total_bytes=disk_total_bytes,
            disk_available_bytes=disk_available_bytes,
            gpus=tuple(gpus),
        )
    
    @staticmethod
    def _detect_gpus() -> List[GPUInfo]:
        """Detect NVIDIA GPUs using pynvml."""
        gpus: List[GPUInfo] = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_bytes = memory.total
                
                # Get compute capability
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                
                # Get CUDA version
                cuda_version = None
                try:
                    cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                    cuda_version = f"{cuda_ver // 1000}.{(cuda_ver % 1000) // 10}"
                except Exception:
                    pass
                
                # Get driver version
                driver_version = None
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode("utf-8")
                except Exception:
                    pass
                
                gpus.append(GPUInfo(
                    name=name,
                    vram_bytes=vram_bytes,
                    compute_capability=(major, minor),
                    cuda_version=cuda_version,
                    driver_version=driver_version,
                ))
            
            pynvml.nvmlShutdown()
        except ImportError:
            pass  # pynvml not installed
        except Exception:
            pass  # GPU detection failed
        
        return gpus
    
    def summary(self) -> str:
        """Human-readable summary of capabilities."""
        lines = [
            f"CPU: {self.cpu_cores} cores / {self.cpu_threads} threads @ {self.cpu_freq_mhz} MHz",
            f"RAM: {self.ram_gb:.1f} GB ({self.available_ram_gb:.1f} GB available)",
            f"Disk: {self.disk_total_bytes / BYTES_PER_GB:.1f} GB ({self.disk_available_bytes / BYTES_PER_GB:.1f} GB available)",
        ]
        
        if self.gpus:
            for i, gpu in enumerate(self.gpus):
                lines.append(f"GPU {i}: {gpu.name} ({gpu.vram_gb:.1f} GB VRAM)")
        else:
            lines.append("GPU: None detected")
        
        return "\n".join(lines)


# =============================================================================
# Task Requirements (Poka Yoke: Fixed-Value Method - Ensure Completeness)
# =============================================================================


@dataclass(frozen=True)
class TaskRequirements:
    """
    Requirements for a training task (immutable).
    
    Poka Yoke: Defines minimum requirements that MUST be met.
    Tasks are only shown if hardware meets ALL requirements.
    """
    task_id: str
    task_name: str
    
    # Memory requirements
    min_ram_bytes: int = 0
    min_vram_bytes: int = 0
    
    # Storage requirements
    min_disk_bytes: int = 0
    
    # Compute requirements
    min_compute_capability: Optional[Tuple[int, int]] = None
    requires_gpu: bool = False
    requires_fp16: bool = False
    requires_bf16: bool = False
    
    # Reward info
    reward_per_batch: float = 0.0
    estimated_batches_per_hour: float = 0.0
    
    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id is required")
        if len(self.task_id) > MAX_TASK_ID_LEN:
            raise ValueError(f"task_id exceeds {MAX_TASK_ID_LEN} characters")
        if not self.task_name:
            raise ValueError("task_name is required")
        if len(self.task_name) > MAX_TASK_NAME_LEN:
            raise ValueError(f"task_name exceeds {MAX_TASK_NAME_LEN} characters")
        if self.min_ram_bytes < 0:
            raise ValueError("min_ram_bytes cannot be negative")
        if self.min_vram_bytes < 0:
            raise ValueError("min_vram_bytes cannot be negative")
        if self.min_disk_bytes < 0:
            raise ValueError("min_disk_bytes cannot be negative")
        # Validate compute capability shape
        if self.min_compute_capability is not None:
            if len(self.min_compute_capability) != 2:
                raise ValueError("min_compute_capability must be (major, minor)")
            if any(v < 0 for v in self.min_compute_capability):
                raise ValueError("min_compute_capability values must be non-negative")
        # Validate reward values (prevent NaN/overflow)
        import math
        if math.isnan(self.reward_per_batch) or math.isinf(self.reward_per_batch):
            raise ValueError("reward_per_batch must be finite")
        if self.reward_per_batch < 0 or self.reward_per_batch > MAX_REWARD_VALUE:
            raise ValueError(f"reward_per_batch must be 0-{MAX_REWARD_VALUE}")
        if math.isnan(self.estimated_batches_per_hour) or math.isinf(self.estimated_batches_per_hour):
            raise ValueError("estimated_batches_per_hour must be finite")
        if self.estimated_batches_per_hour < 0:
            raise ValueError("estimated_batches_per_hour cannot be negative")
        # GPU-specific requirements imply requires_gpu
        if (self.min_vram_bytes > 0 or self.min_compute_capability or 
            self.requires_fp16 or self.requires_bf16) and not self.requires_gpu:
            raise ValueError("GPU-specific requirements set but requires_gpu is False")
    
    @property
    def estimated_reward_per_hour(self) -> float:
        return self.reward_per_batch * self.estimated_batches_per_hour


# =============================================================================
# Task Matcher (Poka Yoke: Only Show Compatible Tasks)
# =============================================================================


class TaskMatcher:
    """
    Match hardware capabilities to compatible tasks.
    
    Poka Yoke principle: Users can ONLY see tasks they can complete.
    This prevents:
    - Joining tasks that will OOM
    - Wasting time on incompatible work
    - Frustrating failed attempts
    """
    
    def can_run_task(
        self,
        hardware: HardwareCapabilities,
        task: TaskRequirements,
    ) -> bool:
        """
        Check if hardware can run task.
        
        Returns True only if ALL requirements are met.
        """
        # RAM check
        if task.min_ram_bytes > hardware.available_ram_bytes:
            return False
        
        # Disk check
        if task.min_disk_bytes > hardware.disk_available_bytes:
            return False
        
        # GPU required check
        if task.requires_gpu and not hardware.has_gpu:
            return False
        
        # VRAM check (use best single GPU, not total across all GPUs)
        # This prevents OOM when task needs single large GPU but node has multiple small GPUs
        if task.min_vram_bytes > 0:
            if not hardware.has_gpu or not hardware.best_gpu:
                return False
            if task.min_vram_bytes > hardware.best_gpu.vram_bytes:
                return False
        
        # Compute capability check
        if task.min_compute_capability and hardware.best_gpu:
            if hardware.best_gpu.compute_capability < task.min_compute_capability:
                return False
        
        # FP16 check
        if task.requires_fp16 and hardware.best_gpu:
            if not hardware.best_gpu.supports_fp16():
                return False
        
        # BF16 check
        if task.requires_bf16 and hardware.best_gpu:
            if not hardware.best_gpu.supports_bf16():
                return False
        
        return True
    
    def filter_compatible_tasks(
        self,
        hardware: HardwareCapabilities,
        tasks: List[TaskRequirements],
    ) -> List[TaskRequirements]:
        """
        Filter to only compatible tasks.
        
        Poka Yoke: Returns ONLY tasks the hardware can complete.
        """
        return [t for t in tasks if self.can_run_task(hardware, t)]
    
    def rank_tasks(
        self,
        hardware: HardwareCapabilities,
        tasks: List[TaskRequirements],
    ) -> List[TaskRequirements]:
        """
        Rank compatible tasks by estimated reward.
        
        Returns tasks sorted by reward_per_hour descending.
        """
        compatible = self.filter_compatible_tasks(hardware, tasks)
        return sorted(
            compatible,
            key=lambda t: t.estimated_reward_per_hour,
            reverse=True,
        )
    
    def auto_select_best_task(
        self,
        hardware: HardwareCapabilities,
        tasks: List[TaskRequirements],
    ) -> Optional[TaskRequirements]:
        """
        Zero-config task selection: Pick optimal task automatically.
        
        Returns the highest reward task that hardware can run,
        or None if no compatible tasks.
        """
        ranked = self.rank_tasks(hardware, tasks)
        return ranked[0] if ranked else None


# =============================================================================
# Network Discovery (Poka Yoke: Auto-discover, No Manual Entry)
# =============================================================================


@dataclass(frozen=True)
class NetworkEndpoint:
    """
    Discovered network endpoint (immutable).
    
    Poka Yoke: Endpoints are validated before use.
    Users cannot manually enter endpoints in zero-config mode.
    """
    host: str
    port: int
    node_id: Optional[str] = None
    network_name: str = "mainnet"
    is_bootstrap: bool = False
    
    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host is required")
        if not (1 <= self.port <= 65535):
            raise ValueError("port must be 1-65535")
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class ConnectionTestResult:
    """Result of connection test to an endpoint."""
    endpoint: NetworkEndpoint
    success: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    protocol_version: Optional[str] = None


class NetworkDiscovery:
    """
    Zero-config network discovery.
    
    Discovery order (fallback chain):
    1. mDNS/Bonjour on local network
    2. Well-known DNS records
    3. Hardcoded bootstrap nodes
    
    Poka Yoke: User CANNOT manually enter endpoints.
    This prevents typos and malicious node connections.
    """
    
    MDNS_SERVICE_TYPE = "_ian._tcp.local."
    DNS_SERVICE_DOMAIN = "_ian._tcp.ian.network"
    
    def __init__(
        self,
        timeout_seconds: float = DISCOVERY_TIMEOUT_SECONDS,
    ) -> None:
        self._timeout = timeout_seconds
        self._discovered: List[NetworkEndpoint] = []
    
    async def discover(self) -> List[NetworkEndpoint]:
        """
        Discover network endpoints automatically.
        
        Returns list of discovered endpoints, ordered by preference.
        """
        endpoints: List[NetworkEndpoint] = []
        
        # 1. Try mDNS discovery
        mdns_endpoints = await self._discover_mdns()
        endpoints.extend(mdns_endpoints)
        
        # 2. Try DNS-SD discovery
        dns_endpoints = await self._discover_dns()
        endpoints.extend(dns_endpoints)
        
        # 3. Add bootstrap nodes as fallback
        for host, port in BOOTSTRAP_NODES:
            endpoints.append(NetworkEndpoint(
                host=host,
                port=port,
                is_bootstrap=True,
            ))
        
        # Separate bootstrap nodes (always include them)
        bootstrap_eps: List[NetworkEndpoint] = []
        other_eps: List[NetworkEndpoint] = []
        for ep in endpoints:
            if ep.is_bootstrap:
                bootstrap_eps.append(ep)
            else:
                other_eps.append(ep)
        
        # Deduplicate non-bootstrap by address and cap (DoS prevention)
        seen: Set[str] = set()
        unique_other: List[NetworkEndpoint] = []
        for ep in other_eps:
            if ep.address not in seen:
                seen.add(ep.address)
                unique_other.append(ep)
                # Leave room for bootstrap nodes
                if len(unique_other) >= MAX_DISCOVERED_ENDPOINTS - len(bootstrap_eps):
                    break
        
        # Always append bootstrap nodes at the end (guaranteed inclusion)
        self._discovered = unique_other + bootstrap_eps
        return self._discovered
    
    async def _discover_mdns(self) -> List[NetworkEndpoint]:
        """Discover via mDNS/Bonjour."""
        endpoints: List[NetworkEndpoint] = []
        
        try:
            from zeroconf import ServiceBrowser, Zeroconf
            from zeroconf.asyncio import AsyncZeroconf
            
            discovered: List[NetworkEndpoint] = []
            
            seen_addrs: Set[str] = set()
            
            class Listener:
                def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    # Cap unique discovered endpoints to prevent memory DoS
                    if len(discovered) >= MAX_DISCOVERED_ENDPOINTS:
                        return
                    info = zc.get_service_info(type_, name)
                    if info:
                        for addr in info.addresses:
                            if len(discovered) >= MAX_DISCOVERED_ENDPOINTS:
                                break
                            try:
                                # Handle both IPv4 (4 bytes) and IPv6 (16 bytes)
                                if len(addr) == 4:
                                    ip = socket.inet_ntoa(addr)
                                elif len(addr) == 16:
                                    ip = socket.inet_ntop(socket.AF_INET6, addr)
                                else:
                                    continue  # Unknown address format
                            except (OSError, ValueError):
                                continue  # Invalid address
                            
                            # Dedup at discovery time to prevent cap exhaustion
                            addr_key = f"{ip}:{info.port}"
                            if addr_key in seen_addrs:
                                continue
                            seen_addrs.add(addr_key)
                            
                            discovered.append(NetworkEndpoint(
                                host=ip,
                                port=info.port,
                                node_id=name,
                            ))
                
                def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    pass
                
                def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
                    pass
            
            async with AsyncZeroconf() as azc:
                browser = ServiceBrowser(
                    azc.zeroconf,
                    self.MDNS_SERVICE_TYPE,
                    Listener(),
                )
                await asyncio.sleep(min(self._timeout, 3.0))
                browser.cancel()
            
            endpoints = discovered
            
        except ImportError:
            pass  # zeroconf not installed
        except Exception:
            pass  # mDNS discovery failed
        
        return endpoints
    
    async def _discover_dns(self) -> List[NetworkEndpoint]:
        """Discover via DNS SRV records."""
        endpoints: List[NetworkEndpoint] = []
        
        try:
            import dns.resolver
            
            answers = dns.resolver.resolve(
                self.DNS_SERVICE_DOMAIN,
                "SRV",
            )
            
            for rdata in answers:
                endpoints.append(NetworkEndpoint(
                    host=str(rdata.target).rstrip("."),
                    port=rdata.port,
                ))
        except ImportError:
            pass  # dnspython not installed
        except Exception:
            pass  # DNS discovery failed
        
        return endpoints
    
    async def test_connection(
        self,
        endpoint: NetworkEndpoint,
    ) -> ConnectionTestResult:
        """
        Test connection to an endpoint.
        
        Poka Yoke Detection: Verify before use.
        """
        import time
        
        start = time.monotonic()
        
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(endpoint.host, endpoint.port),
                timeout=CONNECTION_TEST_TIMEOUT_SECONDS,
            )
            
            latency_ms = (time.monotonic() - start) * 1000
            
            writer.close()
            await writer.wait_closed()
            
            return ConnectionTestResult(
                endpoint=endpoint,
                success=True,
                latency_ms=latency_ms,
            )
            
        except asyncio.TimeoutError:
            return ConnectionTestResult(
                endpoint=endpoint,
                success=False,
                error="Connection timeout",
            )
        except Exception as e:
            return ConnectionTestResult(
                endpoint=endpoint,
                success=False,
                error=str(e),
            )
    
    async def find_best_endpoint(self) -> Optional[NetworkEndpoint]:
        """
        Find the best (lowest latency) reachable endpoint.
        
        Zero-config: Automatically selects optimal endpoint.
        
        Security: Caps concurrent connections to prevent DoS from
        malicious mDNS/DNS flooding with many endpoints.
        """
        if not self._discovered:
            await self.discover()
        
        # Cap endpoints to test (DoS prevention)
        # Always include at least one bootstrap node to prevent mDNS flooding
        # from starving trusted endpoints
        non_bootstrap = [ep for ep in self._discovered if not ep.is_bootstrap]
        bootstrap = [ep for ep in self._discovered if ep.is_bootstrap]
        
        # Reserve slots for bootstrap nodes
        max_non_bootstrap = max(0, MAX_ENDPOINTS_TO_TEST - min(2, len(bootstrap)))
        endpoints_to_test = non_bootstrap[:max_non_bootstrap] + bootstrap[:2]
        
        # Use semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(5)
        
        async def test_with_limit(ep: NetworkEndpoint) -> ConnectionTestResult:
            async with semaphore:
                return await self.test_connection(ep)
        
        results = await asyncio.gather(*[
            test_with_limit(ep) for ep in endpoints_to_test
        ])
        
        # Filter successful, sort by latency
        successful = [r for r in results if r.success and r.latency_ms is not None]
        if not successful:
            return None
        
        best = min(successful, key=lambda r: r.latency_ms or float("inf"))
        return best.endpoint


# =============================================================================
# Onboarding Orchestrator (Poka Yoke: Motion-Step - Enforce Sequence)
# =============================================================================


class OnboardingState(Enum):
    """
    Onboarding state machine states.
    
    Poka Yoke Motion-Step: States must progress in order.
    Cannot skip steps or go backwards.
    """
    INIT = auto()
    DETECTING_HARDWARE = auto()
    HARDWARE_DETECTED = auto()
    DISCOVERING_NETWORK = auto()
    NETWORK_DISCOVERED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    SELECTING_TASK = auto()
    READY = auto()
    ERROR = auto()


# Valid state transitions (Poka Yoke: Only forward progress allowed)
_VALID_TRANSITIONS: Dict[OnboardingState, FrozenSet[OnboardingState]] = {
    OnboardingState.INIT: frozenset({OnboardingState.DETECTING_HARDWARE, OnboardingState.ERROR}),
    OnboardingState.DETECTING_HARDWARE: frozenset({OnboardingState.HARDWARE_DETECTED, OnboardingState.ERROR}),
    OnboardingState.HARDWARE_DETECTED: frozenset({OnboardingState.DISCOVERING_NETWORK, OnboardingState.ERROR}),
    OnboardingState.DISCOVERING_NETWORK: frozenset({OnboardingState.NETWORK_DISCOVERED, OnboardingState.ERROR}),
    OnboardingState.NETWORK_DISCOVERED: frozenset({OnboardingState.CONNECTING, OnboardingState.ERROR}),
    OnboardingState.CONNECTING: frozenset({OnboardingState.CONNECTED, OnboardingState.ERROR}),
    OnboardingState.CONNECTED: frozenset({OnboardingState.SELECTING_TASK, OnboardingState.ERROR}),
    OnboardingState.SELECTING_TASK: frozenset({OnboardingState.READY, OnboardingState.ERROR}),
    OnboardingState.READY: frozenset({OnboardingState.ERROR}),
    OnboardingState.ERROR: frozenset(),  # Terminal state
}


class OnboardingOrchestrator:
    """
    Orchestrates zero-config onboarding process.
    
    Poka Yoke principles:
    - State machine prevents invalid transitions
    - Each step validates before proceeding
    - Errors halt progress (fail-safe)
    - Progress callbacks for UI feedback
    """
    
    def __init__(
        self,
        on_state_change: Optional[Callable[[OnboardingState], None]] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._state = OnboardingState.INIT
        self._on_state_change = on_state_change
        self._on_progress = on_progress
        
        self._hardware: Optional[HardwareCapabilities] = None
        self._discovery: Optional[NetworkDiscovery] = None
        self._endpoint: Optional[NetworkEndpoint] = None
        self._task: Optional[TaskRequirements] = None
        self._error: Optional[str] = None
    
    @property
    def state(self) -> OnboardingState:
        return self._state
    
    @property
    def hardware(self) -> Optional[HardwareCapabilities]:
        return self._hardware
    
    @property
    def endpoint(self) -> Optional[NetworkEndpoint]:
        return self._endpoint
    
    @property
    def selected_task(self) -> Optional[TaskRequirements]:
        return self._task
    
    @property
    def error(self) -> Optional[str]:
        return self._error
    
    @property
    def is_ready(self) -> bool:
        return self._state == OnboardingState.READY
    
    def _transition(self, new_state: OnboardingState) -> None:
        """
        Transition to new state (with validation).
        
        Poka Yoke: Rejects invalid transitions.
        """
        # Idempotent: already in target state
        if self._state == new_state:
            return
        
        valid = _VALID_TRANSITIONS.get(self._state, frozenset())
        if new_state not in valid:
            raise OnboardingError(
                f"Invalid transition: {self._state.name} -> {new_state.name}"
            )
        
        self._state = new_state
        if self._on_state_change:
            try:
                self._on_state_change(new_state)
            except Exception:
                pass  # Don't let callback errors break state machine
    
    def _progress(self, message: str) -> None:
        """Report progress to callback."""
        if self._on_progress:
            try:
                self._on_progress(message)
            except Exception:
                pass  # Don't let callback errors break onboarding
    
    async def run(
        self,
        available_tasks: Optional[List[TaskRequirements]] = None,
    ) -> bool:
        """
        Run complete onboarding process.
        
        Returns True if ready, False if error.
        Zero-config: Requires no user input.
        """
        try:
            # Step 1: Detect hardware
            self._transition(OnboardingState.DETECTING_HARDWARE)
            self._progress("Detecting hardware...")
            self._hardware = HardwareCapabilities.auto_detect()
            self._transition(OnboardingState.HARDWARE_DETECTED)
            self._progress(f"Hardware detected: {self._hardware.cpu_cores} cores, {self._hardware.ram_gb:.1f}GB RAM")
            if self._hardware.has_gpu:
                self._progress(f"GPU: {self._hardware.best_gpu.name} ({self._hardware.best_gpu.vram_gb:.1f}GB)")
            
            # Step 2: Discover network
            self._transition(OnboardingState.DISCOVERING_NETWORK)
            self._progress("Discovering network...")
            self._discovery = NetworkDiscovery()
            endpoints = await self._discovery.discover()
            self._transition(OnboardingState.NETWORK_DISCOVERED)
            self._progress(f"Found {len(endpoints)} network endpoints")
            
            # Step 3: Connect to best endpoint
            self._transition(OnboardingState.CONNECTING)
            self._progress("Connecting to network...")
            self._endpoint = await self._discovery.find_best_endpoint()
            if not self._endpoint:
                raise NetworkDiscoveryError("No reachable endpoints found")
            self._transition(OnboardingState.CONNECTED)
            self._progress(f"Connected to {self._endpoint.address}")
            
            # Step 4: Select task (if tasks provided)
            self._transition(OnboardingState.SELECTING_TASK)
            if available_tasks:
                self._progress("Selecting optimal task...")
                matcher = TaskMatcher()
                self._task = matcher.auto_select_best_task(self._hardware, available_tasks)
                if self._task:
                    self._progress(f"Selected: {self._task.task_name}")
                else:
                    self._progress("No compatible tasks available")
            else:
                self._progress("No tasks to select (will fetch from network)")
            
            # Step 5: Ready
            self._transition(OnboardingState.READY)
            self._progress("Ready to contribute!")
            
            return True
            
        except OnboardingError as e:
            self._error = str(e)
            self._transition(OnboardingState.ERROR)
            self._progress(f"Error: {e}")
            return False
        except Exception as e:
            self._error = str(e)
            self._transition(OnboardingState.ERROR)
            self._progress(f"Unexpected error: {e}")
            return False
