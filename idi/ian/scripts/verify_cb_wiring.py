import asyncio
import time
from idi.ian.network.resilience import CircuitBreaker, CircuitBreakerConfig, CircuitState
from idi.ian.network.kernels import circuit_breaker_fsm_ref as kernel

async def verify_wiring():
    print("--- Verifying CircuitBreaker Wiring ---")
    
    # 1. Setup
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=0.1
    )
    cb = CircuitBreaker("test-cb", config)
    
    # Check Initial State
    print(f"Initial State: {cb.state} (Expected: CLOSED)")
    assert cb.state == CircuitState.CLOSED
    assert cb.stats.consecutive_failures == 0
    
    # 2. Trigger Failures
    print("\n[Step 1] Triggering 2 Failures (Threshold 3)")
    await cb._on_failure(Exception("Fail 1"))
    await cb._on_failure(Exception("Fail 2"))
    
    print(f"State: {cb.state}")
    print(f"Failures: {cb._stats.consecutive_failures}")
    
    assert cb.state == CircuitState.CLOSED
    assert cb._kstate.consecutive_failures == 2
    assert cb.stats.consecutive_failures == 2 # Sync check
    
    # 3. Trip Circuit
    print("\n[Step 2] Triggering 3rd Failure -> TRIP")
    await cb._on_failure(Exception("Fail 3"))
    
    print(f"State: {cb.state} (Expected: OPEN)")
    assert cb.state == CircuitState.OPEN
    assert cb.stats.consecutive_failures == 3 # Should be passed from kernel
    
    # 4. Timeout to Half-Open
    print("\n[Step 3] Waiting for timeout...")
    await asyncio.sleep(0.15)
    
    # _before_call triggers the transition check logic
    try:
        await cb._before_call()
    except Exception:
        pass # HALF_OPEN transition happens inside _before_call logic (via kernel apply)
        
    # Wait, in the wired version:
    # _before_call calls _apply_kernel('timeout_to_half_open') if time passed
    # This updates state.
    
    print(f"State: {cb.state} (Expected: HALF_OPEN)")
    assert cb.state == CircuitState.HALF_OPEN
    assert cb._kstate.state == "HALF_OPEN"
    
    # 5. Success in Half-Open
    print("\n[Step 4] Success 1/2 in Half-Open")
    await cb._on_success()
    print(f"State: {cb.state} (Expected: HALF_OPEN)")
    assert cb.state == CircuitState.HALF_OPEN
    
    print("\n[Step 5] Success 2/2 in Half-Open -> CLOSE")
    await cb._on_success()
    print(f"State: {cb.state} (Expected: CLOSED)")
    assert cb.state == CircuitState.CLOSED
    assert cb._kstate.consecutive_successes == 0 # Kernel resets on close
    
    print("\n[SUCCESS] CircuitBreaker logic is fully driven by Verified Kernel.")

if __name__ == "__main__":
    asyncio.run(verify_wiring())
