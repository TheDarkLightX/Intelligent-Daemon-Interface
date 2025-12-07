#!/usr/bin/env python3
"""
PRNG (LFSR) Test Suite

Tests the 16-bit Galois LFSR implementation:
- Period verification (should be 65535)
- Distribution uniformity
- Statistical randomness tests
- Multi-party seed generation

Copyright DarkLightX/Dana Edwards
"""

import sys
from typing import List, Dict
from collections import Counter
import math


class GaloisLFSR:
    """
    16-bit Galois LFSR with polynomial x^16 + x^14 + x^13 + x^11 + 1
    Feedback taps: 0xB400 (bits 15, 13, 12, 10)
    """
    
    POLYNOMIAL = 0xB400  # Taps at bits 15, 13, 12, 10
    
    def __init__(self, seed: int = 1):
        if seed == 0:
            seed = 1  # Avoid zero state (fixed point)
        self.state = seed & 0xFFFF
        self.initial_state = self.state
        
    def step(self) -> int:
        """Generate next random value"""
        lsb = self.state & 1
        self.state >>= 1
        if lsb:
            self.state ^= self.POLYNOMIAL
        return self.state
    
    def get_random(self, extra_entropy: int = 0) -> int:
        """Get random value XORed with extra entropy"""
        return self.step() ^ extra_entropy


def test_period():
    """Verify LFSR period is maximal (2^16 - 1 = 65535)"""
    print("\n--- TEST: LFSR Period ---")
    
    lfsr = GaloisLFSR(seed=1)
    initial = lfsr.state
    count = 0
    max_iter = 70000
    
    while count < max_iter:
        lfsr.step()
        count += 1
        if lfsr.state == initial:
            break
    
    expected = 65535
    print(f"  Period: {count}")
    print(f"  Expected: {expected}")
    
    if count == expected:
        print("  PASS: Maximal period achieved")
        return True
    else:
        print("  FAIL: Period mismatch")
        return False


def test_distribution():
    """Test uniform distribution of random values"""
    print("\n--- TEST: Distribution Uniformity ---")
    
    lfsr = GaloisLFSR(seed=12345)
    samples = 65535  # Full period
    
    # Count values in buckets (16 buckets)
    buckets = Counter()
    for _ in range(samples):
        val = lfsr.step()
        bucket = val // 4096  # 0-15
        buckets[bucket] += 1
    
    expected_per_bucket = samples / 16
    
    print(f"  Samples: {samples}")
    print(f"  Expected per bucket: {expected_per_bucket:.1f}")
    print(f"  Actual distribution:")
    
    max_deviation = 0
    for i in range(16):
        count = buckets[i]
        deviation = abs(count - expected_per_bucket) / expected_per_bucket * 100
        max_deviation = max(max_deviation, deviation)
        print(f"    Bucket {i:2d}: {count:5d} ({deviation:+.1f}%)")
    
    # Allow up to 5% deviation (very generous for uniform)
    if max_deviation < 5:
        print(f"  PASS: Max deviation {max_deviation:.1f}% < 5%")
        return True
    else:
        print(f"  FAIL: Max deviation {max_deviation:.1f}% >= 5%")
        return False


def test_no_short_cycles():
    """Verify no short cycles exist"""
    print("\n--- TEST: No Short Cycles ---")
    
    # Test multiple seeds
    seeds = [1, 100, 1000, 12345, 65535]
    min_cycle = float('inf')
    
    for seed in seeds:
        lfsr = GaloisLFSR(seed=seed)
        initial = lfsr.state
        count = 0
        
        for _ in range(100):  # Check first 100 steps
            lfsr.step()
            count += 1
            if lfsr.state == initial:
                min_cycle = min(min_cycle, count)
                break
    
    if min_cycle > 100:
        print(f"  No short cycles found (checked 100 steps from {len(seeds)} seeds)")
        print("  PASS: No short cycles")
        return True
    else:
        print(f"  FAIL: Short cycle of length {min_cycle} found")
        return False


def test_multi_party_seed():
    """Test multi-party XOR seed generation"""
    print("\n--- TEST: Multi-Party Seed Generation ---")
    
    # Simulate 3 parties
    party1_secret = 0x1234
    party2_secret = 0x5678
    party3_secret = 0x9ABC
    
    # XOR all secrets
    seed = party1_secret ^ party2_secret ^ party3_secret
    
    print(f"  Party 1 secret: 0x{party1_secret:04X}")
    print(f"  Party 2 secret: 0x{party2_secret:04X}")
    print(f"  Party 3 secret: 0x{party3_secret:04X}")
    print(f"  Combined seed:  0x{seed:04X}")
    
    # Generate some random values
    lfsr = GaloisLFSR(seed=seed)
    values = [lfsr.step() for _ in range(10)]
    
    print(f"  First 10 values: {values}")
    
    # Verify determinism
    lfsr2 = GaloisLFSR(seed=seed)
    values2 = [lfsr2.step() for _ in range(10)]
    
    if values == values2:
        print("  PASS: Deterministic with same seed")
        return True
    else:
        print("  FAIL: Non-deterministic")
        return False


def test_delay_distribution():
    """Test distribution of random delays (0-3)"""
    print("\n--- TEST: Delay Distribution (0-3) ---")
    
    lfsr = GaloisLFSR(seed=42)
    samples = 10000
    
    delays = Counter()
    for _ in range(samples):
        delay = lfsr.step() % 4
        delays[delay] += 1
    
    expected = samples / 4
    
    print(f"  Samples: {samples}")
    print(f"  Delay distribution:")
    
    for i in range(4):
        count = delays[i]
        pct = count / samples * 100
        print(f"    Delay {i}: {count:5d} ({pct:.1f}%)")
    
    # Check roughly uniform (within 10%)
    all_close = all(abs(delays[i] - expected) < expected * 0.1 for i in range(4))
    
    if all_close:
        print("  PASS: Uniform delay distribution")
        return True
    else:
        print("  FAIL: Non-uniform delays")
        return False


def test_lottery_probability():
    """Test 10% lottery burn probability"""
    print("\n--- TEST: Lottery Burn Probability (10%) ---")
    
    lfsr = GaloisLFSR(seed=99)
    samples = 100000
    
    burns = sum(1 for _ in range(samples) if lfsr.step() % 10 == 0)
    
    actual_pct = burns / samples * 100
    expected_pct = 10.0
    
    print(f"  Samples: {samples}")
    print(f"  Burns: {burns}")
    print(f"  Actual: {actual_pct:.2f}%")
    print(f"  Expected: {expected_pct}%")
    
    # Allow ±1% deviation
    if abs(actual_pct - expected_pct) < 1:
        print("  PASS: Lottery probability correct")
        return True
    else:
        print("  FAIL: Probability deviation too large")
        return False


def test_perturbation_range():
    """Test threshold perturbation range (±2)"""
    print("\n--- TEST: Perturbation Range (±2) ---")
    
    lfsr = GaloisLFSR(seed=777)
    samples = 10000
    
    perturbations = Counter()
    for _ in range(samples):
        # perturbation = (rand % 5) - 2
        p = (lfsr.step() % 5) - 2
        perturbations[p] += 1
    
    print(f"  Samples: {samples}")
    print(f"  Perturbation distribution:")
    
    for i in range(-2, 3):
        count = perturbations[i]
        pct = count / samples * 100
        print(f"    {i:+d}: {count:5d} ({pct:.1f}%)")
    
    # Should be roughly 20% each
    all_present = all(perturbations[i] > samples * 0.15 for i in range(-2, 3))
    
    if all_present:
        print("  PASS: All perturbation values represented")
        return True
    else:
        print("  FAIL: Missing perturbation values")
        return False


def main():
    """Run all PRNG tests"""
    print("=" * 70)
    print("PRNG (LFSR) TEST SUITE")
    print("=" * 70)
    
    tests = [
        test_period,
        test_distribution,
        test_no_short_cycles,
        test_multi_party_seed,
        test_delay_distribution,
        test_lottery_probability,
        test_perturbation_range,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

