#pragma once

// ============================================================================
// QuotientCache.h — Quotients via precomputed magic multipliers (Granlund & Montgomery).
//
// Replaces floor(n / d) with a single 128-bit multiply + shifts, avoiding
// the expensive hardware division instruction (especially on Intel where
// 64-bit div is ~30-90 cycles vs ~5-7 for 128-bit mul).
//
// For each divisor d in [1, B], we precompute:
//   lg[d]  = ceil(log2(d+1))  (tracked via a doubling counter)
//   m[d]   = ceil(((2^lg - d) << SHIFT) / d)
//
// Then:  floor(n / d) = (n + (m[d-1] * (uint128)n) >> SHIFT) >> lg[d-1]
//
// Domain: with SHIFT = 60 the multiply-shift identity is exact only for
// arguments below 2^60. Callers must keep n (and ceilDiv's n + p - 1)
// under that bound; the Mobius sieve asserts this at its entry point.
//
// Two cache types:
//   QuotientCache      — covers a contiguous range [1..B] of divisors
//   SieveQuotientCache — covers only the primes (indexed by prime index)
//
// Compile-time control:
//   -DUSE_DIVISION_FREE=1  enables division-free quotient methods (default on x86)
//   -DUSE_DIVISION_FREE=0  uses direct hardware division (default on ARM)
// ============================================================================

#include "types.h"
#include <cstdlib>
#include <vector>

#ifndef USE_DIVISION_FREE
#define USE_DIVISION_FREE 0
#endif
static constexpr bool UseDivisionFree = USE_DIVISION_FREE;

// ============================================================================
// QuotientCache: contiguous divisor range [1..B]
//
// Used in S1/S2 to replace n/x for x <= cbrt(2n).
// ============================================================================

struct QuotientCache {
    static constexpr int SHIFT = 60;

    UInt8*  lgs   = nullptr;
    UInt64* ms    = nullptr;
    UInt32  count = 0;

    QuotientCache() = default;
    QuotientCache(const QuotientCache&) = delete;
    QuotientCache& operator=(const QuotientCache&) = delete;

    ~QuotientCache() {
        std::free(lgs);
        std::free(ms);
    }

    void init(UInt32 B) {
        std::free(lgs);
        std::free(ms);
        count = B;
        lgs = static_cast<UInt8*>(std::calloc(B, sizeof(UInt8)));
        ms  = static_cast<UInt64*>(std::calloc(B, sizeof(UInt64)));

        // d = 1: lg = 0
        UInt8 lg = 0;
        lgs[0] = lg;
        __uint128_t diff = (1ULL << lg) - 1;
        ms[0] = static_cast<UInt64>(diff << SHIFT) + 1;

        // d = 2..B: lg tracks ceil(log2(d+1)) via doubling
        lg = 1;
        UInt64 lgat = 0, lgtarget = 1;
        for (UInt64 i = 1; i < B; ++i) {
            lgs[i] = lg;
            __uint128_t diff2 = (1ULL << lg) - i - 1;
            ms[i] = static_cast<UInt64>((diff2 << SHIFT) / (i + 1)) + 1;
            lgat++;
            if (lgat == lgtarget) {
                lgat = 0;
                lgtarget <<= 1;
                lg++;
            }
        }
    }

    // floor(n / d) for 1 <= d <= count, using 128-bit multiply + shift.
    inline UInt64 quotient(UInt64 n, UInt64 d) const {
        return (n + static_cast<UInt64>(
            (static_cast<__uint128_t>(ms[d - 1]) * static_cast<__uint128_t>(n)) >> SHIFT
        )) >> lgs[d - 1];
    }
};

// ============================================================================
// SieveQuotientCache: magic multipliers indexed by prime index.
//
// Used in the Mobius sieve to replace ceil(lo / p) for each prime p.
// The shift amount is computed on-the-fly via __builtin_clzll (1 cycle).
// ============================================================================

struct SieveQuotientCache {
    static constexpr int SHIFT = 60;

    UInt64* ms    = nullptr;
    UInt32  count = 0;

    SieveQuotientCache() = default;
    SieveQuotientCache(const SieveQuotientCache&) = delete;
    SieveQuotientCache& operator=(const SieveQuotientCache&) = delete;

    ~SieveQuotientCache() {
        clear();
    }

    void clear() {
        std::free(ms);
        ms = nullptr;
        count = 0;
    }

    void init(const UInt32* primes, UInt32 numPrimes) {
        std::free(ms);
        count = numPrimes;
        ms = static_cast<UInt64*>(std::calloc(count, sizeof(UInt64)));

        // Each (diff << SHIFT) / p is a 128-bit division (__udivti3 on
        // x86-64, ~50-100 cycles). Iterations are independent, and at large
        // sieve ranges there are millions of primes — parallelize rather
        // than pay ~a second of serial setup.
        #pragma omp parallel for schedule(static)
        for (UInt32 i = 0; i < count; ++i) {
            UInt64 p = primes[i];
            UInt8 l = static_cast<UInt8>(64 - __builtin_clzll(p));
            __uint128_t diff = (1ULL << l) - p;
            ms[i] = static_cast<UInt64>((diff << SHIFT) / p) + 1;
        }
    }

    // ceil(lo / p) where primeIdx is the index into the primes array.
    // Equivalent to (lo + p - 1) / p but division-free.
    inline UInt64 ceilDiv(UInt64 lo, UInt32 primeIdx, UInt64 p) const {
        UInt8 slg = static_cast<UInt8>(64 - __builtin_clzll(p));
        UInt64 val = lo + p - 1;
        return (val + static_cast<UInt64>(
            (static_cast<__uint128_t>(ms[primeIdx]) * static_cast<__uint128_t>(val)) >> SHIFT
        )) >> slg;
    }
};
