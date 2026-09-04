#pragma once

// ============================================================================
// MertensHurst.h — Compute the Mertens function M(n).
//
// Single entry point: MertensHurst(n) for 10^8 <= n <= 10^25.
//
// Internally uses an O(n^{2/3}) combinatorial algorithm:
//   1. Initial sieve of mu over [1, nu] to identify squarefree indices
//   2. Loop 0/1: segmented sieve with S2+S1 updates (16-bit then 32-bit M)
//   3. Loop 2: large-segment sieve with S1 updates only (R aliases mu)
//   4. Back substitution to recover M(n) from partial values
// ============================================================================

#include "types.h"

// segmentCap:  cap on the sieve segment length for the large-segment phase,
//              in integers (the sieve costs about one byte per integer).
//              Default 12e9 (~12 GB); increase for very large n (e.g. 10^25).
//              Rounded up to nearest multiple of STENCIL_PERIOD (13860).
//
// uOverride:  set the sieve truncation point directly (0 = use formula).
//              Must satisfy 0 < u < n and the build-specific sieve cap.
//
// uFactor:    override the scaling factor in the u formula (0.0 = use default).
//              u = ceil(uFactor * (n / log(log(n)))^{2/3}).
//              Mutually exclusive with uOverride.
//
// nuRatio:    S1/S2 split ratio. get_nu(x) = floor(nuRatio * sqrt(x)).
//              Default 1.5. Must be > 0. Affects performance, not correctness.
Int64 MertensHurst(UInt128 n, bool profile = false, UInt64 segmentCap = 12000000000ULL,
                   UInt64 uOverride = 0, double uFactor = 0.0, double nuRatio = 1.5);
