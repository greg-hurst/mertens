#pragma once

// ============================================================================
// MertensHurst.h — Compute the Mertens function M(n).
//
// Single entry point: MertensHurst(n), with absolute input bounds
// 10^8 <= n <= 10^26. Depending on the compiled Loop 2 sieve mode, the
// default split additionally needs n >= 1.921e8 to 2.372e8.
// The q210-coupled-record build gives a rigorous Int8 residual bound.
//
// Internally uses an O(n^{2/3}) combinatorial algorithm:
//   1. Initial sieve of mu over [1, nu] to identify squarefree indices
//   2. Loop 0/1: segmented sieve with S2+S1 updates (16-bit then 32-bit M)
//   3. Loop 2: large-segment sieve with S1 updates only (R aliases mu)
//   4. Back substitution to recover M(n) from partial values
// ============================================================================

#include "types.h"

// segmentCap:  cap on stored sieve entries for the large-segment phase
//              (the sieve costs about one byte per entry).
//              Default 12e9 (~12 GB); increase for very large n (e.g. 10^26).
//              Rounded up to nearest multiple of STENCIL_PERIOD (13860).
//
// uOverride:  set the sieve truncation point directly (0 = use formula).
//              Must satisfy 0 < u < n and the build-specific sieve cap.
//
// uFactor:    override the scaling factor in the u formula (0.0 = use default).
//              u = ceil(uFactor * (n / log(log(n)))^{2/3}).
//              Defaults by LOOP2_SIEVE_P:
//                P1: clamp(0.55 - 0.025*(log10(n) - 16), 0.30, 0.55)
//                P2: clamp(0.70 - 0.025*(log10(n) - 18), 0.30, 0.70)
//                P6: clamp(0.75 - 0.025*(log10(n) - 18), 0.30, 0.75)
//              Mutually exclusive with uOverride.
//
// nuRatio:    S1/S2 split ratio. get_nu(x) = floor(nuRatio * sqrt(x)).
//              The compiled default is 0.90 for P1, 0.95 for P2, and 1.00
//              for P6. Explicit values must be > 0. Affects performance,
//              not correctness.
double MertensHurstDefaultNuRatio();

Int64 MertensHurst(UInt128 n, bool profile = false, UInt64 segmentCap = 12000000000ULL,
                   UInt64 uOverride = 0, double uFactor = 0.0,
                   double nuRatio = MertensHurstDefaultNuRatio());
