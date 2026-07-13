#include "MertensHurst.h"
#include "S2Q6.h"
#include "S1.h"
#include "S1Q6.h"
#include "OuterRecovery.h"
#include "SegmentedMertensSieve.h"

// Mirrors the default in SegmentedMobiusSieve.cpp; override with -DUSE_BUCKET_SIEVE=0.
#ifndef USE_BUCKET_SIEVE
#define USE_BUCKET_SIEVE 1
#endif

// The normal path uses exact outer Q=6 transforms for S1 and S2. Compile with
// both controls disabled to retain the original all-Q=2 path as an oracle.
#ifndef MERTENSHURST_S1_OUTER_Q6
#define MERTENSHURST_S1_OUTER_Q6 1
#endif
static constexpr bool UseS1OuterQ6 = MERTENSHURST_S1_OUTER_Q6;
static_assert(MERTENSHURST_S1_OUTER_Q6 == 0 || MERTENSHURST_S1_OUTER_Q6 == 1,
              "MERTENSHURST_S1_OUTER_Q6 must be 0 or 1");

#ifndef MERTENSHURST_S2_OUTER_Q6
#define MERTENSHURST_S2_OUTER_Q6 1
#endif
static constexpr bool UseS2OuterQ6 = MERTENSHURST_S2_OUTER_Q6;
static_assert(MERTENSHURST_S2_OUTER_Q6 == 0 || MERTENSHURST_S2_OUTER_Q6 == 1,
              "MERTENSHURST_S2_OUTER_Q6 must be 0 or 1");
static_assert(!UseS2OuterQ6 || UseS1OuterQ6,
              "S2 outer Q=6 requires the Q=6 global/S1 state");

#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <sys/time.h>

// ============================================================================
// Timing helpers
// ============================================================================

static inline void getDayTime(timeval& t) { gettimeofday(&t, NULL); }

static inline double getDuration(timeval& start, timeval& end) {
    return (Int64)(end.tv_sec) - (Int64)(start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0;
}

// Runtime-conditional profiling. These macros are only invoked once per sieve
// segment (not in inner loops), so the branch cost is negligible.
#define START_PROFILE() do { if (profile) getDayTime(start); } while(0)
#define END_PROFILE(t)  do { if (profile) { \
    getDayTime(end); (t) += getDuration(start, end); } } while(0)

namespace {

// ============================================================================
// Bounds on n: see INPUT_BOUNDS.md for the full analysis.
// The code enforces 10^8 <= n <= 10^25.
// ============================================================================

// Compute primes up to ceil(sqrt(limit)), respecting the sieve's minimum bound.
static std::vector<UInt32> sievePrimesUpToSqrt(UInt64 limit) {
    return SegmentedMobiusSieveCore::primesUpTo(std::max(
        (UInt32)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
        (UInt32)std::ceil(std::sqrt((double)limit))));
}

// ============================================================================
// MertensComputer — internal class that orchestrates the full computation.
// Hidden in an anonymous namespace; exposed via the MertensHurst() free function.
// ============================================================================

class MertensComputer {
public:
    Int64 compute(UInt128 n, bool profile, UInt64 segmentCap,
                  UInt64 uOverride, double uFactor, double nuRatio);

private:
    SegmentedMertensSieveCore mSieve;
    double mNuRatio;

    static UInt64 getSegmentSize(UInt128 n, UInt64 u);

    void initializeBounds(
        UInt128 n, UInt64 u,
        const Int8* mu,
        std::vector<UInt128>& partialArgs128,
        std::vector<UInt64>& partialArgs,
        std::vector<UInt64>& nus,
        std::vector<UInt64>& kappas,
        std::vector<UInt64>& kappas2,
        std::vector<UInt64>& partialArgsDivU,
        std::vector<UInt32>& hash,
        std::vector<UInt32>& hash2,
        std::vector<UInt64>& s2SplitCache
    );

    struct Chunk {
        UInt32 i;
        UInt64 a;
        UInt64 b;
    };

    static UInt64 isqrt_u128(const UInt128& a);
    UInt64 get_nu(const UInt128& x);
};

// ============================================================================
// Utility functions
// ============================================================================

UInt64 MertensComputer::isqrt_u128(const UInt128& a) {
    if (a == 0) return 0;
    long double da = (long double)a;
    UInt64 x = UInt64(sqrtl(da));

    while (UInt128(x+1) * UInt128(x+1) <= a) ++x;
    while (UInt128(x)   * UInt128(x)   >  a) --x;

    return x;
}

// nu_y = floor(c * sqrt(y)), c = 1.5 by default. Bigger c means more
// work in S1 (cheap per-term: just an array lookup) and less in S2
// (expensive: sums over M). 1.5 roughly balances sieve, S1, and S2.
UInt64 MertensComputer::get_nu(const UInt128& x) {
    return static_cast<UInt64>(mNuRatio * static_cast<double>(isqrt_u128(x)));
}

UInt64 MertensComputer::getSegmentSize(UInt128 n, UInt64 u) {
    constexpr UInt64 BF = SegmentedMobiusSieveCore::STENCIL_PERIOD;
    UInt64 len = static_cast<UInt64>(std::log10((double)n + 3));

    if (len < 14) {
        return std::max(UInt64(BF), UInt64(BF) * (static_cast<UInt64>(std::sqrt((double)n)) / BF));
    }

    UInt64 B = BF * static_cast<UInt64>((std::ceil(std::sqrt(2.0 * u)) + 1) / BF + 1);

    return std::max(UInt64(3104640ULL), len < 15 ? 4*B : 2*B);
}

// ============================================================================
// Initialize cached bounds
// ============================================================================

void MertensComputer::initializeBounds(
    UInt128 n, UInt64 u,
    const Int8* mu,
    std::vector<UInt128>& partialArgs128,
    std::vector<UInt64>& partialArgs,
    std::vector<UInt64>& nus,
    std::vector<UInt64>& kappas,
    std::vector<UInt64>& kappas2,
    std::vector<UInt64>& partialArgsDivU,
    std::vector<UInt32>& hash,
    std::vector<UInt32>& hash2,
    std::vector<UInt64>& s2SplitCache
) {
    UInt32 end = hash.size();
    UInt32 i = 1;

    for (UInt32 j = 1; j < end; ++j) {
        if (mu[j-1]) {
            hash[j] = i;
            hash2[i] = j;
            const UInt128 quotient = n / j;

            if (quotient > UInt128(1000000000000000000ULL))
                partialArgs128.push_back(quotient);
            else
                partialArgs[i] = UInt64(quotient);

            partialArgsDivU[i] = quotient / u;

            nus[i] = get_nu(quotient);

            const UInt128 q2 = quotient / 2;
            const UInt64  n2 = get_nu(q2);

            kappas[i]  =   quotient / (nus[i] + 1);
            kappas2[i] = 2 * (q2 / (n2 + 1));

            s2SplitCache[i] = n2;

            ++i;
        }
    }
}

// ============================================================================
// Main computation
// ============================================================================

Int64 MertensComputer::compute(UInt128 n, bool profile, UInt64 segmentCap,
                               UInt64 uOverride, double uFactor, double nuRatio) {
    // 10^8 <= n
    if (__builtin_expect(n < 100000000ULL, false)) {
        std::cerr << "Error: MertensComputer::compute requires n >= 10^8." << std::endl;
        std::abort();
    }

    // n <= 10^25
    if (__builtin_expect(n > UInt128(1000000000000ULL) * UInt128(10000000000000ULL), false)) {
        std::cerr << "Error: MertensComputer::compute requires n <= 10^25. "
                  << "See \"Upper bounds on n\" in MertensHurst.cpp to extend." << std::endl;
        std::abort();
    }

    // Validate mutually exclusive u parameters
    if (uOverride > 0 && uFactor > 0.0) {
        std::cerr << "Error: uOverride and uFactor are mutually exclusive." << std::endl;
        std::abort();
    }

    // Validate nuRatio
    if (nuRatio <= 0.0) {
        std::cerr << "Error: nuRatio must be positive (got " << nuRatio << ")." << std::endl;
        std::abort();
    }
    mNuRatio = nuRatio;

    struct timeval start, end;
    double t[8] = {0.0};

    constexpr UInt64 BF = SegmentedMobiusSieveCore::STENCIL_PERIOD;
    constexpr UInt64 min_B = BF * ((10000000ULL + BF - 1) / BF);

    // Compute u: direct override, factor override, or default formula
    UInt64 u;
    if (uOverride > 0) {
        u = uOverride;
    } else {
        double fac;
        if (uFactor > 0.0) {
            fac = uFactor;
        } else {
            fac = std::clamp(1.85 - 0.05 * std::log10((double)n), 0.75, 1.05);
        }
        u = std::ceil(fac * std::pow(std::cbrt((double)n / std::log(std::log((double)n))), 2));
    }

    // Enforce hard sieve caps on u, from any source (see INPUT_BOUNDS.md).
    {
        // UInt32 prime storage: sqrt(u) < 2^32. This also guarantees the
        // ceil-log2 byte-encoding requirement u < 2^64.
        UInt64 uMax = 4294967295ULL * 4294967295ULL;
#if USE_BUCKET_SIEVE
        // Bucket scheduler reach: sqrt(u) < LP_SIZE * M2.
        constexpr UInt64 reach = SegmentedMobiusSieveCore::schedulerReach();
        uMax = std::min(uMax, reach * reach);
#endif
        if (u > uMax) {
            std::cerr << "Error: sieve bound u = " << u << " exceeds the hard cap "
                      << uMax << " for this build (see INPUT_BOUNDS.md)." << std::endl;
            std::abort();
        }
    }

    // Validate u bounds
    if (u >= n) {
        std::cerr << "Error: u must be less than n (got u=" << u << ")." << std::endl;
        std::abort();
    }
    // Log-compactification sieve cap (see INPUT_BOUNDS.md constraint 5)
    if (u > 1157000000000000000ULL) {
        std::cerr << "Error: u exceeds log-compactification sieve cap of ~1.157e18 "
                  << "(got u=" << u << "). Decrease --u or --u-factor." << std::endl;
        std::abort();
    }
    // nu = n/u controls the S1/S2 boundary: S2 sums over [y/u, kappa_y],
    // S1 sums over [1, nu_y]. Sieve range is [1, nu_max] where nu_max = nu_1 = n/u.
    UInt64 nu = n / u;
    UInt64 B = std::min(min_B, getSegmentSize(n, u));

    // Round up to stencil alignment
    const UInt64 nuCAP = BF * (nu / BF + 1);

    // Initial sieve of mu over [1, nuCAP] for squarefree detection
    SegmentedMobiusSieveCore initialSieve;
    std::vector<UInt32> primes = sievePrimesUpToSqrt(nuCAP);
    initialSieve.initialize(nuCAP);
    initialSieve.sieve(1, nuCAP, primes);

    // mx = number of squarefree integers in [1, nu]. Only squarefree k
    // contribute to the outer sum (since mu(k)=0 otherwise).
    Int32 mx = 0;
    for (UInt32 i = 0; i < nu; ++i) {
        mx += initialSieve[i] & 1;
    }

    // hash[j] = compressed index of squarefree j, hash2[i] = original j.
    // only squarefree j get entries — skips the ~39% with mu(j)=0.
    std::vector<UInt32> hash(nu+1, 0);
    std::vector<UInt32> hash2(nu+1, 0);
    std::vector<UInt64> partialArgs(mx+1), nusVec(mx+1), kappas(mx+1), kappas2(mx+1), partialArgsDivU(mx+1);
    std::vector<UInt64> s2SplitCache(mx + 1, 0);

    std::vector<UInt128> partialArgs128(1, UInt128(0));

    std::vector<UInt64> qcache(2*mx+2, 0);

    initializeBounds(n, u, initialSieve.data(),
                     partialArgs128, partialArgs, nusVec, kappas, kappas2,
                     partialArgsDivU, hash, hash2, s2SplitCache);

    const UInt64 nuMax = nusVec[1];
    const UInt32 cnt128 = partialArgs128.size()-1;

    // The hot Q=6 S1 mapping is separate from the all-squarefree recovery map.
    // C and D are stored only for active k coprime to 6; A and B already live
    // in kappas/kappas2 for the current path.
    std::vector<S1Q6WorkItem> s1Q6Worklist;
    std::vector<UInt64> s1Q6Kappa3;
    std::vector<UInt64> s1Q6Kappa6;
    std::vector<UInt64> s2Q6Nu3;
    std::vector<UInt64> s2Q6Nu6;
    std::vector<UInt32> q6WorkIndexByCompact;
    if constexpr (UseS1OuterQ6) {
        s1Q6Worklist = buildS1Q6Worklist(
            hash2.data(), static_cast<UInt32>(mx), nu
        );
        s1Q6Kappa3.resize(s1Q6Worklist.size());
        s1Q6Kappa6.resize(s1Q6Worklist.size());
        if constexpr (UseS2OuterQ6) {
            s2Q6Nu3.resize(s1Q6Worklist.size());
            s2Q6Nu6.resize(s1Q6Worklist.size());
            q6WorkIndexByCompact.assign(mx + 1, 0);
        }
        for (std::size_t workIndex = 0; workIndex < s1Q6Worklist.size(); ++workIndex) {
            const UInt32 index = s1Q6Worklist[workIndex].compactIndex;
            const UInt128 y = index <= cnt128
                ? partialArgs128[index]
                : UInt128(partialArgs[index]);
            const UInt128 y3 = y / 3;
            const UInt128 y6 = y / 6;
            const UInt64 nu3 = get_nu(y3);
            const UInt64 nu6 = get_nu(y6);
            s1Q6Kappa3[workIndex] = static_cast<UInt64>(UInt128(3) * (y3 / (nu3 + 1)));
            s1Q6Kappa6[workIndex] = static_cast<UInt64>(UInt128(6) * (y6 / (nu6 + 1)));
            if constexpr (UseS2OuterQ6) {
                s2Q6Nu3[workIndex] = nu3;
                s2Q6Nu6[workIndex] = nu6;
                q6WorkIndexByCompact[index] = static_cast<UInt32>(workIndex + 1);
            }
        }
    }

    // Exact stored-state invariant. For square-free j, let
    //
    //   A_j  = 1 + kappa_j*M(nu_j), T1_j = S1(x/j,u), T2_j = S2(x/j).
    //
    // The all-Q=2 oracle stores
    //
    //   P_j = A_j - 1_{j odd} sum_{d|2, dj<=N} mu(d)(T1_{dj}+T2_{dj}).
    //
    // The mixed S1-Q6/S2-Q2 validation path stores
    //
    //   P_j = A_j
    //       - 1_{(j,6)=1} sum_{d|6, dj<=N} mu(d) T1_{dj}
    //       - 1_{j odd}   sum_{d|2, dj<=N} mu(d) T2_{dj}.
    //
    // The normal outer-Q6 path stores the complete transform
    //
    //   P_j = A_j
    //       - 1_{(j,6)=1} sum_{d|6, dj<=N} mu(d)(T1_{dj}+T2_{dj}).
    //
    // Every square-free slot, including auxiliary slots divisible by 2 or 3,
    // retains A_j. This makes sum_j mu(j)P_j equal to M(x) and permits the
    // same square-free Möbius back substitution. Compacting storage to the hot
    // worklist would lose the constant/kappa correction terms.
    //
    // partialValues starts with the leading 1 in A_j. The kappa term is added
    // incrementally below; the S1/S2 transformed terms are subtracted by their
    // respective hot loops. The cnt128 largest x/j use 128-bit accumulators.
    std::vector<Int64> partialValues(mx+1, 1);
    std::vector<Int128> partialValues128(cnt128+1, 1);
    partialValues[0] = 0;
    partialValues128[0] = 0;

    // initialize the primes used in SegmentedMertensSieveCore
    primes = sievePrimesUpToSqrt(u);

    // initialize the sieve for the main loops
    mSieve.initialize(B);

    // Granlund-Montgomery quotient cache: precompute magic multipliers for
    // d <= cbrt(2n), turning ~40-cycle 128-bit divisions into ~6-cycle
    // multiply-and-correct in the S1/S2 inner loops.
    // (no-op when USE_DIVISION_FREE=0)
    QuotientCache qCache;
    UInt64 dCAP = 0;
    if constexpr (UseDivisionFree) {
        dCAP = static_cast<UInt64>(std::ceil(std::cbrt(2.0 * (double)n)));
        qCache.init(dCAP);
    }

    // Compressed M: coarse[i] = M at every 256th position, R[i] = Int8 offset
    // from the nearest coarse sample. Lookup is M(k) = coarse[k>>8] + R[k].
    // 4x smaller than full Int32, which matters a lot for S1 cache behavior.
    //
    // Loop 0 uses Int16 coarse (safe up to n ~ 7.6e9),
    // Loop 1 switches to Int32 for the rest of [1, nuMax].
    constexpr int M_LOG_STRIDE = SegmentedMertensSieveCore::STRIDE_LOG;
    constexpr UInt64 M_STRIDE = UInt64(1) << M_LOG_STRIDE;
    auto coarseLength = [](UInt64 length) {
        return (length + M_STRIDE - 1) >> M_LOG_STRIDE;
    };

    std::vector<Int16> M16(coarseLength(B), 0);
    Int16 M16Prev = 0;

    std::vector<Int32> M32(coarseLength(B), 0);
    Int32 MPrev = 0;

    std::vector<Int8> R(B, 0);

    // pointers
    Int16* M16P = M16.data();
    Int32* MP   = M32.data();
    Int8*  RP   = R.data();
    Int8*  MuP  = mSieve.mobiusSieve().data();

    UInt64 L1 = 1;
    UInt64 L2 = B;

    // kappa_y * M(nu_y) correction term. picked up incrementally as nu
    // values land in processed sieve segments.
    Int32 j = mx;
    UInt64 osqrt = nusVec[j];

    // S2 work gets split into CHUNK_LEN-sized chunks for OpenMP.
    // needs to be > cbrt(n) so chunks stay in the quotient predictor range.
    std::vector<Chunk> chunks;
    UInt64 CHUNK_LEN = BF * (static_cast<UInt64>(4.0 * std::cbrt((double)n)) / BF + 1);
    Int32 mx0 = 1, mx1 = mx;
    while (mx0 < mx && nusVec[mx0] > CHUNK_LEN) { ++mx0; }

    if (mx0 >= (Int32)(nu/2)) {
        std::cerr << "Error: u is too large (u=" << u << ", nu=" << nu << ", mx0=" << mx0
                  << "). Decrease --u or --u-factor." << std::endl;
        std::abort();
    }

    auto applyS1Segment = [&](auto* mertensCoarse, const Int8* residual,
                              UInt64 segmentLo, UInt64 segmentHi) {
        if constexpr (UseS1OuterQ6) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (UInt64 workIndex = 0; workIndex < s1Q6Worklist.size(); ++workIndex) {
                const S1Q6WorkItem& item = s1Q6Worklist[workIndex];
                const UInt32 index = item.compactIndex;
                if (__builtin_expect(index > cnt128, true)) {
                    partialValues[index] -= evaluateS1OuterQ6(
                        item.mode, partialArgs[index], partialArgsDivU[index],
                        kappas[index], kappas2[index],
                        s1Q6Kappa3[workIndex], s1Q6Kappa6[workIndex],
                        segmentLo, segmentHi, mertensCoarse, residual, qCache, dCAP
                    );
                } else {
                    partialValues128[index] -= evaluateS1OuterQ6(
                        item.mode, partialArgs128[index], partialArgsDivU[index],
                        kappas[index], kappas2[index],
                        s1Q6Kappa3[workIndex], s1Q6Kappa6[workIndex],
                        segmentLo, segmentHi, mertensCoarse, residual, qCache, dCAP
                    );
                }
            }
        } else {
            #pragma omp parallel for schedule(dynamic, 1)
            for (UInt64 index = 1; index <= static_cast<UInt64>(mx); ++index) {
                const UInt64 outer = hash2[index];
                if ((outer & 1ULL) == 0) continue;

                const bool doAll = outer > nu / 2;
                if (__builtin_expect(index > cnt128, true)) {
                    apply_S1_updates(doAll, partialValues[index], partialArgs[index],
                                     partialArgsDivU[index], kappas[index], kappas2[index],
                                     segmentLo, segmentHi, mertensCoarse, residual,
                                     qCache, dCAP);
                } else {
                    const UInt64 evenIndex = doAll ? 0 : hash[2 * outer];
                    apply_S1_updates(doAll, partialValues128[index], partialArgs128[index],
                                     partialArgsDivU[index], kappas[index], kappas2[index],
                                     segmentLo, segmentHi, mertensCoarse, residual,
                                     qCache, dCAP,
                                     &qcache[2 * index], &qcache[2 * index + 1],
                                     &qcache[2 * evenIndex], &qcache[2 * evenIndex + 1]);
                }
            }
        }
    };

    auto isS2Active = [&](UInt32 index) {
        const UInt64 outer = hash2[index];
        if constexpr (UseS2OuterQ6)
            return outer % 6 == 1 || outer % 6 == 5;
        return (outer & 1ULL) != 0;
    };

    auto q6WorkIndex = [&](UInt32 index) -> UInt32 {
        const UInt32 encoded = q6WorkIndexByCompact[index];
        if (__builtin_expect(encoded == 0, false)) {
            std::cerr << "Internal error: missing outer-Q6 S2 work mapping." << std::endl;
            std::abort();
        }
        return encoded - 1;
    };

    auto q6OuterClass = [&](UInt32 workIndex) -> UInt32 {
        switch (s1Q6Worklist[workIndex].mode) {
            case S1OuterQ6Mode::Full6:         return 0;
            case S1OuterQ6Mode::Minus2Minus3: return 1;
            case S1OuterQ6Mode::Minus2:        return 2;
            case S1OuterQ6Mode::Single:        return 3;
        }
        std::abort();
    };

    auto applyS2_64 = [&](UInt32 index, UInt64 lo, UInt64 hi,
                          bool currentHalf) -> Int64 {
        if constexpr (UseS2OuterQ6) {
            const UInt32 workIndex = q6WorkIndex(index);
            return update_S2_q6<S2Q6Spec>(
                partialArgs[index], L1, lo, hi, MuP,
                q6OuterClass(workIndex), nusVec[index], s2SplitCache[index],
                s2Q6Nu3[workIndex], s2Q6Nu6[workIndex], qCache, dCAP
            );
        }
        return currentHalf
            ? update_S2<true>(partialArgs[index], L1, lo, hi, MuP,
                              nusVec[index], s2SplitCache[index], qCache, dCAP)
            : update_S2<false>(partialArgs[index], L1, lo, hi, MuP,
                               nusVec[index], s2SplitCache[index], qCache, dCAP);
    };

    auto applyS2_128 = [&](UInt32 index, UInt64 lo, UInt64 hi,
                           bool currentHalf) -> Int128 {
        if constexpr (UseS2OuterQ6) {
            const UInt32 workIndex = q6WorkIndex(index);
            return update_S2_q6_128<S2Q6Spec>(
                partialArgs128[index], L1, lo, hi, MuP,
                q6OuterClass(workIndex), nusVec[index], s2SplitCache[index],
                s2Q6Nu3[workIndex], s2Q6Nu6[workIndex]
            );
        }
        return currentHalf
            ? update_S2_128<true>(partialArgs128[index], L1, lo, hi, MuP,
                                  nusVec[index], s2SplitCache[index])
            : update_S2_128<false>(partialArgs128[index], L1, lo, hi, MuP,
                                   nusVec[index], s2SplitCache[index]);
    };

    // ========================================================================
    // Loop 0/1 iteration lambda
    // ========================================================================

    auto doLoop01Iteration = [&](auto& _MP,
                                   auto& _MPrev,
                                   const UInt64 bound,
                                   const int prof_base) {
        while (L2 < bound) {
            // ------------ Sieve Step ------------
            START_PROFILE();
            L2 = L1 + B - 1;
            mSieve.sieve(L1, L2, _MPrev, _MP, RP, primes);
            _MPrev = GET_M(_MP, RP, L1, L2);
            END_PROFILE(t[prof_base + 0]);

            // ------------ S2 Step ------------
            START_PROFILE();
            chunks.clear();
            for (UInt32 i = 1; i <= (UInt32)mx0; ++i) {
                if (isS2Active(i)) {
                    const UInt64 m = std::min(L2, nusVec[i]);
                    if (L1 > m) { mx0 = i-1; break; }

                    for (UInt64 a = L1; a <= m; a += CHUNK_LEN)
                        chunks.push_back(Chunk{i, a, std::min(a + CHUNK_LEN - 1, m)});
                }
            }

            // dynamic: cost per chunk varies wildly (small k is O(sqrt(x/k)),
            // large k is basically free)
            #pragma omp parallel for schedule(dynamic, 1)
            for (std::size_t tt = 0; tt < chunks.size(); ++tt) {
                const Chunk& c = chunks[tt];
                const UInt32 i = c.i;
                if (__builtin_expect(i > cnt128, true)) {
                    const Int64 v = applyS2_64(i, c.a, c.b, true);
                    #pragma omp atomic
                    partialValues[i] -= v;
                } else {
                    const Int128 v = applyS2_128(i, c.a, c.b, true);
                    #pragma omp critical
                    partialValues128[i] -= v;
                }
            }

            if (mx1 > mx0) {
                while (mx1 > mx0 && nusVec[mx1] < L1) { --mx1; }

                #pragma omp parallel for schedule(dynamic, 1)
                for (UInt64 i = mx0+1; i <= (UInt64)mx1; ++i) {
                    const UInt64 jj = hash2[i];
                    if (isS2Active(static_cast<UInt32>(i))) {
                        const bool currentHalf = jj <= nu / 2;
                        if (__builtin_expect(i > cnt128, true)) {
                            const Int64 v = applyS2_64(
                                static_cast<UInt32>(i), L1,
                                std::min(L2, nusVec[i]), currentHalf
                            );

                            partialValues[i] -= v;
                        } else {
                            const Int128 v = applyS2_128(
                                static_cast<UInt32>(i), L1,
                                std::min(L2, nusVec[i]), currentHalf
                            );

                            partialValues128[i] -= v;
                        }
                    }
                }
            }
            END_PROFILE(t[prof_base + 1]);

            // ------------ S1 Step ------------
            START_PROFILE();
            applyS1Segment(_MP, RP, L1, L2);
            END_PROFILE(t[prof_base + 2]);

            // ------------ Extra term Step ------------
            while (j && osqrt <= L2) {
                const Int64 mval = static_cast<Int64>(GET_M(_MP, RP, L1, osqrt));
                if (__builtin_expect((UInt32)j > cnt128, true))
                    partialValues[j] += static_cast<Int64>(kappas[j]) * mval;
                else
                    partialValues128[j] += kappas[j] * static_cast<Int128>(mval);

                osqrt = nusVec[--j];
            }

            L1 = L2 + 1;
        }
    };

    // |M(n)| < 128 for all n <= 7,613,644,886, so Int16 is safe for Loop 0.
    // MFRAC adds a little safety margin.
#define MFRAC 0.97
#define M16BITMAX UInt64(MFRAC * 7613644886ULL)

    // Loop 0: sieve [1, min(nuMax, M16BITMAX)] with Int16 M accumulators.
    // Both S1 and S2 updates are performed per segment.
    doLoop01Iteration(M16P, M16Prev, std::min(nuMax, M16BITMAX), 0);

    // Loop 1: continue with Int32 M accumulators up to nuMax.
    MPrev = M16Prev;
    doLoop01Iteration(MP, MPrev, nuMax, 3);

#undef M16BITMAX
#undef MFRAC

    nusVec = std::vector<UInt64>({});
    s2SplitCache = std::vector<UInt64>({});
    chunks = std::vector<Chunk>({});

    // Loop 2 only does S1 (S2 is done after Loop 0/1), so segments can be
    // much larger — up to ~12 billion elements — to cut down on the
    // per-segment cost of sweeping over partial values.
    const UInt64 segmentCapRounded = BF * ((segmentCap + BF - 1) / BF);
    B = 20 * 96 * BF * static_cast<UInt64>((std::ceil(std::sqrt(2.0 * u)) + 1) / BF + 1);
    B = std::min(B, segmentCapRounded);
    M32.resize(coarseLength(B));

    mSieve.mobiusSieve().fillFromStencil(B);

    // pointers — R aliases Mu (in-place prefix sum, saves ~12GB at large n)
    MP  = M32.data();
    RP  = mSieve.mobiusSieve().data();
    MuP = mSieve.mobiusSieve().data();

    // ========================================================================
    // Main loop #2
    // ========================================================================

    while (L2 < u) {
        START_PROFILE();
        L2 = std::min(L1 + B - 1, u);
        mSieve.sieveInPlace(L1, L2, MPrev, MP, primes);
        MPrev = GET_M(MP, RP, L1, L2);
        END_PROFILE(t[6]);

        START_PROFILE();
        applyS1Segment(MP, RP, L1, L2);
        END_PROFILE(t[7]);

        L1 = L2 + 1;
    }

    if (profile) {
        double tot = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];

        std::cout << std::endl;
        std::cout << "-------------- Parameters -------------------" << std::endl;
        std::cout << "              u: " << u << std::endl;
        std::cout << "        nuRatio: " << mNuRatio << std::endl;
        std::cout << std::endl;
        if (t[0] + t[1] + t[2] > 0.0) {
            std::cout << "--------------- Loop 1 16-bit ---------------" << std::endl;
            std::cout << "          Sieve: " << t[0] << ", " << (100.0*t[0]/tot) << "%" << std::endl;
            std::cout << "             S1: " << t[2] << ", " << (100.0*t[2]/tot) << "%" << std::endl;
            std::cout << "             S2: " << t[1] << ", " << (100.0*t[1]/tot) << "%" << std::endl;
        }
        if (t[3] + t[4] + t[5] > 0.0) {
            std::cout << "--------------- Loop 1 32-bit ---------------" << std::endl;
            std::cout << "          Sieve: " << t[3] << ", " << (100.0*t[3]/tot) << "%" << std::endl;
            std::cout << "             S1: " << t[5] << ", " << (100.0*t[5]/tot) << "%" << std::endl;
            std::cout << "             S2: " << t[4] << ", " << (100.0*t[4]/tot) << "%" << std::endl;
        }
        if (t[6] + t[7] > 0.0) {
            std::cout << "--------------- Loop 2 32-bit ---------------" << std::endl;
            std::cout << "          Sieve: " << t[6] << ", " << (100.0*t[6]/tot) << "%" << std::endl;
            std::cout << "             S1: " << t[7] << ", " << (100.0*t[7]/tot) << "%" << std::endl;
        }
        std::cout << "------------------ Totals -------------------" << std::endl;
        std::cout << "          Sieve: " << (t[0]+t[3]+t[6]) << ", " << (100.0*(t[0]+t[3]+t[6])/tot) << "%" << std::endl;
        std::cout << "             S1: " << (t[2]+t[5]+t[7]) << ", " << (100.0*(t[2]+t[5]+t[7])/tot) << "%" << std::endl;
        std::cout << "             S2: " << (t[1]+t[4]     ) << ", " << (100.0*(t[1]+t[4]     )/tot) << "%" << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << std::endl;
    }

    // ========================================================================
    // Square-free Mobius recovery. For both the Q2 and Q6 invariants,
    //
    //   P_i = sum_{m: i*m<=nu, i*m square-free} W_{i*m}.
    //
    // Decreasing-i back substitution recovers W_i, and W_1=M(x).
    // ========================================================================

    if (cnt128 > 0) {
        partialValues128.resize(partialValues.size());
        for (UInt32 i = cnt128+1; i <= (UInt32)mx; ++i)
            partialValues128[i] = (Int128)partialValues[i];

        recoverSquarefreeInPlace(partialValues128, hash, static_cast<UInt32>(nu));
        return (Int64)partialValues128[1];
    }

    recoverSquarefreeInPlace(partialValues, hash, static_cast<UInt32>(nu));
    return partialValues[1];
}

} // anonymous namespace

Int64 MertensHurst(UInt128 n, bool profile, UInt64 segmentCap,
                   UInt64 uOverride, double uFactor, double nuRatio) {
    MertensComputer computer;
    return computer.compute(n, profile, segmentCap, uOverride, uFactor, nuRatio);
}

#undef START_PROFILE
#undef END_PROFILE
