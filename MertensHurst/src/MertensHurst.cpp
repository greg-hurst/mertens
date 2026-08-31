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

// Coherent Q=6 uses one split point for every active divisor in an outer
// group. Keep it separate from the production default until the complete
// optimized contract has passed its exactness and timing gates.
#ifndef MERTENSHURST_COHERENT_Q6
#define MERTENSHURST_COHERENT_Q6 0
#endif
static constexpr bool UseCoherentQ6 = MERTENSHURST_COHERENT_Q6;
static_assert(MERTENSHURST_COHERENT_Q6 == 0 || MERTENSHURST_COHERENT_Q6 == 1,
              "MERTENSHURST_COHERENT_Q6 must be 0 or 1");
static_assert(!UseCoherentQ6 || (UseS1OuterQ6 && UseS2OuterQ6),
              "coherent Q=6 requires both outer Q=6 transforms");
static_assert(!MERTENSHURST_COHERENT_PERIOD36 || UseCoherentQ6,
              "coherent period-36 requires coherent Q=6 splits");

// Production needs only M(x), which follows directly from a signed sum of the
// compact partial values. Retain full back substitution as an exact oracle.
#ifndef MERTENSHURST_FULL_RECOVERY
#define MERTENSHURST_FULL_RECOVERY 0
#endif
static constexpr bool UseFullRecovery = MERTENSHURST_FULL_RECOVERY;
static_assert(MERTENSHURST_FULL_RECOVERY == 0 || MERTENSHURST_FULL_RECOVERY == 1,
              "MERTENSHURST_FULL_RECOVERY must be 0 or 1");
static_assert(!UseCoherentQ6 || !UseFullRecovery,
              "coherent Q=6 is closed to final-value recovery");

#ifndef MERTENSHURST_UNORDERED_S2
#define MERTENSHURST_UNORDERED_S2 0
#endif
static constexpr bool UseUnorderedS2 = MERTENSHURST_UNORDERED_S2;
static_assert(MERTENSHURST_UNORDERED_S2 == 0
              || MERTENSHURST_UNORDERED_S2 == 1,
              "MERTENSHURST_UNORDERED_S2 must be 0 or 1");
static_assert(!UseUnorderedS2 || (UseCoherentQ6
                                  && MERTENSHURST_COHERENT_PERIOD36
                                  && !UseFullRecovery),
              "unordered S2 requires final-value coherent period-36 Q=6");

// Completing missing outer-Q6 members adds only scalar zeros when the common
// split remains below y/6. If any hot group misses that guard, the invocation
// falls back as a whole to the retained truncated coherent invariant.
#ifndef MERTENSHURST_Q6_ZERO_COMPLETION
#define MERTENSHURST_Q6_ZERO_COMPLETION 0
#endif
static constexpr bool UseQ6ZeroCompletion =
    MERTENSHURST_Q6_ZERO_COMPLETION;
static_assert(MERTENSHURST_Q6_ZERO_COMPLETION == 0
              || MERTENSHURST_Q6_ZERO_COMPLETION == 1,
              "MERTENSHURST_Q6_ZERO_COMPLETION must be 0 or 1");
static_assert(!UseQ6ZeroCompletion || (UseCoherentQ6 && !UseFullRecovery),
              "Q6 zero completion requires final-value coherent Q6");

// Completed groups have no auxiliary partial-value interpretation. The
// optimized profile stores only their hot accumulators in worklist order.
#ifndef MERTENSHURST_Q6_COMPACT_HOT_STATE
#define MERTENSHURST_Q6_COMPACT_HOT_STATE 0
#endif
static constexpr bool UseQ6CompactHotState =
    MERTENSHURST_Q6_COMPACT_HOT_STATE;
static_assert(MERTENSHURST_Q6_COMPACT_HOT_STATE == 0
              || MERTENSHURST_Q6_COMPACT_HOT_STATE == 1,
              "MERTENSHURST_Q6_COMPACT_HOT_STATE must be 0 or 1");
static_assert(!UseQ6CompactHotState
              || (UseQ6ZeroCompletion && UseUnorderedS2),
              "compact Q6 state requires zero completion and unordered S2");

#ifndef MERTENSHURST_VALIDATE_UNORDERED_S2
#define MERTENSHURST_VALIDATE_UNORDERED_S2 0
#endif
static constexpr bool ValidateUnorderedS2 =
    MERTENSHURST_VALIDATE_UNORDERED_S2;
static_assert(MERTENSHURST_VALIDATE_UNORDERED_S2 == 0
              || MERTENSHURST_VALIDATE_UNORDERED_S2 == 1,
              "MERTENSHURST_VALIDATE_UNORDERED_S2 must be 0 or 1");
static_assert(!ValidateUnorderedS2 || UseUnorderedS2,
              "unordered S2 validation requires unordered S2");

#ifndef MERTENSHURST_VALIDATE_UNORDERED_S2_WIDE
#define MERTENSHURST_VALIDATE_UNORDERED_S2_WIDE 0
#endif
static constexpr bool ValidateUnorderedS2Wide =
    MERTENSHURST_VALIDATE_UNORDERED_S2_WIDE;
static_assert(MERTENSHURST_VALIDATE_UNORDERED_S2_WIDE == 0
              || MERTENSHURST_VALIDATE_UNORDERED_S2_WIDE == 1,
              "MERTENSHURST_VALIDATE_UNORDERED_S2_WIDE must be 0 or 1");
static_assert(!ValidateUnorderedS2Wide || ValidateUnorderedS2,
              "forced-wide unordered S2 requires its ordered validator");
static constexpr bool NeedsOuterHash = !UseS1OuterQ6 || UseFullRecovery;

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <sys/time.h>
#include <vector>

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

template <typename T>
void releaseVector(std::vector<T>& values) {
    std::vector<T>().swap(values);
}

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

    template<bool StoreHash, bool StoreRecoverySigns>
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
        std::vector<UInt64>& s2SplitCache,
        std::vector<UInt64>& negativeRecoverySigns
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

// nu_y = floor(c * sqrt(y)), c = 1.4 by default. Bigger c means more
// work in S1 (cheap per-term: just an array lookup) and less in S2
// (expensive: sums over M). 1.4 roughly balances sieve, S1, and S2.
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

template<bool StoreHash, bool StoreRecoverySigns>
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
    std::vector<UInt64>& s2SplitCache,
    std::vector<UInt64>& negativeRecoverySigns
) {
    const UInt32 end = static_cast<UInt32>(n / u) + 1;
    UInt32 i = 1;

    for (UInt32 j = 1; j < end; ++j) {
        if (mu[j-1]) {
            if constexpr (StoreHash)
                hash[j] = i;
            hash2[i] = j;
            if constexpr (StoreRecoverySigns) {
                if (mu[j-1] < 0)
                    negativeRecoverySigns[i >> 6] |= UInt64(1) << (i & 63);
            }
            const UInt128 quotient = n / j;

            if (quotient > UInt128(1000000000000000000ULL))
                partialArgs128.push_back(quotient);
            else
                partialArgs[i] = UInt64(quotient);

            partialArgsDivU[i] = quotient / u;

            nus[i] = get_nu(quotient);
            if constexpr (!UseCoherentQ6) {
                const UInt128 q2 = quotient / 2;
                const UInt64 n2 = get_nu(q2);
                kappas[i] = quotient / (nus[i] + 1);
                kappas2[i] = 2 * (q2 / (n2 + 1));
                s2SplitCache[i] = n2;
            }

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
    double t[10] = {0.0};

    constexpr UInt64 BF = SegmentedMobiusSieveCore::STENCIL_PERIOD;
    constexpr UInt64 min_B = BF * ((10000000ULL + BF - 1) / BF);

    // Compute u: direct override, factor override, or tuned default
    UInt64 u;
    if (uOverride > 0) {
        u = uOverride;
    } else {
        double fac = 0.75;
        if (uFactor > 0.0)
            fac = uFactor;
        u = std::ceil(fac * std::pow(std::cbrt((double)n / std::log(std::log((double)n))), 2));
    }

    // Enforce hard sieve caps on u, from any source (see INPUT_BOUNDS.md).
    {
        // UInt32 prime storage: sqrt(u) < 2^32. This also guarantees the
        // ceil-log2 byte-encoding requirement u < 2^64.
        UInt64 uMax = 4294967295ULL * 4294967295ULL;
#if USE_BUCKET_SIEVE
        // Bucket scheduler reach: sqrt(u) <= (LP_SIZE - 1) * M2.
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

    Int32 mx = 0;
    std::vector<UInt32> primes;
    std::vector<UInt32> hash;
    std::vector<UInt32> hash2;
    std::vector<UInt64> partialArgs;
    std::vector<UInt64> nusVec;
    std::vector<UInt64> kappas;
    std::vector<UInt64> kappas2;
    std::vector<UInt64> partialArgsDivU;
    std::vector<UInt64> s2SplitCache;
    std::vector<UInt64> negativeRecoverySigns;
    std::vector<UInt128> partialArgs128(1, UInt128(0));

    // The initial mu values are needed only while constructing the compact
    // square-free maps and cached bounds. Scope the large buffer accordingly.
    {
        SegmentedMobiusSieveCore initialSieve;
        primes = sievePrimesUpToSqrt(nuCAP);
        initialSieve.initialize(nuCAP);
        initialSieve.sieve(1, nuCAP, primes);

        // mx = number of squarefree integers in [1, nu]. Only squarefree k
        // contribute to the outer sum (since mu(k)=0 otherwise).
        for (UInt32 i = 0; i < nu; ++i)
            mx += initialSieve[i] & 1;

        // hash[j] maps original j to its compact square-free index; hash2 is
        // the reverse map and therefore needs only mx + 1 entries.
        if constexpr (NeedsOuterHash)
            hash.assign(nu+1, 0);
        hash2.assign(mx+1, 0);
        partialArgs.resize(mx+1);
        nusVec.resize(mx+1);
        if constexpr (!UseCoherentQ6) {
            kappas.resize(mx+1);
            kappas2.resize(mx+1);
            s2SplitCache.resize(mx+1);
        }
        partialArgsDivU.resize(mx+1);
        if constexpr (!UseFullRecovery)
            negativeRecoverySigns.resize(static_cast<UInt32>(mx) / 64 + 1);

        initializeBounds<NeedsOuterHash, !UseFullRecovery>(
            n, u, initialSieve.data(), partialArgs128, partialArgs, nusVec,
            kappas, kappas2, partialArgsDivU, hash, hash2, s2SplitCache,
            negativeRecoverySigns
        );
    }

    std::vector<UInt64> qcache;
    if constexpr (!UseS1OuterQ6)
        qcache.resize(2*mx+2, 0);

    const UInt64 nuMax = nusVec[1];
    const UInt32 cnt128 = partialArgs128.size()-1;

    // Keep S1/S2 bounds for active Q=6 entries contiguous. State needed by
    // every square-free entry remains in the full compact arrays.
    std::vector<S1Q6WorkItem> s1Q6Worklist;
    std::vector<UInt64> q6PartialArgs;
    std::vector<UInt128> q6PartialArgs128;
    std::vector<UInt64> q6PartialArgsDivU;
    std::vector<UInt64> q6Kappa1;
    std::vector<UInt64> q6Kappa2;
    std::vector<UInt64> q6S2SplitCache;
    std::vector<UInt64> q6CommonNu;
    std::vector<UInt64> q6CommonKappa;
    std::vector<UInt64> q6BoundaryFactor;
    std::vector<UInt32> q6Bases;
    std::vector<Int8> q6Signs;
    std::vector<UInt64> s1Q6Kappa3;
    std::vector<UInt64> s1Q6Kappa6;
    std::vector<UInt64> s2Q6Nu3;
    std::vector<UInt64> s2Q6Nu6;
    std::vector<UInt32> q6WorkIndexByCompact;
    UInt64 q6WideCount = 0;
    UInt32 compactHalf = 0;
    bool useQ6ZeroCompletion = UseQ6ZeroCompletion;
    bool q6ZeroCompletionFallback = false;
    bool useQ6CompactHotState = false;
    if constexpr (UseS1OuterQ6) {
        s1Q6Worklist = buildS1Q6Worklist(
            hash2.data(), static_cast<UInt32>(mx), nu
        );
        if constexpr (UseQ6ZeroCompletion) {
            for (const S1Q6WorkItem& item : s1Q6Worklist) {
                const UInt32 index = item.compactIndex;
                const UInt128 y = index <= cnt128
                    ? partialArgs128[index]
                    : UInt128(partialArgs[index]);
                if (UInt128(nusVec[index]) >= y / 6) {
                    useQ6ZeroCompletion = false;
                    q6ZeroCompletionFallback = true;
                    break;
                }
            }
        }
        if constexpr (UseQ6CompactHotState)
            useQ6CompactHotState = useQ6ZeroCompletion;
        if constexpr (!UseCoherentQ6) {
            s1Q6Kappa3.resize(s1Q6Worklist.size());
            s1Q6Kappa6.resize(s1Q6Worklist.size());
            if constexpr (UseS2OuterQ6) {
                s2Q6Nu3.resize(s1Q6Worklist.size());
                s2Q6Nu6.resize(s1Q6Worklist.size());
            }
        }
        if constexpr (UseCoherentQ6) {
            q6CommonNu.resize(s1Q6Worklist.size());
            q6CommonKappa.resize(s1Q6Worklist.size());
            q6BoundaryFactor.resize(s1Q6Worklist.size());
        }
        for (std::size_t workIndex = 0; workIndex < s1Q6Worklist.size(); ++workIndex) {
            const UInt32 index = s1Q6Worklist[workIndex].compactIndex;
            const UInt128 y = index <= cnt128
                ? partialArgs128[index]
                : UInt128(partialArgs[index]);
            if constexpr (UseCoherentQ6) {
                const UInt64 commonNu = nusVec[index];
                if (commonNu == 0 || commonNu >= u) {
                    std::cerr << "Error: coherent Q=6 requires 1 <= nu < u"
                              << " (nu=" << commonNu << ", u=" << u << ")."
                              << std::endl;
                    std::abort();
                }
                const UInt64 commonKappa = static_cast<UInt64>(
                    y / UInt128(commonNu + 1)
                );
                q6CommonNu[workIndex] = commonNu;
                q6CommonKappa[workIndex] = commonKappa;
                q6BoundaryFactor[workIndex] = coherentS1BoundaryFactor(
                    useQ6ZeroCompletion
                        ? S1OuterQ6Mode::Full6
                        : s1Q6Worklist[workIndex].mode,
                    commonKappa
                );
            }
            if constexpr (!UseCoherentQ6) {
                const UInt128 y3 = y / 3;
                const UInt128 y6 = y / 6;
                const UInt64 nu3 = get_nu(y3);
                const UInt64 nu6 = get_nu(y6);
                s1Q6Kappa3[workIndex] = static_cast<UInt64>(
                    UInt128(3) * (y3 / (nu3 + 1))
                );
                s1Q6Kappa6[workIndex] = static_cast<UInt64>(
                    UInt128(6) * (y6 / (nu6 + 1))
                );
                if constexpr (UseS2OuterQ6) {
                    s2Q6Nu3[workIndex] = nu3;
                    s2Q6Nu6[workIndex] = nu6;
                }
            }
        }
        if constexpr (UseS2OuterQ6 && !UseCoherentQ6) {
            compactHalf = static_cast<UInt32>(std::upper_bound(
                hash2.begin(), hash2.end(), static_cast<UInt32>(nu / 2)
            ) - hash2.begin() - 1);
        }

        if constexpr (UseUnorderedS2) {
            q6Bases.resize(s1Q6Worklist.size());
            q6Signs.resize(s1Q6Worklist.size());
            #pragma omp parallel for schedule(static) if(s1Q6Worklist.size() >= 1000000)
            for (UInt64 workIndex = 0;
                 workIndex < s1Q6Worklist.size();
                 ++workIndex) {
                const UInt32 compactIndex =
                    s1Q6Worklist[workIndex].compactIndex;
                q6Bases[workIndex] = hash2[compactIndex];
                const bool negative = (
                    negativeRecoverySigns[compactIndex >> 6]
                    >> (compactIndex & 63)
                ) & 1ULL;
                q6Signs[workIndex] = negative ? -1 : 1;
            }
        }

        auto compactQ6Bounds = [&](std::vector<UInt64>& compact,
                                   std::vector<UInt64>& full,
                                   bool releaseFull) {
            compact.resize(s1Q6Worklist.size());
            #pragma omp parallel for schedule(static) if(s1Q6Worklist.size() >= 1000000)
            for (UInt64 workIndex = 0; workIndex < s1Q6Worklist.size(); ++workIndex) {
                compact[workIndex] = full[s1Q6Worklist[workIndex].compactIndex];
            }
            if (releaseFull)
                releaseVector(full);
        };

        compactQ6Bounds(q6PartialArgs, partialArgs, UseS2OuterQ6);
        while (q6WideCount < s1Q6Worklist.size()
               && s1Q6Worklist[q6WideCount].compactIndex <= cnt128) {
            ++q6WideCount;
        }
        q6PartialArgs128.resize(q6WideCount);
        #pragma omp parallel for schedule(static) if(q6WideCount >= 1000000)
        for (UInt64 workIndex = 0; workIndex < q6WideCount; ++workIndex) {
            q6PartialArgs128[workIndex]
                = partialArgs128[s1Q6Worklist[workIndex].compactIndex];
        }
        if constexpr (ValidateUnorderedS2Wide) {
            q6PartialArgs128.resize(s1Q6Worklist.size());
            #pragma omp parallel for schedule(static) if(s1Q6Worklist.size() >= 1000000)
            for (UInt64 workIndex = q6WideCount;
                 workIndex < s1Q6Worklist.size();
                 ++workIndex) {
                q6PartialArgs128[workIndex] = UInt128(
                    q6PartialArgs[workIndex]
                );
            }
        }
        if constexpr (UseS2OuterQ6)
            releaseVector(partialArgs128);
        compactQ6Bounds(q6PartialArgsDivU, partialArgsDivU, true);
        if constexpr (!UseCoherentQ6) {
            compactQ6Bounds(q6Kappa1, kappas, false);
            compactQ6Bounds(q6Kappa2, kappas2, true);
        }
        if constexpr (UseS2OuterQ6) {
            if constexpr (!UseCoherentQ6)
                compactQ6Bounds(q6S2SplitCache, s2SplitCache, true);
            if (!useQ6CompactHotState) {
                q6WorkIndexByCompact.assign(mx + 1, 0);
                #pragma omp parallel for schedule(static) if(s1Q6Worklist.size() >= 1000000)
                for (UInt64 workIndex = 0;
                     workIndex < s1Q6Worklist.size();
                     ++workIndex) {
                    q6WorkIndexByCompact[s1Q6Worklist[workIndex].compactIndex]
                        = static_cast<UInt32>(workIndex + 1);
                }
            }
            releaseVector(hash2);
        }
        if (useQ6CompactHotState)
            releaseVector(negativeRecoverySigns);
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
    // The coherent final-value path keeps the leading 1 in every square-free
    // slot, but replaces the individual kappa corrections of one Q=6 group
    // by H_D(Q)M(v) in its coprime base slot. Signed final recovery therefore
    // sees the exact grouped boundary term. This representation is purposely
    // incompatible with full per-slot back substitution.
    //
    // Zero completion extends every hot group to the full divisor set. Its
    // four scalar constants cancel, so every slot starts at zero, auxiliary
    // slots stay zero, and signed recovery sees only the completed hot groups.
    // A failed completion guard restores the entire retained invariant above.
    //
    // The natural paths retain A_j in every square-free slot, including the
    // auxiliary slots divisible by 2 or 3, and therefore also permit complete
    // back substitution. All paths retain the leading constant in those slots;
    // the coherent grouped correction is closed only under signed final-value
    // recovery as stated above.
    //
    // The retained paths start partialValues with the leading 1 in A_j; zero
    // completion starts them at zero. The boundary term is added incrementally
    // below, and the S1/S2 terms are subtracted by their hot loops. The cnt128
    // largest x/j use 128-bit accumulators.
    std::vector<Int64> partialValues;
    std::vector<Int128> partialValues128;
    std::vector<Int64> q6CompactValues;
    std::vector<Int128> q6CompactValues128;
    if (useQ6CompactHotState) {
        q6CompactValues.resize(s1Q6Worklist.size() - q6WideCount, 0);
        q6CompactValues128.resize(q6WideCount, 0);
    } else {
        const Int64 initialValue = useQ6ZeroCompletion ? 0 : 1;
        partialValues.assign(mx + 1, initialValue);
        partialValues128.assign(cnt128 + 1, initialValue);
        partialValues[0] = 0;
        partialValues128[0] = 0;
    }

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
    Int64 coherentBoundaryIndex = static_cast<Int64>(q6CommonNu.size()) - 1;

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
                    if constexpr (UseCoherentQ6) {
                        if (useQ6ZeroCompletion) {
                            const Int64 value = evaluateCompletedS1OuterQ6(
                                q6PartialArgs[workIndex],
                                q6PartialArgsDivU[workIndex],
                                q6CommonKappa[workIndex], segmentLo, segmentHi,
                                mertensCoarse, residual, qCache, dCAP
                            );
                            if (useQ6CompactHotState)
                                q6CompactValues[workIndex - q6WideCount] -= value;
                            else
                                partialValues[index] -= value;
                        } else {
                            partialValues[index] -= evaluateCoherentS1OuterQ6(
                                item.mode, q6PartialArgs[workIndex],
                                q6PartialArgsDivU[workIndex],
                                q6CommonKappa[workIndex], segmentLo, segmentHi,
                                mertensCoarse, residual, qCache, dCAP
                            );
                        }
                    } else {
                        partialValues[index] -= evaluateS1OuterQ6(
                            item.mode, q6PartialArgs[workIndex],
                            q6PartialArgsDivU[workIndex], q6Kappa1[workIndex],
                            q6Kappa2[workIndex],
                            s1Q6Kappa3[workIndex], s1Q6Kappa6[workIndex],
                            segmentLo, segmentHi, mertensCoarse, residual,
                            qCache, dCAP
                        );
                    }
                } else {
                    if constexpr (UseCoherentQ6) {
                        if (useQ6ZeroCompletion) {
                            const Int128 value = evaluateCompletedS1OuterQ6(
                                q6PartialArgs128[workIndex],
                                q6PartialArgsDivU[workIndex],
                                q6CommonKappa[workIndex], segmentLo, segmentHi,
                                mertensCoarse, residual, qCache, dCAP
                            );
                            if (useQ6CompactHotState)
                                q6CompactValues128[workIndex] -= value;
                            else
                                partialValues128[index] -= value;
                        } else {
                            partialValues128[index] -= evaluateCoherentS1OuterQ6(
                                item.mode, q6PartialArgs128[workIndex],
                                q6PartialArgsDivU[workIndex],
                                q6CommonKappa[workIndex], segmentLo, segmentHi,
                                mertensCoarse, residual, qCache, dCAP
                            );
                        }
                    } else {
                        partialValues128[index] -= evaluateS1OuterQ6(
                            item.mode, q6PartialArgs128[workIndex],
                            q6PartialArgsDivU[workIndex], q6Kappa1[workIndex],
                            q6Kappa2[workIndex],
                            s1Q6Kappa3[workIndex], s1Q6Kappa6[workIndex],
                            segmentLo, segmentHi, mertensCoarse, residual,
                            qCache, dCAP
                        );
                    }
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
        if constexpr (UseS2OuterQ6)
            return q6WorkIndexByCompact[index] != 0;
        return (hash2[index] & 1U) != 0;
    };

    auto isCurrentHalf = [&](UInt32 index) {
        if constexpr (UseS2OuterQ6)
            return index <= compactHalf;
        return hash2[index] <= nu / 2;
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
        if (useQ6ZeroCompletion) return 0;
        switch (s1Q6Worklist[workIndex].mode) {
            case S1OuterQ6Mode::Full6:         return 0;
            case S1OuterQ6Mode::Minus2Minus3: return 1;
            case S1OuterQ6Mode::Minus2:        return 2;
            case S1OuterQ6Mode::Single:        return 3;
        }
        std::abort();
    };

    Int128 unorderedS2Square = 0;
    if constexpr (UseUnorderedS2) {
        START_PROFILE();

        const UInt64 unorderedWideCount = ValidateUnorderedS2Wide
                                        ? q6Bases.size()
                                        : q6WideCount;

#ifndef NDEBUG
        for (std::size_t index = 1; index < q6Bases.size(); ++index) {
            assert(q6Bases[index - 1] < q6Bases[index]);
            assert(q6CommonNu[index - 1] >= q6CommonNu[index]);
        }
#endif

        auto countBasesAtMost = [&](UInt64 value, UInt32 limit) -> UInt32 {
            if (value >= std::numeric_limits<UInt32>::max()) return limit;
            return static_cast<UInt32>(std::upper_bound(
                q6Bases.begin(), q6Bases.begin() + limit,
                static_cast<UInt32>(value)
            ) - q6Bases.begin());
        };

        auto countSplitAtLeast = [&](UInt64 value, UInt32 limit) -> UInt32 {
            UInt32 lo = 0;
            UInt32 hi = limit;
            while (lo < hi) {
                const UInt32 middle = lo + (hi - lo) / 2;
                if (q6CommonNu[middle] >= value)
                    lo = middle + 1;
                else
                    hi = middle;
            }
            return lo;
        };

        const std::array<UInt32, 3> outerClassCutoffs = {
            countBasesAtMost(
                nu / 6, static_cast<UInt32>(q6Bases.size())
            ),
            countBasesAtMost(
                nu / 3, static_cast<UInt32>(q6Bases.size())
            ),
            countBasesAtMost(
                nu / 2, static_cast<UInt32>(q6Bases.size())
            )
        };
        UInt32 cacheCutoff = 0;
        if constexpr (UseDivisionFree) {
            cacheCutoff = countBasesAtMost(
                dCAP, static_cast<UInt32>(q6Bases.size())
            );
        }

        auto evaluateRow = [&](UInt32 bi) -> Int128 {
            const UInt64 b = q6Bases[bi];
            const UInt64 reverseNu = q6CommonNu[bi];
            const UInt32 forwardActive = countSplitAtLeast(b, bi);
            const UInt32 reverseActive = countBasesAtMost(reverseNu, bi);
            const UInt32 rowEnd = std::max(forwardActive, reverseActive);

            const bool wide = bi < unorderedWideCount;
            Int128 wideDiagonal = 0;
            Int64 narrowDiagonal = 0;
            const std::size_t diagonalInner = CoherentS2Q6::innerClass(
                b, reverseNu
            );
            if (diagonalInner < CoherentS2Q6::ClassCount) {
                const std::size_t diagonalMode = CoherentS2Q6::modeIndex(
                    q6OuterClass(bi), diagonalInner
                );
                if (wide) {
                    const Int128 quotient = static_cast<Int128>(
                        q6PartialArgs128[bi] / b
                    );
                    wideDiagonal = CoherentS2Q6::evaluatePeriodKernelValue(
                        CoherentS2Q6::PeriodTable[diagonalMode], quotient
                    );
                } else {
                    const UInt64 quotient = q6PartialArgs[bi] / b;
                    narrowDiagonal = CoherentS2Q6::evaluatePeriodKernel(
                        CoherentS2Q6::PeriodTable[diagonalMode], quotient
                    );
                }
            }
            if (rowEnd == 0)
                return wide ? wideDiagonal : Int128(narrowDiagonal);

            std::array<UInt32, 14> endpoints{};
            std::size_t endpointCount = 0;
            auto addEndpoint = [&](UInt32 endpoint) {
                endpoints[endpointCount++] = std::min(endpoint, rowEnd);
            };

            addEndpoint(0);
            addEndpoint(rowEnd);
            addEndpoint(countSplitAtLeast(6 * b, bi));
            addEndpoint(countSplitAtLeast(3 * b, bi));
            addEndpoint(countSplitAtLeast(2 * b, bi));
            addEndpoint(forwardActive);
            addEndpoint(countBasesAtMost(reverseNu / 6, bi));
            addEndpoint(countBasesAtMost(reverseNu / 3, bi));
            addEndpoint(countBasesAtMost(reverseNu / 2, bi));
            addEndpoint(reverseActive);
            if (!useQ6ZeroCompletion) {
                addEndpoint(outerClassCutoffs[0]);
                addEndpoint(outerClassCutoffs[1]);
                addEndpoint(outerClassCutoffs[2]);
            }
            if constexpr (UseDivisionFree)
                addEndpoint(cacheCutoff);

            std::sort(endpoints.begin(), endpoints.begin() + endpointCount);
            endpointCount = static_cast<std::size_t>(std::unique(
                endpoints.begin(), endpoints.begin() + endpointCount
            ) - endpoints.begin());

            Int128 wideOffDiagonal = 0;
            Int64 narrowOffDiagonal = 0;
            for (std::size_t cell = 1; cell < endpointCount; ++cell) {
                const UInt32 cellBegin = endpoints[cell - 1];
                const UInt32 cellEnd = endpoints[cell];
                if (cellBegin == cellEnd) continue;

                const UInt64 a = q6Bases[cellBegin];
                const std::size_t forwardInner = CoherentS2Q6::innerClass(
                    b, q6CommonNu[cellBegin]
                );
                const std::size_t reverseInner = CoherentS2Q6::innerClass(
                    a, reverseNu
                );
                const bool forward =
                    forwardInner < CoherentS2Q6::ClassCount;
                const bool reverse =
                    reverseInner < CoherentS2Q6::ClassCount;
                if (!forward && !reverse) continue;

                std::size_t forwardMode = 0;
                std::size_t reverseMode = 0;
                if (forward) {
                    forwardMode = CoherentS2Q6::modeIndex(
                        q6OuterClass(cellBegin), forwardInner
                    );
                }
                if (reverse) {
                    reverseMode = CoherentS2Q6::modeIndex(
                        q6OuterClass(bi), reverseInner
                    );
                }

                const CoherentS2Q6::PeriodKernel& kernel =
                    forward && reverse
                    ? CoherentS2Q6::PairPeriodTable[forwardMode][reverseMode]
                    : CoherentS2Q6::PeriodTable[
                        forward ? forwardMode : reverseMode
                    ];

                if (wide) {
                    const UInt128 numerator = q6PartialArgs128[bi];
                    for (UInt32 ai = cellBegin; ai < cellEnd; ++ai) {
                        const Int128 quotient = static_cast<Int128>(
                            numerator / q6Bases[ai]
                        );
                        const Int128 value =
                            CoherentS2Q6::evaluatePeriodKernelValue(
                                kernel, quotient
                            );
                        wideOffDiagonal += q6Signs[ai] < 0 ? -value : value;
                    }
                } else {
                    const UInt64 numerator = q6PartialArgs[bi];
                    auto accumulateCell = [&](auto&& quotientAt) {
                        #pragma clang loop unroll_count(4)
                        for (UInt32 ai = cellBegin; ai < cellEnd; ++ai) {
                            const UInt64 quotient = quotientAt(q6Bases[ai]);
                            const Int64 value =
                                CoherentS2Q6::evaluatePeriodKernel(
                                    kernel, quotient
                                );
                            narrowOffDiagonal += q6Signs[ai] < 0
                                ? -value
                                : value;
                        }
                    };

                    if constexpr (UseDivisionFree) {
                        if (q6Bases[cellBegin] <= dCAP) {
                            accumulateCell([&](UInt64 denominator) {
                                return qCache.quotient(numerator, denominator);
                            });
                        } else {
                            accumulateCell([&](UInt64 denominator) {
                                return numerator / denominator;
                            });
                        }
                    } else {
                        accumulateCell([&](UInt64 denominator) {
                            return numerator / denominator;
                        });
                    }
                }
            }

            if (wide) {
                return wideDiagonal + (q6Signs[bi] < 0
                    ? -wideOffDiagonal
                    : wideOffDiagonal);
            }
            const Int128 narrowOffDiagonal128 = Int128(narrowOffDiagonal);
            return Int128(narrowDiagonal) + (q6Signs[bi] < 0
                ? -narrowOffDiagonal128
                : narrowOffDiagonal128);
        };

        std::vector<Int128> threadTotals(omp_get_max_threads(), Int128(0));
        #pragma omp parallel
        {
            Int128 local = 0;
            #pragma omp for schedule(dynamic, 4)
            for (UInt64 bi = 0; bi < q6Bases.size(); ++bi)
                local += evaluateRow(static_cast<UInt32>(bi));
            threadTotals[omp_get_thread_num()] = local;
        }
        for (Int128 total : threadTotals)
            unorderedS2Square += total;

        if constexpr (ValidateUnorderedS2) {
            std::vector<Int128> orderedThreadTotals(
                omp_get_max_threads(), Int128(0)
            );
            #pragma omp parallel
            {
                Int128 local = 0;
                #pragma omp for schedule(dynamic, 1)
                for (UInt64 ai = 0; ai < q6Bases.size(); ++ai) {
                    const UInt64 split = q6CommonNu[ai];
                    const UInt32 innerCount = countBasesAtMost(
                        std::min<UInt64>(nu, split),
                        static_cast<UInt32>(q6Bases.size())
                    );
                    const UInt32 outerClass = q6OuterClass(
                        static_cast<UInt32>(ai)
                    );
                    Int128 row = 0;
                    for (UInt32 bi = 0; bi < innerCount; ++bi) {
                        const std::size_t innerClass =
                            CoherentS2Q6::innerClass(q6Bases[bi], split);
                        if (innerClass >= CoherentS2Q6::ClassCount) continue;
                        Int128 value;
                        if (ai < unorderedWideCount) {
                            const Int128 quotient = static_cast<Int128>(
                                q6PartialArgs128[ai] / q6Bases[bi]
                            );
                            value = CoherentS2Q6::evaluateDivisorClasses(
                                outerClass, innerClass, quotient
                            );
                        } else {
                            const Int64 quotient = static_cast<Int64>(
                                q6PartialArgs[ai] / q6Bases[bi]
                            );
                            value = Int128(CoherentS2Q6::evaluateDivisorClasses(
                                outerClass, innerClass, quotient
                            ));
                        }
                        row += Int128(q6Signs[bi]) * value;
                    }
                    local += Int128(q6Signs[ai]) * row;
                }
                orderedThreadTotals[omp_get_thread_num()] = local;
            }

            Int128 orderedS2Square = 0;
            for (Int128 total : orderedThreadTotals)
                orderedS2Square += total;
            if (orderedS2Square != unorderedS2Square) {
                std::cerr << "Internal error: unordered S2 square mismatch."
                          << std::endl;
                std::abort();
            }
        }

        releaseVector(q6Bases);
        if (!useQ6CompactHotState)
            releaseVector(q6Signs);
        END_PROFILE(t[9]);
    }

    auto applyS2_64 = [&](UInt32 index, UInt64 lo, UInt64 hi,
                          bool currentHalf) -> Int64 {
        if constexpr (UseUnorderedS2) {
            lo = std::max(lo, nu + 1);
            if (lo > hi) return 0;
        }
        if constexpr (UseS2OuterQ6) {
            const UInt32 workIndex = q6WorkIndex(index);
            if constexpr (UseCoherentQ6) {
                const UInt64 commonNu = q6CommonNu[workIndex];
                return update_S2_coherent_q6(
                    q6PartialArgs[workIndex], L1, lo, hi, MuP,
                    q6OuterClass(workIndex), commonNu, qCache, dCAP
                );
            }
            return update_S2_q6<S2Q6Spec>(
                q6PartialArgs[workIndex], L1, lo, hi, MuP,
                q6OuterClass(workIndex), nusVec[index], q6S2SplitCache[workIndex],
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
        if constexpr (UseUnorderedS2) {
            lo = std::max(lo, nu + 1);
            if (lo > hi) return 0;
        }
        if constexpr (UseS2OuterQ6) {
            const UInt32 workIndex = q6WorkIndex(index);
            if constexpr (UseCoherentQ6) {
                const UInt64 commonNu = q6CommonNu[workIndex];
                return update_S2_coherent_q6_128(
                    q6PartialArgs128[workIndex], L1, lo, hi, MuP,
                    q6OuterClass(workIndex), commonNu
                );
            }
            return update_S2_q6_128<S2Q6Spec>(
                q6PartialArgs128[workIndex], L1, lo, hi, MuP,
                q6OuterClass(workIndex), nusVec[index], q6S2SplitCache[workIndex],
                s2Q6Nu3[workIndex], s2Q6Nu6[workIndex]
            );
        }
        return currentHalf
            ? update_S2_128<true>(partialArgs128[index], L1, lo, hi, MuP,
                                  nusVec[index], s2SplitCache[index])
            : update_S2_128<false>(partialArgs128[index], L1, lo, hi, MuP,
                                   nusVec[index], s2SplitCache[index]);
    };

    auto applyCompactS2_64 = [&](UInt32 workIndex, UInt64 lo,
                                  UInt64 hi) -> Int64 {
        return update_S2_coherent_q6(
            q6PartialArgs[workIndex], L1, lo, hi, MuP,
            0, q6CommonNu[workIndex], qCache, dCAP
        );
    };

    auto applyCompactS2_128 = [&](UInt32 workIndex, UInt64 lo,
                                   UInt64 hi) -> Int128 {
        return update_S2_coherent_q6_128(
            q6PartialArgs128[workIndex], L1, lo, hi, MuP,
            0, q6CommonNu[workIndex]
        );
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
            const UInt64 s2SegmentLo = UseUnorderedS2
                ? std::max(L1, nu + 1)
                : L1;
            if (useQ6CompactHotState) {
                UInt64 activeEnd = 0;
                UInt64 chunkedEnd = 0;
                if (s2SegmentLo <= L2) {
                    activeEnd = static_cast<UInt64>(std::partition_point(
                        q6CommonNu.begin(), q6CommonNu.end(),
                        [=](UInt64 split) { return split >= s2SegmentLo; }
                    ) - q6CommonNu.begin());
                    chunkedEnd = static_cast<UInt64>(std::partition_point(
                        q6CommonNu.begin(),
                        q6CommonNu.begin()
                            + static_cast<std::ptrdiff_t>(activeEnd),
                        [=](UInt64 split) { return split > CHUNK_LEN; }
                    ) - q6CommonNu.begin());

                    for (UInt64 workIndex = 0;
                         workIndex < chunkedEnd;
                         ++workIndex) {
                        const UInt64 rowEnd = std::min(
                            L2, q6CommonNu[workIndex]
                        );
                        for (UInt64 lo = s2SegmentLo;
                             lo <= rowEnd;
                             lo += CHUNK_LEN) {
                            chunks.push_back(Chunk{
                                static_cast<UInt32>(workIndex), lo,
                                std::min(lo + CHUNK_LEN - 1, rowEnd)
                            });
                        }
                    }
                }

                #pragma omp parallel for schedule(dynamic, 1)
                for (std::size_t tt = 0; tt < chunks.size(); ++tt) {
                    const Chunk& chunk = chunks[tt];
                    const UInt32 workIndex = chunk.i;
                    if (workIndex < q6WideCount) {
                        const Int128 value = applyCompactS2_128(
                            workIndex, chunk.a, chunk.b
                        );
                        #pragma omp critical
                        q6CompactValues128[workIndex] -= value;
                    } else {
                        const Int64 value = applyCompactS2_64(
                            workIndex, chunk.a, chunk.b
                        );
                        #pragma omp atomic
                        q6CompactValues[workIndex - q6WideCount] -= value;
                    }
                }

                #pragma omp parallel for schedule(dynamic, 1)
                for (UInt64 workIndex = chunkedEnd;
                     workIndex < activeEnd;
                     ++workIndex) {
                    const UInt64 rowEnd = std::min(
                        L2, q6CommonNu[workIndex]
                    );
                    if (workIndex < q6WideCount) {
                        q6CompactValues128[workIndex] -= applyCompactS2_128(
                            static_cast<UInt32>(workIndex), s2SegmentLo, rowEnd
                        );
                    } else {
                        q6CompactValues[workIndex - q6WideCount]
                            -= applyCompactS2_64(
                                static_cast<UInt32>(workIndex),
                                s2SegmentLo, rowEnd
                            );
                    }
                }
            } else if (s2SegmentLo <= L2) {
                for (UInt32 i = 1; i <= (UInt32)mx0; ++i) {
                    if (isS2Active(i)) {
                        const UInt64 m = std::min(L2, nusVec[i]);
                        if (s2SegmentLo > m) { mx0 = i-1; break; }

                        for (UInt64 a = s2SegmentLo; a <= m; a += CHUNK_LEN) {
                            chunks.push_back(Chunk{
                                i, a, std::min(a + CHUNK_LEN - 1, m)
                            });
                        }
                    }
                }

                // dynamic: cost per chunk varies wildly (small k is
                // O(sqrt(x/k)), large k is basically free)
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
                    while (mx1 > mx0 && nusVec[mx1] < s2SegmentLo) { --mx1; }

                    #pragma omp parallel for schedule(dynamic, 1)
                    for (UInt64 i = mx0+1; i <= (UInt64)mx1; ++i) {
                        if (isS2Active(static_cast<UInt32>(i))) {
                            const bool currentHalf = isCurrentHalf(
                                static_cast<UInt32>(i)
                            );
                            if (__builtin_expect(i > cnt128, true)) {
                                const Int64 v = applyS2_64(
                                    static_cast<UInt32>(i), s2SegmentLo,
                                    std::min(L2, nusVec[i]), currentHalf
                                );

                                partialValues[i] -= v;
                            } else {
                                const Int128 v = applyS2_128(
                                    static_cast<UInt32>(i), s2SegmentLo,
                                    std::min(L2, nusVec[i]), currentHalf
                                );

                                partialValues128[i] -= v;
                            }
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
            if constexpr (UseCoherentQ6) {
                while (coherentBoundaryIndex >= 0
                       && q6CommonNu[coherentBoundaryIndex] <= L2) {
                    const UInt64 workIndex = static_cast<UInt64>(
                        coherentBoundaryIndex
                    );
                    const UInt32 index = s1Q6Worklist[workIndex].compactIndex;
                    const Int64 mval = static_cast<Int64>(GET_M(
                        _MP, RP, L1, q6CommonNu[workIndex]
                    ));
                    if (useQ6CompactHotState) {
                        if (workIndex < q6WideCount) {
                            q6CompactValues128[workIndex] += Int128(
                                q6BoundaryFactor[workIndex]
                            ) * Int128(mval);
                        } else {
                            q6CompactValues[workIndex - q6WideCount]
                                += static_cast<Int64>(
                                    q6BoundaryFactor[workIndex]
                                ) * mval;
                        }
                    } else if (__builtin_expect(index > cnt128, true)) {
                        partialValues[index] += static_cast<Int64>(
                            q6BoundaryFactor[workIndex]
                        ) * mval;
                    } else {
                        partialValues128[index] += Int128(
                            q6BoundaryFactor[workIndex]
                        ) * Int128(mval);
                    }
                    --coherentBoundaryIndex;
                }
            } else {
                while (j && osqrt <= L2) {
                    const Int64 mval = static_cast<Int64>(
                        GET_M(_MP, RP, L1, osqrt)
                    );
                    if (__builtin_expect((UInt32)j > cnt128, true))
                        partialValues[j] += static_cast<Int64>(kappas[j]) * mval;
                    else
                        partialValues128[j] += kappas[j] * static_cast<Int128>(mval);

                    osqrt = nusVec[--j];
                }
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

    releaseVector(nusVec);
    releaseVector(s2SplitCache);
    releaseVector(chunks);
    releaseVector(R);
    releaseVector(M16);
    if constexpr (UseS1OuterQ6) {
        releaseVector(partialArgs);
        releaseVector(partialArgs128);
        releaseVector(kappas);
        releaseVector(hash2);
    }
    if constexpr (UseS2OuterQ6) {
        releaseVector(q6S2SplitCache);
        releaseVector(s2Q6Nu3);
        releaseVector(s2Q6Nu6);
        releaseVector(q6WorkIndexByCompact);
    }
    if constexpr (UseCoherentQ6) {
        releaseVector(q6CommonNu);
        releaseVector(q6BoundaryFactor);
    }

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

    // ========================================================================
    // Square-free Mobius finalization. For both the Q2 and Q6 invariants,
    //
    //   P_i = sum_{m: i*m<=nu, i*m square-free} W_{i*m}.
    //
    // Production obtains W_1=M(x) directly by Mobius inversion. The oracle
    // build retains decreasing-i back substitution of every W_i.
    // ========================================================================

#ifndef NDEBUG
    if (useQ6CompactHotState) {
        assert(q6CompactValues128.size() == q6WideCount);
        assert(q6CompactValues.size()
               == s1Q6Worklist.size() - q6WideCount);
        assert(q6Signs.size() == s1Q6Worklist.size());
        assert(partialValues.empty());
        assert(partialValues128.empty());
        assert(negativeRecoverySigns.empty());
        assert(q6WorkIndexByCompact.empty());
    } else if (useQ6ZeroCompletion) {
        std::size_t hotPosition = 0;
        for (UInt32 index = 1; index <= static_cast<UInt32>(mx); ++index) {
            if (hotPosition < s1Q6Worklist.size()
                && s1Q6Worklist[hotPosition].compactIndex == index) {
                ++hotPosition;
                continue;
            }
            const Int128 value = index <= cnt128
                ? partialValues128[index]
                : Int128(partialValues[index]);
            assert(value == 0);
        }
        assert(hotPosition == s1Q6Worklist.size());
    }
#endif

    Int64 result;
    if constexpr (!UseFullRecovery) {
        Int128 recovered = 0;
        if (useQ6CompactHotState) {
            for (UInt64 workIndex = 0;
                 workIndex < q6WideCount;
                 ++workIndex) {
                recovered += q6Signs[workIndex] < 0
                    ? -q6CompactValues128[workIndex]
                    : q6CompactValues128[workIndex];
            }
            for (UInt64 workIndex = q6WideCount;
                 workIndex < s1Q6Worklist.size();
                 ++workIndex) {
                const Int128 value = Int128(
                    q6CompactValues[workIndex - q6WideCount]
                );
                recovered += q6Signs[workIndex] < 0 ? -value : value;
            }
        } else {
            recovered = recoverSquarefreeFinalValue(
                partialValues, partialValues128, cnt128, negativeRecoverySigns,
                static_cast<UInt32>(mx)
            );
        }
        if constexpr (UseUnorderedS2)
            recovered -= unorderedS2Square;
#ifndef NDEBUG
        assert(recovered >= Int128(std::numeric_limits<Int64>::min()));
        assert(recovered <= Int128(std::numeric_limits<Int64>::max()));
#endif
        result = static_cast<Int64>(recovered);
    } else {
        START_PROFILE();
        if (cnt128 > 0) {
            partialValues128.resize(partialValues.size());
            for (UInt32 i = cnt128+1; i <= (UInt32)mx; ++i)
                partialValues128[i] = (Int128)partialValues[i];

            recoverSquarefreeInPlace(partialValues128, hash, static_cast<UInt32>(nu));
            result = (Int64)partialValues128[1];
        } else {
            recoverSquarefreeInPlace(partialValues, hash, static_cast<UInt32>(nu));
            result = partialValues[1];
        }
        END_PROFILE(t[8]);
    }

    if (profile) {
        double tot = t[0] + t[1] + t[2] + t[3] + t[4]
                   + t[5] + t[6] + t[7] + t[8] + t[9];

        std::cout << std::endl;
        std::cout << "-------------- Parameters -------------------" << std::endl;
        std::cout << "              u: " << u << std::endl;
        std::cout << "        nuRatio: " << mNuRatio << std::endl;
        if constexpr (UseQ6ZeroCompletion) {
            std::cout << "Q6 zero completion: "
                      << (useQ6ZeroCompletion ? "active" : "retained fallback")
                      << std::endl;
            if (q6ZeroCompletionFallback)
                std::cout << "  fallback reason: common split reached y/6"
                          << std::endl;
        }
        if constexpr (UseQ6CompactHotState) {
            std::cout << "Q6 compact hot state: "
                      << (useQ6CompactHotState ? "active" : "retained fallback")
                      << std::endl;
        }
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
        std::cout << "             S2: " << (t[1]+t[4]+t[9]) << ", " << (100.0*(t[1]+t[4]+t[9])/tot) << "%" << std::endl;
        if constexpr (UseUnorderedS2) {
            std::cout << "   Unordered S2: " << t[9] << ", "
                      << (100.0*t[9]/tot) << "%" << std::endl;
        }
        if constexpr (UseFullRecovery) {
            std::cout << "Back substitution: " << t[8] << ", " << (100.0*t[8]/tot) << "%" << std::endl;
        }
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << std::endl;
    }

    return result;
}

} // anonymous namespace

Int64 MertensHurst(UInt128 n, bool profile, UInt64 segmentCap,
                   UInt64 uOverride, double uFactor, double nuRatio) {
    MertensComputer computer;
    return computer.compute(n, profile, segmentCap, uOverride, uFactor, nuRatio);
}

#undef START_PROFILE
#undef END_PROFILE
