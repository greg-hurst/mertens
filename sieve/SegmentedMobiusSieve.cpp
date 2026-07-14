#include "SegmentedMobiusSieve.h"
#include "simd_defs.h"
#include "stencil_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <omp.h>

// Toggle the large-prime bucket scheduler. When disabled, all primes above
// SMALL_PRIME_CAP are sieved as "medium" primes (direct iteration per sub-segment).
// Override at build time: -DUSE_BUCKET_SIEVE=0
#ifndef USE_BUCKET_SIEVE
#define USE_BUCKET_SIEVE 1
#endif
static constexpr bool UseBucketSieve = USE_BUCKET_SIEVE;

// Cutoffs for the fused raw-byte path. At high sieve bounds,
// a larger direct tile exposes more independent memory traffic and the sparse
// top prime band is cheaper to traverse directly than to schedule in buckets.
static constexpr UInt64 MEDIUM_TILE_SWITCH     = UInt64(1) << 36;
static constexpr UInt64 MEDIUM_TILE_LARGE      = UInt64(168) * SegmentedMobiusSieveCore::STENCIL_PERIOD;
static constexpr UInt64 RAW_LARGE_PRIME_CUTOFF = UInt64(200) * SegmentedMobiusSieveCore::STENCIL_PERIOD;

// Experimental: split sieveSubSegment's hit loop into a pure-scatter RMW pass and
// a pure-ALU forwarding pass. Override at build time: -DSIEVE_TWO_PASS=1
#ifndef SIEVE_TWO_PASS
#define SIEVE_TWO_PASS 0
#endif

// Adaptive threshold for large-prime routing strategy.
// When numSegments * LP_ROUTE_THRESHOLD >= numLargePrimes, use single-pass
// serial routing (Option A); otherwise use parallel per-thread constructors.
// On ARM the threshold defaults to 0 (parallel constructors always): serial
// routing runs on one core while the rest idle, and with fast hardware
// division the redundant parallel scan wins beyond a few threads. On x86 the
// division-free routing iterations are cheaper, so Option A keeps a threshold.
// Override at build time: -DLP_ROUTE_THRESHOLD=15000
#ifndef LP_ROUTE_THRESHOLD
#if defined(__aarch64__) || defined(__arm64__)
#define LP_ROUTE_THRESHOLD 0
#else
#define LP_ROUTE_THRESHOLD 15000
#endif
#endif

// ============================================================================
// Construction
// ============================================================================

SegmentedMobiusSieveCore::SegmentedMobiusSieveCore()
    : mStencilData(stencil, stencil + STENCIL_PERIOD)
    , mSmallPrimeStopIdx(2)   // skip past primes 2,3 (indices 0,1 in typical prime list)
    , mSqrtPrimeStopIdx(2)
{}

SegmentedMobiusSieveCore::SegmentedMobiusSieveCore(UInt64 segmentSize)
    : SegmentedMobiusSieveCore()
{
    initialize(segmentSize);
}

// ============================================================================
// Setup
// ============================================================================

void SegmentedMobiusSieveCore::initialize(UInt64 segmentSize) {
    mBuckets.resize(0);
    mBuckets.resize(omp_get_max_threads());

    fillFromStencil(segmentSize);
    mMu.resize(segmentSize);

    mPreMu.resize(M1 + STENCIL_PERIOD);
    for (UInt64 off = 0; off < M1 + STENCIL_PERIOD; off += STENCIL_PERIOD)
        std::memcpy(mPreMu.data() + off, mStencilData.data(), STENCIL_PERIOD * sizeof(Int8));

    mSmallPrimeStopIdx = 2;  // NUM_PRIME_SQUARES = 2 (skip 2, 3)
    mSqrtPrimeStopIdx = 2;
}

void SegmentedMobiusSieveCore::initSieveQuotientCache(const std::vector<UInt32>& primes) {
    if constexpr (UseDivisionFree) {
        mSieveQCache.init(primes.data(), primes.size());
    }
}

void SegmentedMobiusSieveCore::fillFromStencil(UInt64 size) {
    mMu.resize(size);
    Int8* dst = mMu.data();

    // Pick a tile size that makes memcpy hit peak throughput (~4 MiB).
    UInt64 tile = STENCIL_PERIOD * ((4u << 20) / STENCIL_PERIOD + 1);
    if (tile == 0) tile = STENCIL_PERIOD;
    tile = std::min(tile, size);

    // Build tile: one stencil copy + doubling-fill inside tile
    std::vector<Int8> T(tile);
    UInt64 initCopy = std::min(tile, (UInt64)STENCIL_PERIOD);
    std::memcpy(T.data(), mStencilData.data(), initCopy);
    UInt64 filled = initCopy;
    while (filled < tile) {
        UInt64 n = std::min(filled, tile - filled);
        std::memcpy(T.data() + filled, T.data(), n);
        filled += n;
    }

    // Parallel tile copies into destination
    const UInt64 nt = (size + tile - 1) / tile;

    #pragma omp parallel for schedule(static)
    for (UInt64 i = 0; i < nt; ++i) {
        UInt64 off = i * tile;
        UInt64 n   = std::min(tile, size - off);
        std::memcpy(dst + off, T.data(), n);
    }
}

// ============================================================================
// Utility: prime generation
// ============================================================================

std::vector<UInt32> SegmentedMobiusSieveCore::primesUpTo(UInt32 n) {
    std::vector<UInt32> S(n);

    for (Int32 i = 2; i <= (Int32)n; ++i)
        S[i - 2] = i;

    const Int32 sqrtn = static_cast<Int32>(std::sqrt((double)n));
    for (Int32 i = 1; i <= sqrtn; ++i) {
        if (S[i - 1] != 0) {
            for (Int32 k = 2 * i + 1; k <= (Int32)n - 1; k += i + 1)
                S[k - 1] = 0;
        }
    }

    S.erase(std::remove(S.begin(), S.end(), (UInt32)0), S.end());
    return S;
}

// ============================================================================
// Main sieve entry point
// ============================================================================

template<bool Finalize>
void SegmentedMobiusSieveCore::sieve(UInt64 lo, UInt64 hi, const std::vector<UInt32>& P) {
    assert(P.size() >= 71 && "prime list must have at least 71 entries (primes up to 353)");

    // Auto-initialize quotient cache on first use if needed
    if constexpr (UseDivisionFree) {
        if (mSieveQCache.count < P.size())
            initSieveQuotientCache(P);

        // The Granlund-Montgomery formulation with SHIFT=60 is exact only
        // for arguments below 2^60. ceilDiv computes val = x + p - 1 with
        // x <= hi and p < 2^32, so require hi < 2^60 - 2^32. The default
        // bucket-sieve range cap (2.06e17) is far below this; only
        // BUCKET_SIEVE=0 builds, where the encoding stays exact to ~1.8e19,
        // can reach it — those must use DIVISION_FREE=0 past this bound.
        assert(hi < (1ULL << 60) - (1ULL << 32) &&
               "DIVISION_FREE=1 requires hi < 2^60 - 2^32 (quotient cache domain)");
    }

    const UInt64 len = hi - lo + 1;
    Int8* MuP = mMu.data();
    const UInt64 stencilOff = (lo <= 1) ? 0 : (lo - 1) % STENCIL_PERIOD;

    const UInt64 sqrtHi = std::round(std::sqrt((double)hi));

    if (sqrtHi < SMALL_PRIME_CAP) {
        while (mSmallPrimeStopIdx < P.size() && sqrtHi >= P[mSmallPrimeStopIdx])
            ++mSmallPrimeStopIdx;

        mSqrtPrimeStopIdx = mSmallPrimeStopIdx;

        #pragma omp parallel for schedule(dynamic, 1)
        for (UInt64 l = 0; l < len; l += M1) {
            std::memcpy(MuP + l, mPreMu.data() + stencilOff,
                        sizeof(Int8) * std::min(static_cast<UInt64>(M1), len - l));
            sieveHelper1(MuP + l, P.data(), lo + l, std::min(lo + l + M1 - 1, hi),
                         mSmallPrimeStopIdx, &mSieveQCache);
        }

        sieveSquaresPass(MuP, P.data(), lo, hi, 71, mSqrtPrimeStopIdx, &mSieveQCache);

        if constexpr (Finalize) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (UInt64 l = 0; l < len; l += M1)
                finalizeMu(MuP + l, lo + l, std::min(lo + l + M1 - 1, hi));
        }

        return;
    }

    // establish mSmallPrimeStopIdx = first prime index with p > SMALL_PRIME_CAP
    while (mSmallPrimeStopIdx < P.size()
           && (UInt64)P[mSmallPrimeStopIdx] <= (UInt64)SMALL_PRIME_CAP)
        ++mSmallPrimeStopIdx;

    // establish mSqrtPrimeStopIdx = first prime index with p > sqrtHi
    mSqrtPrimeStopIdx = mSmallPrimeStopIdx;
    while (mSqrtPrimeStopIdx < P.size() && (UInt64)P[mSqrtPrimeStopIdx] <= sqrtHi)
        ++mSqrtPrimeStopIdx;

    // Guard: the largest sieved prime must fit within the circular bucket
    // scheduler's capacity. If this fires, increase LP_SIZE to the next
    // power of two.
    assert(mSqrtPrimeStopIdx <= mSmallPrimeStopIdx ||
           (UInt64)P[mSqrtPrimeStopIdx - 1] <= (LargePrimeHitScheduler::LP_SIZE - 1) * M2);

    UInt32 idxLargeBegin = mSmallPrimeStopIdx;
    if constexpr (UseBucketSieve) {
        // MertensHurst benefits from keeping its sparse
        // upper prime band on the direct four-stream path. Finalized callers
        // retain the lower cutoff that wins with the rolling block walk.
        constexpr UInt64 largePrimeCutoff = Finalize ? M3 : RAW_LARGE_PRIME_CUTOFF;
        while (idxLargeBegin < mSqrtPrimeStopIdx
               && static_cast<UInt64>(P[idxLargeBegin]) <= largePrimeCutoff)
            ++idxLargeBegin;
    } else {
        idxLargeBegin = mSqrtPrimeStopIdx;  // treat all primes as medium
    }

    const UInt64 numSeg = (len + M2 - 1) / M2;
    if (!Finalize || numSeg < BLOCK_WALK_MIN_SEGMENTS) {
        // MertensHurst finalizes during its prefix scan, so its best locality
        // is small-prime initialization followed immediately by the medium
        // pass. Short standalone ranges also stay here because reseeding many
        // small bucket blocks costs more than rolling offsets save.
        const UInt64 mediumTile = hi >= MEDIUM_TILE_SWITCH
            ? MEDIUM_TILE_LARGE
            : M2;
        #pragma omp parallel for schedule(dynamic, 1)
        for (UInt64 tile = 0; tile < len; tile += mediumTile) {
            const UInt64 tileEnd = std::min(tile + mediumTile, len);
            for (UInt64 l = tile; l < tileEnd; l += M1) {
                const UInt64 L = lo + l;
                const UInt64 H = std::min({L + M1 - 1, lo + tileEnd - 1, hi});
                const UInt64 n = H - L + 1;
                std::memcpy(MuP + l, mPreMu.data() + stencilOff,
                            sizeof(Int8) * static_cast<size_t>(n));
                sieveSmallPrimes(MuP + l, P.data(), L, H);
            }

            if (mSmallPrimeStopIdx < idxLargeBegin) {
                const UInt64 L = lo + tile;
                const UInt64 H = lo + tileEnd - 1;
                sieveMediumSubSegment(
                    MuP + tile, P.data(), L, H,
                    mSmallPrimeStopIdx, idxLargeBegin, &mSieveQCache
                );
            }
        }
        sieveSquaresPass(MuP, P.data(), lo, hi, 71, mSqrtPrimeStopIdx, &mSieveQCache);
        if (idxLargeBegin < mSqrtPrimeStopIdx) {
            const UInt64 numSeg = (len + M2 - 1) / M2;
            const UInt32 numLP = mSqrtPrimeStopIdx - idxLargeBegin;
            const bool useSerialRouting = numSeg * LP_ROUTE_THRESHOLD >= numLP;

            if (useSerialRouting) {
                const int nt = omp_get_max_threads();
                std::vector<UInt64> threadStart(nt + 1);
                for (int t = 0; t <= nt; ++t) {
                    threadStart[t] = (numSeg * static_cast<UInt64>(t))
                                   / static_cast<UInt64>(nt);
                }

                for (int t = 0; t < nt; ++t) {
                    auto& buckets = mBuckets[t];
                    if (buckets.size() < LargePrimeHitScheduler::LP_NBUCKETS)
                        buckets.resize(LargePrimeHitScheduler::LP_NBUCKETS);
                    for (auto& bucket : buckets)
                        bucket.clear();
                }

                for (UInt32 i = idxLargeBegin; i < mSqrtPrimeStopIdx; ++i) {
                    const UInt64 p = P[i];
                    if (p > hi)
                        break;

                    for (int t = 0; t < nt; ++t) {
                        const UInt64 threadLo = lo + threadStart[t] * M2;
                        const UInt64 threadHi = (t + 1 < nt)
                            ? lo + threadStart[t + 1] * M2 - 1
                            : hi;

                        UInt64 m;
                        if constexpr (UseDivisionFree) {
                            m = p * mSieveQCache.ceilDiv(threadLo, i, p);
                        } else {
                            m = p * (((threadLo - 1) / p) + 1);
                        }
                        if (m > threadHi)
                            continue;

                        const UInt64 localSubSeg = (m - threadLo) / M2;
                        using Scheduler = LargePrimeHitScheduler;
                        const Scheduler::EntryT entry = Scheduler::packEntry(
                            static_cast<UInt32>(p), (m - threadLo) - localSubSeg * M2
                        );
                        mBuckets[t][Scheduler::ringIndex(localSubSeg, entry)].push_back(entry);
                    }
                }

                #pragma omp parallel num_threads(nt)
                {
                    const int tid = omp_get_thread_num();
                    const UInt64 s0 = threadStart[tid];
                    const UInt64 s1 = threadStart[tid + 1];

                    if (s0 < s1) {
                        LargePrimeHitScheduler scheduler(
                            LargePrimeHitScheduler::PrePopulated{},
                            lo, lo + s0 * M2, std::min(lo + s1 * M2 - 1, hi),
                            M2, P, idxLargeBegin, mSqrtPrimeStopIdx - 1,
                            mBuckets[tid], &mSieveQCache
                        );
                        for (UInt64 s = s0; s < s1; ++s) {
                            scheduler.sieveSubSegment(MuP);
                            if constexpr (Finalize) {
                                const UInt64 L = lo + s * M2;
                                finalizeMu(MuP + s * M2, L, std::min(L + M2 - 1, hi));
                            }
                        }
                    }
                }
            } else {
                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    const int nt = omp_get_num_threads();
                    const UInt64 s0 = (numSeg * static_cast<UInt64>(tid))
                                          / static_cast<UInt64>(nt);
                    const UInt64 s1 = (numSeg * static_cast<UInt64>(tid + 1))
                                          / static_cast<UInt64>(nt);

                    if (s0 < s1) {
                        LargePrimeHitScheduler scheduler(
                            lo, lo + s0 * M2, std::min(lo + s1 * M2 - 1, hi),
                            M2, P, idxLargeBegin, mSqrtPrimeStopIdx - 1,
                            mBuckets[tid], &mSieveQCache
                        );
                        for (UInt64 s = s0; s < s1; ++s) {
                            scheduler.sieveSubSegment(MuP);
                            if constexpr (Finalize) {
                                const UInt64 L = lo + s * M2;
                                finalizeMu(MuP + s * M2, L, std::min(L + M2 - 1, hi));
                            }
                        }
                    }
                }
            }
        } else if constexpr (Finalize) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (UInt64 l = 0; l < len; l += M2)
                finalizeMu(MuP + l, lo + l, std::min(lo + l + M2 - 1, hi));
        }
        return;
    }

    // Phase 1: copy stencil + apply small primes on M1 chunks.
    #pragma omp parallel for schedule(dynamic, 1)
    for (UInt64 l = 0; l < len; l += M1) {
        const UInt64 L = lo + l;
        const UInt64 H = std::min(L + M1 - 1, hi);
        const UInt64 n = H - L + 1;
        std::memcpy(MuP + l, mPreMu.data() + stencilOff,
                    sizeof(Int8) * static_cast<size_t>(n));
        sieveSmallPrimes(MuP + l, P.data(), L, H);
    }

    // Zero squareful entries before the contiguous medium/bucket walk. Later
    // log additions leave them nonnegative, so finalization still maps them
    // to zero.
    sieveSquaresPass(MuP, P.data(), lo, hi, 71, mSqrtPrimeStopIdx, &mSieveQCache);

    if (idxLargeBegin >= mSqrtPrimeStopIdx) {
        if (mSmallPrimeStopIdx < idxLargeBegin) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (UInt64 l = 0; l < len; l += M2) {
                const UInt64 L = lo + l;
                const UInt64 H = std::min(L + M2 - 1, hi);
                sieveMediumSubSegment(MuP + l, P.data(), L, H,
                                      mSmallPrimeStopIdx, idxLargeBegin, &mSieveQCache);
                if constexpr (Finalize)
                    finalizeMu(MuP + l, L, H);
            }
        } else if constexpr (Finalize) {
            #pragma omp parallel for schedule(dynamic, 1)
            for (UInt64 l = 0; l < len; l += M2)
                finalizeMu(MuP + l, lo + l, std::min(lo + l + M2 - 1, hi));
        }

    } else {
        // Dynamic contiguous blocks keep medium-prime next-hit state hot and
        // carry each large-prime scheduler only across the range it owns.
        const UInt64 targetBlocks = 2 * static_cast<UInt64>(omp_get_max_threads());
        const UInt64 balancedBlock = (numSeg + targetBlocks - 1) / targetBlocks;
        const UInt64 blockSegments = std::max<UInt64>(
            1, std::min(BLOCK_SEGMENTS, balancedBlock)
        );
        const UInt64 numBlocks = (numSeg + blockSegments - 1) / blockSegments;

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            std::vector<UInt32> nextOff;

            #pragma omp for schedule(dynamic, 1)
            for (UInt64 block = 0; block < numBlocks; ++block) {
                const UInt64 s0 = block * blockSegments;
                const UInt64 s1 = std::min(s0 + blockSegments, numSeg);
                const UInt64 blockLo = lo + s0 * M2;
                const UInt64 blockHi = std::min(lo + s1 * M2 - 1, hi);

                LargePrimeHitScheduler sched(
                    lo, blockLo, blockHi, M2, P,
                    idxLargeBegin, mSqrtPrimeStopIdx - 1,
                    mBuckets[tid], &mSieveQCache
                );

                const UInt64 mediumCount = idxLargeBegin - mSmallPrimeStopIdx;
                nextOff.resize(static_cast<size_t>(mediumCount));
                for (UInt64 i = mSmallPrimeStopIdx; i < idxLargeBegin; ++i) {
                    const UInt64 p = P[i];
                    UInt64 first;
                    if constexpr (UseDivisionFree) {
                        first = p * mSieveQCache.ceilDiv(blockLo, i, p);
                    } else {
                        first = p * (((blockLo - 1) / p) + 1);
                    }
                    nextOff[i - mSmallPrimeStopIdx] = static_cast<UInt32>(first - blockLo);
                }

                for (UInt64 s = s0; s < s1; ++s) {
                    const UInt64 L = lo + s * M2;
                    const UInt64 H = std::min(L + M2 - 1, hi);
                    sieveMediumSubSegmentRolling(
                        MuP + s * M2, P.data(), H - L + 1,
                        mSmallPrimeStopIdx, idxLargeBegin, nextOff
                    );
                    sched.sieveSubSegment(MuP);
                    if constexpr (Finalize)
                        finalizeMu(MuP + s * M2, L, H);
                }
            }
        }
    }
}

// Explicit instantiations: <true> is the normal mu-finalizing path used by all
// existing callers; <false> leaves the raw packed bytes for the fused
// finalize+prefix scan (SegmentedMertensSieve.h).
template void SegmentedMobiusSieveCore::sieve<true>(UInt64, UInt64, const std::vector<UInt32>&);
template void SegmentedMobiusSieveCore::sieve<false>(UInt64, UInt64, const std::vector<UInt32>&);

// ============================================================================
// Phase 1: Small primes (manually unrolled sieve for primes 13..353)
// ============================================================================

void SegmentedMobiusSieveCore::sieveSmallPrimes(Int8* mu, const UInt32* primes,
                                                UInt64 lo, UInt64 hi) {
    sieveHelper1(mu, primes, lo, hi, mSmallPrimeStopIdx, &mSieveQCache);
}

void SegmentedMobiusSieveCore::sieveHelper1(Int8* __restrict Mu, const UInt32* __restrict primes,
                                            UInt64 lo, UInt64 hi, UInt32 stoppos,
                                            const SieveQuotientCache* cache) {
    const UInt64 len = hi - lo + 1;

    // Log-space sieve: each entry gets (ceil(log2(p)) | 1) added per prime
    // factor p. For an odd prime ceil(log2 p) == numbits(p), so each group
    // below shares one constant (the bit width, made odd):
    //   add5: primes 13..31   -> 5
    //   add7: primes 37..127  -> 7
    //   add9: primes 131..353 -> 9
    // OR with 1 keeps the value odd so finalizeMu can read the factor count
    // parity from bit 0 (that's how it figures out the sign of mu). The ceil
    // weighting is what lets finalizeMu compare against floor(log2 N) exactly,
    // with no slack margin. The stencil (primes 2,3,5,7,11) uses the same
    // ceil(log2 p)|1 weights so the encoding is uniform across all primes.
    auto add5Stride = [&](UInt64 st, UInt64 step) {
        UInt64 i = st;
        for (; i + 4*step < len; i += 4*step) {
            Mu[i] += 5;
            Mu[i + step] += 5;
            Mu[i + 2*step] += 5;
            Mu[i + 3*step] += 5;
        }
        for (; i < len; i += step) Mu[i] += 5;
    };

    auto add7Stride = [&](UInt64 st, UInt64 step) {
        UInt64 i = st;
        for (; i + 4*step < len; i += 4*step) {
            Mu[i] += 7;
            Mu[i + step] += 7;
            Mu[i + 2*step] += 7;
            Mu[i + 3*step] += 7;
        }
        for (; i < len; i += step) Mu[i] += 7;
    };

    auto add9Stride = [&](UInt64 st, UInt64 step) {
        UInt64 i = st;
        for (; i + 4*step < len; i += 4*step) {
            Mu[i] += 9;
            Mu[i + step] += 9;
            Mu[i + 2*step] += 9;
            Mu[i + 3*step] += 9;
        }
        for (; i < len; i += step) Mu[i] += 9;
    };

    auto zerooutStride = [&](UInt64 st, UInt64 step) {
        UInt64 i = st;
        for (; i + 4*step < len; i += 4*step) {
            Mu[i] = 0;
            Mu[i + step] = 0;
            Mu[i + 2*step] = 0;
            Mu[i + 3*step] = 0;
        }
        for (; i < len; i += step) Mu[i] = 0;
    };

    // ----------- Sieve multiples of p & p^2 < 350 -----------
    // Primes 13..353 are fully unrolled to avoid loop/branch overhead.
    // These dominate the inner loop since small prime multiples are dense.
    // Primes 2,3,5,7,11 are handled by the pre-computed stencil (period 13860).
    add5Stride(13*((lo + 12)/13) - lo, 13);      add5Stride(17*((lo + 16)/17) - lo, 17);
    add5Stride(19*((lo + 18)/19) - lo, 19);      add5Stride(23*((lo + 22)/23) - lo, 23);
    add5Stride(29*((lo + 28)/29) - lo, 29);      add5Stride(31*((lo + 30)/31) - lo, 31);
    add7Stride(37*((lo + 36)/37) - lo, 37);      add7Stride(41*((lo + 40)/41) - lo, 41);
    add7Stride(43*((lo + 42)/43) - lo, 43);      add7Stride(47*((lo + 46)/47) - lo, 47);
    add7Stride(53*((lo + 52)/53) - lo, 53);      add7Stride(59*((lo + 58)/59) - lo, 59);
    add7Stride(61*((lo + 60)/61) - lo, 61);      add7Stride(67*((lo + 66)/67) - lo, 67);
    add7Stride(71*((lo + 70)/71) - lo, 71);      add7Stride(73*((lo + 72)/73) - lo, 73);
    add7Stride(79*((lo + 78)/79) - lo, 79);      add7Stride(83*((lo + 82)/83) - lo, 83);
    add7Stride(89*((lo + 88)/89) - lo, 89);      add7Stride(97*((lo + 96)/97) - lo, 97);
    add7Stride(101*((lo + 100)/101) - lo, 101);  add7Stride(103*((lo + 102)/103) - lo, 103);
    add7Stride(107*((lo + 106)/107) - lo, 107);  add7Stride(109*((lo + 108)/109) - lo, 109);
    add7Stride(113*((lo + 112)/113) - lo, 113);  add7Stride(127*((lo + 126)/127) - lo, 127);
    add9Stride(131*((lo + 130)/131) - lo, 131);  add9Stride(137*((lo + 136)/137) - lo, 137);
    add9Stride(139*((lo + 138)/139) - lo, 139);  add9Stride(149*((lo + 148)/149) - lo, 149);
    add9Stride(151*((lo + 150)/151) - lo, 151);  add9Stride(157*((lo + 156)/157) - lo, 157);
    add9Stride(163*((lo + 162)/163) - lo, 163);  add9Stride(167*((lo + 166)/167) - lo, 167);
    add9Stride(173*((lo + 172)/173) - lo, 173);  add9Stride(179*((lo + 178)/179) - lo, 179);
    add9Stride(181*((lo + 180)/181) - lo, 181);  add9Stride(191*((lo + 190)/191) - lo, 191);
    add9Stride(193*((lo + 192)/193) - lo, 193);  add9Stride(197*((lo + 196)/197) - lo, 197);
    add9Stride(199*((lo + 198)/199) - lo, 199);  add9Stride(211*((lo + 210)/211) - lo, 211);
    add9Stride(223*((lo + 222)/223) - lo, 223);  add9Stride(227*((lo + 226)/227) - lo, 227);
    add9Stride(229*((lo + 228)/229) - lo, 229);  add9Stride(233*((lo + 232)/233) - lo, 233);
    add9Stride(239*((lo + 238)/239) - lo, 239);  add9Stride(241*((lo + 240)/241) - lo, 241);
    add9Stride(251*((lo + 250)/251) - lo, 251);  add9Stride(257*((lo + 256)/257) - lo, 257);
    add9Stride(263*((lo + 262)/263) - lo, 263);  add9Stride(269*((lo + 268)/269) - lo, 269);
    add9Stride(271*((lo + 270)/271) - lo, 271);  add9Stride(277*((lo + 276)/277) - lo, 277);
    add9Stride(281*((lo + 280)/281) - lo, 281);  add9Stride(283*((lo + 282)/283) - lo, 283);
    add9Stride(293*((lo + 292)/293) - lo, 293);  add9Stride(307*((lo + 306)/307) - lo, 307);
    add9Stride(311*((lo + 310)/311) - lo, 311);  add9Stride(313*((lo + 312)/313) - lo, 313);
    add9Stride(317*((lo + 316)/317) - lo, 317);  add9Stride(331*((lo + 330)/331) - lo, 331);
    add9Stride(337*((lo + 336)/337) - lo, 337);  add9Stride(347*((lo + 346)/347) - lo, 347);
    add9Stride(349*((lo + 348)/349) - lo, 349);  add9Stride(353*((lo + 352)/353) - lo, 353);

    // Zero out multiples of p^2 for p=5,7,11,13,17,19 (the squares < 361).
    // mu(n) = 0 whenever n has a squared prime factor.
    zerooutStride(25*((lo  + 24 )/25 ) - lo,  25);
    zerooutStride(49*((lo  + 48 )/49 ) - lo,  49);
    zerooutStride(121*((lo + 120)/121) - lo, 121);
    zerooutStride(169*((lo + 168)/169) - lo, 169);
    zerooutStride(289*((lo + 288)/289) - lo, 289);
    zerooutStride(361*((lo + 360)/361) - lo, 361);

    // ----------- Sieve the above's prime squares -----------

    for (UInt64 i = 8; i < std::min((UInt32)71, stoppos); ++i) {
        const UInt64 p = primes[i];
        const UInt64 p2 = p * p;

        UInt64 st2;
        if constexpr (UseDivisionFree) {
            // ceil(lo / p^2) = ceil(ceil(lo / p) / p)
            const UInt64 q1 = cache->ceilDiv(lo, i, p);
            const UInt64 q2 = cache->ceilDiv(q1, i, p);
            st2 = p2 * q2 - lo;
        } else {
            st2 = p2 * (((lo - 1) / p2) + 1) - lo;
        }

        for (UInt64 pos2 = st2; pos2 < len; pos2 += p2)
            Mu[pos2] = 0;
    }

    // ----------- Sieve remaining small primes -----------

    for (UInt64 i = 71; i < stoppos; ++i) {
        const UInt64 p = primes[i];

        UInt64 st;
        if constexpr (UseDivisionFree) {
            const UInt64 q1 = cache->ceilDiv(lo, i, p);
            st = p * q1 - lo;
        } else {
            st = p * (((lo - 1) / p) + 1) - lo;
        }

        if (__builtin_expect(st >= len, false))
            continue;

        const Int8 l = static_cast<Int8>((64 - __builtin_clzll(p)) | 1ULL);

        // 4-way unrolled like the add5/7/9 lambdas above; the four stores
        // are independent so they pipeline.
        UInt64 pos = st;
        for (; pos + 3 * p < len; pos += 4 * p) {
            Mu[pos] += l;
            Mu[pos + p] += l;
            Mu[pos + 2 * p] += l;
            Mu[pos + 3 * p] += l;
        }
        for (; pos < len; pos += p)
            Mu[pos] += l;

        // p^2 multiples (p >= 359 means p^2 > M1, at most one hit per sub-segment)
        // are handled once per segment by sieveSquaresPass, not per sub-segment.
    }
}

// ============================================================================
// Squares pass: zero p^2 multiples for primes >= 359, once per segment
// ============================================================================

// For p >= 359, p^2 > M1, so per-sub-segment square handling pays a quotient per
// prime per sub-segment for almost no hits; one pass over the whole segment pays
// one quotient per prime total. Zeroing commutes with the log-prime adds of
// other primes: a zeroed byte plus small positive adds stays non-negative,
// which finalizeMu decodes as mu = 0. (The per-sub-segment code already relied on
// this ordering tolerance — primes later in the loop add onto positions
// zeroed by earlier primes.)
void SegmentedMobiusSieveCore::sieveSquaresPass(Int8* __restrict Mu,
                                                const UInt32* __restrict primes,
                                                UInt64 lo, UInt64 hi,
                                                UInt32 fromIdx, UInt32 toIdx,
                                                const SieveQuotientCache* cache) {
    const UInt64 len = hi - lo + 1;

    // Work per prime is ~len/p^2, so a flat parallel-for over primes leaves
    // one thread with nearly all the hits while the rest idle (~30% of sieve
    // time at large segments). Split instead:
    //   dense primes  (p <= P_SPLIT): parallelize over range chunks;
    //   sparse primes (p >  P_SPLIT): parallelize over primes.
    static constexpr UInt32 P_SPLIT  = 1u << 16;
    static constexpr UInt64 SQ_CHUNK = UInt64(1) << 26;   // 64 MB

    UInt32 splitIdx = fromIdx;
    while (splitIdx < toIdx && primes[splitIdx] <= P_SPLIT)
        ++splitIdx;

    // Dense group: range-chunked. Costs one extra quotient per (prime, chunk),
    // negligible next to the removed serialization.
    const UInt64 nChunks = (len + SQ_CHUNK - 1) / SQ_CHUNK;
    #pragma omp parallel for schedule(dynamic, 1)
    for (UInt64 c = 0; c < nChunks; ++c) {
        const UInt64 cLo   = lo + c * SQ_CHUNK;                       // absolute
        const UInt64 cEndR = std::min(c * SQ_CHUNK + SQ_CHUNK, len);  // relative, exclusive

        for (UInt32 i = fromIdx; i < splitIdx; ++i) {
            const UInt64 p = primes[i];
            const UInt64 p2 = p * p;

            UInt64 st2;
            if constexpr (UseDivisionFree) {
                // ceil(cLo / p^2) = ceil(ceil(cLo / p) / p)
                const UInt64 q1 = cache->ceilDiv(cLo, i, p);
                const UInt64 q2 = cache->ceilDiv(q1, i, p);
                st2 = p2 * q2 - lo;
            } else {
                st2 = p2 * (((cLo - 1) / p2) + 1) - lo;
            }

            for (UInt64 pos2 = st2; pos2 < cEndR; pos2 += p2)
                Mu[pos2] = 0;
        }
    }

    // Sparse group: per-prime over the whole segment.
    #pragma omp parallel for schedule(dynamic, 256)
    for (UInt32 i = splitIdx; i < toIdx; ++i) {
        const UInt64 p = primes[i];
        const UInt64 p2 = p * p;

        UInt64 st2;
        if constexpr (UseDivisionFree) {
            const UInt64 q1 = cache->ceilDiv(lo, i, p);
            const UInt64 q2 = cache->ceilDiv(q1, i, p);
            st2 = p2 * q2 - lo;
        } else {
            st2 = p2 * (((lo - 1) / p2) + 1) - lo;
        }

        for (UInt64 pos2 = st2; pos2 < len; pos2 += p2)
            Mu[pos2] = 0;
    }
}

// ============================================================================
// Phase 2: Medium primes
// ============================================================================

void SegmentedMobiusSieveCore::sieveMediumSubSegment(Int8* __restrict Mu, const UInt32* __restrict primes,
                                                     UInt64 lo, UInt64 hi,
                                                     UInt64 fromIdx, UInt64 toIdx,
                                                     const SieveQuotientCache* cache) {
    const UInt64 len = hi - lo + 1;

    auto firstOffset = [&](UInt64 i, UInt64 p) {
        if constexpr (UseDivisionFree) {
            return p * cache->ceilDiv(lo, i, p) - lo;
        } else {
            return p * (((lo - 1) / p) + 1) - lo;
        }
    };

    auto finishStream = [&](UInt64 pos, UInt64 p, Int8 logp) {
        for (; pos + 3 * p < len; pos += 4 * p) {
            Mu[pos] += logp;
            Mu[pos + p] += logp;
            Mu[pos + 2 * p] += logp;
            Mu[pos + 3 * p] += logp;
        }
        for (; pos < len; pos += p)
            Mu[pos] += logp;
    };

    UInt64 i = fromIdx;
    for (; i + 3 < toIdx; i += 4) {
        const UInt64 p0 = primes[i];
        const UInt64 p1 = primes[i + 1];
        const UInt64 p2 = primes[i + 2];
        const UInt64 p3 = primes[i + 3];
        UInt64 pos0 = firstOffset(i, p0);
        UInt64 pos1 = firstOffset(i + 1, p1);
        UInt64 pos2 = firstOffset(i + 2, p2);
        UInt64 pos3 = firstOffset(i + 3, p3);
        const Int8 l0 = static_cast<Int8>((64 - __builtin_clzll(p0)) | 1ULL);
        const Int8 l1 = static_cast<Int8>((64 - __builtin_clzll(p1)) | 1ULL);
        const Int8 l2 = static_cast<Int8>((64 - __builtin_clzll(p2)) | 1ULL);
        const Int8 l3 = static_cast<Int8>((64 - __builtin_clzll(p3)) | 1ULL);

        while (pos0 < len && pos1 < len && pos2 < len && pos3 < len) {
            Mu[pos0] += l0;
            Mu[pos1] += l1;
            Mu[pos2] += l2;
            Mu[pos3] += l3;
            pos0 += p0;
            pos1 += p1;
            pos2 += p2;
            pos3 += p3;
        }
        finishStream(pos0, p0, l0);
        finishStream(pos1, p1, l1);
        finishStream(pos2, p2, l2);
        finishStream(pos3, p3, l3);
    }
    for (; i < toIdx; ++i) {
        const UInt64 p = primes[i];
        const UInt64 pos = firstOffset(i, p);
        const Int8 logp = static_cast<Int8>((64 - __builtin_clzll(p)) | 1ULL);
        finishStream(pos, p, logp);
    }
}

void SegmentedMobiusSieveCore::sieveMediumSubSegmentRolling(
    Int8* __restrict Mu, const UInt32* __restrict primes, UInt64 len,
    UInt64 fromIdx, UInt64 toIdx, std::vector<UInt32>& nextOff) {
    auto finishStream = [&](UInt64 pos, UInt64 p, Int8 logp) {
        for (; pos + 3 * p < len; pos += 4 * p) {
            Mu[pos] += logp;
            Mu[pos + p] += logp;
            Mu[pos + 2 * p] += logp;
            Mu[pos + 3 * p] += logp;
        }
        for (; pos < len; pos += p)
            Mu[pos] += logp;
        return pos;
    };

    UInt64 i = fromIdx;
    for (; i + 3 < toIdx; i += 4) {
        const UInt64 p0 = primes[i];
        const UInt64 p1 = primes[i + 1];
        const UInt64 p2 = primes[i + 2];
        const UInt64 p3 = primes[i + 3];
        UInt64 pos0 = nextOff[i - fromIdx];
        UInt64 pos1 = nextOff[i + 1 - fromIdx];
        UInt64 pos2 = nextOff[i + 2 - fromIdx];
        UInt64 pos3 = nextOff[i + 3 - fromIdx];
        const Int8 l0 = static_cast<Int8>((64 - __builtin_clzll(p0)) | 1ULL);
        const Int8 l1 = static_cast<Int8>((64 - __builtin_clzll(p1)) | 1ULL);
        const Int8 l2 = static_cast<Int8>((64 - __builtin_clzll(p2)) | 1ULL);
        const Int8 l3 = static_cast<Int8>((64 - __builtin_clzll(p3)) | 1ULL);

        while (pos0 < len && pos1 < len && pos2 < len && pos3 < len) {
            Mu[pos0] += l0;
            Mu[pos1] += l1;
            Mu[pos2] += l2;
            Mu[pos3] += l3;
            pos0 += p0;
            pos1 += p1;
            pos2 += p2;
            pos3 += p3;
        }
        pos0 = finishStream(pos0, p0, l0);
        pos1 = finishStream(pos1, p1, l1);
        pos2 = finishStream(pos2, p2, l2);
        pos3 = finishStream(pos3, p3, l3);
        nextOff[i - fromIdx] = static_cast<UInt32>(pos0 - len);
        nextOff[i + 1 - fromIdx] = static_cast<UInt32>(pos1 - len);
        nextOff[i + 2 - fromIdx] = static_cast<UInt32>(pos2 - len);
        nextOff[i + 3 - fromIdx] = static_cast<UInt32>(pos3 - len);
    }
    for (; i < toIdx; ++i) {
        const UInt64 p = primes[i];
        const Int8 logp = static_cast<Int8>((64 - __builtin_clzll(p)) | 1ULL);
        const UInt64 pos = finishStream(nextOff[i - fromIdx], p, logp);
        nextOff[i - fromIdx] = static_cast<UInt32>(pos - len);
    }
}

// ============================================================================
// Finalization: packed sieve values -> mu ∈ {-1, 0, +1}
// ============================================================================

#if SIEVE_SIMD_NEON || SIEVE_SIMD_SSE
// Constant-threshold finalize kernel (ceil-log2 scheme).
//
// For each byte v: squarefree iff v < 0 (bit 7 set). Then S = v & 0x7F is the
// summed (ceil(log2 p) | 1) weights and par = v & 1 the factor-count parity.
// fl = floor(log2 N) is constant across [Mu, Mu+n) — the caller splits the
// segment at powers of two so a single broadcast threshold is exact here.
//
//   mu = squarefree ? ( (S > fl) ? 1 - 2*par : 2*par - 1 ) : 0
//
// Both S (0..127) and fl (0..63) are non-negative, so the S > fl test is a
// plain signed-byte compare — no unsigned bias trick needed. The caller
// guarantees N >= 3 for every lane (N <= 2 is handled separately).
void SegmentedMobiusSieveCore::finalizeMuVec(
    Int8* __restrict Mu, int fl, size_t n) {
    size_t i = 0;
    const Int8 flb = static_cast<Int8>(fl);

#if SIEVE_SIMD_SVE2
    const svint8_t svZero  = svdup_n_s8(0);
    const svint8_t svOne   = svdup_n_s8(1);
    const svint8_t svMask7 = svdup_n_s8(0x7F);
    const svint8_t svFl    = svdup_n_s8(flb);

    for (; i < n; i += svcntb()) {
        svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)n);

        svint8_t v = svld1_s8(pg, Mu + i);

        // sf = (v < 0) — squarefree
        svbool_t sf = svcmplt_s8(pg, v, svZero);

        // S = v & 0x7F
        svint8_t S = svand_s8_x(pg, v, svMask7);

        // base = 2*(v & 1) - 1
        svint8_t lsb  = svand_s8_x(pg, v, svOne);
        svint8_t base = svsub_s8_x(pg, svadd_s8_x(pg, lsb, lsb), svOne);
        svint8_t negBase = svneg_s8_x(pg, base);

        // ff = (S > fl) — fully factored; r = ff ? -base : base
        svbool_t ff = svcmpgt_s8(pg, S, svFl);
        svint8_t r  = svsel_s8(ff, negBase, base);

        svint8_t out = svsel_s8(sf, r, svZero);
        svst1_s8(pg, Mu + i, out);
    }

#elif SIEVE_SIMD_NEON
    const int8x16_t vZero  = vdupq_n_s8(0);
    const int8x16_t vOne   = vdupq_n_s8(1);
    const int8x16_t vMask7 = vdupq_n_s8(0x7F);
    const int8x16_t vFl    = vdupq_n_s8(flb);

    for (; i + 16 <= n; i += 16) {
        int8x16_t v = vld1q_s8(Mu + i);

        uint8x16_t sf = vcltq_s8(v, vZero);          // v < 0
        int8x16_t  S  = vandq_s8(v, vMask7);

        int8x16_t lsb  = vandq_s8(v, vOne);
        int8x16_t base = vsubq_s8(vshlq_n_s8(lsb, 1), vOne);
        int8x16_t negBase = vnegq_s8(base);

        uint8x16_t ff = vcgtq_s8(S, vFl);            // S > fl (signed)
        int8x16_t  r  = vbslq_s8(ff, negBase, base);

        int8x16_t out = vbslq_s8(sf, r, vZero);
        vst1q_s8(Mu + i, out);
    }

#elif SIEVE_SIMD_AVX512
    const __m512i vZero  = _mm512_setzero_si512();
    const __m512i vOne   = _mm512_set1_epi8(1);
    const __m512i vMask7 = _mm512_set1_epi8(0x7F);
    const __m512i vFl    = _mm512_set1_epi8(flb);

    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512((const __m512i*)(Mu + i));

        __mmask64 sf = _mm512_cmplt_epi8_mask(v, vZero);   // v < 0
        __m512i   S  = _mm512_and_si512(v, vMask7);

        __m512i lsb  = _mm512_and_si512(v, vOne);
        __m512i base = _mm512_sub_epi8(_mm512_add_epi8(lsb, lsb), vOne);
        __m512i negBase = _mm512_sub_epi8(vZero, base);

        __mmask64 ff = _mm512_cmpgt_epi8_mask(S, vFl);     // S > fl
        // r = ff ? negBase : base
        __m512i r = _mm512_mask_blend_epi8(ff, base, negBase);

        // out = sf ? r : 0
        __m512i out = _mm512_maskz_mov_epi8(sf, r);
        _mm512_storeu_si512((__m512i*)(Mu + i), out);
    }

#elif SIEVE_SIMD_AVX2
    const __m256i vZero  = _mm256_setzero_si256();
    const __m256i vOne   = _mm256_set1_epi8(1);
    const __m256i vMask7 = _mm256_set1_epi8(0x7F);
    const __m256i vFl    = _mm256_set1_epi8(flb);

    for (; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(Mu + i));

        __m256i sf = _mm256_cmpgt_epi8(vZero, v);          // v < 0
        __m256i S  = _mm256_and_si256(v, vMask7);

        __m256i lsb  = _mm256_and_si256(v, vOne);
        __m256i base = _mm256_sub_epi8(_mm256_add_epi8(lsb, lsb), vOne);
        __m256i negBase = _mm256_sub_epi8(vZero, base);

        __m256i ff = _mm256_cmpgt_epi8(S, vFl);            // S > fl (signed; both >= 0)
        __m256i r  = _mm256_or_si256(_mm256_and_si256(ff, negBase),
                                     _mm256_andnot_si256(ff, base));

        __m256i out = _mm256_and_si256(sf, r);
        _mm256_storeu_si256((__m256i*)(Mu + i), out);
    }

#elif SIEVE_SIMD_SSE
    const __m128i vZero  = _mm_setzero_si128();
    const __m128i vOne   = _mm_set1_epi8(1);
    const __m128i vMask7 = _mm_set1_epi8(0x7F);
    const __m128i vFl    = _mm_set1_epi8(flb);

    for (; i + 16 <= n; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i*)(Mu + i));

        __m128i sf = _mm_cmplt_epi8(v, vZero);             // v < 0
        __m128i S  = _mm_and_si128(v, vMask7);

        __m128i lsb  = _mm_and_si128(v, vOne);
        __m128i base = _mm_sub_epi8(_mm_add_epi8(lsb, lsb), vOne);
        __m128i negBase = _mm_sub_epi8(vZero, base);

        __m128i ff = _mm_cmpgt_epi8(S, vFl);               // S > fl (signed; both >= 0)
        __m128i r  = _mm_or_si128(_mm_and_si128(ff, negBase),
                                  _mm_andnot_si128(ff, base));

        __m128i out = _mm_and_si128(sf, r);
        _mm_storeu_si128((__m128i*)(Mu + i), out);
    }
#endif

    // Scalar tail for SIMD remainder (N >= 3 guaranteed by caller).
    for (; i < n; ++i) {
        const Int8 v = Mu[i];
        if (v >= 0) { Mu[i] = 0; continue; }     // square factor -> mu = 0
        const int S    = v & 0x7F;
        const int base = ((v & 1) << 1) - 1;     // 2*par - 1 in {-1,+1}
        Mu[i] = static_cast<Int8>((S > fl) ? -base : base);
    }
}
#endif // SIEVE_SIMD_NEON || SIEVE_SIMD_SSE

// Convert packed log-prime sums to mu in {-1, 0, +1} (ceil-log2 scheme).
//
// Each Mu[i] holds 0x80 | S for squarefree lo+i (bit 7 set, low 7 bits hold
// S = sum of (ceil(log2 p) | 1) over the sieved prime factors), or 0 if a
// square factor was found. With ceil weights, S > floor(log2 N) exactly when
// N is fully factored by the sieved primes; otherwise an unsieved prime
// cofactor (> sqrt N) remains. par = bit 0 carries the factor-count parity.
//
//   mu = !squarefree ? 0
//      : N <= 2       ? (N == 1 ? 1 : -1)            // S-vs-floor misreads these
//      : (S > floor_log2(N)) ? 1 - 2*par : 2*par - 1
//
// floor_log2(N) is constant between consecutive powers of two, so the segment
// is finalized in constant-threshold runs; the SIMD kernel applies one
// broadcast threshold per run. Within a typical large-lo sub-segment the run is the
// whole sub-segment (powers of two are spaced far wider than M2).
void SegmentedMobiusSieveCore::finalizeMu(Int8* __restrict Mu, UInt64 lo, UInt64 hi) {
    const UInt64 len = hi - lo + 1;
    UInt64 pos = 0;

    // Small-N guard: mu(1) = 1, mu(2) = -1 (only ever the very first segment).
    for (; pos < len && lo + pos <= 2; ++pos)
        Mu[pos] = (lo + pos == 1) ? Int8(1) : Int8(-1);

    // Constant-floor(log2 N) runs, split at each power of two.
    while (pos < len) {
        const UInt64 N0 = lo + pos;
        const int fl = 63 - __builtin_clzll(N0);
        const UInt64 runHi = (fl < 63) ? std::min(hi, (UInt64(1) << (fl + 1)) - 1) : hi;
        const size_t runLen = static_cast<size_t>(runHi - N0 + 1);

#if SIEVE_SIMD_NEON || SIEVE_SIMD_SSE
        finalizeMuVec(Mu + pos, fl, runLen);
#else
        for (size_t j = 0; j < runLen; ++j) {
            const Int8 v = Mu[pos + j];
            if (v >= 0) { Mu[pos + j] = 0; continue; }   // square factor -> mu = 0
            const int S    = v & 0x7F;
            const int base = ((v & 1) << 1) - 1;         // 2*par - 1 in {-1,+1}
            Mu[pos + j] = static_cast<Int8>((S > fl) ? -base : base);
        }
#endif
        pos += runLen;
    }
}

// ============================================================================
// LargePrimeHitScheduler — bucket sieve for primes > segment size
// ============================================================================

SegmentedMobiusSieveCore::LargePrimeHitScheduler::LargePrimeHitScheduler(
    const UInt64& baseLo,
    const UInt64& lo,
    const UInt64& hi,
    const UInt64& subSegmentLen,
    const PrimesVecT& primes,
    const UInt32& pInd0,
    const UInt32& pInd1,
    std::vector<PVecT>& lpBucket,
    const SieveQuotientCache* cache
)
    : mBaseLo(baseLo)
    , mLo(lo)
    , mHi(hi)
    , mSubSegmentLen(subSegmentLen)
    , mCurrentSubSegIndex(0)
    , mFinalSubSegIndex((hi - lo) / subSegmentLen)
    , mP(primes.data())
    , mPInd0(pInd0)
    , mPInd1(pInd1)
    , mBuckets(lpBucket)
    , mCache(cache)
{
    if (mBuckets.size() < LP_NBUCKETS)
        mBuckets.resize(LP_NBUCKETS);

    const UInt64 L = subSegLo(mCurrentSubSegIndex);

    for (UInt32 i = mPInd0; i <= mPInd1; ++i) {
        const UInt32 p = mP[i];

        if (p > mHi)
            break;

        UInt64 m;
        if constexpr (UseDivisionFree) {
            m = p * mCache->ceilDiv(L, i, p);
        } else {
            m = p * (((L - 1) / p) + 1);
        }
        const UInt64 subSeg = subSegIndexOf(m);
        if (subSeg < mFinalSubSegIndex || (subSeg == mFinalSubSegIndex && m <= mHi))
            bucketPush(subSeg, packEntry(p, (m - mLo) - subSeg * M2));
    }
}

// Pre-populated constructor: bookkeeping only, no prime scan.
SegmentedMobiusSieveCore::LargePrimeHitScheduler::LargePrimeHitScheduler(
    PrePopulated,
    const UInt64& baseLo,
    const UInt64& lo,
    const UInt64& hi,
    const UInt64& subSegmentLen,
    const PrimesVecT& primes,
    const UInt32& pInd0,
    const UInt32& pInd1,
    std::vector<PVecT>& lpBucket,
    const SieveQuotientCache* cache
)
    : mBaseLo(baseLo)
    , mLo(lo)
    , mHi(hi)
    , mSubSegmentLen(subSegmentLen)
    , mCurrentSubSegIndex(0)
    , mFinalSubSegIndex((hi - lo) / subSegmentLen)
    , mP(primes.data())
    , mPInd0(pInd0)
    , mPInd1(pInd1)
    , mBuckets(lpBucket)
    , mCache(cache)
{
    // Buckets are already populated by the external routing pass.
    // Just ensure the array is sized.
    if (mBuckets.size() < LP_NBUCKETS)
        mBuckets.resize(LP_NBUCKETS);
}

// subSegLo/subSegIndexOf use the compile-time constant M2 rather than the
// mSubSegmentLen member (always M2 in practice — both construction sites pass
// it). With a constant divisor the compiler strength-reduces the division
// in subSegIndexOf, which sits on the per-hit path of sieveSubSegment, to a
// multiply + shift.
UInt64 SegmentedMobiusSieveCore::LargePrimeHitScheduler::subSegLo(
    const UInt64& subSegIndex) const noexcept {
    return mLo + subSegIndex * M2;
}

UInt64 SegmentedMobiusSieveCore::LargePrimeHitScheduler::subSegIndexOf(
    const UInt64& x) const noexcept {
    return (x - mLo) / M2;
}

UInt64 SegmentedMobiusSieveCore::LargePrimeHitScheduler::ringIndex(
    const UInt64& subSeg, EntryT e) noexcept {
#if SIEVE_SUBS_ACTIVE
    return (subSeg & (LP_SIZE - 1)) * LP_SUBS + ((e & LP_OFF_MASK) >> LP_SUB_SHIFT);
#else
    (void)e;
    return subSeg & (LP_SIZE - 1);
#endif
}

bool SegmentedMobiusSieveCore::LargePrimeHitScheduler::emptySubSegment(
    const UInt64& subSeg) const noexcept {
#if SIEVE_SUBS_ACTIVE
    const UInt64 b0 = (subSeg & (LP_SIZE - 1)) * LP_SUBS;
    for (UInt64 sb = 0; sb < LP_SUBS; ++sb)
        if (!mBuckets[b0 + sb].empty()) return false;
    return true;
#else
    return mBuckets[subSeg & (LP_SIZE - 1)].empty();
#endif
}

void SegmentedMobiusSieveCore::LargePrimeHitScheduler::bucketPush(
    const UInt64& subSeg, EntryT e) noexcept {
    mBuckets[ringIndex(subSeg, e)].push_back(e);
}

Int8 SegmentedMobiusSieveCore::LargePrimeHitScheduler::logprime(UInt32 p) noexcept {
    return static_cast<Int8>((32 - __builtin_clz(p)) | 1U);
}

SegmentedMobiusSieveCore::LargePrimeHitScheduler::EntryT
SegmentedMobiusSieveCore::LargePrimeHitScheduler::packEntry(UInt32 p, UInt64 off) noexcept {
#if SIEVE_NARROW_ENTRY
    (void)off;                      // offset recomputed per hit from p
    return (EntryT)p;
#else
    const UInt64 s = p / M2;        // constant divisor — strength-reduced
    const UInt64 r = p - s * M2;
    return (EntryT(UInt8(logprime(p))) << LP_LOG_SHIFT)
         | (s << LP_S_SHIFT) | (r << LP_R_SHIFT) | off;
#endif
}

void SegmentedMobiusSieveCore::LargePrimeHitScheduler::sieveSubSegment(
    Int8* __restrict muBase) noexcept {
    if (__builtin_expect(mCurrentSubSegIndex > mFinalSubSegIndex, false)) {
        return;
    }

    if (__builtin_expect(emptySubSegment(mCurrentSubSegIndex), false)) {
        ++mCurrentSubSegIndex;
        return;
    }

    const UInt64 L = subSegLo(mCurrentSubSegIndex);

    // Process each prime in this sub-segment's bucket: apply its hit, then
    // forward the prime to the bucket containing its next multiple.
    // This ensures each large prime is touched at most once per sub-segment.
    //
    // Squared factors are handled once per segment by sieveSquaresPass,
    // so no per-hit divisibility check is needed here.
    //
    // Entries are self-contained (off, p mod M2, p / M2, log weight), so a
    // hit is a byte RMW with the entry's log field, and forwarding is the
    // Bresenham step off' = off + r with the carry advancing the sub-segment
    // stride — no division, no multiply, no CLZ, no prime-array load.
    const UInt64 base    = L - mBaseLo;    // muBase index of this sub-segment's start
    const UInt64 lastOff = (mHi - mLo) - mFinalSubSegIndex * M2;  // last valid
                                           // offset within the final sub-segment

    // The mu RMW targets are random offsets within the sub-segment — invisible to
    // hardware prefetchers — but the entry stream is sequential, so each
    // future target is known PF entries in advance. Prefetch with write
    // intent; the line is reused by later hits and by the fused finalize.
    constexpr size_t PF = 16;

    // With sub-buckets, process the sub-segment's LP_SUBS offset bands in order:
    // each band's RMWs land in a ~128 KB window, so the mu lines stay
    // L1-resident across the band. Without them, LP_SUBS == 1 and this
    // loop runs once over the sub-segment's single bucket.
    const UInt64 ring0 = (mCurrentSubSegIndex & (LP_SIZE - 1)) * LP_SUBS;
    for (UInt64 sb = 0; sb < LP_SUBS; ++sb) {
        PVecT& pvec = mBuckets[ring0 + sb];
        const EntryT* e = pvec.data();
        const size_t  n = pvec.size();

#if SIEVE_NARROW_ENTRY
        // Narrow entries hold only the prime: one raw divide per hit
        // recomputes the offset, CLZ recomputes the log weight. The quotient
        // cache cannot help — it is keyed by prime index, which the entry
        // does not store, and carrying the index would grow the entry back
        // toward 8 bytes. No prefetch (the target needs the divide).
        for (size_t i = 0; i < n; ++i) {
            const UInt64 p = e[i];
            const UInt64 m = p * ((L - 1) / p + 1);   // smallest multiple of p >= L
            muBase[base + (m - L)] += logprime((UInt32)p);
            const UInt64 nextM = m + p;
            if (nextM <= mHi)
                bucketPush((nextM - mLo) / M2, (EntryT)p);
        }
#elif SIEVE_TWO_PASS
        // Pass 1: apply all mu RMWs — a pure scatter loop, so the out-of-order
        // window holds the maximum number of outstanding misses.
        for (size_t i = 0; i < n; ++i) {
            const size_t ipf = (i + PF < n) ? i + PF : n - 1;
            __builtin_prefetch(&muBase[base + (e[ipf] & LP_OFF_MASK)], 1, 3);
            const EntryT entry = e[i];
            muBase[base + (entry & LP_OFF_MASK)] += (Int8)(entry >> LP_LOG_SHIFT);
        }
        // Pass 2: forward entries — pure ALU plus bucket appends; the entry
        // stream is cache-hot from pass 1.
        for (size_t i = 0; i < n; ++i) {
            const EntryT entry = e[i];
            UInt64 noff = (entry & LP_OFF_MASK) + ((entry >> LP_R_SHIFT) & LP_OFF_MASK);
            const UInt64 carry = noff >= M2;
            noff -= M2 & (UInt64(0) - carry);
            const UInt64 nSubSeg = mCurrentSubSegIndex
                                + ((entry >> LP_S_SHIFT) & LP_S_MASK) + carry;

            if (nSubSeg < mFinalSubSegIndex || (nSubSeg == mFinalSubSegIndex && noff <= lastOff))
                bucketPush(nSubSeg, (entry & ~LP_OFF_MASK) | noff);
        }
#else
        for (size_t i = 0; i < n; ++i) {
            const size_t ipf = (i + PF < n) ? i + PF : n - 1;
            __builtin_prefetch(&muBase[base + (e[ipf] & LP_OFF_MASK)], 1, 3);

            const EntryT entry = e[i];
            const UInt64 off = entry & LP_OFF_MASK;

            muBase[base + off] += (Int8)(entry >> LP_LOG_SHIFT);

            // The carry is data-random (~50/50), so keep it branchless.
            UInt64 noff = off + ((entry >> LP_R_SHIFT) & LP_OFF_MASK);
            const UInt64 carry = noff >= M2;
            noff -= M2 & (UInt64(0) - carry);
            const UInt64 nSubSeg = mCurrentSubSegIndex
                                + ((entry >> LP_S_SHIFT) & LP_S_MASK) + carry;

            if (nSubSeg < mFinalSubSegIndex || (nSubSeg == mFinalSubSegIndex && noff <= lastOff))
                bucketPush(nSubSeg, (entry & ~LP_OFF_MASK) | noff);
        }
#endif

        pvec.clear();
    }  // sub-bucket loop

    ++mCurrentSubSegIndex;
}
