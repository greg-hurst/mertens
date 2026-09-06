#include "SegmentedOddMobiusSieve.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>

#ifndef USE_BUCKET_SIEVE
#define USE_BUCKET_SIEVE 1
#endif
static constexpr bool UseOddBucketSieve = USE_BUCKET_SIEVE;

// Keep these byte counts equal to the established full-sieve defaults.  Their
// multipliers are four times larger only because the odd stencil is four times
// shorter in packed slots.
#ifndef ODD_SIEVE_MEDIUM_TILE_MULT
#define ODD_SIEVE_MEDIUM_TILE_MULT 672
#endif
#ifndef ODD_SIEVE_DIRECT_CUTOFF_MULT
#define ODD_SIEVE_DIRECT_CUTOFF_MULT 800
#endif
static constexpr UInt64 MEDIUM_TILE_SWITCH = UInt64(1) << 36;
static constexpr UInt64 MEDIUM_TILE_LARGE =
    UInt64(ODD_SIEVE_MEDIUM_TILE_MULT) * SegmentedOddMobiusSieveCore::STENCIL_PERIOD;
static constexpr UInt64 TILED_DIRECT_SIEVE_CUTOFF =
    UInt64(ODD_SIEVE_DIRECT_CUTOFF_MULT) * SegmentedOddMobiusSieveCore::STENCIL_PERIOD;

// ============================================================================
// Construction and coordinate helpers
// ============================================================================

SegmentedOddMobiusSieveCore::SegmentedOddMobiusSieveCore()
    : mStencilData(STENCIL_PERIOD)
{
    // n = 1 + 2*i.  The odd stencil handles 3,5,7,11 and the 3^2 square.
    // Squares of 5,7,11 are handled in the M1 pass, keeping this short period.
    for (UInt64 i = 0; i < STENCIL_PERIOD; ++i) {
        const UInt64 n = 1 + 2 * i;
        if (n % 9 == 0) {
            mStencilData[i] = 0;
            continue;
        }

        UInt8 v = 0x80;
        if (n % 3  == 0) v += 3;
        if (n % 5  == 0) v += 3;
        if (n % 7  == 0) v += 3;
        if (n % 11 == 0) v += 5;
        mStencilData[i] = static_cast<Int8>(v);
    }
}

SegmentedOddMobiusSieveCore::SegmentedOddMobiusSieveCore(UInt64 segmentSize)
    : SegmentedOddMobiusSieveCore()
{
    initialize(segmentSize);
}

void SegmentedOddMobiusSieveCore::initialize(UInt64 segmentSize) {
    const UInt64 packedCapacity = (segmentSize >> 1) + (segmentSize & 1);

    mBuckets.clear();
    mBuckets.resize(static_cast<size_t>(omp_get_max_threads()));
    mMu.resize(packedCapacity);

    mPreMu.resize(M1 + STENCIL_PERIOD);
    for (UInt64 off = 0; off < mPreMu.size(); off += STENCIL_PERIOD) {
        const UInt64 n = std::min<UInt64>(STENCIL_PERIOD, mPreMu.size() - off);
        std::memcpy(mPreMu.data() + off, mStencilData.data(), n * sizeof(Int8));
    }
}

void SegmentedOddMobiusSieveCore::fillPackedFromStencil(UInt64 packedCount) {
    mMu.resize(packedCount);
    if (packedCount == 0) return;

    const UInt64 targetTile = UInt64(4) << 20;
    UInt64 tile = STENCIL_PERIOD * (targetTile / STENCIL_PERIOD + 1);
    tile = std::min(tile, packedCount);

    std::vector<Int8> stencilTile(tile);
    const UInt64 firstCopy = std::min<UInt64>(tile, STENCIL_PERIOD);
    std::memcpy(stencilTile.data(), mStencilData.data(), firstCopy);
    UInt64 filled = firstCopy;
    while (filled < tile) {
        const UInt64 n = std::min(filled, tile - filled);
        std::memcpy(stencilTile.data() + filled, stencilTile.data(), n);
        filled += n;
    }

    const UInt64 numTiles = (packedCount + tile - 1) / tile;
    #pragma omp parallel for schedule(static)
    for (UInt64 i = 0; i < numTiles; ++i) {
        const UInt64 off = i * tile;
        const UInt64 n = std::min(tile, packedCount - off);
        std::memcpy(mMu.data() + off, stencilTile.data(), n);
    }
}

UInt64 SegmentedOddMobiusSieveCore::integerSquareRoot(UInt64 n) noexcept {
    UInt64 r = static_cast<UInt64>(std::sqrt(static_cast<long double>(n)));
    while (static_cast<__uint128_t>(r + 1) * (r + 1) <= n) ++r;
    while (static_cast<__uint128_t>(r) * r > n) --r;
    return r;
}

UInt64 SegmentedOddMobiusSieveCore::firstPackedOffset(UInt64 firstOdd,
                                                       UInt64 p) noexcept {
    UInt64 q = (firstOdd - 1) / p + 1;
    q += (q & 1) ^ 1;
    return (p * q - firstOdd) / 2;
}

UInt64 SegmentedOddMobiusSieveCore::firstPackedOffset(
    UInt64 firstOdd, UInt64 p, UInt32 primeIndex,
    const SieveQuotientCache* cache) noexcept {
    UInt64 q;
    if constexpr (UseDivisionFree)
        q = cache->ceilDiv(firstOdd, primeIndex, p);
    else
        q = (firstOdd - 1) / p + 1;
    q += (q & 1) ^ 1;
    return (p * q - firstOdd) / 2;
}

UInt64 SegmentedOddMobiusSieveCore::firstPackedSquareOffset(
    UInt64 firstOdd, UInt64 p, UInt32 primeIndex,
    const SieveQuotientCache* cache) noexcept {
    const UInt64 p2 = p * p;
    UInt64 q;
    if constexpr (UseDivisionFree) {
        const UInt64 q1 = cache->ceilDiv(firstOdd, primeIndex, p);
        q = cache->ceilDiv(q1, primeIndex, p);
    } else {
        q = (firstOdd - 1) / p2 + 1;
    }
    q += (q & 1) ^ 1;
    return (p2 * q - firstOdd) / 2;
}

void SegmentedOddMobiusSieveCore::initSieveQuotientCache(
    const std::vector<UInt32>& primes) {
    if constexpr (UseDivisionFree)
        mSieveQCache.init(primes.data(), static_cast<UInt32>(primes.size()));
}

std::vector<UInt32> SegmentedOddMobiusSieveCore::primesUpTo(UInt32 n) {
    std::vector<UInt32> values(n);
    for (UInt64 i = 2; i <= n; ++i)
        values[i - 2] = static_cast<UInt32>(i);

    const UInt32 sqrtN = static_cast<UInt32>(std::sqrt(static_cast<double>(n)));
    for (UInt32 i = 1; i <= sqrtN; ++i) {
        if (values[i - 1] != 0) {
            for (UInt64 k = UInt64(2) * i + 1; k <= UInt64(n) - 1; k += i + 1)
                values[k - 1] = 0;
        }
    }

    const auto end = std::remove(values.begin(), values.end(), UInt32(0));
    return std::vector<UInt32>(values.begin(), end);
}

// ============================================================================
// Main sieve entry point
// ============================================================================

template<bool Finalize>
void SegmentedOddMobiusSieveCore::sieve(
    UInt64 lo, UInt64 hi, const std::vector<UInt32>& primes) {
    mLo = lo;
    mHi = hi;
    mFirstOdd = firstOddAtOrAbove(lo);
    mOddCount = countOdds(lo, hi);

    if (mOddCount == 0) return;

    if (primes.size() < 71) {
        std::cerr << "Error: odd Mobius sieve requires primes through 353."
                  << std::endl;
        std::abort();
    }

    const UInt64 sqrtHi = integerSquareRoot(hi);
    if (mMu.size() < mOddCount) mMu.resize(mOddCount);
    if (mPreMu.empty()) initialize(2 * mOddCount);
    if (mBuckets.size() < static_cast<UInt64>(omp_get_max_threads()))
        mBuckets.resize(static_cast<size_t>(omp_get_max_threads()));

    if constexpr (UseDivisionFree) {
        if (mSieveQCache.count < primes.size())
            initSieveQuotientCache(primes);

        constexpr UInt64 quotientCacheLimit =
            (UInt64(1) << 60) - (UInt64(1) << 32);
        if (__builtin_expect(hi >= quotientCacheLimit, false)) {
            std::cerr << "Error: DIVISION_FREE=1 requires sieve endpoint < "
                      << quotientCacheLimit << "." << std::endl;
            std::abort();
        }
    }

    const UInt32 sqrtPrimeStop = static_cast<UInt32>(
        std::upper_bound(primes.begin(), primes.end(), static_cast<UInt32>(sqrtHi))
        - primes.begin());

    const UInt64 m1Limit = std::min<UInt64>(sqrtHi, M1_PRIME_CAP);
    const UInt32 m1PrimeStop = static_cast<UInt32>(
        std::upper_bound(primes.begin(), primes.end(), static_cast<UInt32>(m1Limit))
        - primes.begin());

    UInt32 largePrimeBegin = m1PrimeStop;
    if constexpr (UseOddBucketSieve) {
        while (largePrimeBegin < sqrtPrimeStop
               && primes[largePrimeBegin] <= TILED_DIRECT_SIEVE_CUTOFF)
            ++largePrimeBegin;
    } else {
        largePrimeBegin = sqrtPrimeStop;
    }

    if constexpr (UseOddBucketSieve) {
        if (__builtin_expect(
                sqrtPrimeStop > largePrimeBegin
                && UInt64(primes[sqrtPrimeStop - 1]) > schedulerReach(), false)) {
            std::cerr << "Error: largest odd-sieve prime exceeds bucket scheduler "
                      << "reach " << schedulerReach() << ". Increase LP_SIZE."
                      << std::endl;
            std::abort();
        }
    }

    Int8* mu = mMu.data();
    const UInt64 stencilOffset = ((mFirstOdd - 1) / 2) % STENCIL_PERIOD;
    const UInt64 mediumTile = hi >= MEDIUM_TILE_SWITCH ? MEDIUM_TILE_LARGE : M2;

    #pragma omp parallel for schedule(dynamic, 1)
    for (UInt64 tile = 0; tile < mOddCount; tile += mediumTile) {
        const UInt64 tileEnd = std::min(tile + mediumTile, mOddCount);

        for (UInt64 l = tile; l < tileEnd; l += M1) {
            const UInt64 n = std::min(M1, tileEnd - l);
            std::memcpy(mu + l, mPreMu.data() + stencilOffset, n);
            sieveM1PrimeRange(mu + l, primes.data(), mFirstOdd + 2 * l,
                              n, m1PrimeStop, &mSieveQCache);
        }

        if (m1PrimeStop < largePrimeBegin) {
            sieveMediumSubSegment(mu + tile, primes.data(),
                                  mFirstOdd + 2 * tile, tileEnd - tile,
                                  m1PrimeStop, largePrimeBegin,
                                  &mSieveQCache);
        }
    }

    // Zeroing commutes with the positive log-weight additions.  Doing this
    // once over the whole packed interval avoids a quotient per M1 chunk.
    sieveSquaresPass(mu, primes.data(), mFirstOdd, mOddCount,
                     71, sqrtPrimeStop, &mSieveQCache);

    if (largePrimeBegin < sqrtPrimeStop) {
        const UInt64 numSegments = (mOddCount + M2 - 1) / M2;

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int nt = omp_get_num_threads();
            const UInt64 s0 = numSegments * static_cast<UInt64>(tid)
                            / static_cast<UInt64>(nt);
            const UInt64 s1 = numSegments * static_cast<UInt64>(tid + 1)
                            / static_cast<UInt64>(nt);

            if (s0 < s1) {
                const UInt64 packedBase = s0 * M2;
                const UInt64 packedEnd = std::min(s1 * M2, mOddCount);
                LargePrimeHitScheduler scheduler(
                    packedBase, mFirstOdd + 2 * packedBase,
                    packedEnd - packedBase, primes,
                    largePrimeBegin, sqrtPrimeStop,
                    mBuckets[static_cast<size_t>(tid)], &mSieveQCache
                );

                for (UInt64 s = s0; s < s1; ++s) {
                    scheduler.sieveSubSegment(mu);
                    if constexpr (Finalize) {
                        const UInt64 off = s * M2;
                        const UInt64 n = std::min(M2, mOddCount - off);
                        finalizeMu(mu + off, mFirstOdd + 2 * off, n);
                    }
                }
            }
        }
    } else if constexpr (Finalize) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (UInt64 off = 0; off < mOddCount; off += M2) {
            const UInt64 n = std::min(M2, mOddCount - off);
            finalizeMu(mu + off, mFirstOdd + 2 * off, n);
        }
    }
}

template void SegmentedOddMobiusSieveCore::sieve<true>(
    UInt64, UInt64, const std::vector<UInt32>&);
template void SegmentedOddMobiusSieveCore::sieve<false>(
    UInt64, UInt64, const std::vector<UInt32>&);

// ============================================================================
// Packed M1 and medium-prime walks
// ============================================================================

void SegmentedOddMobiusSieveCore::sieveM1PrimeRange(
    Int8* __restrict mu, const UInt32* __restrict primes,
    UInt64 firstOdd, UInt64 packedCount, UInt32 stoppos,
    const SieveQuotientCache* cache) {
    auto addStride = [&](UInt64 pos, UInt64 p, Int8 logp) {
        for (; pos + 3 * p < packedCount; pos += 4 * p) {
            mu[pos]         += logp;
            mu[pos + p]     += logp;
            mu[pos + 2 * p] += logp;
            mu[pos + 3 * p] += logp;
        }
        for (; pos < packedCount; pos += p)
            mu[pos] += logp;
    };

    auto zeroStride = [&](UInt64 pos, UInt64 p2) {
        for (; pos + 3 * p2 < packedCount; pos += 4 * p2) {
            mu[pos]          = 0;
            mu[pos + p2]     = 0;
            mu[pos + 2 * p2] = 0;
            mu[pos + 3 * p2] = 0;
        }
        for (; pos < packedCount; pos += p2)
            mu[pos] = 0;
    };

    // The compiler fully unrolls these compact constant tables.  Consequently
    // each first-hit division is strength-reduced just as in the full sieve's
    // explicitly written small-prime kernel.
    static constexpr UInt16 smallPrimes[] = {
         13,  17,  19,  23,  29,  31,  37,  41,  43,  47,  53,  59,
         61,  67,  71,  73,  79,  83,  89,  97, 101, 103, 107, 109,
        113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
        181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241,
        251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
        317, 331, 337, 347, 349, 353,
    };

#if defined(__clang__)
    #pragma clang loop unroll(full)
#elif defined(__GNUC__)
    #pragma GCC unroll 128
#endif
    for (UInt64 i = 0; i < sizeof(smallPrimes) / sizeof(smallPrimes[0]); ++i) {
        const UInt64 p = smallPrimes[i];
        addStride(firstPackedOffset(firstOdd, p), p,
                  primeLogWeight(static_cast<UInt32>(p)));
    }

    static constexpr UInt16 constantPrimes[] = {
         359,  367,  373,  379,  383,  389,  397,  401,  409,  419,
         421,  431,  433,  439,  443,  449,  457,  461,  463,  467,
         479,  487,  491,  499,  503,  509,  521,  523,  541,  547,
         557,  563,  569,  571,  577,  587,  593,  599,  601,  607,
         613,  617,  619,  631,  641,  643,  647,  653,  659,  661,
         673,  677,  683,  691,  701,  709,  719,  727,  733,  739,
         743,  751,  757,  761,  769,  773,  787,  797,  809,  811,
         821,  823,  827,  829,  839,  853,  857,  859,  863,  877,
         881,  883,  887,  907,  911,  919,  929,  937,  941,  947,
         953,  967,  971,  977,  983,  991,  997, 1009, 1013, 1019,
        1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087,
        1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153,
        1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229,
        1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297,
        1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381,
        1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453,
        1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523,
        1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597,
        1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663,
        1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741,
        1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823,
        1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901,
        1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993,
        1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039,
    };

    static constexpr UInt64 constantPrimeCount = []() constexpr {
        UInt64 count = 0;
        while (count < sizeof(constantPrimes) / sizeof(constantPrimes[0])
               && constantPrimes[count] <= M1_CONSTANT_PRIME_CAP)
            ++count;
        return count;
    }();

#if defined(__clang__)
    #pragma clang loop unroll(full)
#elif defined(__GNUC__)
    #pragma GCC unroll 256
#endif
    for (UInt64 i = 0; i < constantPrimeCount; ++i) {
        const UInt64 p = constantPrimes[i];
        addStride(firstPackedOffset(firstOdd, p), p,
                  primeLogWeight(static_cast<UInt32>(p)));
    }

    // The prime list is ordered and starts at 2.  This is computed at compile
    // time from the constant kernel rather than searched once per M1 chunk.
    static constexpr UInt32 firstVariablePrimeIndex =
        static_cast<UInt32>(71 + constantPrimeCount);

    auto firstOffset = [&](UInt32 i, UInt64 p) {
        return firstPackedOffset(firstOdd, p, i, cache);
    };

    auto finishStream = [&](UInt64 pos, UInt64 p, Int8 logp) {
        addStride(pos, p, logp);
    };

    UInt32 i = firstVariablePrimeIndex;
    for (; i + 3 < stoppos; i += 4) {
        const UInt64 p0 = primes[i];
        const UInt64 p1 = primes[i + 1];
        const UInt64 p2 = primes[i + 2];
        const UInt64 p3 = primes[i + 3];
        UInt64 pos0 = firstOffset(i, p0);
        UInt64 pos1 = firstOffset(i + 1, p1);
        UInt64 pos2 = firstOffset(i + 2, p2);
        UInt64 pos3 = firstOffset(i + 3, p3);
        const Int8 l0 = primeLogWeight(static_cast<UInt32>(p0));
        const Int8 l1 = primeLogWeight(static_cast<UInt32>(p1));
        const Int8 l2 = primeLogWeight(static_cast<UInt32>(p2));
        const Int8 l3 = primeLogWeight(static_cast<UInt32>(p3));

        while (pos0 < packedCount && pos1 < packedCount
               && pos2 < packedCount && pos3 < packedCount) {
            mu[pos0] += l0;
            mu[pos1] += l1;
            mu[pos2] += l2;
            mu[pos3] += l3;
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
    for (; i < stoppos; ++i) {
        const UInt64 p = primes[i];
        finishStream(firstOffset(i, p), p,
                     primeLogWeight(static_cast<UInt32>(p)));
    }

    static constexpr UInt16 stencilSquarePrimes[] = {5, 7, 11, 13, 17, 19};
#if defined(__clang__)
    #pragma clang loop unroll(full)
#endif
    for (UInt64 j = 0;
         j < sizeof(stencilSquarePrimes) / sizeof(stencilSquarePrimes[0]); ++j) {
        const UInt64 p = stencilSquarePrimes[j];
        const UInt64 p2 = p * p;
        zeroStride(firstPackedOffset(firstOdd, p2), p2);
    }

    // Remaining squares below 359.  Their factor additions above are fully
    // unrolled; square hits are sparse enough that this compact loop is best.
    for (UInt32 j = 8; j < 71; ++j) {
        const UInt64 p = primes[j];
        const UInt64 p2 = p * p;
        zeroStride(firstPackedSquareOffset(firstOdd, p, j, cache), p2);
    }
}

void SegmentedOddMobiusSieveCore::sieveMediumSubSegment(
    Int8* __restrict mu, const UInt32* __restrict primes,
    UInt64 firstOdd, UInt64 packedCount, UInt32 fromIdx, UInt32 toIdx,
    const SieveQuotientCache* cache) {
    auto finishStream = [&](UInt64 pos, UInt64 p, Int8 logp) {
        for (; pos + 3 * p < packedCount; pos += 4 * p) {
            mu[pos]         += logp;
            mu[pos + p]     += logp;
            mu[pos + 2 * p] += logp;
            mu[pos + 3 * p] += logp;
        }
        for (; pos < packedCount; pos += p)
            mu[pos] += logp;
    };

    UInt32 i = fromIdx;
    for (; i + 3 < toIdx; i += 4) {
        const UInt64 p0 = primes[i];
        const UInt64 p1 = primes[i + 1];
        const UInt64 p2 = primes[i + 2];
        const UInt64 p3 = primes[i + 3];
        UInt64 pos0 = firstPackedOffset(firstOdd, p0, i, cache);
        UInt64 pos1 = firstPackedOffset(firstOdd, p1, i + 1, cache);
        UInt64 pos2 = firstPackedOffset(firstOdd, p2, i + 2, cache);
        UInt64 pos3 = firstPackedOffset(firstOdd, p3, i + 3, cache);
        const Int8 l0 = primeLogWeight(static_cast<UInt32>(p0));
        const Int8 l1 = primeLogWeight(static_cast<UInt32>(p1));
        const Int8 l2 = primeLogWeight(static_cast<UInt32>(p2));
        const Int8 l3 = primeLogWeight(static_cast<UInt32>(p3));

        while (pos0 < packedCount && pos1 < packedCount
               && pos2 < packedCount && pos3 < packedCount) {
            mu[pos0] += l0;
            mu[pos1] += l1;
            mu[pos2] += l2;
            mu[pos3] += l3;
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
        finishStream(firstPackedOffset(firstOdd, p, i, cache), p,
                     primeLogWeight(static_cast<UInt32>(p)));
    }
}

// ============================================================================
// Packed square pass
// ============================================================================

void SegmentedOddMobiusSieveCore::sieveSquaresPass(
    Int8* __restrict mu, const UInt32* __restrict primes,
    UInt64 firstOdd, UInt64 packedCount, UInt32 fromIdx, UInt32 toIdx,
    const SieveQuotientCache* cache) {
    static constexpr UInt32 P_SPLIT = 1u << 16;
    static constexpr UInt64 SQ_CHUNK = UInt64(1) << 26;  // packed bytes

    UInt32 splitIdx = fromIdx;
    while (splitIdx < toIdx && primes[splitIdx] <= P_SPLIT)
        ++splitIdx;

    const UInt64 numChunks = (packedCount + SQ_CHUNK - 1) / SQ_CHUNK;
    #pragma omp parallel for schedule(dynamic, 1)
    for (UInt64 c = 0; c < numChunks; ++c) {
        const UInt64 cStart = c * SQ_CHUNK;
        const UInt64 cEnd = std::min(cStart + SQ_CHUNK, packedCount);
        const UInt64 cFirstOdd = firstOdd + 2 * cStart;

        for (UInt32 i = fromIdx; i < splitIdx; ++i) {
            const UInt64 p = primes[i];
            const UInt64 p2 = p * p;
            UInt64 pos = cStart + firstPackedSquareOffset(cFirstOdd, p, i, cache);
            for (; pos < cEnd; pos += p2)
                mu[pos] = 0;
        }
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (UInt32 i = splitIdx; i < toIdx; ++i) {
        const UInt64 p = primes[i];
        const UInt64 p2 = p * p;
        UInt64 pos = firstPackedSquareOffset(firstOdd, p, i, cache);
        for (; pos < packedCount; pos += p2)
            mu[pos] = 0;
    }
}

// ============================================================================
// Finalization: packed log bytes -> mu for the represented odd integers
// ============================================================================

#if SIEVE_SIMD_NEON || SIEVE_SIMD_SSE
void SegmentedOddMobiusSieveCore::finalizeMuVec(
    Int8* __restrict mu, int floorLog2, size_t count) {
    size_t i = 0;
    const Int8 fl = static_cast<Int8>(floorLog2);

#if SIEVE_SIMD_SVE2
    const svint8_t z = svdup_n_s8(0);
    const svint8_t one = svdup_n_s8(1);
    const svint8_t mask7 = svdup_n_s8(0x7F);
    const svint8_t vfl = svdup_n_s8(fl);
    for (; i < count; i += svcntb()) {
        const svbool_t pg = svwhilelt_b8(static_cast<uint64_t>(i),
                                         static_cast<uint64_t>(count));
        const svint8_t v = svld1_s8(pg, mu + i);
        const svbool_t sf = svcmplt_s8(pg, v, z);
        const svint8_t sum = svand_s8_x(pg, v, mask7);
        const svint8_t lsb = svand_s8_x(pg, v, one);
        const svint8_t base = svsub_s8_x(pg, svadd_s8_x(pg, lsb, lsb), one);
        const svint8_t result = svsel_s8(svcmpgt_s8(pg, sum, vfl),
                                         svneg_s8_x(pg, base), base);
        svst1_s8(pg, mu + i, svsel_s8(sf, result, z));
    }
#elif SIEVE_SIMD_NEON
    const int8x16_t z = vdupq_n_s8(0);
    const int8x16_t one = vdupq_n_s8(1);
    const int8x16_t mask7 = vdupq_n_s8(0x7F);
    const int8x16_t vfl = vdupq_n_s8(fl);
    for (; i + 16 <= count; i += 16) {
        const int8x16_t v = vld1q_s8(mu + i);
        const uint8x16_t sf = vcltq_s8(v, z);
        const int8x16_t sum = vandq_s8(v, mask7);
        const int8x16_t lsb = vandq_s8(v, one);
        const int8x16_t base = vsubq_s8(vshlq_n_s8(lsb, 1), one);
        const int8x16_t result = vbslq_s8(vcgtq_s8(sum, vfl), vnegq_s8(base), base);
        vst1q_s8(mu + i, vbslq_s8(sf, result, z));
    }
#elif SIEVE_SIMD_AVX512
    const __m512i z = _mm512_setzero_si512();
    const __m512i one = _mm512_set1_epi8(1);
    const __m512i mask7 = _mm512_set1_epi8(0x7F);
    const __m512i vfl = _mm512_set1_epi8(fl);
    for (; i + 64 <= count; i += 64) {
        const __m512i v = _mm512_loadu_si512((const __m512i*)(mu + i));
        const __mmask64 sf = _mm512_cmplt_epi8_mask(v, z);
        const __m512i sum = _mm512_and_si512(v, mask7);
        const __m512i lsb = _mm512_and_si512(v, one);
        const __m512i base = _mm512_sub_epi8(_mm512_add_epi8(lsb, lsb), one);
        const __mmask64 fullyFactored = _mm512_cmpgt_epi8_mask(sum, vfl);
        const __m512i result = _mm512_mask_blend_epi8(
            fullyFactored, base, _mm512_sub_epi8(z, base));
        _mm512_storeu_si512((__m512i*)(mu + i),
                            _mm512_maskz_mov_epi8(sf, result));
    }
#elif SIEVE_SIMD_AVX2
    const __m256i z = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i mask7 = _mm256_set1_epi8(0x7F);
    const __m256i vfl = _mm256_set1_epi8(fl);
    for (; i + 32 <= count; i += 32) {
        const __m256i v = _mm256_loadu_si256((const __m256i*)(mu + i));
        const __m256i sf = _mm256_cmpgt_epi8(z, v);
        const __m256i sum = _mm256_and_si256(v, mask7);
        const __m256i lsb = _mm256_and_si256(v, one);
        const __m256i base = _mm256_sub_epi8(_mm256_add_epi8(lsb, lsb), one);
        const __m256i fullyFactored = _mm256_cmpgt_epi8(sum, vfl);
        const __m256i result = _mm256_or_si256(
            _mm256_and_si256(fullyFactored, _mm256_sub_epi8(z, base)),
            _mm256_andnot_si256(fullyFactored, base));
        _mm256_storeu_si256((__m256i*)(mu + i), _mm256_and_si256(sf, result));
    }
#elif SIEVE_SIMD_SSE
    const __m128i z = _mm_setzero_si128();
    const __m128i one = _mm_set1_epi8(1);
    const __m128i mask7 = _mm_set1_epi8(0x7F);
    const __m128i vfl = _mm_set1_epi8(fl);
    for (; i + 16 <= count; i += 16) {
        const __m128i v = _mm_loadu_si128((const __m128i*)(mu + i));
        const __m128i sf = _mm_cmplt_epi8(v, z);
        const __m128i sum = _mm_and_si128(v, mask7);
        const __m128i lsb = _mm_and_si128(v, one);
        const __m128i base = _mm_sub_epi8(_mm_add_epi8(lsb, lsb), one);
        const __m128i fullyFactored = _mm_cmpgt_epi8(sum, vfl);
        const __m128i result = _mm_or_si128(
            _mm_and_si128(fullyFactored, _mm_sub_epi8(z, base)),
            _mm_andnot_si128(fullyFactored, base));
        _mm_storeu_si128((__m128i*)(mu + i), _mm_and_si128(sf, result));
    }
#endif

    for (; i < count; ++i) {
        const Int8 v = mu[i];
        if (v >= 0) {
            mu[i] = 0;
            continue;
        }
        const int sum = v & 0x7F;
        const int base = ((v & 1) << 1) - 1;
        mu[i] = static_cast<Int8>(sum > floorLog2 ? -base : base);
    }
}
#endif

void SegmentedOddMobiusSieveCore::finalizeMu(
    Int8* __restrict mu, UInt64 firstOdd, UInt64 packedCount) {
    UInt64 pos = 0;
    if (firstOdd == 1 && packedCount != 0) {
        mu[0] = 1;
        pos = 1;
    }

    while (pos < packedCount) {
        const UInt64 n0 = firstOdd + 2 * pos;
        const int floorLog2 = 63 - __builtin_clzll(n0);
        const UInt64 lastOdd = firstOdd + 2 * (packedCount - 1);
        const UInt64 powerRunHi = floorLog2 < 63
            ? (UInt64(1) << (floorLog2 + 1)) - 1
            : lastOdd;
        const UInt64 runHi = std::min(lastOdd, powerRunHi);
        const size_t runCount = static_cast<size_t>((runHi - n0) / 2 + 1);

#if SIEVE_SIMD_NEON || SIEVE_SIMD_SSE
        finalizeMuVec(mu + pos, floorLog2, runCount);
#else
        for (size_t j = 0; j < runCount; ++j) {
            const Int8 v = mu[pos + j];
            if (v >= 0) {
                mu[pos + j] = 0;
                continue;
            }
            const int sum = v & 0x7F;
            const int base = ((v & 1) << 1) - 1;
            mu[pos + j] = static_cast<Int8>(sum > floorLog2 ? -base : base);
        }
#endif
        pos += runCount;
    }
}

// ============================================================================
// Packed large-prime bucket scheduler
// ============================================================================

SegmentedOddMobiusSieveCore::LargePrimeHitScheduler::LargePrimeHitScheduler(
    UInt64 packedBase, UInt64 firstOdd, UInt64 packedCount,
    const std::vector<UInt32>& primes, UInt32 pInd0, UInt32 pInd1,
    std::vector<PVecT>& buckets, const SieveQuotientCache* cache)
    : mPackedBase(packedBase)
    , mFirstOdd(firstOdd)
    , mPackedCount(packedCount)
    , mFinalSubSegIndex((packedCount - 1) / M2)
    , mPrimes(primes.data())
    , mPInd0(pInd0)
    , mPInd1(pInd1)
    , mBuckets(buckets)
    , mCache(cache)
{
    if (mBuckets.size() < LP_NBUCKETS) mBuckets.resize(LP_NBUCKETS);
    for (auto& bucket : mBuckets) bucket.clear();

    for (UInt32 i = mPInd0; i < mPInd1; ++i) {
        const UInt64 p = mPrimes[i];
        const UInt64 off = SegmentedOddMobiusSieveCore::firstPackedOffset(
            mFirstOdd, p, i, mCache);
        if (off < mPackedCount)
            bucketPush(off / M2, packEntry(static_cast<UInt32>(p), off % M2));
    }
}

UInt64 SegmentedOddMobiusSieveCore::LargePrimeHitScheduler::subSegFirstOdd(
    UInt64 subSeg) const noexcept {
    return mFirstOdd + 2 * subSeg * M2;
}

UInt64 SegmentedOddMobiusSieveCore::LargePrimeHitScheduler::ringIndex(
    UInt64 subSeg, EntryT entry) noexcept {
#if ODD_SIEVE_SUBS_ACTIVE
    return (subSeg & (LP_SIZE - 1)) * LP_SUBS
         + ((entry & LP_OFF_MASK) >> LP_SUB_SHIFT);
#else
    (void)entry;
    return subSeg & (LP_SIZE - 1);
#endif
}

bool SegmentedOddMobiusSieveCore::LargePrimeHitScheduler::emptySubSegment(
    UInt64 subSeg) const noexcept {
#if ODD_SIEVE_SUBS_ACTIVE
    const UInt64 begin = (subSeg & (LP_SIZE - 1)) * LP_SUBS;
    for (UInt64 i = 0; i < LP_SUBS; ++i)
        if (!mBuckets[begin + i].empty()) return false;
    return true;
#else
    return mBuckets[subSeg & (LP_SIZE - 1)].empty();
#endif
}

void SegmentedOddMobiusSieveCore::LargePrimeHitScheduler::bucketPush(
    UInt64 subSeg, EntryT entry) noexcept {
    mBuckets[ringIndex(subSeg, entry)].push_back(entry);
}

SegmentedOddMobiusSieveCore::LargePrimeHitScheduler::EntryT
SegmentedOddMobiusSieveCore::LargePrimeHitScheduler::packEntry(
    UInt32 p, UInt64 off) noexcept {
#if SIEVE_NARROW_ENTRY
    (void)off;
    return p;
#else
    const UInt64 stride = p / M2;
    const UInt64 rem = p - stride * M2;
    return (EntryT(UInt8(SegmentedOddMobiusSieveCore::primeLogWeight(p)))
                << LP_LOG_SHIFT)
         | (stride << LP_S_SHIFT) | (rem << LP_R_SHIFT) | off;
#endif
}

void SegmentedOddMobiusSieveCore::LargePrimeHitScheduler::sieveSubSegment(
    Int8* __restrict muBase) noexcept {
    if (__builtin_expect(mCurrentSubSegIndex > mFinalSubSegIndex, false)) return;
    if (__builtin_expect(emptySubSegment(mCurrentSubSegIndex), false)) {
        ++mCurrentSubSegIndex;
        return;
    }

    const UInt64 base = mPackedBase + mCurrentSubSegIndex * M2;
#if !SIEVE_NARROW_ENTRY
    const UInt64 lastOff = (mPackedCount - 1) - mFinalSubSegIndex * M2;
#endif
    const UInt64 ring0 = (mCurrentSubSegIndex & (LP_SIZE - 1)) * LP_SUBS;

    for (UInt64 sb = 0; sb < LP_SUBS; ++sb) {
        PVecT& entries = mBuckets[ring0 + sb];
        const EntryT* entryData = entries.data();
        const size_t n = entries.size();

#if SIEVE_NARROW_ENTRY
        const UInt64 firstOdd = subSegFirstOdd(mCurrentSubSegIndex);
#if ODD_SIEVE_NARROW_SUBS_ACTIVE
        for (size_t i = 0; i < n; ++i) {
            const UInt64 p = entryData[i];
            const UInt64 off = SegmentedOddMobiusSieveCore::firstPackedOffset(firstOdd, p);
            const UInt32 hit = static_cast<UInt32>(off)
                | (static_cast<UInt32>(SegmentedOddMobiusSieveCore::primeLogWeight(
                       static_cast<UInt32>(p))) << LP_OFF_BITS);
            mTransientHits[off >> LP_TRANSIENT_SHIFT].push_back(hit);

            const UInt64 next = mCurrentSubSegIndex * M2 + off + p;
            if (next < mPackedCount)
                bucketPush(next / M2, static_cast<EntryT>(p));
        }

        for (TransientHitVecT& hits : mTransientHits) {
            for (const UInt32 hit : hits)
                muBase[base + (hit & LP_OFF_MASK)]
                    += static_cast<Int8>(hit >> LP_OFF_BITS);
            hits.clear();
        }
#else
        for (size_t i = 0; i < n; ++i) {
            const UInt64 p = entryData[i];
            const UInt64 off = SegmentedOddMobiusSieveCore::firstPackedOffset(firstOdd, p);
            muBase[base + off] += SegmentedOddMobiusSieveCore::primeLogWeight(
                static_cast<UInt32>(p));
            const UInt64 next = mCurrentSubSegIndex * M2 + off + p;
            if (next < mPackedCount)
                bucketPush(next / M2, static_cast<EntryT>(p));
        }
#endif
#else
        constexpr size_t PF = 16;
        for (size_t i = 0; i < n; ++i) {
            const size_t ipf = i + PF < n ? i + PF : n - 1;
            __builtin_prefetch(&muBase[base + (entryData[ipf] & LP_OFF_MASK)], 1, 3);

            const EntryT entry = entryData[i];
            const UInt64 off = entry & LP_OFF_MASK;
            muBase[base + off] += static_cast<Int8>(entry >> LP_LOG_SHIFT);

            const UInt64 rem = (entry >> LP_R_SHIFT) & LP_OFF_MASK;
            const UInt64 stride = (entry >> LP_S_SHIFT) & LP_S_MASK;
            UInt64 nextOff = off + rem;
            const UInt64 carry = nextOff >= M2;
            nextOff -= M2 & (UInt64(0) - carry);
            const UInt64 nextSegment = mCurrentSubSegIndex + stride + carry;
            if (nextSegment < mFinalSubSegIndex
                || (nextSegment == mFinalSubSegIndex && nextOff <= lastOff))
                bucketPush(nextSegment, (entry & ~LP_OFF_MASK) | nextOff);
        }
#endif
        entries.clear();
    }

    ++mCurrentSubSegIndex;
}
