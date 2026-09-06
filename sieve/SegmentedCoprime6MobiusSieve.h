#pragma once

// ============================================================================
// SegmentedCoprime6MobiusSieve.h — Native packed sieve of mu(n), (n, 6) = 1.
//
// Public interval bounds remain ordinary integer coordinates.  The packed
// coordinate is global:
//
//   originalAt(k) = 6*(k/2) + (k odd ? 5 : 1),
//
// so the stored sequence is 1, 5, 7, 11, 13, 17, ... .  For the most recent
// inclusive interval [lo, hi], data()[i] is mu(originalAt(firstPackedIndex()+i)).
// No multiples of 2 or 3 are allocated or visited.
//
// The native sieve has a packed stencil, two constant-stride direct lanes per
// prime, and one alternating-hit bucket stream per large prime.  It does not
// lift, filter, or copy either the full or odd-only sieve.
// ============================================================================

#include "types.h"
#include "simd_defs.h"
#include "QuotientCache.h"

#include <array>
#include <cstdint>
#include <vector>

class SegmentedCoprime6MobiusSieveCore {
    struct LargePrimeHitScheduler;

public:
    // Two residue lanes times lcm(5, 7, 11) in packed coordinates.
    static constexpr UInt64 STENCIL_PERIOD = 770;
    static constexpr UInt32 MIN_PRIMES_BOUND = 360;

    SegmentedCoprime6MobiusSieveCore();
    explicit SegmentedCoprime6MobiusSieveCore(UInt64 segmentSize);

    // segmentSize is an original-integer span, not a packed slot count.
    void initialize(UInt64 segmentSize);

    // Sieve the represented values in inclusive original coordinates [lo, hi].
    // Finalize=false leaves the packed ceil-log bytes in data().
    template<bool Finalize = true>
    void sieve(UInt64 lo, UInt64 hi, const std::vector<UInt32>& primes);

    // Fill the first packedCount slots with the native stencil beginning at 1.
    void fillPackedFromStencil(UInt64 packedCount);

    Int8 operator[](UInt64 i) const { return mMu[i]; }
    const Int8* data() const { return mMu.data(); }
    Int8* data()             { return mMu.data(); }

    UInt64 capacity() const         { return mMu.size(); }
    UInt64 lo() const               { return mLo; }
    UInt64 hi() const               { return mHi; }
    UInt64 firstPackedIndex() const { return mFirstPackedIndex; }
    UInt64 packedCount() const      { return mPackedCount; }
    UInt64 coprime6Count() const    { return mPackedCount; }
    UInt64 firstCoprime6() const {
        return mPackedCount == 0 ? 0 : originalAt(mFirstPackedIndex);
    }

    UInt64 originalAtLocal(UInt64 i) const {
        return originalAt(mFirstPackedIndex + i);
    }

    // Number of positive integers <= x that are coprime to 6.
    static constexpr UInt64 countThrough(UInt64 x) {
        const UInt64 q = x / 6;
        const UInt64 r = x - 6 * q;
        return 2 * q + (r >= 1) + (r >= 5);
    }

    static constexpr UInt64 countCoprime6Through(UInt64 x) {
        return countThrough(x);
    }

    static constexpr UInt64 countRange(UInt64 lo, UInt64 hi) {
        if (hi < lo) return 0;
        return countThrough(hi) - (lo <= 1 ? 0 : countThrough(lo - 1));
    }


    static constexpr UInt64 countCoprime6(UInt64 lo, UInt64 hi) {
        return countRange(lo, hi);
    }

    // Exact maximum represented-entry count over every original interval of
    // this length.  The quotient-first form cannot overflow UInt64.
    static constexpr UInt64 maxCountForSpan(UInt64 length) {
        const UInt64 q = length / 6;
        const UInt64 r = length - 6 * q;
        return 2 * q + (r != 0) + (r >= 3);
    }

    // Global packed index of the first represented value >= lo.
    static constexpr UInt64 firstPackedIndex(UInt64 lo) {
        return lo <= 1 ? 0 : countThrough(lo - 1);
    }

    static constexpr UInt64 originalAt(UInt64 packedIndex) {
        return 3 * packedIndex + 1 + (packedIndex & 1);
    }

    static std::vector<UInt32> primesUpTo(UInt32 n);

    static constexpr UInt64 schedulerReach() {
        // The larger of the alternating packed gaps is at most (4*p+1)/3.
        return (3 * ((LargePrimeHitScheduler::LP_SIZE - 1) * M2) - 1) / 4;
    }

private:
    struct PackedHit {
        UInt64 offset;
        bool quotientResidueOne;
    };

    static inline Int8 primeLogWeight(UInt32 p) noexcept {
        return static_cast<Int8>((32 - __builtin_clz(p)) | 1U);
    }

    static UInt64 integerSquareRoot(UInt64 n) noexcept;
    static PackedHit firstPackedHit(UInt64 firstPackedIndex, UInt64 p) noexcept;
    static PackedHit firstPackedHit(UInt64 firstPackedIndex, UInt64 p,
                                    UInt32 primeIndex,
                                    const SieveQuotientCache* cache) noexcept;
    static std::array<UInt64, 2> firstPackedOffsets(
        UInt64 firstPackedIndex, UInt64 p) noexcept;
    static std::array<UInt64, 2> firstPackedOffsets(
        UInt64 firstPackedIndex, UInt64 p, UInt32 primeIndex,
        const SieveQuotientCache* cache) noexcept;
    static std::array<UInt64, 2> firstPackedSquareOffsets(
        UInt64 firstPackedIndex, UInt64 p) noexcept;
    static std::array<UInt64, 2> firstPackedSquareOffsets(
        UInt64 firstPackedIndex, UInt64 p, UInt32 primeIndex,
        const SieveQuotientCache* cache) noexcept;
    static UInt64 nextPackedGap(UInt64 p, bool quotientResidueOne) noexcept;

    void initSieveQuotientCache(const std::vector<UInt32>& primes);

    static void sieveM1PrimeRange(Int8* mu, const UInt32* primes,
                                  UInt64 firstPackedIndex, UInt64 packedCount,
                                  UInt32 stoppos,
                                  const SieveQuotientCache* cache);
    static void sieveSquaresPass(Int8* mu, const UInt32* primes,
                                 UInt64 firstPackedIndex, UInt64 packedCount,
                                 UInt32 fromIdx, UInt32 toIdx,
                                 const SieveQuotientCache* cache);
    static void sieveMediumSubSegment(Int8* mu, const UInt32* primes,
                                      UInt64 firstPackedIndex,
                                      UInt64 packedCount,
                                      UInt32 fromIdx, UInt32 toIdx,
                                      const SieveQuotientCache* cache);
    static void finalizeMu(Int8* mu, UInt64 firstPackedIndex,
                           UInt64 packedCount);

#if SIEVE_SIMD_NEON || SIEVE_SIMD_SSE
    static void finalizeMuVec(Int8* mu, int floorLog2, size_t count);
#endif

    struct LargePrimeHitScheduler {
#ifndef SIEVE_NARROW_ENTRY
#define SIEVE_NARROW_ENTRY 1
#endif
#if SIEVE_NARROW_ENTRY
        using EntryT = UInt32;
#else
        using EntryT = UInt64;
#endif
        using PVecT = std::vector<EntryT>;

        static constexpr int    LP_OFF_BITS   = 21;
        static constexpr UInt64 LP_OFF_MASK   = (UInt64(1) << LP_OFF_BITS) - 1;
        static constexpr int    LP_P_SHIFT    = LP_OFF_BITS;
        static constexpr UInt64 LP_P_MASK     = (UInt64(1) << 32) - 1;
        static constexpr int    LP_PHASE_SHIFT = LP_P_SHIFT + 32;
        static constexpr UInt64 LP_PHASE_MASK = UInt64(1) << LP_PHASE_SHIFT;
        static constexpr int    LP_LOG_SHIFT  = LP_PHASE_SHIFT + 1;

#ifndef SIEVE_LP_SIZE
#define SIEVE_LP_SIZE 512
#endif
        static constexpr UInt64 LP_SIZE = SIEVE_LP_SIZE;
        static_assert(LP_SIZE >= 2 && (LP_SIZE & (LP_SIZE - 1)) == 0,
                      "LP_SIZE must be a power of two of at least 2");

#ifndef SIEVE_SUB_BUCKETS
#define SIEVE_SUB_BUCKETS 1
#endif
#if SIEVE_SUB_BUCKETS && !SIEVE_NARROW_ENTRY
#ifndef SIEVE_SUB_SHIFT
#define SIEVE_SUB_SHIFT 17
#endif
        static constexpr int LP_SUB_SHIFT = SIEVE_SUB_SHIFT;
        static_assert(LP_SUB_SHIFT > 0 && LP_SUB_SHIFT <= LP_OFF_BITS,
                      "wide sub-bucket shift must fit the packed offset field");
        static constexpr UInt64 LP_SUBS = UInt64(1) << (LP_OFF_BITS - LP_SUB_SHIFT);
#define COPRIME6_SIEVE_SUBS_ACTIVE 1
#else
        static constexpr UInt64 LP_SUBS = 1;
#define COPRIME6_SIEVE_SUBS_ACTIVE 0
#endif
        static constexpr UInt64 LP_NBUCKETS = LP_SIZE * LP_SUBS;

#if SIEVE_SUB_BUCKETS && SIEVE_NARROW_ENTRY
#ifndef SIEVE_NARROW_SUB_SHIFT
#define SIEVE_NARROW_SUB_SHIFT 16
#endif
        static_assert(SIEVE_NARROW_SUB_SHIFT > 0
                   && SIEVE_NARROW_SUB_SHIFT <= LP_OFF_BITS,
                      "narrow sub-bucket shift must fit the offset field");
        static constexpr int LP_TRANSIENT_SHIFT = SIEVE_NARROW_SUB_SHIFT;
        static constexpr UInt64 LP_TRANSIENT_SUBS =
            UInt64(1) << (LP_OFF_BITS - LP_TRANSIENT_SHIFT);
        using TransientHitVecT = std::vector<UInt32>;
#define COPRIME6_SIEVE_NARROW_SUBS_ACTIVE 1
#else
#define COPRIME6_SIEVE_NARROW_SUBS_ACTIVE 0
#endif

        LargePrimeHitScheduler(UInt64 packedBase, UInt64 firstPackedIndex,
                               UInt64 packedCount,
                               const std::vector<UInt32>& primes,
                               UInt32 pInd0, UInt32 pInd1,
                               std::vector<PVecT>& buckets,
                               const SieveQuotientCache* cache);

        void sieveSubSegment(Int8* muBase) noexcept;
        UInt64 subSegFirstPackedIndex(UInt64 subSeg) const noexcept;
        bool emptySubSegment(UInt64 subSeg) const noexcept;
        void bucketPush(UInt64 subSeg, EntryT entry) noexcept;
        static UInt64 ringIndex(UInt64 subSeg, EntryT entry) noexcept;
        static EntryT packEntry(UInt32 p, UInt64 off,
                                bool quotientResidueOne) noexcept;

        UInt64 mPackedBase;
        UInt64 mFirstPackedIndex;
        UInt64 mPackedCount;
        UInt64 mFinalSubSegIndex;
        UInt64 mCurrentSubSegIndex = 0;
        const UInt32* mPrimes;
        UInt32 mPInd0;
        UInt32 mPInd1;
        std::vector<PVecT>& mBuckets;
#if COPRIME6_SIEVE_NARROW_SUBS_ACTIVE
        std::array<TransientHitVecT, LP_TRANSIENT_SUBS> mTransientHits;
#endif
        const SieveQuotientCache* mCache;
    };

#ifndef COPRIME6_SIEVE_M1_MULT
#define COPRIME6_SIEVE_M1_MULT 144
#endif
#ifndef COPRIME6_SIEVE_M2_MULT
#define COPRIME6_SIEVE_M2_MULT 1152
#endif
    static constexpr UInt64 M1 = COPRIME6_SIEVE_M1_MULT * STENCIL_PERIOD;
    static constexpr UInt64 M2 = COPRIME6_SIEVE_M2_MULT * STENCIL_PERIOD;
    static_assert(M2 <= (UInt64(1) << LargePrimeHitScheduler::LP_OFF_BITS),
                  "packed sub-segment offset must fit in LP_OFF_BITS");

#ifndef COPRIME6_SIEVE_M1_PRIME_CAP
#ifdef SIEVE_M1_PRIME_CAP
#define COPRIME6_SIEVE_M1_PRIME_CAP SIEVE_M1_PRIME_CAP
#else
#define COPRIME6_SIEVE_M1_PRIME_CAP 32000
#endif
#endif
    static constexpr UInt32 M1_PRIME_CAP = COPRIME6_SIEVE_M1_PRIME_CAP;

#ifndef SIEVE_M1_CONSTANT_PRIME_CAP
#define SIEVE_M1_CONSTANT_PRIME_CAP 1021
#endif
    static constexpr UInt32 M1_CONSTANT_PRIME_CAP = SIEVE_M1_CONSTANT_PRIME_CAP;
    static_assert(M1_PRIME_CAP >= M1_CONSTANT_PRIME_CAP,
                  "coprime-6 M1-prime cap must cover the constant-prime kernel");

    std::vector<Int8> mMu;
    std::vector<Int8> mPreMu;
    std::vector<Int8> mStencilData;
    std::vector<std::vector<LargePrimeHitScheduler::PVecT>> mBuckets;
    SieveQuotientCache mSieveQCache;

    UInt64 mLo = 1;
    UInt64 mHi = 0;
    UInt64 mFirstPackedIndex = 0;
    UInt64 mPackedCount = 0;
};
