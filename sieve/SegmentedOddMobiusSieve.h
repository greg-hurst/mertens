#pragma once

// ============================================================================
// SegmentedOddMobiusSieve.h — Native packed sieve of mu(n) for odd n.
//
// Public interval bounds remain ordinary integer coordinates.  For the most
// recent inclusive interval [lo, hi], the packed buffer contains
//
//   data[i] = mu(firstOdd() + 2*i),  0 <= i < oddCount().
//
// No even entries are allocated or visited.  initialize() takes an
// original-integer span, while capacity() explicitly reports packed odd slots.
//
// The native sieve mirrors the full sieve's three hot phases:
//   1. packed odd stencil plus small and M1 primes;
//   2. tiled direct sieving for medium primes;
//   3. a packed-coordinate bucket scheduler for large primes.
//
// Prime 2 and the full sieve's mod-4 forwarding skip disappear entirely.
// ============================================================================

#include "types.h"
#include "simd_defs.h"
#include "QuotientCache.h"

#include <array>
#include <cstdint>
#include <vector>

class SegmentedOddMobiusSieveCore {
    struct LargePrimeHitScheduler;

public:
    static constexpr UInt64 STENCIL_PERIOD = 3465;  // LCM(9, 5, 7, 11) odd slots
    static constexpr UInt32 MIN_PRIMES_BOUND = 360;

    SegmentedOddMobiusSieveCore();
    explicit SegmentedOddMobiusSieveCore(UInt64 segmentSize);

    // segmentSize is an original-integer span, not a packed slot count.
    void initialize(UInt64 segmentSize);

    // Sieve odd mu values in the inclusive original-coordinate interval
    // [lo, hi].  Finalize=false leaves the packed ceil-log bytes in data().
    template<bool Finalize = true>
    void sieve(UInt64 lo, UInt64 hi, const std::vector<UInt32>& primes);

    // Fill the first packedCount slots with the native odd stencil beginning
    // at n=1.  The packed unit is explicit because this is a low-level helper.
    void fillPackedFromStencil(UInt64 packedCount);

    Int8 operator[](UInt64 i) const { return mMu[i]; }
    const Int8* data() const { return mMu.data(); }
    Int8* data()             { return mMu.data(); }

    // Allocated packed odd slots (approximately half initialize()'s span).
    UInt64 capacity() const { return mMu.size(); }

    UInt64 lo() const       { return mLo; }
    UInt64 hi() const       { return mHi; }
    UInt64 firstOdd() const { return mFirstOdd; }
    UInt64 oddCount() const { return mOddCount; }

    UInt64 originalAt(UInt64 i) const { return mFirstOdd + 2 * i; }

    static constexpr UInt64 firstOddAtOrAbove(UInt64 lo) {
        return lo | UInt64(1);
    }

    static constexpr UInt64 countOdds(UInt64 lo, UInt64 hi) {
        if (hi < lo) return 0;
        const UInt64 first = firstOddAtOrAbove(lo);
        return first > hi ? 0 : (hi - first) / 2 + 1;
    }

    static constexpr UInt64 originalAt(UInt64 firstOdd, UInt64 i) {
        return firstOdd + 2 * i;
    }

    static std::vector<UInt32> primesUpTo(UInt32 n);

    static constexpr UInt64 schedulerReach() {
        return (LargePrimeHitScheduler::LP_SIZE - 1) * M2;
    }

private:
    static inline Int8 primeLogWeight(UInt32 p) noexcept {
        return static_cast<Int8>((32 - __builtin_clz(p)) | 1U);
    }

    static UInt64 integerSquareRoot(UInt64 n) noexcept;
    static UInt64 firstPackedOffset(UInt64 firstOdd, UInt64 p) noexcept;
    static UInt64 firstPackedOffset(UInt64 firstOdd, UInt64 p, UInt32 primeIndex,
                                    const SieveQuotientCache* cache) noexcept;
    static UInt64 firstPackedSquareOffset(UInt64 firstOdd, UInt64 p,
                                          UInt32 primeIndex,
                                          const SieveQuotientCache* cache) noexcept;

    void initSieveQuotientCache(const std::vector<UInt32>& primes);

    static void sieveM1PrimeRange(Int8* mu, const UInt32* primes,
                                  UInt64 firstOdd, UInt64 packedCount,
                                  UInt32 stoppos,
                                  const SieveQuotientCache* cache);
    static void sieveSquaresPass(Int8* mu, const UInt32* primes,
                                 UInt64 firstOdd, UInt64 packedCount,
                                 UInt32 fromIdx, UInt32 toIdx,
                                 const SieveQuotientCache* cache);
    static void sieveMediumSubSegment(Int8* mu, const UInt32* primes,
                                      UInt64 firstOdd, UInt64 packedCount,
                                      UInt32 fromIdx, UInt32 toIdx,
                                      const SieveQuotientCache* cache);
    static void finalizeMu(Int8* mu, UInt64 firstOdd, UInt64 packedCount);

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

        static constexpr int    LP_OFF_BITS  = 21;
        static constexpr UInt64 LP_OFF_MASK  = (UInt64(1) << LP_OFF_BITS) - 1;
        static constexpr int    LP_R_SHIFT   = 21;
        static constexpr int    LP_S_SHIFT   = 42;
        static constexpr UInt64 LP_S_MASK    = (UInt64(1) << 10) - 1;
        static constexpr int    LP_LOG_SHIFT = 52;

#ifndef SIEVE_LP_SIZE
#define SIEVE_LP_SIZE 512
#endif
        static constexpr UInt64 LP_SIZE = SIEVE_LP_SIZE;
        static_assert(LP_SIZE >= 2 && (LP_SIZE & (LP_SIZE - 1)) == 0,
                      "LP_SIZE must be a power of two of at least 2");
        static_assert(LP_SIZE <= LP_S_MASK + 1,
                      "stride field must hold p / M2 for every scheduled prime");

        // Set to 0 to retain the full LP_SIZE ring in the narrow backend.
#ifndef SIEVE_ODD_ACTIVE_RING
#define SIEVE_ODD_ACTIVE_RING 1
#endif

#if SIEVE_NARROW_ENTRY
        static constexpr UInt32 LP_PRIME_MASK = (UInt32(1) << 29) - 1;
        static constexpr UInt32 LP_PHASE_TICK = UInt32(1) << 29;
#endif

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
#define ODD_SIEVE_SUBS_ACTIVE 1
#else
        static constexpr UInt64 LP_SUBS = 1;
#define ODD_SIEVE_SUBS_ACTIVE 0
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
#define ODD_SIEVE_NARROW_SUBS_ACTIVE 1
#else
#define ODD_SIEVE_NARROW_SUBS_ACTIVE 0
#endif

        LargePrimeHitScheduler(UInt64 packedBase, UInt64 firstOdd,
                               UInt64 packedCount,
                               const std::vector<UInt32>& primes,
                               UInt32 pInd0, UInt32 pInd1,
                               std::vector<PVecT>& buckets,
                               const SieveQuotientCache* cache);

        void sieveSubSegment(Int8* muBase) noexcept;
#if SIEVE_NARROW_ENTRY
        template<bool Skip9>
        void sieveSubSegmentImpl(Int8* muBase) noexcept;
#endif
        UInt64 subSegFirstOdd(UInt64 subSeg) const noexcept;
        bool emptySubSegment(UInt64 subSeg) const noexcept;
        void bucketPush(UInt64 subSeg, EntryT entry) noexcept;
        UInt64 ringIndex(UInt64 subSeg, EntryT entry) const noexcept;
        static EntryT packEntry(UInt32 p, UInt64 off) noexcept;

        UInt64 mPackedBase;
        UInt64 mFirstOdd;
        UInt64 mPackedCount;
        UInt64 mFinalSubSegIndex;
        UInt64 mCurrentSubSegIndex = 0;
        const UInt32* mPrimes;
        UInt32 mPInd0;
        UInt32 mPInd1;
        std::vector<PVecT>& mBuckets;
#if SIEVE_NARROW_ENTRY && SIEVE_ODD_ACTIVE_RING
        UInt64 mActiveMask = LP_SIZE - 1;
#endif
#if SIEVE_NARROW_ENTRY
        bool mSkip9 = false;
#endif
#if ODD_SIEVE_NARROW_SUBS_ACTIVE
        std::array<TransientHitVecT, LP_TRANSIENT_SUBS> mTransientHits;
#endif
        const SieveQuotientCache* mCache;
    };

#ifndef ODD_SIEVE_M1_MULT
#define ODD_SIEVE_M1_MULT 32
#endif
#ifndef ODD_SIEVE_M2_MULT
#define ODD_SIEVE_M2_MULT 256
#endif
    // Keep the byte/cache geometry equal to the full sieve despite the shorter
    // packed stencil period: 110,880- and 887,040-byte work units.
    static constexpr UInt64 M1 = ODD_SIEVE_M1_MULT * STENCIL_PERIOD;
    static constexpr UInt64 M2 = ODD_SIEVE_M2_MULT * STENCIL_PERIOD;
    // A hit offset is in [0,M2), so equality is safe.  A scheduled prime is
    // additionally checked at runtime against (LP_SIZE-1)*M2; consequently
    // p/M2 <= LP_SIZE-1 <= 1023 and the wide entry's ten-bit stride is exact.
    static_assert(M2 <= (UInt64(1) << LargePrimeHitScheduler::LP_OFF_BITS),
                  "packed sub-segment offset must fit in LP_OFF_BITS");

#ifndef ODD_SIEVE_M1_PRIME_CAP
#ifdef SIEVE_M1_PRIME_CAP
#define ODD_SIEVE_M1_PRIME_CAP SIEVE_M1_PRIME_CAP
#else
#define ODD_SIEVE_M1_PRIME_CAP 32000
#endif
#endif
    static constexpr UInt32 M1_PRIME_CAP = ODD_SIEVE_M1_PRIME_CAP;

#ifndef SIEVE_M1_CONSTANT_PRIME_CAP
#define SIEVE_M1_CONSTANT_PRIME_CAP 1021
#endif
    static constexpr UInt32 M1_CONSTANT_PRIME_CAP = SIEVE_M1_CONSTANT_PRIME_CAP;
    static_assert(M1_PRIME_CAP >= M1_CONSTANT_PRIME_CAP,
                  "odd M1-prime cap must cover the constant-prime kernel");

    std::vector<Int8> mMu;
    std::vector<Int8> mPreMu;
    std::vector<Int8> mStencilData;
    std::vector<std::vector<LargePrimeHitScheduler::PVecT>> mBuckets;
    SieveQuotientCache mSieveQCache;

    UInt64 mLo = 1;
    UInt64 mHi = 0;
    UInt64 mFirstOdd = 1;
    UInt64 mOddCount = 0;
};
