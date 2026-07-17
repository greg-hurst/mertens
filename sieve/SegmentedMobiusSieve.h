#pragma once

// ============================================================================
// SegmentedMobiusSieve.h — Segmented sieve of the Mobius function mu(n).
//
// Computes mu(k) for k in [lo, hi] using a three-phase approach:
//   Phase 1: Initialize M1 chunks with stencil, small primes, and M1 medium primes
//   Phase 2: Apply the remaining medium primes via tiled direct sieving
//   Phase 3: Apply large primes via a bucket scheduler
//
// Prime-range terminology:
//   stencil primes:  p <= 11
//   small primes:    13 <= p <= 353
//   M1 medium primes: 359 <= p <= M1_PRIME_CAP
//   M1-stage primes: collectively, p <= M1_PRIME_CAP
//   tiled medium primes: M1_PRIME_CAP < p <= the direct-sieve cutoff
//   large primes: p > the direct-sieve cutoff
//
// The sieve buffer stores packed Int8 values during sieving. A finalization
// step (finalizeMuVec when SIMD is available) converts these to actual mu ∈ {-1, 0, +1}.
//
// Arbitrary intervals [lo, hi] are supported — no alignment constraints.
//
// The prime list passed to sieve() must contain at least 71 entries (primes
// up to >= 353) due to the hardcoded sieveM1PrimeRange unrolling. Use
// primesUpTo(std::max(360u, sqrt_hi)) to guarantee this.
// ============================================================================

#include "types.h"
#include "simd_defs.h"
#include "QuotientCache.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

class SegmentedMobiusSieveCore {
public:
    // --- Constants ---
    static constexpr UInt64 STENCIL_PERIOD = 13860;  // LCM(4, 9, 5, 7, 11)
    static constexpr UInt32 MIN_PRIMES_BOUND = 360;  // primesUpTo must include at least this

    // --- Construction ---
    SegmentedMobiusSieveCore();
    explicit SegmentedMobiusSieveCore(UInt64 segmentSize);

    // --- Setup ---

    // Resize buffers and prepare the pre-sieved stencil for the given segment size.
    void initialize(UInt64 segmentSize);

    // --- Sieving ---

    // Sieve mu over [lo, hi] using the given primes list.
    // After this call, operator[](k - lo) == mu(k) for k in [lo, hi].
    // Requires: primes.size() >= 71 (primes up to at least 353).
    //
    // Finalize (compile-time, default true): when true the packed sieve bytes
    // are converted to mu in {-1,0,+1} before returning (the normal path). When
    // false, finalization is skipped and the buffer is left holding the raw
    // packed log-prime sums; the caller must finalize (e.g. the fused
    // finalize+prefix scan in SegmentedMertensSieve). Both instantiations are
    // explicitly emitted in the .cpp.
    template<bool Finalize = true>
    void sieve(UInt64 lo, UInt64 hi, const std::vector<UInt32>& primes);

    // Fill the buffer with the stencil pattern (for Loop 2's large segments).
    // Fills the first `size` bytes of the mu buffer with repeating stencil data.
    void fillFromStencil(UInt64 size);

    // --- Access ---

    // mu at relative index i (0-based from the last sieve's lo).
    Int8 operator[](UInt64 i) const { return mMu[i]; }

    // Raw pointer access to the sieve buffer.
    const Int8* data() const { return mMu.data(); }
    Int8* data()             { return mMu.data(); }

    UInt64 capacity() const  { return mMu.size(); }

    // --- Utility ---

    // Generate all primes up to n using a simple sieve of Eratosthenes.
    // For use with sieve(), pass at least primesUpTo(max(MIN_PRIMES_BOUND, sqrt(hi))).
    static std::vector<UInt32> primesUpTo(UInt32 n);

    // Largest prime the bucket scheduler can reach (LP_SIZE * M2). When the
    // bucket scheduler is enabled, the sieve endpoint must satisfy
    // sqrt(u) < schedulerReach() — see MertensHurst/INPUT_BOUNDS.md.
    static constexpr UInt64 schedulerReach() {
        return LargePrimeHitScheduler::LP_SIZE * M2;
    }

private:
    // Odd-valued ceil-log2 weight stored for each odd prime factor in the
    // packed sieve. Requires odd prime p. Keep synchronized with the stencil.
    static inline Int8 primeLogWeight(UInt32 p) noexcept {
        return static_cast<Int8>((32 - __builtin_clz(p)) | 1U);
    }

    // Precompute Granlund-Montgomery magic multipliers for the given prime list.
    // Auto-called by sieve() on first use when USE_DIVISION_FREE=1.
    void initSieveQuotientCache(const std::vector<UInt32>& primes);

    // --- Sieve phases ---

    // M1 stage: stencil + hardcoded small primes + M1 medium primes
    void sieveM1Primes(Int8* mu, const UInt32* primes, UInt64 lo, UInt64 hi);

    // Finalization: convert packed sieve values to mu ∈ {-1, 0, +1}
    void finalizeMu(Int8* mu, UInt64 lo, UInt64 hi);

    // Vectorized finalization over a constant-floor(log2 N) run, threshold fl.
#if SIEVE_SIMD_NEON || SIEVE_SIMD_SSE
    static void finalizeMuVec(Int8* mu, int fl, size_t n);
#endif

    // Apply hardcoded small primes 13..353, then M1 medium primes using
    // constant-stride and four-stream kernels.
    static void sieveM1PrimeRange(Int8* mu, const UInt32* primes, UInt64 lo, UInt64 hi,
                                  UInt32 stoppos, const SieveQuotientCache* cache);

    // Zero p^2 multiples for primes[fromIdx..toIdx) over the whole segment.
    // For p >= 359 (index 71+), p^2 exceeds every sub-segment length, so squares are
    // handled once per segment instead of per sub-segment.
    static void sieveSquaresPass(Int8* mu, const UInt32* primes, UInt64 lo, UInt64 hi,
                                 UInt32 fromIdx, UInt32 toIdx,
                                 const SieveQuotientCache* cache);

    // Apply medium-range primes to a sub-segment
    static void sieveMediumSubSegment(Int8* mu, const UInt32* primes, UInt64 lo, UInt64 hi,
                                      UInt64 fromIdx, UInt64 toIdx,
                                      const SieveQuotientCache* cache);

    // Apply medium primes while carrying each prime's next-hit offset across
    // a contiguous run of M2 sub-segments.
    static void sieveMediumSubSegmentRolling(Int8* mu, const UInt32* primes, UInt64 len,
                                             UInt64 fromIdx, UInt64 toIdx,
                                             std::vector<UInt32>& nextOff);

    // --- Bucket sieve for large primes ---
    struct LargePrimeHitScheduler {
        // Bucket entry layout (-DSIEVE_NARROW_ENTRY=0|1):
        //   narrow (default): the entry IS the prime (UInt32). The offset is
        //     recomputed per sub-segment by one divide, the log weight by CLZ.
        //     Half the entry-stream footprint; sub-buckets forced off. Wins
        //     in the bandwidth-bound many-core regime (~20% at 10^16).
        //   wide: self-contained 64-bit entry — bits 0..20 off, 21..41 p mod
        //     M2, 42..51 p / M2, 52..57 log weight. Forwarding is a branchless
        //     Bresenham step: cheapest per hit, but the 8-byte entry stream
        //     saturates memory bandwidth at scale. Machine-specific — on x86
        //     re-measure before trusting the default. See PERFORMANCE.md.
#ifndef SIEVE_NARROW_ENTRY
#define SIEVE_NARROW_ENTRY 1
#endif
#if SIEVE_NARROW_ENTRY
        using EntryT = UInt32;   // narrow: prime only, divide per hit (default)
#else
        using EntryT = UInt64;   // wide: packed, divide-free
#endif
        static constexpr int    LP_OFF_BITS  = 21;
        static constexpr UInt64 LP_OFF_MASK  = (UInt64(1) << LP_OFF_BITS) - 1;
        static constexpr int    LP_R_SHIFT   = 21;
        static constexpr int    LP_S_SHIFT   = 42;
        static constexpr UInt64 LP_S_MASK    = (UInt64(1) << 10) - 1;
        static constexpr int    LP_LOG_SHIFT = 52;

        using PVecT = std::vector<EntryT>;        // bucket storage
        using PrimesVecT = std::vector<UInt32>;   // the prime list

        // Full constructor: initializes bookkeeping AND scans all large primes
        // to populate buckets with first hits.
        LargePrimeHitScheduler(
            const UInt64& baseLo,
            const UInt64& lo,
            const UInt64& hi,
            const UInt64& subSegmentLen,
            const PrimesVecT& primes,
            const UInt32& pInd0,
            const UInt32& pInd1,
            std::vector<PVecT>& lpBucket,
            const SieveQuotientCache* cache
        );

        // Pre-populated constructor: initializes bookkeeping only.
        // Assumes buckets have already been populated by an external routing pass.
        struct PrePopulated {};
        LargePrimeHitScheduler(
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
        );

        UInt64 subSegLo(const UInt64& subSegIndex) const noexcept;
        UInt64 subSegIndexOf(const UInt64& x) const noexcept;
        bool emptySubSegment(const UInt64& subSeg) const noexcept;
        void bucketPush(const UInt64& subSeg, EntryT e) noexcept;
        static UInt64 skipStencilSquare4(UInt64 m, UInt64 p) noexcept;
        // Pack (p, first-hit offset) into the entry layout above. Seed-time
        // only — the per-hit path forwards entries without repacking from p.
        static EntryT packEntry(UInt32 p, UInt64 off) noexcept;
        void sieveSubSegment(Int8* __restrict muBase) noexcept;
        template<bool AllSkipsFit>
        void sieveSubSegmentImpl(Int8* __restrict muBase) noexcept;

        // Number of buckets in the circular buffer. Must be a power of 2
        // because bucket indexing uses (subSeg & (LP_SIZE - 1)) as a fast modulo.
        //
        // LP_SIZE bounds the largest schedulable prime: sqrt(u) < LP_SIZE * M2,
        // so 512 covers u up to ~2.06e17. The wide entry's 10-bit stride field
        // supports LP_SIZE up to exactly 1024, and doubling it is
        // performance-neutral. 512 is the record-run configuration; see
        // PERFORMANCE.md section 6 and Section 7 of the paper.
#ifndef SIEVE_LP_SIZE
#define SIEVE_LP_SIZE 512
#endif
        static constexpr UInt64 LP_SIZE = SIEVE_LP_SIZE;
        static_assert((LP_SIZE & (LP_SIZE - 1)) == 0,
                      "LP_SIZE must be a power of two (bucket index uses masking)");
        static_assert(LP_SIZE <= LP_S_MASK + 1,
                      "stride field must hold p / M2 for the largest schedulable prime");

        // Sub-buckets (default ON; disable with -DSIEVE_SUB_BUCKETS=0):
        // split each sub-segment's bucket into LP_SUBS sub-buckets banded by
        // offset high bits, so hits land in 2^LP_SUB_SHIFT-byte windows and
        // the mu RMW lines stay in L1 across a band (~5-10% end-to-end
        // in bucket-heavy regimes, bit-exact either way). 17 matches Apple's
        // 128 KB L1D; on smaller-L1 x86 cores try 14..16.
#ifndef SIEVE_SUB_BUCKETS
#define SIEVE_SUB_BUCKETS 1
#endif
        // Sub-buckets band by the hit offset, which the narrow entry does not
        // store — so they are available only in the wide build.
#if SIEVE_SUB_BUCKETS && !SIEVE_NARROW_ENTRY
#ifndef SIEVE_SUB_SHIFT
#define SIEVE_SUB_SHIFT 17
#endif
        static constexpr int    LP_SUB_SHIFT = SIEVE_SUB_SHIFT;  // off >> shift selects the sub
        // 2^(LP_OFF_BITS - shift) covers every possible offset for any M2.
        static constexpr UInt64 LP_SUBS = UInt64(1) << (LP_OFF_BITS - LP_SUB_SHIFT);
#define SIEVE_SUBS_ACTIVE 1
#else
        static constexpr UInt64 LP_SUBS = 1;
#define SIEVE_SUBS_ACTIVE 0
#endif
        static constexpr UInt64 LP_NBUCKETS = LP_SIZE * LP_SUBS;

        // Ring slot for an entry headed to `subSeg` (sub-bucket from its offset).
        static UInt64 ringIndex(const UInt64& subSeg, EntryT e) noexcept;

        const UInt64 mBaseLo, mLo, mHi, mSubSegmentLen;
        const UInt64 mFinalSubSegIndex;
        UInt64 mCurrentSubSegIndex;
        const UInt32* mP;
        const UInt32 mPInd0, mPInd1;
        const bool mAllSkipsFit;
        std::vector<PVecT>& mBuckets;
        const SieveQuotientCache* mCache;
    };

    // --- Internal constants for chunk sizes ---
    // Must be multiples of STENCIL_PERIOD for memcpy alignment.
    // M1 ~ L1 cache, M2 ~ L2 cache, M3 = bucket sieve cutoff.
    // M2/M3 multipliers are build-time tunable (keep M3 ~ 1.4x M2):
    //   -DSIEVE_M2_MULT=8 -DSIEVE_M3_MULT=12 gives an L1-sized sub-segment.
    // NOTE: shrinking M2 shrinks the scheduler capacity LP_SIZE * M2 and
    // grows the per-entry stride field s = p / M2 — check both limits
    // (LP_S_MASK and the LP_SIZE comment) before production use.
#ifndef SIEVE_M1_MULT
#define SIEVE_M1_MULT 8
#endif
    static constexpr UInt64 M1 = SIEVE_M1_MULT * STENCIL_PERIOD;   // ~111K — Phase 1 unit
#ifndef SIEVE_M2_MULT
#define SIEVE_M2_MULT 64
#endif
#ifndef SIEVE_M3_MULT
#define SIEVE_M3_MULT 90
#endif
    static constexpr UInt64 M2 = SIEVE_M2_MULT * STENCIL_PERIOD;   // ~887K default
    static_assert(M2 <= (UInt64(1) << LargePrimeHitScheduler::LP_OFF_BITS),
                  "sub-segment offset must fit in LP_OFF_BITS; raise LP_OFF_BITS if M2 grows");
    static constexpr UInt64 M3 = SIEVE_M3_MULT * STENCIL_PERIOD;   // optional block-walk direct cutoff
    static constexpr UInt64 BLOCK_SEGMENTS = 128;                  // contiguous M2 work unit
    static constexpr UInt64 BLOCK_WALK_MIN_SEGMENTS = 4096;        // ~3.6 billion values
    static_assert(M3 > M2, "large primes must exceed the sub-segment length (<= 1 hit per sub-segment)");
    // M1 medium primes through this boundary use the four-stream M1 walk;
    // larger primes use the wider medium tiles. This need not match the
    // stencil period and can be retuned independently for another machine.
#ifndef SIEVE_M1_PRIME_CAP
#define SIEVE_M1_PRIME_CAP 32000
#endif
    static constexpr UInt64 M1_PRIME_CAP = SIEVE_M1_PRIME_CAP;
    static_assert(M1_PRIME_CAP >= 353,
                  "M1-prime cap must include the fully unrolled small primes through 353");

    // M1 medium primes through this boundary use compile-time constant-stride
    // loops; the remaining M1 medium primes use the four-stream walk. The
    // default is tuned for Apple M3 Ultra. Values through 2,039 are available
    // for compile-time retuning on other machines.
#ifndef SIEVE_M1_CONSTANT_PRIME_CAP
#define SIEVE_M1_CONSTANT_PRIME_CAP 1021
#endif
    static constexpr UInt64 M1_CONSTANT_PRIME_CAP = SIEVE_M1_CONSTANT_PRIME_CAP;
    static_assert(M1_CONSTANT_PRIME_CAP >= 353 && M1_CONSTANT_PRIME_CAP <= 2039,
                  "constant-prime cap must be between 353 and 2,039");
    static_assert(M1_CONSTANT_PRIME_CAP <= M1_PRIME_CAP,
                  "constant-prime cap cannot exceed the M1-prime cap");

    // --- Data members ---
    std::vector<Int8> mMu;           // Main sieve buffer
    std::vector<Int8> mPreMu;        // Pre-sieved buffer (M1-sized, stencil copies)
    std::vector<Int8> mStencilData;  // The base stencil pattern (STENCIL_PERIOD entries)

    // Per-thread bucket storage for the large prime scheduler
    std::vector<std::vector<LargePrimeHitScheduler::PVecT>> mBuckets;

    // Index tracking for incremental prime range updates
    UInt32 mM1PrimeStopIdx;
    UInt32 mSqrtPrimeStopIdx;

    // Granlund-Montgomery quotient cache for prime offsets
    SieveQuotientCache mSieveQCache;

};

// ============================================================================
// SegmentedMobiusSieve — Iterator-style wrapper over SegmentedMobiusSieveCore.
//
// Construct with a segment size. Call next() to advance one segment at a time,
// starting from 1. Primes are grown automatically as needed.
//
// After each next() call:
//   getMobius(pos)         — look up mu(pos) within the current segment
//   getSegmentValues()     — all mu values in the current segment as Int8
//   getSegmentData()       — zero-copy pointer to the mu values
//   lo(), hi()             — current segment bounds
//   nextSegment()          — bounds of the upcoming segment
// ============================================================================

class SegmentedMobiusSieve {
public:
    explicit SegmentedMobiusSieve(UInt64 segmentSize)
        : mSegSize(segmentSize)
    {
        mCore.initialize(mSegSize);
    }

    // Advance to the next segment. Returns true if a segment was produced.
    bool next() {
        mLo = mNextLo;
        mHi = mLo + mSegSize - 1;

        growPrimes(mHi);
        mCore.sieve(mLo, mHi, mPrimes);
        mNextLo = mHi + 1;
        return true;
    }

    // Look up mu(pos) within the current segment. pos is absolute.
    Int8 getMobius(UInt64 pos) const {
        return mCore[pos - mLo];
    }

    // Fill vals with all mu values in the current segment.
    // Resizes vals if needed. Caller can reuse the same vector across segments.
    // The copy costs a full pass over the segment — in hot loops prefer
    // getSegmentData(), which is free.
    void getSegmentValues(std::vector<Int8>& vals) const {
        if ((UInt64)vals.size() < mSegSize) vals.resize(mSegSize);

        const Int8* src = mCore.data();
        #pragma omp parallel for schedule(static)
        for (UInt64 i = 0; i < mSegSize; ++i) {
            vals[i] = src[i];
        }
    }

    // Zero-copy access to the mu values in the current segment. Returns a
    // pointer to the core's existing buffer — nothing is copied. Valid until
    // the next next() call. getSegmentData()[i] = mu(lo() + i).
    const Int8* getSegmentData() const {
        return mCore.data();
    }

    UInt64 lo() const { return mLo; }
    UInt64 hi() const { return mHi; }

    // Inclusive bounds of the segment the next call to next() will sieve.
    std::pair<UInt64, UInt64> nextSegment() const {
        return {mNextLo, mNextLo + mSegSize - 1};
    }

    // Access the underlying core sieve.
    SegmentedMobiusSieveCore& core() { return mCore; }

private:
    void growPrimes(UInt64 hi) {
        UInt32 need = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                       (Int64)std::round(std::sqrt((double)hi)));
        if (need > mPrimeBound) {
            UInt32 bound = std::max(need, mPrimeBound * 2);
            mPrimes = SegmentedMobiusSieveCore::primesUpTo(bound);
            mPrimeBound = bound;
        }
    }

    SegmentedMobiusSieveCore mCore;
    std::vector<UInt32> mPrimes;
    UInt64 mSegSize;
    UInt64 mLo = 1;
    UInt64 mHi = 0;
    UInt64 mNextLo = 1;
    UInt32 mPrimeBound = 0;
};

// ============================================================================
// Free function — simplest API, hides all implementation details.
// ============================================================================

// Compute mu(1), mu(2), ..., mu(N) and return as a vector of Int8.
// Result has size N, with result[k-1] = mu(k).
static inline std::vector<Int8> MobiusSieveValues(UInt64 N, UInt64 segSize = 13860000ULL) {
    if (N < 1) return {};

    std::vector<Int8> result(N);

    SegmentedMobiusSieveCore sieve(segSize);

    const UInt32 sqrtN = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                          (Int64)std::round(std::sqrt((double)N)));
    auto primes = SegmentedMobiusSieveCore::primesUpTo(sqrtN);

    for (UInt64 slo = 1; slo <= N; slo += segSize) {
        UInt64 shi = std::min(slo + segSize - 1, N);
        sieve.sieve(slo, shi, primes);

        const Int8* mu = sieve.data();

        #pragma omp parallel for schedule(static)
        for (UInt64 k = slo; k <= shi; ++k) {
            result[k - 1] = mu[k - slo];
        }
    }

    return result;
}
