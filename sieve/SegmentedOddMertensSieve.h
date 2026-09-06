#pragma once

// ============================================================================
// SegmentedOddMertensSieve.h — Segmented sieve of the odd Mertens function.
//
//   M_2(x) = sum_{n <= x, n odd} mu(n)
//
// Wraps SegmentedOddMobiusSieveCore and prefixes its packed odd values without
// expanding the even positions. Public positions remain ordinary integer
// coordinates: M_2(2k) = M_2(2k - 1).
//
// Two storage modes:
//
//   Compressed:
//     M_2(firstOdd + 2*i) = coarse[i >> STRIDE_LOG] + residual[i]
//
//   Direct:
//     M_2(firstOdd + 2*i) = output[i]
//
// STRIDE is deliberately 256 packed odd values, matching the ordinary Mertens
// representation and its hot lookup shape. Each block chooses a shifted coarse
// base from its widened local minimum. A partial block has at most 255 slots;
// a full block contains at least floor(256 / 9) multiples of 9. Therefore its
// local prefix range is at most 255 and the shifted residual fits Int8 exactly.
// ============================================================================

#include "SegmentedOddMobiusSieve.h"
#include "simd_defs.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>
#include <omp.h>

#ifndef SIEVE_ODD_FUSED_FINALIZE
#define SIEVE_ODD_FUSED_FINALIZE 1
#endif

// Deliberately independent of MertensStorage so this header does not pull in
// the full-integer Mertens implementation merely to select a storage layout.
enum class OddMertensStorage { Compressed, Direct };

namespace OddMertensSieveDetail {
    static constexpr int STRIDE_LOG = 8;
    static constexpr UInt64 STRIDE = UInt64(1) << STRIDE_LOG;
    static_assert(STRIDE == 256, "odd Mertens hot path assumes stride 256");
}

template <OddMertensStorage Storage = OddMertensStorage::Compressed>
class SegmentedOddMertensSieveCoreT {
public:
    static constexpr int STRIDE_LOG = OddMertensSieveDetail::STRIDE_LOG;
    static constexpr UInt64 STRIDE = OddMertensSieveDetail::STRIDE;

    SegmentedOddMertensSieveCoreT() = default;
    explicit SegmentedOddMertensSieveCoreT(UInt64 originalSpanCapacity) {
        initialize(originalSpanCapacity);
    }

    // Capacity is expressed in original integers. The Mobius core allocates
    // only the corresponding packed odd slots.
    void initialize(UInt64 originalSpanCapacity) {
        mMobius.initialize(originalSpanCapacity);
    }

    // Sieve M_2 over the odd values in the inclusive original-coordinate
    // interval [lo, hi], retaining the finalized mu values. oddMertensPrev is
    // M_2(lo - 1). Only available in Compressed mode.
    template<typename MIntT>
    void sieve(UInt64 lo, UInt64 hi, MIntT oddMertensPrev,
               MIntT* __restrict M, Int8* __restrict R,
               const std::vector<UInt32>& primes) {
        static_assert(Storage == OddMertensStorage::Compressed,
                      "sieve() with separate R is only available in Compressed mode");

        mMobius.sieve(lo, hi, primes);
        setRange(lo, hi, oddMertensPrev);
        prefixSum(M, R, mMobius.data(), mOddCount, oddMertensPrev);
        mActiveR = R;
    }

    // Compressed mode prefixes in place: after this call the Mobius buffer
    // contains residuals rather than mu. Direct mode writes packed M_2 values
    // to M and preserves the finalized Mobius buffer.
    template<typename MIntT>
    void sieveInPlace(UInt64 lo, UInt64 hi, MIntT oddMertensPrev,
                      MIntT* __restrict M,
                      const std::vector<UInt32>& primes) {
        if constexpr (Storage == OddMertensStorage::Compressed) {
#if SIEVE_ODD_FUSED_FINALIZE
            mMobius.template sieve<false>(lo, hi, primes);
            setRange(lo, hi, oddMertensPrev);
            Int8* muData = mMobius.data();
            prefixSumFusedFinalize(M, muData, muData, mFirstOdd, mOddCount,
                                   oddMertensPrev);
#else
            mMobius.sieve(lo, hi, primes);
            setRange(lo, hi, oddMertensPrev);
            Int8* muData = mMobius.data();
            prefixSum(M, muData, muData, mOddCount, oddMertensPrev);
#endif
            mActiveR = mMobius.data();
        } else {
            mMobius.sieve(lo, hi, primes);
            setRange(lo, hi, oddMertensPrev);
            prefixSumDirect(M, mMobius.data(), mOddCount, oddMertensPrev);
        }
    }

    // Exact original-coordinate lookup after a sieve call. If the interval
    // begins at an even lo, querying lo returns the incoming carry because the
    // first represented odd value is lo + 1.
    template<typename MIntT>
    inline MIntT getOddMertens(const MIntT* M, UInt64 pos) const {
        if (pos < mFirstOdd || mOddCount == 0)
            return static_cast<MIntT>(mOddMertensPrev);

        const UInt64 off = (pos - mFirstOdd) >> 1;
        if constexpr (Storage == OddMertensStorage::Compressed)
            return M[off >> STRIDE_LOG] + mActiveR[off];
        else
            return M[off];
    }

    SegmentedOddMobiusSieveCore& mobiusSieve() { return mMobius; }
    const SegmentedOddMobiusSieveCore& mobiusSieve() const { return mMobius; }

    UInt64 lo() const { return mLo; }
    UInt64 hi() const { return mHi; }
    UInt64 firstOdd() const { return mFirstOdd; }
    UInt64 oddCount() const { return mOddCount; }
    Int64 oddMertensPrev() const { return mOddMertensPrev; }

private:
    SegmentedOddMobiusSieveCore mMobius;
    UInt64 mLo = 0;
    UInt64 mHi = 0;
    UInt64 mFirstOdd = 1;
    UInt64 mOddCount = 0;
    Int64 mOddMertensPrev = 0;
    const Int8* mActiveR = nullptr;

    template<typename MIntT>
    void setRange(UInt64 lo, UInt64 hi, MIntT oddMertensPrev) {
        mLo = lo;
        mHi = hi;
        mFirstOdd = mMobius.firstOdd();
        mOddCount = mMobius.oddCount();
        mOddMertensPrev = static_cast<Int64>(oddMertensPrev);
    }

#if SIEVE_SIMD_NEON
    static inline int16x8_t scan8_i16_inclusive(int16x8_t x) {
        const int16x8_t z = vdupq_n_s16(0);
        x = vaddq_s16(x, vextq_s16(z, x, 7));
        x = vaddq_s16(x, vextq_s16(z, x, 6));
        x = vaddq_s16(x, vextq_s16(z, x, 4));
        return x;
    }

    static inline void prefix16_i8_to_i16(
        int8x16_t mu, Int16* out, int16x8_t& carry,
        int16x8_t& localMin, int16x8_t& localMax) {
        const int16x8_t loScan = scan8_i16_inclusive(vmovl_s8(vget_low_s8(mu)));
        const int16x8_t lo = vaddq_s16(loScan, carry);
        const int16x8_t hiScan = scan8_i16_inclusive(vmovl_high_s8(mu));
        const int16x8_t hi = vaddq_s16(hiScan, vdupq_laneq_s16(lo, 7));
        carry = vdupq_laneq_s16(hi, 7);
        localMin = vminq_s16(localMin, vminq_s16(lo, hi));
        localMax = vmaxq_s16(localMax, vmaxq_s16(lo, hi));
        vst1q_s16(out, lo);
        vst1q_s16(out + 8, hi);
    }

    static inline void encode16_i16_to_i8(
        const Int16* prefix, Int8* residual, int shift) {
        const int16x8_t vShift = vdupq_n_s16((Int16)shift);
        const int16x8_t lo = vaddq_s16(vld1q_s16(prefix), vShift);
        const int16x8_t hi = vaddq_s16(vld1q_s16(prefix + 8), vShift);
        vst1q_s8(residual, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
    }

    static inline int8x16_t finalize16_i8(int8x16_t raw, int8x16_t vFl) {
        const int8x16_t vZero  = vdupq_n_s8(0);
        const int8x16_t vOne   = vdupq_n_s8(1);
        const int8x16_t vMask7 = vdupq_n_s8(0x7F);
        const uint8x16_t sf = vcltq_s8(raw, vZero);
        const int8x16_t S = vandq_s8(raw, vMask7);
        const int8x16_t lsb = vandq_s8(raw, vOne);
        const int8x16_t base = vsubq_s8(vshlq_n_s8(lsb, 1), vOne);
        const int8x16_t negBase = vnegq_s8(base);
        const uint8x16_t ff = vcgtq_s8(S, vFl);
        const int8x16_t r = vbslq_s8(ff, negBase, base);
        return vbslq_s8(sf, r, vZero);
    }
#elif SIEVE_SIMD_SSE
    static inline __m128i scan8_i16_inclusive_sse(__m128i x) {
        x = _mm_add_epi16(x, _mm_slli_si128(x, 2));
        x = _mm_add_epi16(x, _mm_slli_si128(x, 4));
        x = _mm_add_epi16(x, _mm_slli_si128(x, 8));
        return x;
    }

    static inline __m128i broadcastWord7(__m128i v) {
        v = _mm_shufflehi_epi16(v, _MM_SHUFFLE(3, 3, 3, 3));
        return _mm_unpackhi_epi64(v, v);
    }

    static inline void prefix16_i8_to_i16_sse(
        __m128i mu, Int16* out, __m128i& carry,
        __m128i& localMin, __m128i& localMax) {
        const __m128i zero = _mm_setzero_si128();
        const __m128i sign = _mm_cmpgt_epi8(zero, mu);
        const __m128i loScan = scan8_i16_inclusive_sse(
            _mm_unpacklo_epi8(mu, sign));
        const __m128i lo = _mm_add_epi16(loScan, carry);
        const __m128i hiScan = scan8_i16_inclusive_sse(
            _mm_unpackhi_epi8(mu, sign));
        const __m128i hi = _mm_add_epi16(hiScan, broadcastWord7(lo));
        carry = broadcastWord7(hi);
        localMin = _mm_min_epi16(localMin, _mm_min_epi16(lo, hi));
        localMax = _mm_max_epi16(localMax, _mm_max_epi16(lo, hi));
        _mm_storeu_si128((__m128i*)out, lo);
        _mm_storeu_si128((__m128i*)(out + 8), hi);
    }

    static inline void encode16_i16_to_i8_sse(
        const Int16* prefix, Int8* residual, int shift) {
        const __m128i vShift = _mm_set1_epi16((short)shift);
        const __m128i lo = _mm_add_epi16(
            _mm_loadu_si128((const __m128i*)prefix), vShift);
        const __m128i hi = _mm_add_epi16(
            _mm_loadu_si128((const __m128i*)(prefix + 8)), vShift);
        _mm_storeu_si128((__m128i*)residual, _mm_packs_epi16(lo, hi));
    }

    static inline __m128i finalize16_i8_sse(__m128i raw, __m128i vFl) {
        const __m128i vZero  = _mm_setzero_si128();
        const __m128i vOne   = _mm_set1_epi8(1);
        const __m128i vMask7 = _mm_set1_epi8(0x7F);
        const __m128i sf = _mm_cmplt_epi8(raw, vZero);
        const __m128i S = _mm_and_si128(raw, vMask7);
        const __m128i lsb = _mm_and_si128(raw, vOne);
        const __m128i base = _mm_sub_epi8(_mm_add_epi8(lsb, lsb), vOne);
        const __m128i negBase = _mm_sub_epi8(vZero, base);
        const __m128i ff = _mm_cmpgt_epi8(S, vFl);
        const __m128i r = _mm_or_si128(_mm_and_si128(ff, negBase),
                                      _mm_andnot_si128(ff, base));
        return _mm_and_si128(sf, r);
    }
#endif

    static inline Int8 finalizeOneFl(Int8 v, int fl) {
        if (v >= 0) return 0;
        const int S = v & 0x7F;
        const int base = ((v & 1) << 1) - 1;
        return static_cast<Int8>((S > fl) ? -base : base);
    }

    static inline Int8 finalizeOne(Int8 v, UInt64 n) {
        if (n <= 2) return (n == 1) ? Int8(1) : Int8(-1);
        return finalizeOneFl(v, 63 - (int)__builtin_clzll(n));
    }

    template<typename MIntT>
    static void prefixIntervalSums(MIntT* intervalSums, MIntT* threadSums,
                                   UInt64 numIntervals) {
        #pragma omp parallel
        {
            const UInt32 tid = omp_get_thread_num();
            const UInt32 nthr = omp_get_num_threads();
            MIntT local = 0;

            #pragma omp for schedule(static)
            for (UInt64 i = 0; i < numIntervals; ++i) {
                local += intervalSums[i];
                intervalSums[i] = local;
            }

            threadSums[tid] = local;

            #pragma omp barrier
            #pragma omp single
            {
                MIntT offset = 0;
                for (UInt32 t = 0; t < nthr; ++t) {
                    const MIntT sum = threadSums[t];
                    threadSums[t] = offset;
                    offset += sum;
                }
            }
            #pragma omp barrier

            const MIntT offset = threadSums[tid];
            #pragma omp for schedule(static)
            for (UInt64 i = 0; i < numIntervals; ++i)
                intervalSums[i] += offset;
        }
    }

    template<typename MIntT>
    static void prefixSum(MIntT* __restrict M, Int8* R, Int8* Mu,
                          UInt64 len, MIntT oddMertensPrev) {
        if (len == 0) return;

        const UInt64 numIntervals = (len + STRIDE - 1) >> STRIDE_LOG;
        static std::vector<MIntT> intervalSumsBuf;
        static std::vector<MIntT> threadSumsBuf;
        static std::vector<Int16> blockMinBuf;

        if (intervalSumsBuf.size() < numIntervals) intervalSumsBuf.resize(numIntervals);
        const UInt32 maxThreads = (UInt32)omp_get_max_threads();
        if (threadSumsBuf.size() < maxThreads) threadSumsBuf.resize(maxThreads);
        if (blockMinBuf.size() < numIntervals) blockMinBuf.resize(numIntervals);

        MIntT* intervalSums = intervalSumsBuf.data();
        MIntT* threadSums = threadSumsBuf.data();
        Int16* blockMin = blockMinBuf.data();

        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const UInt64 start = b << STRIDE_LOG;
            const UInt64 end = std::min(start + STRIDE, len);
            alignas(16) Int16 localPrefix[STRIDE];
            int local = 0;
            int localMin = 257;
            int localMax = -257;
            UInt64 k = start;

#if SIEVE_SIMD_NEON
            int16x8_t carry = vdupq_n_s16(0);
            int16x8_t vectorMin = vdupq_n_s16(32767);
            int16x8_t vectorMax = vdupq_n_s16(-32768);
            for (; k + 16 <= end; k += 16)
                prefix16_i8_to_i16(vld1q_s8(Mu + k), localPrefix + (k - start),
                                   carry, vectorMin, vectorMax);
            if (k != start) {
                local = vgetq_lane_s16(carry, 0);
                localMin = vminvq_s16(vectorMin);
                localMax = vmaxvq_s16(vectorMax);
            }
#elif SIEVE_SIMD_SSE
            __m128i carry = _mm_setzero_si128();
            __m128i vectorMin = _mm_set1_epi16(32767);
            __m128i vectorMax = _mm_set1_epi16(-32768);
            for (; k + 16 <= end; k += 16)
                prefix16_i8_to_i16_sse(_mm_loadu_si128((const __m128i*)(Mu + k)),
                                       localPrefix + (k - start),
                                       carry, vectorMin, vectorMax);
            if (k != start) {
                alignas(16) Int16 minLanes[8];
                alignas(16) Int16 maxLanes[8];
                local = (Int16)_mm_cvtsi128_si32(carry);
                _mm_store_si128((__m128i*)minLanes, vectorMin);
                _mm_store_si128((__m128i*)maxLanes, vectorMax);
                for (int lane = 0; lane < 8; ++lane) {
                    localMin = std::min(localMin, int(minLanes[lane]));
                    localMax = std::max(localMax, int(maxLanes[lane]));
                }
            }
#endif
            for (; k < end; ++k) {
                local += int(Mu[k]);
                localPrefix[k - start] = static_cast<Int16>(local);
                localMin = std::min(localMin, local);
                localMax = std::max(localMax, local);
            }

            // A partial block has at most 255 entries. A full block has at
            // least floor(256 / 9) squareful entries. Thus max-min <= 255.
            assert(localMax - localMin <= 255);
            const int shift = -localMin - 128;
            UInt64 write = start;
#if SIEVE_SIMD_NEON
            for (; write + 16 <= end; write += 16)
                encode16_i16_to_i8(localPrefix + (write - start), R + write, shift);
#elif SIEVE_SIMD_SSE
            for (; write + 16 <= end; write += 16)
                encode16_i16_to_i8_sse(localPrefix + (write - start), R + write, shift);
#endif
            for (; write < end; ++write) {
                const int residual = int(localPrefix[write - start]) + shift;
                assert(residual >= -128 && residual <= 127);
                R[write] = static_cast<Int8>(residual);
            }

            intervalSums[b] = MIntT(local);
            blockMin[b] = static_cast<Int16>(localMin);
        }

        prefixIntervalSums(intervalSums, threadSums, numIntervals);

        M[0] = oddMertensPrev + MIntT(blockMin[0]) + MIntT(128);
        #pragma omp parallel for schedule(static)
        for (Int64 b = 1; b < (Int64)numIntervals; ++b)
            M[b] = oddMertensPrev + intervalSums[b - 1]
                 + MIntT(blockMin[b]) + MIntT(128);
    }

    template<typename MIntT>
    static void prefixSumFusedFinalize(MIntT* __restrict M, Int8* R, Int8* Mu,
                                       UInt64 firstOdd, UInt64 len,
                                       MIntT oddMertensPrev) {
        if (len == 0) return;

        const UInt64 numIntervals = (len + STRIDE - 1) >> STRIDE_LOG;
        static std::vector<MIntT> intervalSumsBuf;
        static std::vector<MIntT> threadSumsBuf;
        static std::vector<Int16> blockMinBuf;

        if (intervalSumsBuf.size() < numIntervals) intervalSumsBuf.resize(numIntervals);
        const UInt32 maxThreads = (UInt32)omp_get_max_threads();
        if (threadSumsBuf.size() < maxThreads) threadSumsBuf.resize(maxThreads);
        if (blockMinBuf.size() < numIntervals) blockMinBuf.resize(numIntervals);

        MIntT* intervalSums = intervalSumsBuf.data();
        MIntT* threadSums = threadSumsBuf.data();
        Int16* blockMin = blockMinBuf.data();

        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const UInt64 start = b << STRIDE_LOG;
            const UInt64 end = std::min(start + STRIDE, len);
            const UInt64 nStart = firstOdd + 2 * start;
            const UInt64 nEnd = firstOdd + 2 * (end - 1);
            const int flStart = nStart >= 2 ? 63 - (int)__builtin_clzll(nStart) : 0;
            const int flEnd = nEnd >= 2 ? 63 - (int)__builtin_clzll(nEnd) : 0;
            const bool fastPath = nStart >= 3 && flStart == flEnd;

            alignas(16) Int16 localPrefix[STRIDE];
            int local = 0;
            int localMin = 257;
            int localMax = -257;
            UInt64 k = start;

            if (__builtin_expect(fastPath, 1)) {
#if SIEVE_SIMD_NEON
                const int8x16_t floorLog = vdupq_n_s8((Int8)flStart);
                int16x8_t carry = vdupq_n_s16(0);
                int16x8_t vectorMin = vdupq_n_s16(32767);
                int16x8_t vectorMax = vdupq_n_s16(-32768);
                for (; k + 16 <= end; k += 16) {
                    const int8x16_t mu = finalize16_i8(vld1q_s8(Mu + k), floorLog);
                    prefix16_i8_to_i16(mu, localPrefix + (k - start),
                                       carry, vectorMin, vectorMax);
                }
                if (k != start) {
                    local = vgetq_lane_s16(carry, 0);
                    localMin = vminvq_s16(vectorMin);
                    localMax = vmaxvq_s16(vectorMax);
                }
#elif SIEVE_SIMD_SSE
                const __m128i floorLog = _mm_set1_epi8((char)flStart);
                __m128i carry = _mm_setzero_si128();
                __m128i vectorMin = _mm_set1_epi16(32767);
                __m128i vectorMax = _mm_set1_epi16(-32768);
                for (; k + 16 <= end; k += 16) {
                    const __m128i mu = finalize16_i8_sse(
                        _mm_loadu_si128((const __m128i*)(Mu + k)), floorLog);
                    prefix16_i8_to_i16_sse(mu, localPrefix + (k - start),
                                           carry, vectorMin, vectorMax);
                }
                if (k != start) {
                    alignas(16) Int16 minLanes[8];
                    alignas(16) Int16 maxLanes[8];
                    local = (Int16)_mm_cvtsi128_si32(carry);
                    _mm_store_si128((__m128i*)minLanes, vectorMin);
                    _mm_store_si128((__m128i*)maxLanes, vectorMax);
                    for (int lane = 0; lane < 8; ++lane) {
                        localMin = std::min(localMin, int(minLanes[lane]));
                        localMax = std::max(localMax, int(maxLanes[lane]));
                    }
                }
#endif
            }

            for (; k < end; ++k) {
                const UInt64 n = firstOdd + 2 * k;
                const Int8 mu = __builtin_expect(fastPath, 1)
                              ? finalizeOneFl(Mu[k], flStart)
                              : finalizeOne(Mu[k], n);
                local += int(mu);
                localPrefix[k - start] = static_cast<Int16>(local);
                localMin = std::min(localMin, local);
                localMax = std::max(localMax, local);
            }

            assert(localMax - localMin <= 255);
            const int shift = -localMin - 128;
            UInt64 write = start;
#if SIEVE_SIMD_NEON
            for (; write + 16 <= end; write += 16)
                encode16_i16_to_i8(localPrefix + (write - start), R + write, shift);
#elif SIEVE_SIMD_SSE
            for (; write + 16 <= end; write += 16)
                encode16_i16_to_i8_sse(localPrefix + (write - start), R + write, shift);
#endif
            for (; write < end; ++write) {
                const int residual = int(localPrefix[write - start]) + shift;
                assert(residual >= -128 && residual <= 127);
                R[write] = static_cast<Int8>(residual);
            }

            intervalSums[b] = MIntT(local);
            blockMin[b] = static_cast<Int16>(localMin);
        }

        prefixIntervalSums(intervalSums, threadSums, numIntervals);

        M[0] = oddMertensPrev + MIntT(blockMin[0]) + MIntT(128);
        #pragma omp parallel for schedule(static)
        for (Int64 b = 1; b < (Int64)numIntervals; ++b)
            M[b] = oddMertensPrev + intervalSums[b - 1]
                 + MIntT(blockMin[b]) + MIntT(128);
    }

    template<typename MIntT>
    static void prefixSumDirect(MIntT* __restrict M, const Int8* Mu,
                                UInt64 len, MIntT oddMertensPrev) {
        if (len == 0) return;

        const UInt64 numIntervals = (len + STRIDE - 1) >> STRIDE_LOG;
        static std::vector<MIntT> intervalSumsBuf;
        static std::vector<MIntT> threadSumsBuf;
        if (intervalSumsBuf.size() < numIntervals) intervalSumsBuf.resize(numIntervals);
        const UInt32 maxThreads = (UInt32)omp_get_max_threads();
        if (threadSumsBuf.size() < maxThreads) threadSumsBuf.resize(maxThreads);

        MIntT* intervalSums = intervalSumsBuf.data();
        MIntT* threadSums = threadSumsBuf.data();

        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const UInt64 start = b << STRIDE_LOG;
            const UInt64 end = std::min(start + STRIDE, len);
            MIntT local = 0;
            for (UInt64 k = start; k < end; ++k) {
                local += MIntT(Mu[k]);
                M[k] = local;
            }
            intervalSums[b] = local;
        }

        prefixIntervalSums(intervalSums, threadSums, numIntervals);

        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const MIntT offset = oddMertensPrev + (b ? intervalSums[b - 1] : 0);
            const UInt64 start = b << STRIDE_LOG;
            const UInt64 end = std::min(start + STRIDE, len);
            for (UInt64 k = start; k < end; ++k)
                M[k] += offset;
        }
    }
};

using SegmentedOddMertensSieveCore =
    SegmentedOddMertensSieveCoreT<OddMertensStorage::Compressed>;

// Exact hot lookup in original integer coordinates. oddMertensPrev is
// M_2(lo - 1), and firstOdd is the first packed odd value in the segment.
#define GET_ODD_MERTENS(M, R, FIRST_ODD, ODD_PREV, pos) __extension__({ \
    auto _odd_pos = (pos); \
    auto _odd_first = (FIRST_ODD); \
    auto _odd_prev = (ODD_PREV); \
    decltype(_odd_prev) _odd_result; \
    if (__builtin_expect(_odd_pos < _odd_first, 0)) { \
        _odd_result = _odd_prev; \
    } else { \
        auto _odd_off = (_odd_pos - _odd_first) >> 1; \
        _odd_result = (M)[_odd_off >> OddMertensSieveDetail::STRIDE_LOG] + (R)[_odd_off]; \
    } \
    _odd_result; \
})

// Faster hot lookup when the caller guarantees pos >= FIRST_ODD.
#define GET_ODD_MERTENS_IN_RANGE(M, R, FIRST_ODD, pos) __extension__({ \
    auto _odd_off = ((pos) - (FIRST_ODD)) >> 1; \
    (M)[_odd_off >> OddMertensSieveDetail::STRIDE_LOG] + (R)[_odd_off]; \
})

template <OddMertensStorage Storage = OddMertensStorage::Compressed>
class SegmentedOddMertensSieveT {
public:
    explicit SegmentedOddMertensSieveT(UInt64 originalSegmentSpan)
        : mSegmentSpan(originalSegmentSpan) {
        mCore.initialize(mSegmentSpan);
        const UInt64 packedCapacity = mSegmentSpan / 2 + (mSegmentSpan & 1);
        if constexpr (Storage == OddMertensStorage::Compressed)
            mM.resize((packedCapacity + OddMertensSieveDetail::STRIDE - 1)
                      >> OddMertensSieveDetail::STRIDE_LOG);
        else
            mM.resize(packedCapacity);
    }

    bool next() {
        mLo = mNextLo;
        mHi = mLo + mSegmentSpan - 1;
        growPrimes(mHi);
        mCore.sieveInPlace(mLo, mHi, mOddMertensPrev, mM.data(), mPrimes);
        mOddMertensPrev = mCore.getOddMertens(mM.data(), mHi);
        mNextLo = mHi + 1;
        return true;
    }

    Int32 getOddMertens(UInt64 pos) const {
        return mCore.getOddMertens(mM.data(), pos);
    }

    // Packed values only: vals[i] = M_2(firstOdd() + 2*i).
    void getSegmentValues(std::vector<Int32>& vals) const {
        if (vals.size() < mCore.oddCount()) vals.resize(mCore.oddCount());

        if constexpr (Storage == OddMertensStorage::Compressed) {
            const Int8* R = mCore.mobiusSieve().data();
            const UInt64 numIntervals =
                (mCore.oddCount() + OddMertensSieveDetail::STRIDE - 1)
                >> OddMertensSieveDetail::STRIDE_LOG;
            #pragma omp parallel for schedule(static)
            for (UInt64 b = 0; b < numIntervals; ++b) {
                const Int32 coarse = mM[b];
                const UInt64 start = b << OddMertensSieveDetail::STRIDE_LOG;
                const UInt64 end = std::min(start + OddMertensSieveDetail::STRIDE,
                                            mCore.oddCount());
                for (UInt64 k = start; k < end; ++k)
                    vals[k] = coarse + R[k];
            }
        } else {
            std::memcpy(vals.data(), mM.data(), mCore.oddCount() * sizeof(Int32));
        }
        vals.resize(mCore.oddCount());
    }

    const Int32* getCoarseData() const {
        static_assert(Storage == OddMertensStorage::Compressed,
                      "getCoarseData() is only available in Compressed mode");
        return mM.data();
    }

    const Int8* getResidualData() const {
        static_assert(Storage == OddMertensStorage::Compressed,
                      "getResidualData() is only available in Compressed mode");
        return mCore.mobiusSieve().data();
    }

    const Int32* getSegmentData() const {
        static_assert(Storage == OddMertensStorage::Direct,
                      "getSegmentData() is only available in Direct mode");
        return mM.data();
    }

    UInt64 lo() const { return mLo; }
    UInt64 hi() const { return mHi; }
    UInt64 firstOdd() const { return mCore.firstOdd(); }
    UInt64 oddCount() const { return mCore.oddCount(); }
    std::pair<UInt64, UInt64> nextSegment() const {
        return {mNextLo, mNextLo + mSegmentSpan - 1};
    }

    SegmentedOddMertensSieveCoreT<Storage>& core() { return mCore; }
    const SegmentedOddMertensSieveCoreT<Storage>& core() const { return mCore; }

private:
    void growPrimes(UInt64 hi) {
        const UInt32 need = (UInt32)std::max(
            (Int64)SegmentedOddMobiusSieveCore::MIN_PRIMES_BOUND,
            (Int64)std::round(std::sqrt((double)hi)));
        if (need > mPrimeBound) {
            const UInt32 bound = std::max(need, mPrimeBound * 2);
            mPrimes = SegmentedOddMobiusSieveCore::primesUpTo(bound);
            mPrimeBound = bound;
        }
    }

    SegmentedOddMertensSieveCoreT<Storage> mCore;
    std::vector<Int32> mM;
    std::vector<UInt32> mPrimes;
    UInt64 mSegmentSpan;
    UInt64 mLo = 1;
    UInt64 mHi = 0;
    UInt64 mNextLo = 1;
    Int32 mOddMertensPrev = 0;
    UInt32 mPrimeBound = 0;
};

using SegmentedOddMertensSieve =
    SegmentedOddMertensSieveT<OddMertensStorage::Compressed>;

// Compute M_2(N). Segment sizes are measured in original integers.
static inline Int32 OddMertensSieve(UInt64 N, UInt64 originalSegmentSpan = 0) {
    if (N < 1) return 0;
    if (originalSegmentSpan == 0) originalSegmentSpan = N;

    SegmentedOddMobiusSieveCore sieve(originalSegmentSpan);
    const UInt32 sqrtN = (UInt32)std::max(
        (Int64)SegmentedOddMobiusSieveCore::MIN_PRIMES_BOUND,
        (Int64)std::round(std::sqrt((double)N)));
    const auto primes = SegmentedOddMobiusSieveCore::primesUpTo(sqrtN);

    Int64 sum = 0;
    for (UInt64 lo = 1; lo <= N; lo += originalSegmentSpan) {
        const UInt64 hi = std::min(lo + originalSegmentSpan - 1, N);
        sieve.sieve(lo, hi, primes);
        const Int8* mu = sieve.data();
        const UInt64 count = sieve.oddCount();
        Int64 segmentSum = 0;
        #pragma omp parallel for reduction(+:segmentSum) schedule(static)
        for (UInt64 i = 0; i < count; ++i)
            segmentSum += mu[i];
        sum += segmentSum;
    }
    return (Int32)sum;
}

// Return the packed odd-prefix values through N:
// result[i] = M_2(2*i + 1), 0 <= i < ceil(N/2).
static inline std::vector<Int32> OddMertensSieveValues(
    UInt64 N, UInt64 originalSegmentSpan = 0) {
    if (N < 1) return {};
    if (originalSegmentSpan == 0) originalSegmentSpan = N;

    const UInt64 totalOdd = N / 2 + (N & 1);
    std::vector<Int32> result(totalOdd);
    const UInt64 packedCapacity = originalSegmentSpan / 2 + (originalSegmentSpan & 1);
    std::vector<Int32> M(packedCapacity);
    SegmentedOddMertensSieveCoreT<OddMertensStorage::Direct> sieve(originalSegmentSpan);

    const UInt32 sqrtN = (UInt32)std::max(
        (Int64)SegmentedOddMobiusSieveCore::MIN_PRIMES_BOUND,
        (Int64)std::round(std::sqrt((double)N)));
    const auto primes = SegmentedOddMobiusSieveCore::primesUpTo(sqrtN);

    Int32 oddMertensPrev = 0;
    UInt64 output = 0;
    for (UInt64 lo = 1; lo <= N; lo += originalSegmentSpan) {
        const UInt64 hi = std::min(lo + originalSegmentSpan - 1, N);
        sieve.sieveInPlace(lo, hi, oddMertensPrev, M.data(), primes);
        const UInt64 count = sieve.oddCount();
        std::memcpy(result.data() + output, M.data(), count * sizeof(Int32));
        output += count;
        oddMertensPrev = sieve.getOddMertens(M.data(), hi);
    }
    return result;
}
