#pragma once

// ============================================================================
// SegmentedCoprime6MertensSieve.h — Segmented sieve of the 6-coprime
// Mertens function.
//
//   M_6(x) = sum_{n <= x, (n, 6) = 1} mu(n)
//
// Wraps SegmentedCoprime6MobiusSieveCore and prefixes its packed values
// without expanding the excluded positions. Public positions remain ordinary
// integer coordinates, and M_6 is constant between represented values.
//
// Two storage modes:
//
//   Compressed:
//     M_6(originalAt(firstPackedIndex + i)) =
//         coarse[i >> STRIDE_LOG] + residual[i]
//
//   Direct:
//     M_6(originalAt(firstPackedIndex + i)) = output[i]
//
// Each compressed block shifts its coarse base by the widened local minimum.
// A block has at most 256 values and each mu increment has magnitude at most
// one, so its local prefix range is at most 255 and every residual fits Int8.
// ============================================================================

#include "SegmentedCoprime6MobiusSieve.h"
#include "simd_defs.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <omp.h>

#ifndef SIEVE_COPRIME6_FUSED_FINALIZE
#define SIEVE_COPRIME6_FUSED_FINALIZE 1
#endif

enum class Coprime6MertensStorage { Compressed, Direct };

namespace Coprime6MertensSieveDetail {
    static constexpr int STRIDE_LOG = 8;
    static constexpr UInt64 STRIDE = UInt64(1) << STRIDE_LOG;
    static_assert(STRIDE == 256,
                  "coprime-6 Mertens hot path assumes stride 256");

    static constexpr UInt64 coarseCount(UInt64 packedCount) {
        return (packedCount >> STRIDE_LOG)
             + ((packedCount & (STRIDE - 1)) != 0);
    }

    // Maximum represented values in any original-coordinate interval of the
    // given length. The two wheel residues can both occur in a three-value tail.
    static constexpr UInt64 packedCapacity(UInt64 originalSpan) {
        return SegmentedCoprime6MobiusSieveCore::maxCountForSpan(
            originalSpan
        );
    }

    template<typename MIntT, bool Fused>
    struct PrefixWorkspace {
        std::vector<MIntT> intervalSums;
        std::vector<MIntT> threadSums;
        std::vector<Int16> blockMin;

        void clear() {
            std::vector<MIntT>().swap(intervalSums);
            std::vector<MIntT>().swap(threadSums);
            std::vector<Int16>().swap(blockMin);
        }
    };

    template<typename MIntT, bool Fused>
    static inline PrefixWorkspace<MIntT, Fused>& prefixWorkspace() {
        static PrefixWorkspace<MIntT, Fused> workspace;
        return workspace;
    }
}

template <Coprime6MertensStorage Storage =
          Coprime6MertensStorage::Compressed>
class SegmentedCoprime6MertensSieveCoreT {
public:
    static constexpr int STRIDE_LOG =
        Coprime6MertensSieveDetail::STRIDE_LOG;
    static constexpr UInt64 STRIDE =
        Coprime6MertensSieveDetail::STRIDE;

    SegmentedCoprime6MertensSieveCoreT() = default;
    explicit SegmentedCoprime6MertensSieveCoreT(
        UInt64 originalSpanCapacity) {
        initialize(originalSpanCapacity);
    }

    // Capacity is expressed in original integers. The Mobius core allocates
    // only the corresponding packed coprime-to-6 slots.
    void initialize(UInt64 originalSpanCapacity) {
        mMobius.initialize(originalSpanCapacity);
    }

    // Sieve M_6 over represented values in inclusive original coordinates
    // [lo, hi], retaining the finalized mu values. mertens6Prev is M_6(lo-1).
    // Only available in Compressed mode.
    template<typename MIntT>
    void sieve(UInt64 lo, UInt64 hi, MIntT mertens6Prev,
               MIntT* __restrict M, Int8* __restrict R,
               const std::vector<UInt32>& primes) {
        static_assert(
            Storage == Coprime6MertensStorage::Compressed,
            "sieve() with separate R is only available in Compressed mode");

        mMobius.sieve(lo, hi, primes);
        setRange(lo, hi, mertens6Prev);
        prefixSum(M, R, mMobius.data(), mPackedCount, mertens6Prev);
        mActiveR = R;
    }

    // Compressed mode prefixes in place: afterward the Mobius buffer contains
    // residuals rather than mu. Direct mode writes packed M_6 values to M and
    // preserves the finalized Mobius buffer.
    template<typename MIntT>
    void sieveInPlace(UInt64 lo, UInt64 hi, MIntT mertens6Prev,
                      MIntT* __restrict M,
                      const std::vector<UInt32>& primes) {
        if constexpr (Storage == Coprime6MertensStorage::Compressed) {
#if SIEVE_COPRIME6_FUSED_FINALIZE
            mMobius.template sieve<false>(lo, hi, primes);
            setRange(lo, hi, mertens6Prev);
            Int8* muData = mMobius.data();
            prefixSumFusedFinalize(
                M, muData, muData, mFirstPackedIndex, mPackedCount,
                mertens6Prev
            );
#else
            mMobius.sieve(lo, hi, primes);
            setRange(lo, hi, mertens6Prev);
            Int8* muData = mMobius.data();
            prefixSum(M, muData, muData, mPackedCount, mertens6Prev);
#endif
            mActiveR = mMobius.data();
        } else {
            mMobius.sieve(lo, hi, primes);
            setRange(lo, hi, mertens6Prev);
            prefixSumDirect(M, mMobius.data(), mPackedCount, mertens6Prev);
        }
    }

    // Exact original-coordinate lookup after a sieve call. Positions before
    // this segment's first represented value return the incoming carry.
    template<typename MIntT>
    inline MIntT getCoprime6Mertens(const MIntT* M, UInt64 pos) const {
        const UInt64 through =
            SegmentedCoprime6MobiusSieveCore::countThrough(pos);
        if (mPackedCount == 0 || through <= mFirstPackedIndex)
            return static_cast<MIntT>(mMertens6Prev);

        const UInt64 off = through - mFirstPackedIndex - 1;
        if constexpr (Storage == Coprime6MertensStorage::Compressed)
            return M[off >> STRIDE_LOG] + mActiveR[off];
        else
            return M[off];
    }

    SegmentedCoprime6MobiusSieveCore& mobiusSieve() { return mMobius; }
    const SegmentedCoprime6MobiusSieveCore& mobiusSieve() const {
        return mMobius;
    }

    UInt64 lo() const { return mLo; }
    UInt64 hi() const { return mHi; }
    UInt64 firstPackedIndex() const { return mFirstPackedIndex; }
    UInt64 firstCoprime6() const { return mMobius.firstCoprime6(); }
    UInt64 packedCount() const { return mPackedCount; }
    UInt64 coprime6Count() const { return mPackedCount; }
    Int64 mertens6Prev() const { return mMertens6Prev; }

    template<typename MIntT>
    static void releasePrefixWorkspace() {
        Coprime6MertensSieveDetail::prefixWorkspace<MIntT, false>().clear();
        Coprime6MertensSieveDetail::prefixWorkspace<MIntT, true>().clear();
    }

private:
    SegmentedCoprime6MobiusSieveCore mMobius;
    UInt64 mLo = 0;
    UInt64 mHi = 0;
    UInt64 mFirstPackedIndex = 0;
    UInt64 mPackedCount = 0;
    Int64 mMertens6Prev = 0;
    const Int8* mActiveR = nullptr;

    template<typename MIntT>
    void setRange(UInt64 lo, UInt64 hi, MIntT mertens6Prev) {
        mLo = lo;
        mHi = hi;
        mFirstPackedIndex = mMobius.firstPackedIndex();
        mPackedCount = mMobius.packedCount();
        mMertens6Prev = static_cast<Int64>(mertens6Prev);
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
        const int16x8_t loScan =
            scan8_i16_inclusive(vmovl_s8(vget_low_s8(mu)));
        const int16x8_t lo = vaddq_s16(loScan, carry);
        const int16x8_t hiScan =
            scan8_i16_inclusive(vmovl_high_s8(mu));
        const int16x8_t hi =
            vaddq_s16(hiScan, vdupq_laneq_s16(lo, 7));
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
        const int8x16_t vZero = vdupq_n_s8(0);
        const int8x16_t vOne = vdupq_n_s8(1);
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
            _mm_unpacklo_epi8(mu, sign)
        );
        const __m128i lo = _mm_add_epi16(loScan, carry);
        const __m128i hiScan = scan8_i16_inclusive_sse(
            _mm_unpackhi_epi8(mu, sign)
        );
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
            _mm_loadu_si128((const __m128i*)prefix), vShift
        );
        const __m128i hi = _mm_add_epi16(
            _mm_loadu_si128((const __m128i*)(prefix + 8)), vShift
        );
        _mm_storeu_si128((__m128i*)residual, _mm_packs_epi16(lo, hi));
    }

    static inline __m128i finalize16_i8_sse(__m128i raw, __m128i vFl) {
        const __m128i vZero = _mm_setzero_si128();
        const __m128i vOne = _mm_set1_epi8(1);
        const __m128i vMask7 = _mm_set1_epi8(0x7F);
        const __m128i sf = _mm_cmplt_epi8(raw, vZero);
        const __m128i S = _mm_and_si128(raw, vMask7);
        const __m128i lsb = _mm_and_si128(raw, vOne);
        const __m128i base =
            _mm_sub_epi8(_mm_add_epi8(lsb, lsb), vOne);
        const __m128i negBase = _mm_sub_epi8(vZero, base);
        const __m128i ff = _mm_cmpgt_epi8(S, vFl);
        const __m128i r = _mm_or_si128(
            _mm_and_si128(ff, negBase), _mm_andnot_si128(ff, base)
        );
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
                          UInt64 len, MIntT mertens6Prev) {
        if (len == 0) return;

        const UInt64 numIntervals =
            Coprime6MertensSieveDetail::coarseCount(len);
        auto& workspace =
            Coprime6MertensSieveDetail::prefixWorkspace<MIntT, false>();
        if (workspace.intervalSums.size() < numIntervals)
            workspace.intervalSums.resize(numIntervals);
        const UInt32 maxThreads = (UInt32)omp_get_max_threads();
        if (workspace.threadSums.size() < maxThreads)
            workspace.threadSums.resize(maxThreads);
        if (workspace.blockMin.size() < numIntervals)
            workspace.blockMin.resize(numIntervals);

        MIntT* intervalSums = workspace.intervalSums.data();
        MIntT* threadSums = workspace.threadSums.data();
        Int16* blockMin = workspace.blockMin.data();

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
                prefix16_i8_to_i16(
                    vld1q_s8(Mu + k), localPrefix + (k - start),
                    carry, vectorMin, vectorMax
                );
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
                prefix16_i8_to_i16_sse(
                    _mm_loadu_si128((const __m128i*)(Mu + k)),
                    localPrefix + (k - start), carry, vectorMin, vectorMax
                );
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

            assert(localMax - localMin <= 255);
            const int shift = -localMin - 128;
            UInt64 write = start;
#if SIEVE_SIMD_NEON
            for (; write + 16 <= end; write += 16)
                encode16_i16_to_i8(
                    localPrefix + (write - start), R + write, shift
                );
#elif SIEVE_SIMD_SSE
            for (; write + 16 <= end; write += 16)
                encode16_i16_to_i8_sse(
                    localPrefix + (write - start), R + write, shift
                );
#endif
            for (; write < end; ++write) {
                const int residual =
                    int(localPrefix[write - start]) + shift;
                assert(residual >= -128 && residual <= 127);
                R[write] = static_cast<Int8>(residual);
            }

            intervalSums[b] = MIntT(local);
            blockMin[b] = static_cast<Int16>(localMin);
        }

        prefixIntervalSums(intervalSums, threadSums, numIntervals);

        M[0] = mertens6Prev + MIntT(blockMin[0]) + MIntT(128);
        #pragma omp parallel for schedule(static)
        for (Int64 b = 1; b < (Int64)numIntervals; ++b)
            M[b] = mertens6Prev + intervalSums[b - 1]
                 + MIntT(blockMin[b]) + MIntT(128);
    }

    template<typename MIntT>
    static void prefixSumFusedFinalize(
        MIntT* __restrict M, Int8* R, Int8* Mu,
        UInt64 firstPackedIndex, UInt64 len, MIntT mertens6Prev) {
        if (len == 0) return;

        const UInt64 numIntervals =
            Coprime6MertensSieveDetail::coarseCount(len);
        auto& workspace =
            Coprime6MertensSieveDetail::prefixWorkspace<MIntT, true>();
        if (workspace.intervalSums.size() < numIntervals)
            workspace.intervalSums.resize(numIntervals);
        const UInt32 maxThreads = (UInt32)omp_get_max_threads();
        if (workspace.threadSums.size() < maxThreads)
            workspace.threadSums.resize(maxThreads);
        if (workspace.blockMin.size() < numIntervals)
            workspace.blockMin.resize(numIntervals);

        MIntT* intervalSums = workspace.intervalSums.data();
        MIntT* threadSums = workspace.threadSums.data();
        Int16* blockMin = workspace.blockMin.data();

        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const UInt64 start = b << STRIDE_LOG;
            const UInt64 end = std::min(start + STRIDE, len);
            const UInt64 nStart =
                SegmentedCoprime6MobiusSieveCore::originalAt(
                    firstPackedIndex + start
                );
            const UInt64 nEnd =
                SegmentedCoprime6MobiusSieveCore::originalAt(
                    firstPackedIndex + end - 1
                );
            const int flStart =
                nStart >= 2 ? 63 - (int)__builtin_clzll(nStart) : 0;
            const int flEnd =
                nEnd >= 2 ? 63 - (int)__builtin_clzll(nEnd) : 0;
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
                    const int8x16_t mu = finalize16_i8(
                        vld1q_s8(Mu + k), floorLog
                    );
                    prefix16_i8_to_i16(
                        mu, localPrefix + (k - start),
                        carry, vectorMin, vectorMax
                    );
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
                        _mm_loadu_si128((const __m128i*)(Mu + k)),
                        floorLog
                    );
                    prefix16_i8_to_i16_sse(
                        mu, localPrefix + (k - start),
                        carry, vectorMin, vectorMax
                    );
                }
                if (k != start) {
                    alignas(16) Int16 minLanes[8];
                    alignas(16) Int16 maxLanes[8];
                    local = (Int16)_mm_cvtsi128_si32(carry);
                    _mm_store_si128((__m128i*)minLanes, vectorMin);
                    _mm_store_si128((__m128i*)maxLanes, vectorMax);
                    for (int lane = 0; lane < 8; ++lane) {
                        localMin = std::min(
                            localMin, int(minLanes[lane])
                        );
                        localMax = std::max(
                            localMax, int(maxLanes[lane])
                        );
                    }
                }
#endif
            }

            for (; k < end; ++k) {
                const UInt64 n =
                    SegmentedCoprime6MobiusSieveCore::originalAt(
                        firstPackedIndex + k
                    );
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
                encode16_i16_to_i8(
                    localPrefix + (write - start), R + write, shift
                );
#elif SIEVE_SIMD_SSE
            for (; write + 16 <= end; write += 16)
                encode16_i16_to_i8_sse(
                    localPrefix + (write - start), R + write, shift
                );
#endif
            for (; write < end; ++write) {
                const int residual =
                    int(localPrefix[write - start]) + shift;
                assert(residual >= -128 && residual <= 127);
                R[write] = static_cast<Int8>(residual);
            }

            intervalSums[b] = MIntT(local);
            blockMin[b] = static_cast<Int16>(localMin);
        }

        prefixIntervalSums(intervalSums, threadSums, numIntervals);

        M[0] = mertens6Prev + MIntT(blockMin[0]) + MIntT(128);
        #pragma omp parallel for schedule(static)
        for (Int64 b = 1; b < (Int64)numIntervals; ++b)
            M[b] = mertens6Prev + intervalSums[b - 1]
                 + MIntT(blockMin[b]) + MIntT(128);
    }

    template<typename MIntT>
    static void prefixSumDirect(MIntT* __restrict M, const Int8* Mu,
                                UInt64 len, MIntT mertens6Prev) {
        if (len == 0) return;

        const UInt64 numIntervals =
            Coprime6MertensSieveDetail::coarseCount(len);
        static std::vector<MIntT> intervalSumsBuf;
        static std::vector<MIntT> threadSumsBuf;
        if (intervalSumsBuf.size() < numIntervals)
            intervalSumsBuf.resize(numIntervals);
        const UInt32 maxThreads = (UInt32)omp_get_max_threads();
        if (threadSumsBuf.size() < maxThreads)
            threadSumsBuf.resize(maxThreads);

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
            const MIntT offset =
                mertens6Prev + (b ? intervalSums[b - 1] : 0);
            const UInt64 start = b << STRIDE_LOG;
            const UInt64 end = std::min(start + STRIDE, len);
            for (UInt64 k = start; k < end; ++k)
                M[k] += offset;
        }
    }
};

using SegmentedCoprime6MertensSieveCore =
    SegmentedCoprime6MertensSieveCoreT<
        Coprime6MertensStorage::Compressed>;

// Exact hot lookup in original integer coordinates. FIRST_PACKED is the
// global packed index of this segment's first represented value; M6_PREV is
// M_6(lo - 1). Each argument is evaluated once.
#define GET_COPRIME6_MERTENS(M, R, FIRST_PACKED, M6_PREV, pos) \
    __extension__({ \
        auto _c6_pos = (pos); \
        auto _c6_first_packed = (FIRST_PACKED); \
        auto _c6_prev = (M6_PREV); \
        auto _c6_through = \
            SegmentedCoprime6MobiusSieveCore::countThrough(_c6_pos); \
        decltype(_c6_prev) _c6_result; \
        if (__builtin_expect(_c6_through <= _c6_first_packed, 0)) { \
            _c6_result = _c6_prev; \
        } else { \
            auto _c6_off = _c6_through - _c6_first_packed - 1; \
            _c6_result = \
                (M)[_c6_off >> \
                    Coprime6MertensSieveDetail::STRIDE_LOG] \
                + (R)[_c6_off]; \
        } \
        _c6_result; \
    })

// Faster hot lookup when the caller guarantees the current segment contains
// at least one represented value at or below pos.
#define GET_COPRIME6_MERTENS_IN_RANGE(M, R, FIRST_PACKED, pos) \
    __extension__({ \
        auto _c6_through = \
            SegmentedCoprime6MobiusSieveCore::countThrough(pos); \
        auto _c6_off = _c6_through - (FIRST_PACKED) - 1; \
        (M)[_c6_off >> Coprime6MertensSieveDetail::STRIDE_LOG] \
            + (R)[_c6_off]; \
    })

template <Coprime6MertensStorage Storage =
          Coprime6MertensStorage::Compressed>
class SegmentedCoprime6MertensSieveT {
public:
    explicit SegmentedCoprime6MertensSieveT(UInt64 originalSegmentSpan)
        : mSegmentSpan(originalSegmentSpan) {
        if (mSegmentSpan == 0) {
            std::fprintf(stderr,
                         "Error: coprime-6 Mertens segment span is zero.\n");
            std::abort();
        }

        mCore.initialize(mSegmentSpan);
        const UInt64 packedCapacity =
            Coprime6MertensSieveDetail::packedCapacity(mSegmentSpan);
        if constexpr (Storage == Coprime6MertensStorage::Compressed)
            mM.resize(Coprime6MertensSieveDetail::coarseCount(
                packedCapacity
            ));
        else
            mM.resize(packedCapacity);
    }

    bool next() {
        if (mDone) return false;

        mLo = mNextLo;
        const UInt64 remainder =
            std::numeric_limits<UInt64>::max() - mLo;
        if (mSegmentSpan - 1 > remainder) {
            mHi = std::numeric_limits<UInt64>::max();
            mDone = true;
        } else {
            mHi = mLo + mSegmentSpan - 1;
        }

        growPrimes(mHi);
        mCore.sieveInPlace(
            mLo, mHi, mMertens6Prev, mM.data(), mPrimes
        );
        mMertens6Prev = mCore.getCoprime6Mertens(mM.data(), mHi);
        if (mHi == std::numeric_limits<UInt64>::max())
            mDone = true;
        else
            mNextLo = mHi + 1;
        return true;
    }

    Int32 getCoprime6Mertens(UInt64 pos) const {
        return mCore.getCoprime6Mertens(mM.data(), pos);
    }

    // Packed values only: vals[i] is M_6 at the ith represented value in the
    // current original-coordinate segment.
    void getSegmentValues(std::vector<Int32>& vals) const {
        if (vals.size() < mCore.packedCount())
            vals.resize(mCore.packedCount());

        if constexpr (Storage == Coprime6MertensStorage::Compressed) {
            const Int8* R = mCore.mobiusSieve().data();
            const UInt64 numIntervals =
                Coprime6MertensSieveDetail::coarseCount(
                    mCore.packedCount()
                );
            #pragma omp parallel for schedule(static)
            for (UInt64 b = 0; b < numIntervals; ++b) {
                const Int32 coarse = mM[b];
                const UInt64 start =
                    b << Coprime6MertensSieveDetail::STRIDE_LOG;
                const UInt64 end = std::min(
                    start + Coprime6MertensSieveDetail::STRIDE,
                    mCore.packedCount()
                );
                for (UInt64 k = start; k < end; ++k)
                    vals[k] = coarse + R[k];
            }
        } else if (mCore.packedCount() != 0) {
            std::memcpy(
                vals.data(), mM.data(),
                mCore.packedCount() * sizeof(Int32)
            );
        }
        vals.resize(mCore.packedCount());
    }

    const Int32* getCoarseData() const {
        static_assert(
            Storage == Coprime6MertensStorage::Compressed,
            "getCoarseData() is only available in Compressed mode"
        );
        return mM.data();
    }

    const Int8* getResidualData() const {
        static_assert(
            Storage == Coprime6MertensStorage::Compressed,
            "getResidualData() is only available in Compressed mode"
        );
        return mCore.mobiusSieve().data();
    }

    const Int32* getSegmentData() const {
        static_assert(
            Storage == Coprime6MertensStorage::Direct,
            "getSegmentData() is only available in Direct mode"
        );
        return mM.data();
    }

    UInt64 lo() const { return mLo; }
    UInt64 hi() const { return mHi; }
    UInt64 firstPackedIndex() const { return mCore.firstPackedIndex(); }
    UInt64 firstCoprime6() const { return mCore.firstCoprime6(); }
    UInt64 packedCount() const { return mCore.packedCount(); }
    UInt64 coprime6Count() const { return mCore.packedCount(); }
    bool done() const { return mDone; }

    std::pair<UInt64, UInt64> nextSegment() const {
        if (mDone) return {0, 0};
        const UInt64 remainder =
            std::numeric_limits<UInt64>::max() - mNextLo;
        const UInt64 hi = mSegmentSpan - 1 > remainder
                        ? std::numeric_limits<UInt64>::max()
                        : mNextLo + mSegmentSpan - 1;
        return {mNextLo, hi};
    }

    SegmentedCoprime6MertensSieveCoreT<Storage>& core() {
        return mCore;
    }
    const SegmentedCoprime6MertensSieveCoreT<Storage>& core() const {
        return mCore;
    }

private:
    void growPrimes(UInt64 hi) {
        const UInt32 need = (UInt32)std::max(
            (Int64)SegmentedCoprime6MobiusSieveCore::MIN_PRIMES_BOUND,
            (Int64)std::round(std::sqrt((double)hi))
        );
        if (need > mPrimeBound) {
            const UInt32 bound = std::max(need, mPrimeBound * 2);
            mPrimes =
                SegmentedCoprime6MobiusSieveCore::primesUpTo(bound);
            mPrimeBound = bound;
        }
    }

    SegmentedCoprime6MertensSieveCoreT<Storage> mCore;
    std::vector<Int32> mM;
    std::vector<UInt32> mPrimes;
    UInt64 mSegmentSpan;
    UInt64 mLo = 1;
    UInt64 mHi = 0;
    UInt64 mNextLo = 1;
    Int32 mMertens6Prev = 0;
    UInt32 mPrimeBound = 0;
    bool mDone = false;
};

using SegmentedCoprime6MertensSieve =
    SegmentedCoprime6MertensSieveT<
        Coprime6MertensStorage::Compressed>;

// Compute M_6(N). Segment sizes are measured in original integers.
static inline Int32 Coprime6MertensSieve(
    UInt64 N, UInt64 originalSegmentSpan = 0) {
    if (N < 1) return 0;
    if (originalSegmentSpan == 0) originalSegmentSpan = N;

    SegmentedCoprime6MobiusSieveCore sieve(originalSegmentSpan);
    const UInt32 sqrtN = (UInt32)std::max(
        (Int64)SegmentedCoprime6MobiusSieveCore::MIN_PRIMES_BOUND,
        (Int64)std::round(std::sqrt((double)N))
    );
    const auto primes =
        SegmentedCoprime6MobiusSieveCore::primesUpTo(sqrtN);

    Int64 sum = 0;
    UInt64 lo = 1;
    while (lo <= N) {
        const UInt64 span = std::min(originalSegmentSpan, N - lo + 1);
        const UInt64 hi = lo + span - 1;
        sieve.sieve(lo, hi, primes);
        const Int8* mu = sieve.data();
        const UInt64 count = sieve.packedCount();
        Int64 segmentSum = 0;
        #pragma omp parallel for reduction(+:segmentSum) schedule(static)
        for (UInt64 i = 0; i < count; ++i)
            segmentSum += mu[i];
        sum += segmentSum;
        if (hi == N) break;
        lo = hi + 1;
    }
    return (Int32)sum;
}

// Return packed prefix values through N. Entry k is M_6(originalAt(k)).
static inline std::vector<Int32> Coprime6MertensSieveValues(
    UInt64 N, UInt64 originalSegmentSpan = 0) {
    if (N < 1) return {};
    if (originalSegmentSpan == 0) originalSegmentSpan = N;

    const UInt64 total =
        SegmentedCoprime6MobiusSieveCore::countThrough(N);
    std::vector<Int32> result(total);
    const UInt64 packedCapacity =
        Coprime6MertensSieveDetail::packedCapacity(originalSegmentSpan);
    std::vector<Int32> M(std::max<UInt64>(packedCapacity, 1));
    SegmentedCoprime6MertensSieveCoreT<
        Coprime6MertensStorage::Direct> sieve(originalSegmentSpan);

    const UInt32 sqrtN = (UInt32)std::max(
        (Int64)SegmentedCoprime6MobiusSieveCore::MIN_PRIMES_BOUND,
        (Int64)std::round(std::sqrt((double)N))
    );
    const auto primes =
        SegmentedCoprime6MobiusSieveCore::primesUpTo(sqrtN);

    Int32 mertens6Prev = 0;
    UInt64 output = 0;
    UInt64 lo = 1;
    while (lo <= N) {
        const UInt64 span = std::min(originalSegmentSpan, N - lo + 1);
        const UInt64 hi = lo + span - 1;
        sieve.sieveInPlace(lo, hi, mertens6Prev, M.data(), primes);
        const UInt64 count = sieve.packedCount();
        if (count != 0) {
            std::memcpy(
                result.data() + output, M.data(), count * sizeof(Int32)
            );
            output += count;
        }
        mertens6Prev = sieve.getCoprime6Mertens(M.data(), hi);
        if (hi == N) break;
        lo = hi + 1;
    }
    return result;
}
