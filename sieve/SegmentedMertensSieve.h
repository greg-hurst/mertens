#pragma once

// ============================================================================
// SegmentedMertensSieve.h — Segmented sieve of the Mertens function M(x).
//
// Wraps SegmentedMobiusSieveCore and adds a parallel prefix sum to convert
// mu values into cumulative M(x) values.
//
// Two storage modes (selected via MertensStorage template parameter):
//
//   Compressed (default):
//     M(x) = coarse[x >> STRIDE_LOG] + residual[x]
//     Coarse stores M at every STRIDEth position; residual stores the offset
//     from the interval's coarse sample (as Int8). Ideal for point lookups
//     (e.g., billions of getM calls in S1 hot loops).
//
//   Direct:
//     M(x) = output[x]
//     Full Int32/Int64 array. No compression overhead. Ideal when the
//     consumer wants all values (e.g., MertensSieveValues, Mathematica).
//
// Two sieve modes (Compressed only):
//   sieve()       — separate R buffer (Loop 0/1: mu buffer is preserved)
//   sieveInPlace() — R aliases the mu buffer (Loop 2: saves one segment buffer)
//
// In Direct mode, sieveInPlace() writes M values directly to the output
// array and the mu buffer is always preserved (since output is a separate
// typed buffer).
// ============================================================================

#include "SegmentedMobiusSieve.h"
#include "simd_defs.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <omp.h>

// SIEVE_FUSED_FINALIZE (compile-time, default on): when 1, Loop 2's in-place
// Compressed sieve skips the in-sieve mu finalization (sieve<false>) and
// folds it into the prefix scan, where the decode rides on the scan's
// DRAM-bound load. Bit-identical M either way; define to 0 for the
// separate pass.
#ifndef SIEVE_FUSED_FINALIZE
#define SIEVE_FUSED_FINALIZE 1
#endif

// --- Storage mode enum ---
enum class MertensStorage { Compressed, Direct };

// --- Constants (accessible without template instantiation) ---
namespace MertensSieveDetail {
#ifndef SIEVE_STRIDE_LOG
#define SIEVE_STRIDE_LOG 8
#endif
    static constexpr int STRIDE_LOG = SIEVE_STRIDE_LOG;
    static_assert(STRIDE_LOG >= 4 && STRIDE_LOG <= 12,
                  "see PERFORMANCE.md section 8 for stride/overflow tradeoffs");
    static constexpr UInt64 STRIDE = UInt64(1) << STRIDE_LOG;
}

// ============================================================================
// SegmentedMertensSieveCoreT — Core class, templated on storage mode.
// ============================================================================

template <MertensStorage Storage = MertensStorage::Compressed>
class SegmentedMertensSieveCoreT {
public:
    static constexpr int STRIDE_LOG = MertensSieveDetail::STRIDE_LOG;
    static constexpr UInt64 STRIDE = MertensSieveDetail::STRIDE;

    // --- Construction ---

    SegmentedMertensSieveCoreT() = default;
    explicit SegmentedMertensSieveCoreT(UInt64 segmentSize) { initialize(segmentSize); }

    // --- Setup ---

    void initialize(UInt64 segmentSize) {
        mMobius.initialize(segmentSize);
    }

    // --- Sieving ---

    // Sieve M over [lo, hi] with separate R buffer. Preserves mu data.
    // Only available in Compressed mode.
    template<typename MIntT>
    void sieve(UInt64 lo, UInt64 hi, MIntT MPrev,
               MIntT* __restrict M, Int8* __restrict R,
               const std::vector<UInt32>& primes) {
        static_assert(Storage == MertensStorage::Compressed,
                      "sieve() with separate R is only available in Compressed mode");
        mMobius.sieve(lo, hi, primes);
        prefixSum(M, R, mMobius.data(), hi - lo + 1, MPrev);
        mLo = lo;
        mActiveR = R;
    }

    // Sieve M over [lo, hi].
    // Compressed: R aliases the mu buffer (in-place). After this call,
    //             mMobius.data() contains R, not mu.
    // Direct:     Writes M values directly to the output array.
    //             mu buffer is preserved (output is a separate typed buffer).
    template<typename MIntT>
    void sieveInPlace(UInt64 lo, UInt64 hi, MIntT MPrev,
                      MIntT* __restrict M,
                      const std::vector<UInt32>& primes) {
        if constexpr (Storage == MertensStorage::Compressed) {
#if SIEVE_FUSED_FINALIZE
            // Skip the in-sieve finalize pass and fold it into the prefix scan.
            mMobius.template sieve<false>(lo, hi, primes);
            Int8* muData = mMobius.data();
            prefixSumFusedFinalize(M, muData, muData, lo, hi - lo + 1, MPrev);
#else
            mMobius.sieve(lo, hi, primes);
            Int8* muData = mMobius.data();
            prefixSum(M, muData, muData, hi - lo + 1, MPrev);
#endif
            mActiveR = muData;
        } else {
            mMobius.sieve(lo, hi, primes);
            prefixSumDirect(M, mMobius.data(), hi - lo + 1, MPrev);
        }
        mLo = lo;
    }

    // --- Access ---

    // Look up M(pos) from the representation after a sieve call.
    // M is the array passed to sieve/sieveInPlace; pos is absolute.
    template<typename MIntT>
    inline MIntT getMertens(const MIntT* M, UInt64 pos) const {
        auto off = pos - mLo;
        if constexpr (Storage == MertensStorage::Compressed)
            return M[off >> STRIDE_LOG] + mActiveR[off];
        else
            return M[off];
    }

    // Access the underlying Mobius sieve.
    SegmentedMobiusSieveCore& mobiusSieve() { return mMobius; }
    const SegmentedMobiusSieveCore& mobiusSieve() const { return mMobius; }

private:
    SegmentedMobiusSieveCore mMobius;
    std::vector<Int8> mR;      // Residual buffer (Compressed mode, non-in-place only)
    UInt64 mLo = 0;            // lo from the most recent sieve call
    const Int8* mActiveR = nullptr;  // R buffer from the most recent sieve call (Compressed only)

    // --- SIMD prefix scan helpers (Compressed mode) ---
    //
    // The scan runs directly in int8 — no widening. It cannot overflow: each
    // carried lane is an exact residual R[k], which Int8 must hold anyway,
    // and intra-scan intermediates are bounded by +/-16. The cross-chunk
    // carry stays in a vector register; a vector->scalar->vector round trip
    // would sit on the serial chain and roughly halve Stage A throughput.

#if SIEVE_SIMD_NEON
    static inline int8x16_t scan16_i8_inclusive(int8x16_t x) {
        const int8x16_t z = vdupq_n_s8(0);
        x = vaddq_s8(x, vextq_s8(z, x, 15));
        x = vaddq_s8(x, vextq_s8(z, x, 14));
        x = vaddq_s8(x, vextq_s8(z, x, 12));
        x = vaddq_s8(x, vextq_s8(z, x, 8));
        return x;
    }

    // Finalize 16 packed sieve bytes to mu in {-1,0,+1} (ceil-log2 scheme),
    // for a constant floor(log2 N) threshold broadcast in vFl. Mirrors
    // SegmentedMobiusSieveCore::finalizeMuVec lane-for-lane so the fused scan
    // is bit-identical to finalize-then-scan. Used only by the fused kernel.
    static inline int8x16_t finalize16_i8(int8x16_t raw, int8x16_t vFl) {
        const int8x16_t vZero  = vdupq_n_s8(0);
        const int8x16_t vOne   = vdupq_n_s8(1);
        const int8x16_t vMask7 = vdupq_n_s8(0x7F);
        const uint8x16_t sf  = vcltq_s8(raw, vZero);          // raw < 0 -> squarefree
        const int8x16_t  S   = vandq_s8(raw, vMask7);
        const int8x16_t  lsb = vandq_s8(raw, vOne);
        const int8x16_t  base    = vsubq_s8(vshlq_n_s8(lsb, 1), vOne);  // 2*par - 1
        const int8x16_t  negBase = vnegq_s8(base);
        const uint8x16_t ff  = vcgtq_s8(S, vFl);              // S > fl -> fully factored
        const int8x16_t  r   = vbslq_s8(ff, negBase, base);
        return vbslq_s8(sf, r, vZero);
    }

#elif SIEVE_SIMD_SSE
    static inline __m128i scan16_i8_inclusive_sse(__m128i x) {
        x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
        return x;
    }

    // Broadcast byte 15 of v to all 16 lanes (the cross-chunk carry).
    static inline __m128i broadcast_byte15_sse(__m128i v) {
#if defined(__SSSE3__)
        return _mm_shuffle_epi8(v, _mm_set1_epi8(15));
#else
        v = _mm_unpackhi_epi8(v, v);                          // byte 15 -> word 7
        v = _mm_shufflehi_epi16(v, _MM_SHUFFLE(3, 3, 3, 3));  // word 7 -> words 4..7
        return _mm_unpackhi_epi64(v, v);                      // high half -> both
#endif
    }

    // Finalize 16 packed sieve bytes to mu in {-1,0,+1} (ceil-log2 scheme),
    // constant floor(log2 N) threshold in vFl. Mirrors finalizeMuVec's SSE
    // branch lane-for-lane. Used only by the fused kernel.
    static inline __m128i finalize16_i8_sse(__m128i raw, __m128i vFl) {
        const __m128i vZero  = _mm_setzero_si128();
        const __m128i vOne   = _mm_set1_epi8(1);
        const __m128i vMask7 = _mm_set1_epi8(0x7F);
        const __m128i sf  = _mm_cmplt_epi8(raw, vZero);       // raw < 0
        const __m128i S   = _mm_and_si128(raw, vMask7);
        const __m128i lsb = _mm_and_si128(raw, vOne);
        const __m128i base    = _mm_sub_epi8(_mm_add_epi8(lsb, lsb), vOne);
        const __m128i negBase = _mm_sub_epi8(vZero, base);
        const __m128i ff  = _mm_cmpgt_epi8(S, vFl);           // S > fl (both >= 0)
        const __m128i r   = _mm_or_si128(_mm_and_si128(ff, negBase),
                                         _mm_andnot_si128(ff, base));
        return _mm_and_si128(sf, r);
    }
#endif

    // --- Scalar finalization helpers (fused kernel) ---
    //
    // Decode one packed sieve byte to mu (ceil-log2 scheme), matching
    // SegmentedMobiusSieveCore::finalizeMu. finalizeOneFl takes a precomputed
    // floor(log2 N) threshold (the constant-fl fast path); finalizeOne derives
    // it per element and also handles N <= 2 (the first interval of segment 1).
    static inline Int8 finalizeOneFl(Int8 v, int fl) {
        if (v >= 0) return 0;                       // square factor -> mu = 0
        const int S    = v & 0x7F;
        const int base = ((v & 1) << 1) - 1;        // 2*par - 1 in {-1,+1}
        return static_cast<Int8>((S > fl) ? -base : base);
    }
    static inline Int8 finalizeOne(Int8 v, UInt64 N) {
        if (N <= 2) return (N == 1) ? Int8(1) : Int8(-1);
        return finalizeOneFl(v, 63 - (int)__builtin_clzll(N));
    }

    // --- Parallel prefix sum: Mu -> (M, R) [Compressed mode] ---
    //
    // Converts a flat mu buffer into compressed (coarse M, residual R) form.
    template<typename MIntT>
    static void prefixSum(
        MIntT* __restrict M,
        Int8* R,
        Int8* Mu,
        const UInt64 len,
        const MIntT MPrev) {
        static_assert(STRIDE_LOG > 2, "STRIDE_LOG too small");
        static_assert(STRIDE_LOG < 15, "STRIDE_LOG too big");

        const UInt64 fullIntervals = len >> STRIDE_LOG;
        const UInt64 rem           = len & (STRIDE - 1);
        const UInt64 numIntervals  = fullIntervals + (rem ? 1 : 0);

        if (len == 0) return;

        // Reused work buffers (avoid allocation every segment).
        // Note: not safe if multiple independent sieve instances call
        // prefixSum concurrently from an outer parallel layer.
        static std::vector<MIntT> intervalSumsBuf;
        static std::vector<MIntT> sumaBuf;

        if (intervalSumsBuf.size() < numIntervals) intervalSumsBuf.resize(numIntervals);
        const UInt32 maxThr = (UInt32)omp_get_max_threads();
        if (sumaBuf.size() < maxThr) sumaBuf.resize(maxThr);

        MIntT* intervalSums = intervalSumsBuf.data();
        MIntT* suma         = sumaBuf.data();

        // Stage A: per-interval local prefix, compute R, intervalSums, and Mub.
        // Mub (the leading mu of each interval, needed by Stage C) is captured
        // inside the loop, before R — which may alias Mu in-place — overwrites
        // it. A separate pre-pass would stride one byte per 256, which still
        // touches every other cache line: nearly half a DRAM pass for nothing.
        static std::vector<Int8> Mub;
        if (Mub.size() < numIntervals) Mub.resize(numIntervals);

        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const UInt64 start = b * STRIDE;
            const UInt64 end   = (start + STRIDE <= len ? start + STRIDE : len);

            const Int8 MuStart = Mu[start];
            Mub[b] = MuStart;
            Int8 local = (Int8)(-MuStart);

            UInt64 k = start;

#if SIEVE_SIMD_NEON
            int8x16_t carry = vdupq_n_s8(local);
            // 64-byte unroll: the four scans are independent; only the short
            // broadcast-and-add carry chain is serial between chunks.
            for (; k + 64 <= end; k += 64) {
                const int8x16_t s0 = scan16_i8_inclusive(vld1q_s8((const Int8*)(Mu + k)));
                const int8x16_t s1 = scan16_i8_inclusive(vld1q_s8((const Int8*)(Mu + k + 16)));
                const int8x16_t s2 = scan16_i8_inclusive(vld1q_s8((const Int8*)(Mu + k + 32)));
                const int8x16_t s3 = scan16_i8_inclusive(vld1q_s8((const Int8*)(Mu + k + 48)));
                const int8x16_t o0 = vaddq_s8(s0, carry);
                const int8x16_t o1 = vaddq_s8(s1, vdupq_laneq_s8(o0, 15));
                const int8x16_t o2 = vaddq_s8(s2, vdupq_laneq_s8(o1, 15));
                const int8x16_t o3 = vaddq_s8(s3, vdupq_laneq_s8(o2, 15));
                carry = vdupq_laneq_s8(o3, 15);
                vst1q_s8((Int8*)(R + k),      o0);
                vst1q_s8((Int8*)(R + k + 16), o1);
                vst1q_s8((Int8*)(R + k + 32), o2);
                vst1q_s8((Int8*)(R + k + 48), o3);
            }
            for (; k + 16 <= end; k += 16) {
                const int8x16_t ps = scan16_i8_inclusive(vld1q_s8((const Int8*)(Mu + k)));
                const int8x16_t out = vaddq_s8(ps, carry);
                vst1q_s8((Int8*)(R + k), out);
                carry = vdupq_laneq_s8(out, 15);
            }
            local = vgetq_lane_s8(carry, 0);
            // AVX2 and AVX-512 machines use this SSE (128-bit) path intentionally.
            // Prefix sum is sequential (each output depends on the previous),
            // so wider registers don't help — x86 byte-shifts don't cross the
            // 128-bit lane boundary, making a wider scan no better than
            // multiple 16-wide scans with cross-lane fixups (i.e. what we
            // already do). Similarly, SVE2 machines use the NEON path above.
#elif SIEVE_SIMD_SSE
            __m128i carry = _mm_set1_epi8((char)local);
            for (; k + 64 <= end; k += 64) {
                const __m128i s0 = scan16_i8_inclusive_sse(_mm_loadu_si128((const __m128i*)(Mu + k)));
                const __m128i s1 = scan16_i8_inclusive_sse(_mm_loadu_si128((const __m128i*)(Mu + k + 16)));
                const __m128i s2 = scan16_i8_inclusive_sse(_mm_loadu_si128((const __m128i*)(Mu + k + 32)));
                const __m128i s3 = scan16_i8_inclusive_sse(_mm_loadu_si128((const __m128i*)(Mu + k + 48)));
                const __m128i o0 = _mm_add_epi8(s0, carry);
                const __m128i o1 = _mm_add_epi8(s1, broadcast_byte15_sse(o0));
                const __m128i o2 = _mm_add_epi8(s2, broadcast_byte15_sse(o1));
                const __m128i o3 = _mm_add_epi8(s3, broadcast_byte15_sse(o2));
                carry = broadcast_byte15_sse(o3);
                _mm_storeu_si128((__m128i*)(R + k),      o0);
                _mm_storeu_si128((__m128i*)(R + k + 16), o1);
                _mm_storeu_si128((__m128i*)(R + k + 32), o2);
                _mm_storeu_si128((__m128i*)(R + k + 48), o3);
            }
            for (; k + 16 <= end; k += 16) {
                const __m128i ps = scan16_i8_inclusive_sse(_mm_loadu_si128((const __m128i*)(Mu + k)));
                const __m128i out = _mm_add_epi8(ps, carry);
                _mm_storeu_si128((__m128i*)(R + k), out);
                carry = broadcast_byte15_sse(out);
            }
            local = (Int8)_mm_cvtsi128_si32(carry);
#endif
            // Scalar tail (also the full path when no SIMD is available)
            for (; k < end; ++k) {
                local = (Int8)(local + Mu[k]);
                R[k] = local;
            }

            intervalSums[b] = MIntT(local) + MIntT(MuStart);
        }

        // Stage B: parallel prefix-sum over intervalSums
        #pragma omp parallel
        {
            const UInt32 tid  = omp_get_thread_num();
            const UInt32 nthr = omp_get_num_threads();

            MIntT local = 0;

            #pragma omp for schedule(static)
            for (UInt64 i = 0; i < numIntervals; ++i) {
                local += intervalSums[i];
                intervalSums[i] = local;
            }

            suma[tid] = local;

            #pragma omp barrier

            #pragma omp single
            {
                MIntT offset = 0;
                for (UInt32 t = 0; t < nthr; ++t) {
                    const MIntT tmp = suma[t];
                    suma[t] = offset;
                    offset += tmp;
                }
            }

            #pragma omp barrier
            const MIntT offset = suma[tid];

            #pragma omp for schedule(static)
            for (UInt64 i = 0; i < numIntervals; ++i) {
                intervalSums[i] += offset;
            }
        }

        // Stage C: compute M[b] from intervalSums and Mu
        M[0] = MPrev + (MIntT)Mub[0];
        #pragma omp parallel for schedule(static)
        for (Int64 b = 1; b < (Int64)numIntervals; ++b) {
            M[b] = MPrev + intervalSums[b - 1] + (MIntT)Mub[b];
        }
    }

    // --- Fused finalize + prefix sum: raw packed S -> (M, R) [Compressed] ---
    //
    // Identical contract to prefixSum, but Mu holds raw packed sieve bytes
    // (sieve<false>): the decode to mu is folded into Stage A's scan. Only
    // Stage A differs; B and C are identical. The finalize threshold
    // floor(log2 N) is constant per interval except where an interval
    // straddles a power of two or contains N <= 2 — those take a scalar
    // per-element path, everything else the SIMD fast path.
    template<typename MIntT>
    static void prefixSumFusedFinalize(
        MIntT* __restrict M,
        Int8* R,
        Int8* Mu,
        const UInt64 lo,
        const UInt64 len,
        const MIntT MPrev) {
        static_assert(STRIDE_LOG > 2, "STRIDE_LOG too small");
        static_assert(STRIDE_LOG < 15, "STRIDE_LOG too big");

        const UInt64 fullIntervals = len >> STRIDE_LOG;
        const UInt64 rem           = len & (STRIDE - 1);
        const UInt64 numIntervals  = fullIntervals + (rem ? 1 : 0);

        if (len == 0) return;

        static std::vector<MIntT> intervalSumsBuf;
        static std::vector<MIntT> sumaBuf;

        if (intervalSumsBuf.size() < numIntervals) intervalSumsBuf.resize(numIntervals);
        const UInt32 maxThr = (UInt32)omp_get_max_threads();
        if (sumaBuf.size() < maxThr) sumaBuf.resize(maxThr);

        MIntT* intervalSums = intervalSumsBuf.data();
        MIntT* suma         = sumaBuf.data();

        static std::vector<Int8> Mub;
        if (Mub.size() < numIntervals) Mub.resize(numIntervals);

        // Stage A: per-interval fused finalize + local prefix.
        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const UInt64 start  = b * STRIDE;
            const UInt64 end    = (start + STRIDE <= len ? start + STRIDE : len);
            const UInt64 Nstart = lo + start;

            const int flStart = 63 - (int)__builtin_clzll(Nstart);
            const int flEnd   = 63 - (int)__builtin_clzll(lo + end - 1);
            const bool fastPath = (Nstart >= 3) & (flStart == flEnd);

            // Leading mu for Mub / Stage C, captured before R (which may
            // alias Mu in place) overwrites Mu[start]. N <= 2 only occurs in
            // the first interval of segment 1, so the steady state takes the
            // plain threshold decode.
            const Int8 MuStart = __builtin_expect(Nstart >= 3, 1)
                               ? finalizeOneFl(Mu[start], flStart)
                               : finalizeOne(Mu[start], Nstart);
            Mub[b] = MuStart;
            Int8 local = (Int8)(-MuStart);

            UInt64 k = start;

            if (__builtin_expect(fastPath, 1)) {
                const int fl = flStart;
#if SIEVE_SIMD_NEON
                const int8x16_t vFl = vdupq_n_s8((Int8)fl);
                int8x16_t carry = vdupq_n_s8(local);
                for (; k + 64 <= end; k += 64) {
                    const int8x16_t s0 = scan16_i8_inclusive(finalize16_i8(vld1q_s8((const Int8*)(Mu + k)),      vFl));
                    const int8x16_t s1 = scan16_i8_inclusive(finalize16_i8(vld1q_s8((const Int8*)(Mu + k + 16)), vFl));
                    const int8x16_t s2 = scan16_i8_inclusive(finalize16_i8(vld1q_s8((const Int8*)(Mu + k + 32)), vFl));
                    const int8x16_t s3 = scan16_i8_inclusive(finalize16_i8(vld1q_s8((const Int8*)(Mu + k + 48)), vFl));
                    const int8x16_t o0 = vaddq_s8(s0, carry);
                    const int8x16_t o1 = vaddq_s8(s1, vdupq_laneq_s8(o0, 15));
                    const int8x16_t o2 = vaddq_s8(s2, vdupq_laneq_s8(o1, 15));
                    const int8x16_t o3 = vaddq_s8(s3, vdupq_laneq_s8(o2, 15));
                    carry = vdupq_laneq_s8(o3, 15);
                    vst1q_s8((Int8*)(R + k),      o0);
                    vst1q_s8((Int8*)(R + k + 16), o1);
                    vst1q_s8((Int8*)(R + k + 32), o2);
                    vst1q_s8((Int8*)(R + k + 48), o3);
                }
                for (; k + 16 <= end; k += 16) {
                    const int8x16_t ps  = scan16_i8_inclusive(finalize16_i8(vld1q_s8((const Int8*)(Mu + k)), vFl));
                    const int8x16_t out = vaddq_s8(ps, carry);
                    vst1q_s8((Int8*)(R + k), out);
                    carry = vdupq_laneq_s8(out, 15);
                }
                local = vgetq_lane_s8(carry, 0);
#elif SIEVE_SIMD_SSE
                const __m128i vFl = _mm_set1_epi8((char)fl);
                __m128i carry = _mm_set1_epi8((char)local);
                for (; k + 64 <= end; k += 64) {
                    const __m128i s0 = scan16_i8_inclusive_sse(finalize16_i8_sse(_mm_loadu_si128((const __m128i*)(Mu + k)),      vFl));
                    const __m128i s1 = scan16_i8_inclusive_sse(finalize16_i8_sse(_mm_loadu_si128((const __m128i*)(Mu + k + 16)), vFl));
                    const __m128i s2 = scan16_i8_inclusive_sse(finalize16_i8_sse(_mm_loadu_si128((const __m128i*)(Mu + k + 32)), vFl));
                    const __m128i s3 = scan16_i8_inclusive_sse(finalize16_i8_sse(_mm_loadu_si128((const __m128i*)(Mu + k + 48)), vFl));
                    const __m128i o0 = _mm_add_epi8(s0, carry);
                    const __m128i o1 = _mm_add_epi8(s1, broadcast_byte15_sse(o0));
                    const __m128i o2 = _mm_add_epi8(s2, broadcast_byte15_sse(o1));
                    const __m128i o3 = _mm_add_epi8(s3, broadcast_byte15_sse(o2));
                    carry = broadcast_byte15_sse(o3);
                    _mm_storeu_si128((__m128i*)(R + k),      o0);
                    _mm_storeu_si128((__m128i*)(R + k + 16), o1);
                    _mm_storeu_si128((__m128i*)(R + k + 32), o2);
                    _mm_storeu_si128((__m128i*)(R + k + 48), o3);
                }
                for (; k + 16 <= end; k += 16) {
                    const __m128i ps  = scan16_i8_inclusive_sse(finalize16_i8_sse(_mm_loadu_si128((const __m128i*)(Mu + k)), vFl));
                    const __m128i out = _mm_add_epi8(ps, carry);
                    _mm_storeu_si128((__m128i*)(R + k), out);
                    carry = broadcast_byte15_sse(out);
                }
                local = (Int8)_mm_cvtsi128_si32(carry);
#endif
                // Scalar tail (also the full path when no SIMD is available).
                for (; k < end; ++k) {
                    local = (Int8)(local + finalizeOneFl(Mu[k], fl));
                    R[k] = local;
                }
            } else {
                // Slow path: interval straddles a power of two, or contains N <= 2.
                // floor(log2 N) is derived per element.
                for (; k < end; ++k) {
                    local = (Int8)(local + finalizeOne(Mu[k], lo + k));
                    R[k] = local;
                }
            }

            intervalSums[b] = MIntT(local) + MIntT(MuStart);
        }

        // Stage B: parallel prefix-sum over intervalSums (identical to prefixSum).
        #pragma omp parallel
        {
            const UInt32 tid  = omp_get_thread_num();
            const UInt32 nthr = omp_get_num_threads();

            MIntT local = 0;

            #pragma omp for schedule(static)
            for (UInt64 i = 0; i < numIntervals; ++i) {
                local += intervalSums[i];
                intervalSums[i] = local;
            }

            suma[tid] = local;

            #pragma omp barrier

            #pragma omp single
            {
                MIntT offset = 0;
                for (UInt32 t = 0; t < nthr; ++t) {
                    const MIntT tmp = suma[t];
                    suma[t] = offset;
                    offset += tmp;
                }
            }

            #pragma omp barrier
            const MIntT offset = suma[tid];

            #pragma omp for schedule(static)
            for (UInt64 i = 0; i < numIntervals; ++i) {
                intervalSums[i] += offset;
            }
        }

        // Stage C: compute M[b] from intervalSums and Mu (identical to prefixSum).
        M[0] = MPrev + (MIntT)Mub[0];
        #pragma omp parallel for schedule(static)
        for (Int64 b = 1; b < (Int64)numIntervals; ++b) {
            M[b] = MPrev + intervalSums[b - 1] + (MIntT)Mub[b];
        }
    }

    // --- Parallel prefix sum: Mu -> M [Direct mode] ---
    //
    // Converts a flat mu buffer directly into cumulative M values.
    template<typename MIntT>
    static void prefixSumDirect(
        MIntT* __restrict M,
        const Int8* Mu,
        const UInt64 len,
        const MIntT MPrev) {
        const UInt64 fullIntervals = len >> STRIDE_LOG;
        const UInt64 rem = len & (STRIDE - 1);
        const UInt64 numIntervals = fullIntervals + (rem ? 1 : 0);

        if (len == 0) return;

        // Reused work buffers.
        // Note: not safe if multiple independent sieve instances call
        // prefixSumDirect concurrently from an outer parallel layer.
        static std::vector<MIntT> intervalSumsBuf;
        static std::vector<MIntT> sumaBuf;

        if (intervalSumsBuf.size() < numIntervals) intervalSumsBuf.resize(numIntervals);
        const UInt32 maxThr = (UInt32)omp_get_max_threads();
        if (sumaBuf.size() < maxThr) sumaBuf.resize(maxThr);

        MIntT* intervalSums = intervalSumsBuf.data();
        MIntT* suma = sumaBuf.data();

        // Stage A: per-interval local prefix sum, writing MIntT directly
        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const UInt64 start = b * STRIDE;
            const UInt64 end = (start + STRIDE <= len ? start + STRIDE : len);

            MIntT local = 0;
            for (UInt64 k = start; k < end; ++k) {
                local += (MIntT)Mu[k];
                M[k] = local;
            }
            intervalSums[b] = local;
        }

        // Stage B: parallel prefix-sum over intervalSums (identical to Compressed)
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

            suma[tid] = local;

            #pragma omp barrier
            #pragma omp single
            {
                MIntT offset = 0;
                for (UInt32 t = 0; t < nthr; ++t) {
                    const MIntT tmp = suma[t];
                    suma[t] = offset;
                    offset += tmp;
                }
            }
            #pragma omp barrier

            const MIntT offset = suma[tid];

            #pragma omp for schedule(static)
            for (UInt64 i = 0; i < numIntervals; ++i)
                intervalSums[i] += offset;
        }

        // Stage C: broadcast interval offsets to all positions
        #pragma omp parallel for schedule(static)
        for (UInt64 b = 0; b < numIntervals; ++b) {
            const MIntT offset = MPrev + (b > 0 ? intervalSums[b - 1] : 0);
            const UInt64 start = b * STRIDE;
            const UInt64 end = (start + STRIDE <= len ? start + STRIDE : len);
            for (UInt64 k = start; k < end; ++k)
                M[k] += offset;
        }
    }
};

// Backward-compatible type alias (default = Compressed)
using SegmentedMertensSieveCore = SegmentedMertensSieveCoreT<MertensStorage::Compressed>;

// Retrieve M(pos) from the compressed (coarse M, residual R) representation,
// where pos is an absolute index and L1 is the segment start. The macro
// subtracts L1 internally so callers pass the natural argument.
//
// A macro rather than an inline/template method: the S2 inner loops call it
// billions of times under #pragma unroll, and a method compiles measurably
// slower (inlining and unrolling heuristics).
#define GET_M(M, R, L1, pos) __extension__({ \
    auto _off = (pos) - (L1); \
    (M)[_off >> MertensSieveDetail::STRIDE_LOG] + (R)[_off]; \
})

// ============================================================================
// SegmentedMertensSieveT — Iterator-style wrapper, templated on storage mode.
//
// Construct with a segment size. Call next() to advance one segment at a time,
// starting from 1. Primes are grown automatically as needed.
//
// After each next() call:
//   getMertens(pos)      — look up M(pos) within the current segment
//   getSegmentValues()   — all M values in the current segment as Int32
//   getSegmentData()     — [Direct mode only] zero-copy pointer to M values
//   getCoarseData(),
//   getResidualData()    — [Compressed mode only] zero-copy pointers to the
//                          compressed representation
//   lo(), hi()           — current segment bounds
//   nextSegment()        — bounds of the upcoming segment
// ============================================================================

template <MertensStorage Storage = MertensStorage::Compressed>
class SegmentedMertensSieveT {
public:
    explicit SegmentedMertensSieveT(UInt64 segmentSize)
        : mSegSize(segmentSize)
    {
        mCore.initialize(mSegSize);
        if constexpr (Storage == MertensStorage::Compressed) {
            const UInt64 numIntervals = (mSegSize + MertensSieveDetail::STRIDE - 1)
                                >> MertensSieveDetail::STRIDE_LOG;
            mM.resize(numIntervals);
        } else {
            mM.resize(mSegSize);
        }
    }

    // Advance to the next segment. Returns true if a segment was produced.
    bool next() {
        if (mDone) return false;

        mLo = mNextLo;
        mHi = mLo + mSegSize - 1;

        growPrimes(mHi);
        mCore.sieveInPlace(mLo, mHi, mMPrev, mM.data(), mPrimes);
        mMPrev = mCore.getMertens(mM.data(), mHi);
        mNextLo = mHi + 1;
        return true;
    }

    // Look up M(pos) within the current segment. pos is absolute.
    Int32 getMertens(UInt64 pos) const {
        return mCore.getMertens(mM.data(), pos);
    }

    // Fill vals with all M values in the current segment.
    // Resizes vals if needed. Caller can reuse the same vector across segments.
    // In Compressed mode this decompresses the whole segment to Int32 — in hot
    // loops prefer getCoarseData()/getResidualData(), which are free.
    void getSegmentValues(std::vector<Int32>& vals) const {
        if ((UInt64)vals.size() < mSegSize) vals.resize(mSegSize);

        if constexpr (Storage == MertensStorage::Compressed) {
            const Int8* R = mCore.mobiusSieve().data();
            const UInt64 numIntervals = (mSegSize + MertensSieveDetail::STRIDE - 1)
                                >> MertensSieveDetail::STRIDE_LOG;

            #pragma omp parallel for schedule(static)
            for (UInt64 b = 0; b < numIntervals; ++b) {
                const Int32 coarse = mM[b];
                const UInt64 base = b << MertensSieveDetail::STRIDE_LOG;
                const UInt64 end = std::min(base + MertensSieveDetail::STRIDE, mSegSize);
                for (UInt64 j = 0; j < end - base; ++j) {
                    vals[base + j] = coarse + R[base + j];
                }
            }
        } else {
            std::memcpy(vals.data(), mM.data(), mSegSize * sizeof(Int32));
        }
    }

    // [Direct mode only] Zero-copy access to the M values in the current segment.
    const Int32* getSegmentData() const {
        static_assert(Storage == MertensStorage::Direct,
                      "getSegmentData() is only available in Direct mode");
        return mM.data();
    }

    // [Compressed mode only] Zero-copy access to the compressed representation
    // of the current segment. Both return pointers to buffers that already
    // exist — nothing is copied. Valid until the next next() call.
    // For i = pos - lo():
    //   M(pos) = getCoarseData()[i >> STRIDE_LOG] + getResidualData()[i]
    // with STRIDE_LOG = MertensSieveDetail::STRIDE_LOG.
    const Int32* getCoarseData() const {
        static_assert(Storage == MertensStorage::Compressed,
                      "getCoarseData() is only available in Compressed mode");
        return mM.data();
    }

    const Int8* getResidualData() const {
        static_assert(Storage == MertensStorage::Compressed,
                      "getResidualData() is only available in Compressed mode");
        return mCore.mobiusSieve().data();
    }

    UInt64 lo() const { return mLo; }
    UInt64 hi() const { return mHi; }
    bool done() const { return mDone; }

    // Inclusive bounds of the segment the next call to next() will sieve.
    std::pair<UInt64, UInt64> nextSegment() const {
        return {mNextLo, mNextLo + mSegSize - 1};
    }

    // Access the underlying core sieve.
    SegmentedMertensSieveCoreT<Storage>& core() { return mCore; }

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

    SegmentedMertensSieveCoreT<Storage> mCore;
    std::vector<Int32> mM;
    std::vector<UInt32> mPrimes;
    UInt64 mSegSize;
    UInt64 mLo = 1;
    UInt64 mHi = 0;
    UInt64 mNextLo = 1;
    Int32 mMPrev = 0;
    UInt32 mPrimeBound = 0;
    bool mDone = false;
};

// Backward-compatible type aliases
using SegmentedMertensSieve = SegmentedMertensSieveT<MertensStorage::Compressed>;

// ============================================================================
// Free functions — simplest API, hide all implementation details.
// ============================================================================

// Compute M(N) = sum_{k=1}^{N} mu(k).
// Uses the Mobius sieve directly with a parallel reduction sum —
// no Mertens prefix sum or compressed representation needed.
static inline Int32 MertensSieve(UInt64 N, UInt64 segSize = 0) {
    if (N < 1) return 0;

    if (segSize == 0) segSize = N;

    SegmentedMobiusSieveCore sieve(segSize);

    const UInt32 sqrtN = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                          (Int64)std::round(std::sqrt((double)N)));
    auto primes = SegmentedMobiusSieveCore::primesUpTo(sqrtN);

    Int64 sum = 0;
    for (UInt64 slo = 1; slo <= N; slo += segSize) {
        UInt64 shi = std::min(slo + segSize - 1, N);
        sieve.sieve(slo, shi, primes);

        const Int8* mu = sieve.data();
        Int64 segSum = 0;

        #pragma omp parallel for reduction(+:segSum) schedule(static)
        for (UInt64 k = slo; k <= shi; ++k)
            segSum += mu[k - slo];

        sum += segSum;
    }

    return (Int32)sum;
}

// Compute M(k) for k = 1, 2, ..., N and return as a vector of Int32.
// Result has size N, with result[k-1] = M(k).
// Uses Direct mode internally to avoid compress/decompress overhead.
static inline std::vector<Int32> MertensSieveValues(UInt64 N, UInt64 segSize = 0) {
    if (N < 1) return {};

    std::vector<Int32> result(N);

    if (segSize == 0) segSize = N;

    SegmentedMertensSieveCoreT<MertensStorage::Direct> sieve(segSize);
    std::vector<Int32> M(segSize);
    Int32 MPrev = 0;

    const UInt32 sqrtN = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                          (Int64)std::round(std::sqrt((double)N)));
    auto primes = SegmentedMobiusSieveCore::primesUpTo(sqrtN);

    for (UInt64 slo = 1; slo <= N; slo += segSize) {
        UInt64 shi = std::min(slo + segSize - 1, N);
        sieve.sieveInPlace(slo, shi, MPrev, M.data(), primes);

        const UInt64 count = shi - slo + 1;
        std::memcpy(&result[slo - 1], M.data(), count * sizeof(Int32));

        MPrev = sieve.getMertens(M.data(), shi);
    }

    return result;
}
