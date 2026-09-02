#pragma once

// ============================================================================
// S1.h — S1 summation functions for the Mertens function computation.
//
// S1 computes: sum_{x in range} M(floor(n/x))
//
// where M is retrieved from the compressed (coarse, residual) representation.
//
// Two precision levels:
//   64-bit  (update_S1):      for most of the computation
//   128-bit (update_S1_128):  for the largest arguments
//
// The Q2 128-bit path uses the quotient stepper where supported. Its exact
// fallback retains the small-division and fast-predictor ranges.
//
// Parity modes (ParityMode::All, Even, Odd) control which x values to sum
// over, exploiting the inclusion-exclusion parity structure.
// ============================================================================

#include "types.h"
#include "QuotientCache.h"
#include "QuotientPredictor.h"
#include "QuotientStepper.h"
#include "SegmentedMertensSieve.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

enum class S1QuotientPolicy : UInt8 {
    Legacy,
    Stepper,
};

inline constexpr std::size_t Q2StepperLanes = 16;

template <ParityMode Mode, typename MIntT>
static inline bool sum_S1_128_stepped(
    const UInt128& n,
    UInt64 L1,
    UInt64 from,
    UInt64 to,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    Int128& sum
) {
    constexpr UInt64 Step = Mode == ParityMode::All ? 1 : 2;
    using Stepper = QuotientStepper<Q2StepperLanes>;

    UInt64 first = from;
    if constexpr (Mode == ParityMode::Odd)
        first |= 1ULL;
    else if constexpr (Mode == ParityMode::Even)
        first = (first + 1) & ~1ULL;

    if (first > to) {
        sum = 0;
        return true;
    }

    const UInt64 count = (to - first) / Step + 1;
    if (count < 2 * Stepper::Width
        || !Stepper::supports(n, first, Step * Stepper::Width))
        return false;

    auto denominators = [](UInt64 base) {
        Stepper::Batch values;
        for (std::size_t lane = 0; lane < Stepper::Width; ++lane)
            values[lane] = base + Step * lane;
        return values;
    };
    auto add = [&](Int128& result, const Stepper::Batch& quotients) {
        for (UInt64 quotient : quotients)
            result += static_cast<Int128>(GET_M(M, R, L1, quotient));
    };
    auto scalar = [&](UInt64 begin, UInt64 finish) {
        Int128 result = 0;
        for (UInt64 denominator = begin;; denominator += Step) {
            const UInt64 quotient = static_cast<UInt64>(n / denominator);
            result += static_cast<Int128>(GET_M(M, R, L1, quotient));
            if (finish - denominator < Step) break;
        }
        return result;
    };

    Int128 result = 0;
    Stepper stepper;
    add(result, stepper.initialize(n, denominators(first)));

    UInt64 consumed = Stepper::Width;
    while (count - consumed >= Stepper::Width) {
        const UInt64 base = first + Step * consumed;
        add(result, stepper.step<Step * Stepper::Width>(
            denominators(base)
        ));
        consumed += Stepper::Width;
    }
    if (consumed < count)
        result += scalar(first + Step * consumed, to);

    sum = result;
    return true;
}

// ============================================================================
// 128-bit S1: fast range using quotient predictor
// ============================================================================

template <ParityMode Mode, typename MIntT>
static inline void sum_fast_range2(
    UInt128 n,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    UInt64 start,
    UInt64 from,
    UInt64 to,
    Int128& sum,
    UInt64& qPrev,
    UInt64& qCur
) {
    constexpr bool isAll   = (Mode == ParityMode::All);
    constexpr bool wantOdd = (Mode == ParityMode::Odd);

    if constexpr (isAll) {
        if (from > to) return;

        UInt64 x = to;

        if (__builtin_expect(qPrev == 0, false)) {
            qPrev = static_cast<UInt64>(n / (x + 1));
            qCur  = static_cast<UInt64>(n / x);
        }

        sum += static_cast<Int128>(GET_M(M, R, start, qCur));

        UInt64 qEst;
        while (x >= from + 1) {
            --x;
            update_quotients(n, x, qCur, qPrev, qEst);

            // Deep prefetch: predict ~10 iterations ahead via linear extrapolation.
            // At large n (>= ~10^22), the M and R arrays exceed L2 cache and every
            // getM call is a DRAM access (~100ns). The quotient predictor's recurrence
            // q_{k} = qCur + k*(qCur - qPrev) lets us issue prefetch hints for
            // free, hiding DRAM latency by starting the fetch ~10 iterations early.
            {
                UInt64 delta = qCur - qPrev;
                UInt64 qPf = qCur + 10 * delta;
                UInt64 pfOff = qPf - start;
                __builtin_prefetch(&R[pfOff], 0, 0);
                __builtin_prefetch(&M[pfOff >> SegmentedMertensSieveCore::STRIDE_LOG], 0, 0);
            }

            sum += static_cast<Int128>(GET_M(M, R, start, qEst));
        }

        update_quotients(n, x-1, qCur, qPrev, qEst);
    } else {
        if (__builtin_expect(from > to, false)) return;

        UInt64 x = to;
        if (((x & 1ULL) != static_cast<UInt64>(wantOdd))) {
            if (x == 0) return;
            --x;
        }

        if (__builtin_expect(from > x, false)) return;

        if (__builtin_expect(qPrev == 0, false)) {
            qPrev = static_cast<UInt64>(n / (x + 2));
            qCur  = static_cast<UInt64>(n / x);
        }

        sum += static_cast<Int128>(GET_M(M, R, start, qCur));

        UInt64 qEst = qCur;
        while (x >= from + 2) {
            x -= 2;
            update_quotients(n, x, qCur, qPrev, qEst);

            {
                UInt64 delta = qCur - qPrev;
                UInt64 qPf = qCur + 10 * delta;
                UInt64 pfOff = qPf - start;
                __builtin_prefetch(&R[pfOff], 0, 0);
                __builtin_prefetch(&M[pfOff >> SegmentedMertensSieveCore::STRIDE_LOG], 0, 0);
            }

            sum += static_cast<Int128>(GET_M(M, R, start, qEst));
        }

        update_quotients(n, x-2, qCur, qPrev, qEst);
    }
}

// ============================================================================
// 128-bit S1: top-level segmented routine
// ============================================================================

template <ParityMode Mode, S1QuotientPolicy Policy, typename MIntT>
static inline Int128 update_S1_128(
    UInt128 n, UInt64 L1, UInt64 start, UInt64 end,
    const MIntT* __restrict M, const Int8* __restrict R,
    UInt64& qPrev, UInt64& qCur
) {
    if (__builtin_expect(end < start, false)) return 0;

    if constexpr (Policy == S1QuotientPolicy::Stepper && !UseDivisionFree) {
        Int128 stepped = 0;
        if (sum_S1_128_stepped<Mode>(
                n, L1, start, end, M, R, stepped
            )) {
            qPrev = 0;
            qCur = 0;
            return stepped;
        }
    }

    Int128 sum = 0;

    const long double dn = (long double)n;
    const UInt64 cbrt2nCeil = std::min(end, UInt64(ceill(cbrtl(2.001L * dn))));

    // Small region: real division, can skip by parity
    const UInt64 smallTo = std::min(end, cbrt2nCeil);
    if (start <= smallTo) {
        bool usedStepper = false;
        if constexpr (Policy == S1QuotientPolicy::Stepper && UseDivisionFree) {
            Int128 stepped = 0;
            usedStepper = sum_S1_128_stepped<Mode>(
                n, L1, start, smallTo, M, R, stepped
            );
            if (usedStepper) sum += stepped;
        }

        if (!usedStepper) {
            if constexpr (Mode == ParityMode::All) {
                #pragma clang loop unroll_count(4)
                for (UInt64 x = start; x <= smallTo; ++x) {
                    sum += static_cast<Int128>(GET_M(M, R, L1, static_cast<UInt64>(n / x)));
                }
            } else if constexpr (Mode == ParityMode::Odd) {
                #pragma clang loop unroll_count(4)
                for (UInt64 x = start | 1ULL; x <= smallTo; x += 2) {
                    sum += static_cast<Int128>(GET_M(M, R, L1, static_cast<UInt64>(n / x)));
                }
            } else {
                #pragma clang loop unroll_count(4)
                for (UInt64 x = (start + 1) & ~1ULL; x <= smallTo; x += 2) {
                    sum += static_cast<Int128>(GET_M(M, R, L1, static_cast<UInt64>(n / x)));
                }
            }
        }
    }

    // Fast region: quotient predictor
    const UInt64 fastFrom = std::max(start, cbrt2nCeil + 1);
    if (fastFrom <= end) {
        sum_fast_range2<Mode>(n, M, R, L1, fastFrom, end, sum, qPrev, qCur);
    } else if constexpr (Policy == S1QuotientPolicy::Stepper) {
        qPrev = 0;
        qCur = 0;
    }

    return sum;
}

// ============================================================================
// 64-bit S1
// ============================================================================

template <ParityMode Mode, typename MIntT>
static inline Int64 update_S1(UInt64 n, UInt64 L1, UInt64 start, UInt64 end,
                       const MIntT* __restrict M, const Int8* __restrict R,
                       const QuotientCache& qCache, UInt64 dCAP) {
    Int64 res = 0;

    if constexpr (UseDivisionFree) {
        // Quotient cache for x <= dCAP, then quotient predictor for x > dCAP
        const UInt64 cacheEnd = std::min(end, dCAP);

        if constexpr (Mode == ParityMode::All) {
            UInt64 x = start;
            #pragma clang loop unroll_count(4)
            for (; x <= cacheEnd; ++x) {
                res += static_cast<Int64>(GET_M(M, R, L1, qCache.quotient(n, x)));
            }
            if (x <= end) {
                UInt64 qPrev = (x - 1 <= dCAP) ? qCache.quotient(n, x - 1) : n / (x - 1);
                UInt64 qCur  = n / x;
                UInt64 qEst;
                res += static_cast<Int64>(GET_M(M, R, L1, qCur));
                for (++x; x <= end; ++x) {
                    update_quotients_64(n, x, qCur, qPrev, qEst);
                    res += static_cast<Int64>(GET_M(M, R, L1, qEst));
                }
            }
        } else if constexpr (Mode == ParityMode::Odd) {
            UInt64 x = start | 1ULL;
            #pragma clang loop unroll_count(4)
            for (; x <= cacheEnd; x += 2) {
                res += static_cast<Int64>(GET_M(M, R, L1, qCache.quotient(n, x)));
            }
            if (x <= end) {
                UInt64 qPrev = (x - 2 <= dCAP) ? qCache.quotient(n, x - 2) : n / (x - 2);
                UInt64 qCur  = n / x;
                UInt64 qEst;
                res += static_cast<Int64>(GET_M(M, R, L1, qCur));
                for (x += 2; x <= end; x += 2) {
                    update_quotients_64(n, x, qCur, qPrev, qEst);
                    res += static_cast<Int64>(GET_M(M, R, L1, qEst));
                }
            }
        } else {
            UInt64 x = (start + 1) & ~1ULL;
            #pragma clang loop unroll_count(4)
            for (; x <= cacheEnd; x += 2) {
                res += static_cast<Int64>(GET_M(M, R, L1, qCache.quotient(n, x)));
            }
            if (x <= end) {
                UInt64 qPrev = (x - 2 <= dCAP) ? qCache.quotient(n, x - 2) : n / (x - 2);
                UInt64 qCur  = n / x;
                UInt64 qEst;
                res += static_cast<Int64>(GET_M(M, R, L1, qCur));
                for (x += 2; x <= end; x += 2) {
                    update_quotients_64(n, x, qCur, qPrev, qEst);
                    res += static_cast<Int64>(GET_M(M, R, L1, qEst));
                }
            }
        }
    } else {
        // Direct division path
        if constexpr (Mode == ParityMode::All) {
            #pragma clang loop unroll_count(4)
            for (UInt64 x = start; x <= end; ++x) {
                res += static_cast<Int64>(GET_M(M, R, L1, n / x));
            }
        } else if constexpr (Mode == ParityMode::Odd) {
            #pragma clang loop unroll_count(4)
            for (UInt64 x = start | 1ULL; x <= end; x += 2) {
                res += static_cast<Int64>(GET_M(M, R, L1, n / x));
            }
        } else {
            #pragma clang loop unroll_count(4)
            for (UInt64 x = (start + 1) & ~1ULL; x <= end; x += 2) {
                res += static_cast<Int64>(GET_M(M, R, L1, n / x));
            }
        }
    }

    return res;
}

// ============================================================================
// apply_S1_updates: dispatch to 64-bit or 128-bit S1 based on argument type
// ============================================================================

template <typename TVal, typename TArg, typename MIntT>
static inline void apply_S1_updates(
    bool doAll,
    TVal& partialValue,
    const TArg& partialArg,
    const UInt64& partialArgDivU,
    const UInt64& kappa,
    const UInt64& kappa2,
    UInt64 L1, UInt64 L2,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache, UInt64 dCAP,
    UInt64* qPrev_odd = nullptr, UInt64* qCur_odd = nullptr,
    UInt64* qPrev_even = nullptr, UInt64* qCur_even = nullptr
) {
    const UInt64 loBase = UInt64(partialArg / (L2 + 1)) + 1;
    const UInt64 hiBase = UInt64(partialArg / L1);

    // doAll: even outer k — no parity trick, sum over all j in
    //        [partialArgDivU+1, kappa].
    // !doAll: odd outer k — parity decomposition splits into odd j
    //         (subtracted) and even j in [kappa+1, kappa2] (added),
    //         roughly halving the work.
    if (doAll) {
        const UInt64 lo = std::max(loBase, partialArgDivU + 1);
        const UInt64 hi = std::min(hiBase, kappa);

        if (lo <= hi) {
            if constexpr (std::is_same_v<TArg, UInt128>)
                partialValue -= update_S1_128<
                    ParityMode::All, S1QuotientPolicy::Stepper
                >(
                    partialArg, L1, lo, hi, M, R, *qPrev_odd, *qCur_odd);
            else
                partialValue -= update_S1<ParityMode::All>(
                    partialArg, L1, lo, hi, M, R, qCache, dCAP);
        }
    } else {
        const UInt64 loOdd = std::max(loBase, partialArgDivU + 1);
        const UInt64 hiOdd = std::min(hiBase, kappa);

        if (loOdd <= hiOdd) {
            if constexpr (std::is_same_v<TArg, UInt128>)
                partialValue -= update_S1_128<
                    ParityMode::Odd, S1QuotientPolicy::Stepper
                >(
                    partialArg, L1, loOdd, hiOdd, M, R, *qPrev_odd, *qCur_odd);
            else
                partialValue -= update_S1<ParityMode::Odd>(
                    partialArg, L1, loOdd, hiOdd, M, R, qCache, dCAP);
        }

        const UInt64 loEven = std::max(loBase, kappa + 1);
        const UInt64 hiEven = std::min(hiBase, kappa2);

        if (loEven <= hiEven) {
            if constexpr (std::is_same_v<TArg, UInt128>)
                partialValue += update_S1_128<
                    ParityMode::Even, S1QuotientPolicy::Stepper
                >(
                    partialArg, L1, loEven, hiEven, M, R, *qPrev_even, *qCur_even);
            else
                partialValue += update_S1<ParityMode::Even>(
                    partialArg, L1, loEven, hiEven, M, R, qCache, dCAP);
        }
    }
}
