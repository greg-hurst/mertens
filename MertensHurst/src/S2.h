#pragma once

// ============================================================================
// S2.h — S2 summation functions for the Mertens function computation.
//
// S2 computes: sum_{k in range} mu(k) * S2_term(floor(n/k))
//
// Two precision levels:
//   64-bit  (update_S2):      for n/k fitting in 64 bits (~96% of work)
//   128-bit (update_S2_128):  for the largest n/k values (~4% of work)
//
// The 128-bit path further splits into:
//   - "small" range (k <= cbrt(2n)): uses actual 128-bit division
//   - "fast" range  (k > cbrt(2n)):  uses the quotient predictor (division-free)
//
// Both paths exploit the 10-mode S2_term system to skip vanishing terms,
// and iterate only over odd k coprime to 6 (since mu(k)=0 for even k and k≡3 mod 6).
// ============================================================================

#include "types.h"
#include "QuotientCache.h"
#include "QuotientPredictor.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

// ============================================================================
// S2_term: 10-mode quotient transformation for S2 summation.
//
// S2_term<Mode>(q) computes the number of integers in [1, q] that survive
// a particular subset of the inclusion-exclusion cancellation, parameterized
// by Mode in {0..9}. Higher modes exploit more cancellation (fewer surviving
// terms), and are used in sub-ranges where those cancellations are guaranteed.
// ============================================================================

template<int Mode, typename T>
static inline T S2_term(T q) {

    static_assert(
        (std::is_integral_v<T> && std::is_signed_v<T>) || std::is_same_v<T, __int128_t>,
        "T must be a signed integral type"
    );

    if constexpr (Mode == 0) {
        return q;
    } else if constexpr (Mode == 1) {
        return (q + T(1)) >> 1;
    } else if constexpr (Mode == 2) {
        return (q & T(1));
    } else if constexpr (Mode == 3) {
        return (q >> 2) + (q & T(1));
    } else if constexpr (Mode == 4) {
        return ((q + T(1)) >> 1) - q/3;
    } else if constexpr (Mode == 5) {
        return ((q + T(1)) >> 1) - ((q/3 + T(1)) >> 1);
    } else if constexpr (Mode == 6) {
        return (q >> 2) + (q & T(1)) - q/3;
    } else if constexpr (Mode == 7) {
        return (q >> 2) + (q & T(1)) - (q + T(3))/6;
    } else if constexpr (Mode == 8) {
        return (q >> 2) + (q & T(1)) - ((q % 6) >= 3);
    } else if constexpr (Mode == 9) {
        return (q >> 2) + (q & T(1)) - ((q/3) >> 2) - ((q/3) & T(1));
    }
}

template<int Mode>
struct S2LegacyModeEvaluator {
    template<typename T>
    static inline T eval(T q) { return S2_term<Mode>(q); }
};

// ============================================================================
// Current Q_o=2 / Q_i=6 interval dispatcher.
//
// This owns only interval construction. The callback receives a compile-time
// mode tag plus an inclusive interval; the wheel iterator and mode evaluator
// remain separate. Keeping the mode in the type preserves the fully
// specialized kernels used before this refactor.
// ============================================================================

template<int Mode>
using S2ModeTag = std::integral_constant<int, Mode>;

template<bool Half, typename Callback>
static inline void dispatch_S2_current(
    UInt64 x1,
    UInt64 x2,
    UInt64 nu,
    UInt64 nu2,
    Callback&& callback
) {
    auto interval = [&](auto mode, UInt64 lo, UInt64 hi) {
        lo = std::max(x1, lo);
        hi = std::min(x2, hi);
        if (lo <= hi) callback(mode, lo, hi);
    };

    if constexpr (Half) {
        interval(S2ModeTag<9>{}, 0,                (nu2 >> 1) / 3);
        interval(S2ModeTag<8>{}, (nu2 >> 1) / 3+1,(nu  >> 1) / 3);
        interval(S2ModeTag<7>{}, (nu  >> 1) / 3+1, nu2 / 3);
        interval(S2ModeTag<6>{},  nu2 / 3 + 1,     nu / 3);
        interval(S2ModeTag<3>{},  nu / 3 + 1,      nu2 >> 1);
        interval(S2ModeTag<2>{}, (nu2 >> 1) + 1,   nu >> 1);
        interval(S2ModeTag<1>{}, (nu  >> 1) + 1,   nu2);
        interval(S2ModeTag<0>{},  nu2 + 1,          ~UInt64(0));
    } else {
        interval(S2ModeTag<5>{}, 0,               (nu >> 1) / 3);
        interval(S2ModeTag<4>{}, (nu >> 1) / 3+1,  nu / 3);
        interval(S2ModeTag<1>{},  nu / 3 + 1,      nu >> 1);
        interval(S2ModeTag<0>{}, (nu >> 1) + 1,   ~UInt64(0));
    }
}

// ============================================================================
// 128-bit S2: small/division range (odd k coprime to 6)
// ============================================================================

template<typename Evaluator>
static inline void sum_small_range_S2_128_eval(
    const UInt128& n,
    const Int8* __restrict Mu,
    UInt64 L1,
    UInt64 from,
    UInt64 to,
    Int128& sum
) {
    if (from > to) return;

    UInt64 i = from + ((from & 1ULL) == 0);
    i += (i % 6 == 3) ? 2 : 0;

    if (i <= to && i % 6 == 5) {
        sum += Mu[i - L1] * Evaluator::template eval<Int128>(
            static_cast<Int128>(n / i)
        );
        i += 2;
    }

    if (i > to) return;

    // Unrolled by 36: the period of integers coprime to 6 is 6, with
    // qualifying offsets {0,4,6,10,12,16,18,22,24,28,30,34} (12 per period).
    // 36 = 6*6 covers one full period starting from i ≡ 1 mod 6.
    while (i + 35 <= to) {
        const UInt64 j = i - L1;

        #define S2_ADD(off) do { \
            const Int8 m = Mu[j + (off)]; \
            if (__builtin_expect(m != 0, true)) { \
                const Int128 q = static_cast<Int128>(n / (i + (off))); \
                sum += static_cast<Int128>(m) * Evaluator::template eval<Int128>(q); \
            } \
        } while (0)

        S2_ADD( 0);  S2_ADD( 4);  S2_ADD( 6);  S2_ADD(10);
        S2_ADD(12);  S2_ADD(16);  S2_ADD(18);  S2_ADD(22);
        S2_ADD(24);  S2_ADD(28);  S2_ADD(30);  S2_ADD(34);

        #undef S2_ADD

        i += 36;
    }

    // Tail (odd-only)
    for (; i <= to; i += 2) {
        if (i % 6 != 3) {
            const Int8 m = Mu[i - L1];
            if (__builtin_expect(m != 0, true)) {
                const Int128 q = static_cast<Int128>(n / i);
                sum += static_cast<Int128>(m) * Evaluator::template eval<Int128>(q);
            }
        }
    }
}

// ============================================================================
// 128-bit S2: fast/predictor range (odd k coprime to 6, division-free)
// ============================================================================

template<typename Evaluator>
static inline void sum_fast_range_S2_128_eval(
    const UInt128& n,
    const Int8* __restrict Mu,
    UInt64 start,
    UInt64 from,
    UInt64 to,
    Int128& sum
) {
    if (from > to) return;

    if ((from & 1ULL) == 0) ++from;
    if ((to   & 1ULL) == 0) --to;

    if (from % 6 == 3) from += 2;
    if (to % 6 == 3) to -= 2;

    if (from > to) return;

    UInt64 x = from;

    UInt64 qPrev = static_cast<UInt64>(n / (x - 2));
    UInt64 qCur  = static_cast<UInt64>(n / x);
    UInt64 qEst;

    const Int8 m = Mu[x - start];
    if (m) {
        const Int64 t = Evaluator::template eval<Int64>(static_cast<Int64>(qCur));
        sum += static_cast<Int64>(m) * t;
    }

    if (x % 6 == 1) {
        x += 4;
        update_quotients_coprime6iter<true>(n, x, qCur, qPrev, qEst);

        if (x <= to) {
            Int8 m = Mu[x - start];
            if (m) {
                Int64 t = Evaluator::template eval<Int64>(static_cast<Int64>(qEst));
                sum += static_cast<Int64>(m) * t;
            }
        }
    }

    while (x + 6 <= to) {
        x += 2;
        update_quotients_coprime6iter<false>(n, x, qCur, qPrev, qEst);

        {
            Int8 m = Mu[x - start];
            if (m) {
                Int64 t = Evaluator::template eval<Int64>(static_cast<Int64>(qEst));
                sum += static_cast<Int64>(m) * t;
            }
        }

        x += 4;
        update_quotients_coprime6iter<true>(n, x, qCur, qPrev, qEst);

        {
            Int8 m = Mu[x - start];
            if (m) {
                Int64 t = Evaluator::template eval<Int64>(static_cast<Int64>(qEst));
                sum += static_cast<Int64>(m) * t;
            }
        }
    }

    while (x + 2 <= to) {
        x += 2;
        update_quotients_coprime6iter<false>(n, x, qCur, qPrev, qEst);

        if (x % 6 != 3) {
            Int8 m = Mu[x - start];
            if (m) {
                Int64 t = Evaluator::template eval<Int64>(static_cast<Int64>(qCur));
                sum += static_cast<Int64>(m) * t;
            }
        }
    }
}

// ============================================================================
// 128-bit S2: dispatcher combining small + fast ranges
// ============================================================================

template<typename Evaluator>
static inline void update_S2_128_wheel6(
    UInt128 n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    const UInt64& cbrt2nCeil,
    Int128& sum
) {
    if (lo > hi) return;

    if ((lo & 1ULL) == 0) ++lo;
    if ((hi & 1ULL) == 0) --hi;

    if (lo % 6 == 3) lo += 2;
    if (hi % 6 == 3) hi -= 2;

    if (lo > hi) return;

    const UInt64 smallTo = std::min(hi, cbrt2nCeil);
    UInt64 fastFrom = std::max(lo, cbrt2nCeil + 1);

    if (lo <= smallTo)
        sum_small_range_S2_128_eval<Evaluator>(n, Mu, L1, lo, smallTo, sum);

    if (fastFrom <= hi)
        sum_fast_range_S2_128_eval<Evaluator>(n, Mu, L1, fastFrom, hi, sum);
}

template<int Mode>
static inline void update_S2_128_impl(
    UInt128 n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    const UInt64& cbrt2nCeil,
    Int128& sum
) {
    update_S2_128_wheel6<S2LegacyModeEvaluator<Mode>>(
        n, L1, lo, hi, Mu, cbrt2nCeil, sum
    );
}

// ============================================================================
// 128-bit S2: public entry point
//
// Half=true:  parity decomposition — odd-only sum over the main range plus
//             a short even tail, giving 8 mode boundaries from the
//             interleaving of nu and nu2 thresholds.
// Half=false: no parity split (even k), 4-mode split using nu boundaries only.
// ============================================================================

template<bool Half>
static inline Int128 update_S2_128(
    UInt128 n,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt64 nu,
    UInt64 nu2
) {
    if (x1 > x2) return 0;

    Int128 sum = 0;

    const long double dn = (long double)n;
    const UInt64 cbrt2nCeil = (UInt64)ceill(cbrtl(2.001L * dn));

    dispatch_S2_current<Half>(x1, x2, nu, nu2,
        [&](auto mode, UInt64 lo, UInt64 hi) {
            constexpr int Mode = decltype(mode)::value;
            update_S2_128_impl<Mode>(
                n, L1, lo, hi, Mu, cbrt2nCeil, sum
            );
        });
    return sum;
}

// ============================================================================
// 64-bit S2: inner implementation (single mode)
// ============================================================================

template<typename Evaluator>
static inline Int64 update_S2_wheel6(UInt64 n, UInt64 L1, UInt64 x1, UInt64 x2,
                                     const Int8* __restrict Mu,
                                     const QuotientCache& qCache, UInt64 dCAP) {
    if (x1 > x2) return 0;

    Int64 res = 0;

    // Ensure we start at 1 or 5 mod 6.
    UInt64 i = x1 | 1ULL;
    i += (i % 3 == 0) ? 2 : 0;

    // Quotient helpers: compute floor(n/d) as Int64 for signed S2_term arithmetic.
    // Quotients are always non-negative and bounded by n < 10^18 < Int64 max.
    auto sdiv = [n](UInt64 d) -> Int64 { return static_cast<Int64>(n / d); };

    // The unrolled segments below multiply by Mu[..] unconditionally, even
    // when mu == 0: among integers coprime to 6 only ~9% have mu == 0, and a
    // skip branch would mispredict at that rate — costing more than the few
    // saved divisions. The 128-bit path DOES branch on mu != 0, correctly:
    // its software divisions are ~10x more expensive, flipping the trade-off.

    if constexpr (UseDivisionFree) {
        // Granlund-Montgomery quotient cache path: use cache when d <= dCAP, else hardware div
        auto cdiv = [&](UInt64 d) -> Int64 {
            return static_cast<Int64>(__builtin_expect(d <= dCAP, true) ? qCache.quotient(n, d) : n / d);
        };

        if (i <= x2 && i % 6 == 5) {
            res += Mu[i - L1] * Evaluator::template eval<Int64>(cdiv(i));
            i += 2;
        }

        if (__builtin_expect(i > x2, false)) { goto done; }

        {
            // Cache region: i + 34 <= dCAP (all 12 divisions in unrolled segment use cache)
            const UInt64 cacheXcap = (dCAP >= 34) ? std::min((x2 >= 35 ? x2 - 35 : (UInt64)0), dCAP - 34) : 0;
            auto cquo = [&](UInt64 d) -> Int64 { return static_cast<Int64>(qCache.quotient(n, d)); };

            for (; i <= cacheXcap; i += 36) {
                const UInt64 j = i - L1;

                res += Mu[j +  0] * Evaluator::template eval<Int64>(cquo(i +  0))
                    +  Mu[j +  4] * Evaluator::template eval<Int64>(cquo(i +  4))
                    +  Mu[j +  6] * Evaluator::template eval<Int64>(cquo(i +  6))
                    +  Mu[j + 10] * Evaluator::template eval<Int64>(cquo(i + 10))
                    +  Mu[j + 12] * Evaluator::template eval<Int64>(cquo(i + 12))
                    +  Mu[j + 16] * Evaluator::template eval<Int64>(cquo(i + 16))
                    +  Mu[j + 18] * Evaluator::template eval<Int64>(cquo(i + 18))
                    +  Mu[j + 22] * Evaluator::template eval<Int64>(cquo(i + 22))
                    +  Mu[j + 24] * Evaluator::template eval<Int64>(cquo(i + 24))
                    +  Mu[j + 28] * Evaluator::template eval<Int64>(cquo(i + 28))
                    +  Mu[j + 30] * Evaluator::template eval<Int64>(cquo(i + 30))
                    +  Mu[j + 34] * Evaluator::template eval<Int64>(cquo(i + 34));
            }

            // Division region: beyond cache range
            const UInt64 xcap = (x2 >= 35) ? (x2 - 35) : 0;
            for (; i <= xcap; i += 36) {
                const UInt64 j = i - L1;

                res += Mu[j +  0] * Evaluator::template eval<Int64>(sdiv(i +  0)) + Mu[j +  4] * Evaluator::template eval<Int64>(sdiv(i +  4))
                    +  Mu[j +  6] * Evaluator::template eval<Int64>(sdiv(i +  6)) + Mu[j + 10] * Evaluator::template eval<Int64>(sdiv(i + 10))
                    +  Mu[j + 12] * Evaluator::template eval<Int64>(sdiv(i + 12)) + Mu[j + 16] * Evaluator::template eval<Int64>(sdiv(i + 16))
                    +  Mu[j + 18] * Evaluator::template eval<Int64>(sdiv(i + 18)) + Mu[j + 22] * Evaluator::template eval<Int64>(sdiv(i + 22))
                    +  Mu[j + 24] * Evaluator::template eval<Int64>(sdiv(i + 24)) + Mu[j + 28] * Evaluator::template eval<Int64>(sdiv(i + 28))
                    +  Mu[j + 30] * Evaluator::template eval<Int64>(sdiv(i + 30)) + Mu[j + 34] * Evaluator::template eval<Int64>(sdiv(i + 34));
            }

            // Tail
            for (; i <= x2; i += (6 - 2*(i % 3))) {
                res += Mu[i - L1] * Evaluator::template eval<Int64>(cdiv(i));
            }
        }

    done:
        (void)0;

    } else {
        // Direct division path
        if (i <= x2 && i % 6 == 5) {
            res += Mu[i - L1] * Evaluator::template eval<Int64>(sdiv(i));
            i += 2;
        }

        if (__builtin_expect(i > x2, false)) return res;

        const UInt64 xcap = (x2 >= 35) ? (x2 - 35) : 0;

        for (; i <= xcap; i += 36) {
            const UInt64 j = i - L1;

            res += Mu[j +  0] * Evaluator::template eval<Int64>(sdiv(i +  0)) + Mu[j +  4] * Evaluator::template eval<Int64>(sdiv(i +  4))
                +  Mu[j +  6] * Evaluator::template eval<Int64>(sdiv(i +  6)) + Mu[j + 10] * Evaluator::template eval<Int64>(sdiv(i + 10))
                +  Mu[j + 12] * Evaluator::template eval<Int64>(sdiv(i + 12)) + Mu[j + 16] * Evaluator::template eval<Int64>(sdiv(i + 16))
                +  Mu[j + 18] * Evaluator::template eval<Int64>(sdiv(i + 18)) + Mu[j + 22] * Evaluator::template eval<Int64>(sdiv(i + 22))
                +  Mu[j + 24] * Evaluator::template eval<Int64>(sdiv(i + 24)) + Mu[j + 28] * Evaluator::template eval<Int64>(sdiv(i + 28))
                +  Mu[j + 30] * Evaluator::template eval<Int64>(sdiv(i + 30)) + Mu[j + 34] * Evaluator::template eval<Int64>(sdiv(i + 34));
        }

        // Tail
        for (; i <= x2; i += (6 - 2*(i % 3))) {
            res += Mu[i - L1] * Evaluator::template eval<Int64>(sdiv(i));
        }
    }

    return res;
}

template<int Mode>
static inline Int64 update_S2_impl(UInt64 n, UInt64 L1, UInt64 x1, UInt64 x2,
                                   const Int8* __restrict Mu,
                                   const QuotientCache& qCache, UInt64 dCAP) {
    return update_S2_wheel6<S2LegacyModeEvaluator<Mode>>(
        n, L1, x1, x2, Mu, qCache, dCAP
    );
}

// ============================================================================
// 64-bit S2: public entry point
// ============================================================================

template <bool Half>
static inline Int64 update_S2(UInt64 n, UInt64 L1, UInt64 x1, UInt64 x2,
                       const Int8* __restrict Mu, UInt64 nu, UInt64 nu2,
                       const QuotientCache& qCache, UInt64 dCAP) {
    Int64 sum = 0;
    dispatch_S2_current<Half>(x1, x2, nu, nu2,
        [&](auto mode, UInt64 lo, UInt64 hi) {
            constexpr int Mode = decltype(mode)::value;
            sum += update_S2_impl<Mode>(
                n, L1, lo, hi, Mu, qCache, dCAP
            );
        });
    return sum;
}
