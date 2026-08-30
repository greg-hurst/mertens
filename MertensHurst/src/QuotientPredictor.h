#pragma once

// ============================================================================
// QuotientPredictor.h — Division-free quotient estimation via recurrence.
//
// Given n and a sequence of equally spaced x values, we predict floor(n/x)
// from the two previous quotients using:
//
//   qEst = 2*qCur - qPrev + e
//
// where e is a small correction verified by a single multiply.
// This avoids expensive division in the inner loops.
//
// 128-bit variants (n is UInt128):
//   update_quotients()              — stride-1 (every x)
//   update_quotients_coprime6iter() — stride-2/4 coprime-to-6 iteration
//
// 64-bit variants (n is UInt64, used with QuotientCache path):
//   update_quotients_64()              — stride-1 (every x)
//   update_quotients_coprime6iter_64() — stride-2/4 coprime-to-6 iteration
//
// Proved fixed-stride variants (UInt64 or UInt128, either direction):
//   update_quotients_fixed_stride<Step, DenominatorsDecreasing>()
// ============================================================================

#include "types.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <type_traits>

// For equally spaced denominators, when the real second difference of y/x is
// strictly less than one, the correction to
// 2*qCur-qPrev is one of {-1, 0, 1, 2}.  This predicate checks that
// condition exactly without forming either potentially overflowing product.
template<UInt64 Step, typename TArg>
static inline bool quotient_predictor_has_unit_curvature(
    const TArg& n,
    UInt64 middle
) {
    static_assert(Step > 0, "quotient-predictor stride must be positive");
    static_assert(
        std::is_same_v<TArg, UInt64> || std::is_same_v<TArg, UInt128>,
        "unsupported quotient-predictor numerator type"
    );

    if (middle <= Step) return false;

    constexpr UInt64 scale = 2 * Step * Step;
    const UInt128 wideN = static_cast<UInt128>(n);
    const UInt128 wideMiddle = static_cast<UInt128>(middle);
    const UInt128 squareMinusStep = wideMiddle * middle - Step * Step;

    // Compare scale*n < middle*squareMinusStep.  Write n=q*middle+r and
    // squareMinusStep=a*scale+b; only the q==a boundary needs a product.
    const UInt128 quotient = wideN / middle;
    const UInt64 remainder = static_cast<UInt64>(wideN % middle);
    const UInt128 boundaryQuotient = squareMinusStep / scale;
    const UInt64 boundaryRemainder = static_cast<UInt64>(squareMinusStep % scale);

    if (quotient < boundaryQuotient) return true;
    if (quotient > boundaryQuotient) return false;

    return static_cast<UInt128>(scale) * remainder
         < static_cast<UInt128>(boundaryRemainder) * middle;
}

// Smallest middle denominator for which the bounded correction contract is
// valid.  The floating-point cube root is only a starting estimate; the exact
// predicate above determines the returned boundary.
template<UInt64 Step, typename TArg>
static inline UInt64 quotient_predictor_first_unit_curvature(
    const TArg& n
) {
    static_assert(Step > 0, "quotient-predictor stride must be positive");

    const long double scale = 2.0L * static_cast<long double>(Step * Step);
    UInt64 middle = std::max<UInt64>(
        Step + 1,
        static_cast<UInt64>(std::ceil(std::cbrt(scale * static_cast<long double>(n))))
    );

    while (!quotient_predictor_has_unit_curvature<Step>(n, middle))
        ++middle;
    while (middle > Step + 1
           && quotient_predictor_has_unit_curvature<Step>(n, middle - 1))
        --middle;

    return middle;
}

// Fixed-stride predictor with a proved, bounded correction path.  The caller
// must use it only in the unit-curvature range above.  DenominatorsDecreasing
// selects whether target x follows middle x+Step or middle x-Step.
template<UInt64 Step, bool DenominatorsDecreasing, typename TArg>
static inline void update_quotients_fixed_stride(
    const TArg& n,
    UInt64 x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    static_assert(Step > 0, "quotient-predictor stride must be positive");
    static_assert(
        std::is_same_v<TArg, UInt64> || std::is_same_v<TArg, UInt128>,
        "unsupported quotient-predictor numerator type"
    );

#ifndef NDEBUG
    const UInt64 middle = DenominatorsDecreasing ? x + Step : x - Step;
    assert(quotient_predictor_has_unit_curvature<Step>(n, middle));
    assert(qCur <= (~UInt64(0) >> 1));
    assert(qPrev <= 2 * qCur);
#endif

    // Contract: callers keep qCur below 2^63 and the linear estimate
    // non-negative.  The debug assertions above enforce this without adding
    // range checks to the release hot path.
    qEst = 2 * qCur - qPrev;
    TArg product = static_cast<TArg>(qEst) * x;

    if (__builtin_expect(product > n, false)) {
        --qEst;
    } else {
        TArg remainder = n - product;
        if (__builtin_expect(remainder >= x, false)) {
            ++qEst;
            remainder -= x;
            if (__builtin_expect(remainder >= x, false))
                ++qEst;
        }
    }

#ifndef NDEBUG
    assert(qEst == static_cast<UInt64>(n / x));
#endif

    qPrev = qCur;
    qCur = qEst;
}

static inline void update_quotients(
    const UInt128& n,
    const UInt64& x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    qEst = 2*qCur - qPrev;

    UInt128 mul = static_cast<UInt128>(qEst) * x;

    if (__builtin_expect(mul <= n, true)) {
        if (__builtin_expect(mul + x <= n, false)) {
            do {
                ++qEst;
                mul += x;
            } while (mul + x <= n);
        }
    } else {
        --qEst;
    }

    qPrev = qCur;
    qCur  = qEst;
}

// DoubleJump=true: advance by 4 (coprime-to-6 skip pattern: +4, +2, +4, +2, ...)
// DoubleJump=false: advance by 2
template<bool DoubleJump>
static inline void update_quotients_coprime6iter(
    const UInt128& n,
    const UInt64& x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    if constexpr (DoubleJump) {
        qEst = 2*qCur - qPrev;

        qPrev = qCur;
        qCur  = qEst;
    }

    qEst = 2*qCur - qPrev;

    UInt128 mul = static_cast<UInt128>(qEst) * x;

    if (__builtin_expect(mul <= n, true)) {
        if (__builtin_expect(mul + x <= n, false)) {
            do {
                ++qEst;
                mul += x;
            } while (mul + x <= n);
        }
    } else {
        do {
            --qEst;
            mul -= x;
        } while (mul > n);
    }

    qPrev = qCur;
    qCur  = qEst;
}

// ============================================================================
// 64-bit variants: same recurrence, but n fits in 64 bits.
// For the 64-bit path, partial_args < 10^18 < 2^60, so qEst * x fits in
// 64 bits (avoiding expensive 128-bit arithmetic). Verification uses a single
// 64-bit multiply.
// ============================================================================

static inline void update_quotients_64(
    const UInt64 n,
    const UInt64 x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    qEst = 2*qCur - qPrev;

    UInt64 mul = qEst * x;

    if (__builtin_expect(mul <= n, true)) {
        if (__builtin_expect(mul + x <= n, false)) {
            do {
                ++qEst;
                mul += x;
            } while (mul + x <= n);
        }
    } else {
        --qEst;
    }

    qPrev = qCur;
    qCur  = qEst;
}

template<bool DoubleJump>
static inline void update_quotients_coprime6iter_64(
    const UInt64 n,
    const UInt64 x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    if constexpr (DoubleJump) {
        qEst = 2*qCur - qPrev;

        qPrev = qCur;
        qCur  = qEst;
    }

    qEst = 2*qCur - qPrev;

    UInt64 mul = qEst * x;

    if (__builtin_expect(mul <= n, true)) {
        if (__builtin_expect(mul + x <= n, false)) {
            do {
                ++qEst;
                mul += x;
            } while (mul + x <= n);
        }
    } else {
        do {
            --qEst;
            mul -= x;
        } while (mul > n);
    }

    qPrev = qCur;
    qCur  = qEst;
}
