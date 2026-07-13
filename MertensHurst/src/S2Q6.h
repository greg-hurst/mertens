#pragma once

// ============================================================================
// S2Q6.h — exact outer-Q6 S2 dispatch.
//
// The static mode table provides canonical coefficients and the cutoff pair
// removed after each symbolic cell. Production computes every integer cutoff
// exactly. The normal path verifies that the table order is
// nondecreasing (merging ties by skipping empty intervals); any floor-induced
// inversion takes a deliberately slow sorted coefficient-vector fallback.
// Thus specialization is fast for normal inputs without making exactness rely
// on an asymptotic ordering assumption.
// ============================================================================

#include "S2.h"
#include "S2Q6Modes.h"

#include <algorithm>
#include <array>

struct S2Q6Spec {
    using Mode = S2Q6Modes::Mode;
    static constexpr std::size_t BasisCount = S2Q6Modes::kBasisCount;
    static constexpr UInt64 InnerWheel = 6;

    template<Mode M>
    struct Evaluator {
        template<typename T>
        static inline T eval(T q) {
            return S2Q6Modes::Term<M, T>::eval(q);
        }
    };

    static inline Mode mode(std::size_t cell) {
        return S2Q6Modes::kRawCellModes[cell];
    }
    static inline UInt64 outerDivisor(std::size_t cell) {
        return S2Q6Modes::kCutoffOuterDivisors[cell];
    }
    static inline UInt64 innerDivisor(std::size_t cell) {
        return S2Q6Modes::kCutoffInnerDivisors[cell];
    }
    static inline Int8 removedSign(std::size_t cell) {
        return S2Q6Modes::kRemovedSigns[cell];
    }
    static inline std::size_t removedBasis(std::size_t cell) {
        return S2Q6Modes::kRemovedBasisIndices[cell];
    }
    static inline std::size_t classBegin(std::size_t outerClass) {
        return S2Q6Modes::kOuterClassCellOffsets[outerClass];
    }
    static inline std::size_t classEnd(std::size_t outerClass) {
        return S2Q6Modes::kOuterClassCellOffsets[outerClass + 1];
    }
    static inline Int16 coefficient(Mode modeValue, std::size_t basisIndex) {
        return static_cast<Int16>(S2Q6Modes::kCoefficients[
            static_cast<std::size_t>(modeValue)
        ][basisIndex]);
    }
    static inline UInt64 denominator(std::size_t basisIndex) {
        return S2Q6Modes::kDenominators[basisIndex];
    }
    template<typename Callback>
    static inline void dispatchMode(Mode modeValue, Callback&& callback) {
        S2Q6Modes::dispatchMode(modeValue, callback);
    }
};

template<typename Spec>
using S2Q6DynamicCoefficients = std::array<Int16, Spec::BasisCount>;

namespace S2Q6Detail {

static inline UInt64 nuForOuterDivisor(
    UInt64 divisor,
    UInt64 nu1,
    UInt64 nu2,
    UInt64 nu3,
    UInt64 nu6
) {
    switch (divisor) {
        case 1: return nu1;
        case 2: return nu2;
        case 3: return nu3;
        case 6: return nu6;
    }
    return 0;
}

template<typename Spec, typename SpecializedCallback, typename DynamicCallback>
static inline void dispatchExact(
    UInt32 outerClass,
    UInt64 x1,
    UInt64 x2,
    UInt64 nu1,
    UInt64 nu2,
    UInt64 nu3,
    UInt64 nu6,
    SpecializedCallback&& specializedCallback,
    DynamicCallback&& dynamicCallback
) {
    if (x1 > x2) return;

    const std::size_t begin = Spec::classBegin(outerClass);
    const std::size_t end = Spec::classEnd(outerClass);
    UInt64 previous = 0;
    bool ordered = true;
    for (std::size_t cell = begin; cell < end; ++cell) {
        const UInt64 cutoff = nuForOuterDivisor(
            Spec::outerDivisor(cell), nu1, nu2, nu3, nu6
        ) / Spec::innerDivisor(cell);
        if (cutoff < previous) ordered = false;
        previous = cutoff;
    }

    if (__builtin_expect(ordered, true)) {
        previous = 0;
        for (std::size_t cell = begin; cell < end; ++cell) {
            const UInt64 cutoff = nuForOuterDivisor(
                Spec::outerDivisor(cell), nu1, nu2, nu3, nu6
            ) / Spec::innerDivisor(cell);
            const UInt64 lo = std::max(x1, previous + 1);
            const UInt64 hi = std::min(x2, cutoff);
            if (lo <= hi) {
                Spec::dispatchMode(Spec::mode(cell), [&](auto modeTag) {
                    specializedCallback(modeTag, lo, hi);
                });
            }
            previous = cutoff;
        }
        return;
    }

    struct Event {
        UInt64 cutoff;
        Int8 sign;
        UInt8 basis;
    };
    std::array<Event, 32> events{};
    const std::size_t count = end - begin;
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t cell = begin + index;
        events[index] = Event{
            nuForOuterDivisor(
                Spec::outerDivisor(cell), nu1, nu2, nu3, nu6
            ) / Spec::innerDivisor(cell),
            Spec::removedSign(cell),
            static_cast<UInt8>(Spec::removedBasis(cell)),
        };
    }
    std::sort(events.begin(), events.begin() + count,
              [](const Event& left, const Event& right) {
                  return left.cutoff < right.cutoff;
              });

    S2Q6DynamicCoefficients<Spec> coefficients{};
    const typename Spec::Mode initialMode = Spec::mode(begin);
    for (std::size_t basis = 0; basis < Spec::BasisCount; ++basis)
        coefficients[basis] = Spec::coefficient(initialMode, basis);

    previous = 0;
    std::size_t index = 0;
    while (index < count) {
        const UInt64 cutoff = events[index].cutoff;
        const UInt64 lo = std::max(x1, previous + 1);
        const UInt64 hi = std::min(x2, cutoff);
        if (lo <= hi) dynamicCallback(coefficients, lo, hi);

        do {
            coefficients[events[index].basis] -= events[index].sign;
            ++index;
        } while (index < count && events[index].cutoff == cutoff);
        previous = cutoff;
    }
}

template<typename Spec, typename T>
static inline T evalDynamic(
    const S2Q6DynamicCoefficients<Spec>& coefficients,
    T quotient
) {
    T result = 0;
    for (std::size_t basis = 0; basis < Spec::BasisCount; ++basis)
        result += T(coefficients[basis]) * (quotient / T(Spec::denominator(basis)));
    return result;
}

template<typename Spec>
static inline bool innerWheelAccepts(UInt64 value) {
    static_assert(Spec::InnerWheel == 6, "only the wheel-6 backend is supported");
    return (value & 1ULL) != 0 && value % 3 != 0;
}

template<typename Spec>
static inline Int64 updateDynamic64(
    UInt64 n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    const S2Q6DynamicCoefficients<Spec>& coefficients
) {
    Int64 result = 0;
    for (UInt64 value = lo;; ++value) {
        if (innerWheelAccepts<Spec>(value)) {
            const Int8 mu = Mu[value - L1];
            if (mu != 0) {
                result += Int64(mu) * evalDynamic<Spec, Int64>(
                    coefficients, static_cast<Int64>(n / value)
                );
            }
        }
        if (value == hi) break;
    }
    return result;
}

template<typename Spec>
static inline Int128 updateDynamic128(
    const UInt128& n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    const S2Q6DynamicCoefficients<Spec>& coefficients
) {
    Int128 result = 0;
    for (UInt64 value = lo;; ++value) {
        if (innerWheelAccepts<Spec>(value)) {
            const Int8 mu = Mu[value - L1];
            if (mu != 0) {
                result += Int128(mu) * evalDynamic<Spec, Int128>(
                    coefficients, static_cast<Int128>(n / value)
                );
            }
        }
        if (value == hi) break;
    }
    return result;
}

} // namespace S2Q6Detail

template<typename Spec>
static inline Int64 update_S2_q6(
    UInt64 n,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt32 outerClass,
    UInt64 nu1,
    UInt64 nu2,
    UInt64 nu3,
    UInt64 nu6,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    static_assert(Spec::InnerWheel == 6, "wheel-6 updater requires Q_i=6");
    Int64 sum = 0;
    S2Q6Detail::dispatchExact<Spec>(
        outerClass, x1, x2, nu1, nu2, nu3, nu6,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr typename Spec::Mode Mode = decltype(modeTag)::value;
            using Evaluator = typename Spec::template Evaluator<Mode>;
            sum += update_S2_wheel6<Evaluator>(
                n, L1, lo, hi, Mu, qCache, dCAP
            );
        },
        [&](const S2Q6DynamicCoefficients<Spec>& coefficients,
            UInt64 lo, UInt64 hi) {
            sum += S2Q6Detail::updateDynamic64<Spec>(
                n, L1, lo, hi, Mu, coefficients
            );
        }
    );
    return sum;
}

template<typename Spec>
static inline Int128 update_S2_q6_128(
    const UInt128& n,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt32 outerClass,
    UInt64 nu1,
    UInt64 nu2,
    UInt64 nu3,
    UInt64 nu6
) {
    static_assert(Spec::InnerWheel == 6, "wheel-6 updater requires Q_i=6");
    Int128 sum = 0;
    const UInt64 cbrt2nCeil = static_cast<UInt64>(
        ceill(cbrtl(2.001L * static_cast<long double>(n)))
    );
    const UInt64 step6Boundary = quotient_predictor_first_unit_curvature<6>(n);
    S2Q6Detail::dispatchExact<Spec>(
        outerClass, x1, x2, nu1, nu2, nu3, nu6,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr typename Spec::Mode Mode = decltype(modeTag)::value;
            using Evaluator = typename Spec::template Evaluator<Mode>;
            update_S2_128_wheel6<Evaluator>(
                n, L1, lo, hi, Mu, cbrt2nCeil, step6Boundary, sum
            );
        },
        [&](const S2Q6DynamicCoefficients<Spec>& coefficients,
            UInt64 lo, UInt64 hi) {
            sum += S2Q6Detail::updateDynamic128<Spec>(
                n, L1, lo, hi, Mu, coefficients
            );
        }
    );
    return sum;
}
