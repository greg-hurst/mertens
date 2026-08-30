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
#include <cassert>
#include <limits>
#include <type_traits>

#ifndef MERTENSHURST_COHERENT_PERIOD36
#define MERTENSHURST_COHERENT_PERIOD36 0
#endif
static_assert(
    MERTENSHURST_COHERENT_PERIOD36 == 0
    || MERTENSHURST_COHERENT_PERIOD36 == 1,
    "MERTENSHURST_COHERENT_PERIOD36 must be 0 or 1"
);

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

    // Every generated denominator divides 36.  On the dominant UInt128
    // fixed-stride path, the more involved modes are cheaper as an exact
    // period-36 quasi-polynomial than as several independent constant
    // quotients.  Keep the three simple modes in their existing form.
    template<Mode M>
    struct FixedStrideEvaluator {
        static constexpr bool UsePeriod36 =
            M != Mode::M15 && M != Mode::M24 && M != Mode::M25;

        inline static constexpr bool PeriodIsValid = [] {
            constexpr std::size_t modeIndex = static_cast<std::size_t>(M);
            for (std::size_t basis = 0; basis < BasisCount; ++basis) {
                if (36 % S2Q6Modes::kDenominators[basis] != 0)
                    return false;
            }
            for (std::size_t remainder = 0; remainder < 36; ++remainder) {
                Int64 value = 0;
                for (std::size_t basis = 0; basis < BasisCount; ++basis) {
                    value += Int64(
                        S2Q6Modes::kCoefficients[modeIndex][basis]
                    ) * Int64(
                        remainder / S2Q6Modes::kDenominators[basis]
                    );
                }
                if (value < std::numeric_limits<Int8>::min()
                    || value > std::numeric_limits<Int8>::max())
                    return false;
            }
            return true;
        }();
        static_assert(
            !UsePeriod36 || PeriodIsValid,
            "period-36 evaluator requires exact Int8 remainder tables"
        );

        inline static constexpr Int64 PeriodSlope = [] {
            Int64 slope = 0;
            constexpr std::size_t modeIndex = static_cast<std::size_t>(M);
            for (std::size_t basis = 0; basis < BasisCount; ++basis) {
                slope += Int64(S2Q6Modes::kCoefficients[modeIndex][basis])
                    * Int64(36 / S2Q6Modes::kDenominators[basis]);
            }
            return slope;
        }();

        inline static constexpr std::array<Int8, 36> PeriodRemainders = [] {
            std::array<Int8, 36> remainders{};
            constexpr std::size_t modeIndex = static_cast<std::size_t>(M);
            for (std::size_t remainder = 0;
                 remainder < remainders.size();
                 ++remainder) {
                Int64 value = 0;
                for (std::size_t basis = 0; basis < BasisCount; ++basis) {
                    value += Int64(
                        S2Q6Modes::kCoefficients[modeIndex][basis]
                    ) * Int64(
                        remainder / S2Q6Modes::kDenominators[basis]
                    );
                }
                remainders[remainder] = static_cast<Int8>(value);
            }
            return remainders;
        }();

        template<typename T>
        static inline T eval(T quotient) {
            static_assert(
                std::is_same_v<T, Int64>,
                "fixed-stride evaluator requires a nonnegative Int64 quotient"
            );
            assert(quotient >= 0);
            if constexpr (!UsePeriod36) {
                return S2Q6Modes::Term<M, T>::eval(quotient);
            } else {
                const UInt64 q = static_cast<UInt64>(quotient);
                const UInt64 blocks = q / 36;
                const UInt64 remainder = q - blocks * 36;
                return T(PeriodSlope) * T(blocks)
                    + T(PeriodRemainders[remainder]);
            }
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

// Coherent outer groups use the same four divisor classes for the outer and
// inner Q=6 transforms. Their ten unordered class products are generated
// directly from mu(1), mu(2), mu(3), and mu(6); no coefficient or remainder
// table below depends on hand-transcribed constants.
namespace CoherentS2Q6 {

enum class Mode : UInt8 {
    FullFull,
    FullMinus2Minus3,
    FullMinus2,
    FullSingle,
    Minus2Minus3Squared,
    Minus2Minus3Minus2,
    Minus2Minus3Single,
    Minus2Squared,
    Minus2Single,
    SingleSquared,
};

inline constexpr std::size_t ClassCount = 4;
inline constexpr std::size_t ModeCount = 10;
inline constexpr std::size_t BasisCount = 9;
inline constexpr std::array<UInt64, 4> Divisors = {1, 2, 3, 6};
inline constexpr std::array<Int8, 4> Mobius = {1, -1, -1, 1};
inline constexpr std::array<UInt8, ClassCount> ClassMasks = [] {
    std::array<UInt8, ClassCount> masks{};
    for (std::size_t activeClass = 0;
         activeClass < ClassCount;
         ++activeClass) {
        masks[activeClass] = static_cast<UInt8>(
            (UInt8(1) << (ClassCount - activeClass)) - 1
        );
    }
    return masks;
}();
inline constexpr std::array<UInt64, BasisCount> Denominators = {
    1, 2, 3, 4, 6, 9, 12, 18, 36
};
using Coefficients = std::array<Int8, BasisCount>;

static constexpr std::size_t basisIndex(UInt64 denominator) {
    for (std::size_t index = 0; index < BasisCount; ++index) {
        if (Denominators[index] == denominator) return index;
    }
    return BasisCount;
}

static constexpr std::size_t modeIndex(
    std::size_t leftClass,
    std::size_t rightClass
) {
    if (leftClass > rightClass) {
        const std::size_t swap = leftClass;
        leftClass = rightClass;
        rightClass = swap;
    }
    const std::size_t rowOffset =
        leftClass * (2 * ClassCount - leftClass + 1) / 2;
    return rowOffset + rightClass - leftClass;
}

static constexpr Coefficients makeCoefficients(
    std::size_t leftClass,
    std::size_t rightClass
) {
    Coefficients coefficients{};
    for (std::size_t left = 0; left < Divisors.size(); ++left) {
        if ((ClassMasks[leftClass] & (UInt8(1) << left)) == 0) continue;
        for (std::size_t right = 0; right < Divisors.size(); ++right) {
            if ((ClassMasks[rightClass] & (UInt8(1) << right)) == 0) continue;
            const std::size_t basis = basisIndex(
                Divisors[left] * Divisors[right]
            );
            if (basis < BasisCount)
                coefficients[basis] += Mobius[left] * Mobius[right];
        }
    }
    return coefficients;
}

inline constexpr std::array<Coefficients, ModeCount> CoefficientTable = [] {
    std::array<Coefficients, ModeCount> table{};
    for (std::size_t left = 0; left < ClassCount; ++left) {
        for (std::size_t right = left; right < ClassCount; ++right) {
            table[modeIndex(left, right)] = makeCoefficients(left, right);
        }
    }
    return table;
}();

struct PeriodKernel {
    Int16 slope;
    std::array<Int8, 36> remainders;
};

static constexpr Int64 evaluateCoefficients(
    const Coefficients& coefficients,
    UInt64 quotient
) {
    Int64 result = 0;
    for (std::size_t basis = 0; basis < BasisCount; ++basis) {
        result += Int64(coefficients[basis]) * Int64(
            quotient / Denominators[basis]
        );
    }
    return result;
}

static constexpr PeriodKernel makePeriodKernel(
    const Coefficients& coefficients
) {
    PeriodKernel kernel{};
    kernel.slope = static_cast<Int16>(evaluateCoefficients(coefficients, 36));
    for (UInt64 remainder = 0; remainder < 36; ++remainder) {
        kernel.remainders[remainder] = static_cast<Int8>(
            evaluateCoefficients(coefficients, remainder)
        );
    }
    return kernel;
}

inline constexpr std::array<PeriodKernel, ModeCount> PeriodTable = [] {
    std::array<PeriodKernel, ModeCount> table{};
    for (std::size_t mode = 0; mode < ModeCount; ++mode)
        table[mode] = makePeriodKernel(CoefficientTable[mode]);
    return table;
}();

inline constexpr std::array<std::array<PeriodKernel, ModeCount>, ModeCount>
PairPeriodTable = [] {
    std::array<std::array<PeriodKernel, ModeCount>, ModeCount> table{};
    for (std::size_t left = 0; left < ModeCount; ++left) {
        for (std::size_t right = 0; right < ModeCount; ++right) {
            Coefficients coefficients{};
            for (std::size_t basis = 0; basis < BasisCount; ++basis) {
                coefficients[basis] = CoefficientTable[left][basis]
                                    + CoefficientTable[right][basis];
            }
            table[left][right] = makePeriodKernel(coefficients);
        }
    }
    return table;
}();

static constexpr Int64 evaluatePeriodKernel(
    const PeriodKernel& kernel,
    UInt64 quotient
) {
    const UInt64 blocks = quotient / 36;
    const UInt64 remainder = quotient - blocks * 36;
    return Int64(kernel.slope) * Int64(blocks)
         + Int64(kernel.remainders[remainder]);
}

static constexpr Int64 evaluateDirectClasses(
    std::size_t leftClass,
    std::size_t rightClass,
    UInt64 quotient
) {
    Int64 result = 0;
    for (std::size_t left = 0; left < Divisors.size(); ++left) {
        if ((ClassMasks[leftClass] & (UInt8(1) << left)) == 0) continue;
        for (std::size_t right = 0; right < Divisors.size(); ++right) {
            if ((ClassMasks[rightClass] & (UInt8(1) << right)) == 0) continue;
            result += Int64(Mobius[left]) * Int64(Mobius[right]) * Int64(
                quotient / (Divisors[left] * Divisors[right])
            );
        }
    }
    return result;
}

template<typename T>
static inline T evaluateDivisorClasses(
    std::size_t leftClass,
    std::size_t rightClass,
    T quotient
) {
    T result = 0;
    for (std::size_t left = 0; left < Divisors.size(); ++left) {
        if ((ClassMasks[leftClass] & (UInt8(1) << left)) == 0) continue;
        for (std::size_t right = 0; right < Divisors.size(); ++right) {
            if ((ClassMasks[rightClass] & (UInt8(1) << right)) == 0) continue;
            result += T(Mobius[left]) * T(Mobius[right])
                    * (quotient / T(Divisors[left] * Divisors[right]));
        }
    }
    return result;
}

template<Mode M, typename T>
static constexpr T evaluateSimpleMode(T quotient) {
    static_assert(
        M == Mode::Minus2Squared
        || M == Mode::Minus2Single
        || M == Mode::SingleSquared,
        "direct coherent evaluator requires a simple generated mode"
    );
    if constexpr (M == Mode::Minus2Squared)
        return (quotient >> 2) + (quotient & T(1));
    if constexpr (M == Mode::Minus2Single)
        return quotient - quotient / T(2);
    return quotient;
}

static constexpr bool periodKernelFits(const Coefficients& coefficients) {
    const Int64 slope = evaluateCoefficients(coefficients, 36);
    if (slope < std::numeric_limits<Int16>::min()
        || slope > std::numeric_limits<Int16>::max())
        return false;
    for (UInt64 remainder = 0; remainder < 36; ++remainder) {
        const Int64 value = evaluateCoefficients(coefficients, remainder);
        if (value < std::numeric_limits<Int8>::min()
            || value > std::numeric_limits<Int8>::max())
            return false;
    }
    return true;
}

static constexpr bool verifyGeneratedTables() {
    for (std::size_t basis = 0; basis < BasisCount; ++basis) {
        if (36 % Denominators[basis] != 0) return false;
    }
    for (UInt64 left : Divisors) {
        for (UInt64 right : Divisors) {
            if (basisIndex(left * right) >= BasisCount) return false;
        }
    }
    if (static_cast<std::size_t>(Mode::FullFull) != modeIndex(0, 0)
        || static_cast<std::size_t>(Mode::FullMinus2Minus3) != modeIndex(0, 1)
        || static_cast<std::size_t>(Mode::FullMinus2) != modeIndex(0, 2)
        || static_cast<std::size_t>(Mode::FullSingle) != modeIndex(0, 3)
        || static_cast<std::size_t>(Mode::Minus2Minus3Squared) != modeIndex(1, 1)
        || static_cast<std::size_t>(Mode::Minus2Minus3Minus2) != modeIndex(1, 2)
        || static_cast<std::size_t>(Mode::Minus2Minus3Single) != modeIndex(1, 3)
        || static_cast<std::size_t>(Mode::Minus2Squared) != modeIndex(2, 2)
        || static_cast<std::size_t>(Mode::Minus2Single) != modeIndex(2, 3)
        || static_cast<std::size_t>(Mode::SingleSquared) != modeIndex(3, 3))
        return false;
    for (std::size_t left = 0; left < ClassCount; ++left) {
        for (std::size_t right = 0; right < ClassCount; ++right) {
            const std::size_t mode = modeIndex(left, right);
            if (mode >= ModeCount || mode != modeIndex(right, left))
                return false;
            const Coefficients direct = makeCoefficients(left, right);
            for (std::size_t basis = 0; basis < BasisCount; ++basis) {
                if (CoefficientTable[mode][basis] != direct[basis])
                    return false;
            }
        }
    }
    for (UInt64 quotient = 0; quotient < 72; ++quotient) {
        const Int64 signedQuotient = static_cast<Int64>(quotient);
        if (evaluateSimpleMode<Mode::Minus2Squared>(signedQuotient)
            != evaluateDirectClasses(2, 2, quotient))
            return false;
        if (evaluateSimpleMode<Mode::Minus2Single>(signedQuotient)
            != evaluateDirectClasses(2, 3, quotient))
            return false;
        if (evaluateSimpleMode<Mode::SingleSquared>(signedQuotient)
            != evaluateDirectClasses(3, 3, quotient))
            return false;
    }
    for (std::size_t left = 0; left < ClassCount; ++left) {
        for (std::size_t right = left; right < ClassCount; ++right) {
            const std::size_t mode = modeIndex(left, right);
            if (!periodKernelFits(CoefficientTable[mode])) return false;
            for (UInt64 quotient = 0; quotient < 72; ++quotient) {
                const Int64 direct = evaluateDirectClasses(
                    left, right, quotient
                );
                if (evaluateCoefficients(CoefficientTable[mode], quotient)
                    != direct)
                    return false;
                if (evaluatePeriodKernel(PeriodTable[mode], quotient) != direct)
                    return false;
            }
        }
    }
    return true;
}
static_assert(verifyGeneratedTables(),
              "coherent Q=6 period tables must match direct divisor sums");

template<std::size_t LeftMode>
static constexpr bool verifyPairPeriodRow() {
    for (std::size_t right = 0; right < ModeCount; ++right) {
        Coefficients combined{};
        for (std::size_t basis = 0; basis < BasisCount; ++basis) {
            combined[basis] = CoefficientTable[LeftMode][basis]
                            + CoefficientTable[right][basis];
        }
        if (!periodKernelFits(combined)) return false;
        for (UInt64 quotient = 0; quotient < 72; ++quotient) {
            const Int64 direct = evaluateCoefficients(
                CoefficientTable[LeftMode], quotient
            ) + evaluateCoefficients(CoefficientTable[right], quotient);
            if (evaluatePeriodKernel(
                    PairPeriodTable[LeftMode][right], quotient
                ) != direct)
                return false;
        }
    }
    return true;
}
static_assert(verifyPairPeriodRow<0>(), "invalid coherent pair period row 0");
static_assert(verifyPairPeriodRow<1>(), "invalid coherent pair period row 1");
static_assert(verifyPairPeriodRow<2>(), "invalid coherent pair period row 2");
static_assert(verifyPairPeriodRow<3>(), "invalid coherent pair period row 3");
static_assert(verifyPairPeriodRow<4>(), "invalid coherent pair period row 4");
static_assert(verifyPairPeriodRow<5>(), "invalid coherent pair period row 5");
static_assert(verifyPairPeriodRow<6>(), "invalid coherent pair period row 6");
static_assert(verifyPairPeriodRow<7>(), "invalid coherent pair period row 7");
static_assert(verifyPairPeriodRow<8>(), "invalid coherent pair period row 8");
static_assert(verifyPairPeriodRow<9>(), "invalid coherent pair period row 9");

template<typename T>
static inline T evaluateDirect(std::size_t mode, T quotient) {
    T result = 0;
    for (std::size_t basis = 0; basis < BasisCount; ++basis) {
        result += T(CoefficientTable[mode][basis])
                * (quotient / T(Denominators[basis]));
    }
    return result;
}

template<typename T>
static inline T evaluatePeriod(std::size_t mode, T quotient) {
    const T blocks = quotient / T(36);
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * T(36)
    );
    return T(PeriodTable[mode].slope) * blocks
         + T(PeriodTable[mode].remainders[remainder]);
}

template<typename T>
static inline T evaluatePeriodKernelValue(
    const PeriodKernel& kernel,
    T quotient
) {
    const T blocks = quotient / T(36);
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * T(36)
    );
    return T(kernel.slope) * blocks + T(kernel.remainders[remainder]);
}

static constexpr std::size_t innerClass(UInt64 value, UInt64 commonNu) {
    if (value <= commonNu / 6) return 0;
    if (value <= commonNu / 3) return 1;
    if (value <= commonNu / 2) return 2;
    if (value <= commonNu) return 3;
    return ClassCount;
}

static constexpr bool innerClassesAreExact() {
    for (UInt64 commonNu = 0; commonNu < 72; ++commonNu) {
        for (UInt64 value = 1; value <= commonNu + 1; ++value) {
            UInt8 activeMask = 0;
            for (std::size_t divisor = 0; divisor < Divisors.size(); ++divisor) {
                if (Divisors[divisor] * value <= commonNu)
                    activeMask |= UInt8(1) << divisor;
            }
            const std::size_t activeClass = innerClass(value, commonNu);
            if (activeClass == ClassCount) {
                if (activeMask != 0) return false;
            } else if (ClassMasks[activeClass] != activeMask) {
                return false;
            }
        }
    }
    return true;
}
static_assert(innerClassesAreExact(),
              "coherent Q=6 cutoff classes must preserve every endpoint");

template<typename T>
static inline T evaluatePairPeriod(
    std::size_t leftMode,
    std::size_t rightMode,
    T quotient
) {
    const T blocks = quotient / T(36);
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * T(36)
    );
    const PeriodKernel& kernel = PairPeriodTable[leftMode][rightMode];
    return T(kernel.slope) * blocks + T(kernel.remainders[remainder]);
}

template<Mode M>
struct Evaluator {
    template<typename T>
    static inline T eval(T quotient) {
        constexpr std::size_t mode = static_cast<std::size_t>(M);
#if MERTENSHURST_COHERENT_PERIOD36
        if constexpr (M == Mode::Minus2Squared
                      || M == Mode::Minus2Single
                      || M == Mode::SingleSquared)
            return evaluateSimpleMode<M>(quotient);
        return evaluatePeriod(mode, quotient);
#else
        return evaluateDirect(mode, quotient);
#endif
    }
};

template<typename Callback>
static inline void dispatchMode(Mode mode, Callback&& callback) {
    switch (mode) {
        case Mode::FullFull:
            callback(std::integral_constant<Mode, Mode::FullFull>{}); return;
        case Mode::FullMinus2Minus3:
            callback(std::integral_constant<Mode, Mode::FullMinus2Minus3>{}); return;
        case Mode::FullMinus2:
            callback(std::integral_constant<Mode, Mode::FullMinus2>{}); return;
        case Mode::FullSingle:
            callback(std::integral_constant<Mode, Mode::FullSingle>{}); return;
        case Mode::Minus2Minus3Squared:
            callback(std::integral_constant<Mode, Mode::Minus2Minus3Squared>{}); return;
        case Mode::Minus2Minus3Minus2:
            callback(std::integral_constant<Mode, Mode::Minus2Minus3Minus2>{}); return;
        case Mode::Minus2Minus3Single:
            callback(std::integral_constant<Mode, Mode::Minus2Minus3Single>{}); return;
        case Mode::Minus2Squared:
            callback(std::integral_constant<Mode, Mode::Minus2Squared>{}); return;
        case Mode::Minus2Single:
            callback(std::integral_constant<Mode, Mode::Minus2Single>{}); return;
        case Mode::SingleSquared:
            callback(std::integral_constant<Mode, Mode::SingleSquared>{}); return;
    }
}

} // namespace CoherentS2Q6

struct CoherentS2Q6Spec {
    using Mode = CoherentS2Q6::Mode;
    static constexpr UInt64 InnerWheel = 6;

    template<Mode M>
    using Evaluator = CoherentS2Q6::Evaluator<M>;

    template<Mode M>
    using FixedStrideEvaluator = CoherentS2Q6::Evaluator<M>;

    template<typename Callback>
    static inline void dispatchMode(Mode mode, Callback&& callback) {
        CoherentS2Q6::dispatchMode(mode, callback);
    }
};

template<typename Spec>
using S2Q6DynamicCoefficients = std::array<Int16, Spec::BasisCount>;

namespace S2Q6Detail {

// Keep each generated mode in a separate kernel so the dispatcher does not
// become one large inlined function with poor instruction locality.
template<typename Spec, typename Spec::Mode Mode>
__attribute__((noinline)) static Int64 updateMode64(
    UInt64 n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Evaluator = typename Spec::template Evaluator<Mode>;
    return update_S2_wheel6<Evaluator>(n, L1, lo, hi, Mu, qCache, dCAP);
}

template<typename Spec, typename Spec::Mode Mode>
__attribute__((noinline)) static Int128 updateMode128(
    const UInt128& n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    UInt64 cbrt2nCeil,
    UInt64 step6Boundary
) {
    using Evaluator = typename Spec::template Evaluator<Mode>;
    using FixedStrideEvaluator =
        typename Spec::template FixedStrideEvaluator<Mode>;
    Int128 result = 0;
    update_S2_128_wheel6<Evaluator, FixedStrideEvaluator>(
        n, L1, lo, hi, Mu, cbrt2nCeil, step6Boundary, result
    );
    return result;
}

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

template<typename Callback>
static inline void dispatchCoherentS2Q6(
    UInt64 x1,
    UInt64 x2,
    UInt64 commonNu,
    UInt32 outerClass,
    Callback&& callback
) {
    if (x1 > x2) return;
    assert(outerClass < CoherentS2Q6::ClassCount);

    const std::array<UInt64, CoherentS2Q6::ClassCount> cutoffs = {
        commonNu / 6, commonNu / 3, commonNu / 2, commonNu
    };
    UInt64 previous = 0;
    for (std::size_t innerClass = 0;
         innerClass < CoherentS2Q6::ClassCount;
         ++innerClass) {
        const UInt64 lo = std::max(x1, previous + 1);
        const UInt64 hi = std::min(x2, cutoffs[innerClass]);
        if (lo <= hi) {
            const CoherentS2Q6::Mode mode = static_cast<CoherentS2Q6::Mode>(
                CoherentS2Q6::modeIndex(outerClass, innerClass)
            );
            CoherentS2Q6::dispatchMode(mode, [&](auto modeTag) {
                callback(modeTag, lo, hi);
            });
        }
        previous = cutoffs[innerClass];
    }
}

static inline Int64 update_S2_coherent_q6(
    UInt64 n,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt32 outerClass,
    UInt64 commonNu,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    Int64 sum = 0;
    dispatchCoherentS2Q6(
        x1, x2, commonNu, outerClass,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr CoherentS2Q6::Mode Mode = decltype(modeTag)::value;
            sum += S2Q6Detail::updateMode64<CoherentS2Q6Spec, Mode>(
                n, L1, lo, hi, Mu, qCache, dCAP
            );
        }
    );
    return sum;
}

static inline Int128 update_S2_coherent_q6_128(
    const UInt128& n,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt32 outerClass,
    UInt64 commonNu
) {
    Int128 sum = 0;
    const UInt64 cbrt2nCeil = static_cast<UInt64>(
        ceill(cbrtl(2.001L * static_cast<long double>(n)))
    );
    const UInt64 step6Boundary = quotient_predictor_first_unit_curvature<6>(n);
    dispatchCoherentS2Q6(
        x1, x2, commonNu, outerClass,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr CoherentS2Q6::Mode Mode = decltype(modeTag)::value;
            sum += S2Q6Detail::updateMode128<CoherentS2Q6Spec, Mode>(
                n, L1, lo, hi, Mu, cbrt2nCeil, step6Boundary
            );
        }
    );
    return sum;
}

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
            sum += S2Q6Detail::updateMode64<Spec, Mode>(
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
            sum += S2Q6Detail::updateMode128<Spec, Mode>(
                n, L1, lo, hi, Mu, cbrt2nCeil, step6Boundary
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
