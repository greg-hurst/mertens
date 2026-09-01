#pragma once

// ============================================================================
// S2Q30.h — completed outer-Q=30 S2 kernels.
//
// A completed outer group has divisor set {1,2,3,5,6,10,15,30}.  Factoring
// the inner Mobius variable into the same divisor set leaves eight exact
// cutoff classes.  Their product kernels have period 900; the tables below
// are generated from the divisor signs and verified at compile time.
// ============================================================================

#include "S2.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <type_traits>
#include <utility>

namespace CoherentS2Q30 {

enum class Mode : UInt8 {
    P1,
    P2,
    P3,
    P5,
    P6,
    P10,
    P15,
    P30,
};

inline constexpr std::size_t ClassCount = 8;
inline constexpr std::size_t BasisCount = 27;
inline constexpr UInt64 Period = 900;
inline constexpr Int64 RemainderBias = 10;
inline constexpr std::array<UInt64, ClassCount> Divisors = {
    1, 2, 3, 5, 6, 10, 15, 30
};
inline constexpr std::array<Int8, ClassCount> Mobius = {
    1, -1, -1, -1, 1, 1, 1, -1
};
inline constexpr std::array<UInt64, BasisCount> Denominators = {
    1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 25, 30, 36,
    45, 50, 60, 75, 90, 100, 150, 180, 225, 300, 450, 900
};
using Coefficients = std::array<Int8, BasisCount>;

static constexpr std::size_t basisIndex(UInt64 denominator) {
    for (std::size_t index = 0; index < BasisCount; ++index) {
        if (Denominators[index] == denominator) return index;
    }
    return BasisCount;
}

static constexpr Coefficients makeCoefficients(std::size_t innerClass) {
    Coefficients coefficients{};
    for (std::size_t outer = 0; outer < ClassCount; ++outer) {
        for (std::size_t inner = 0; inner <= innerClass; ++inner) {
            const std::size_t basis = basisIndex(
                Divisors[outer] * Divisors[inner]
            );
            if (basis < BasisCount)
                coefficients[basis] += Mobius[outer] * Mobius[inner];
        }
    }
    return coefficients;
}

inline constexpr std::array<Coefficients, ClassCount> CoefficientTable = [] {
    std::array<Coefficients, ClassCount> table{};
    for (std::size_t mode = 0; mode < ClassCount; ++mode)
        table[mode] = makeCoefficients(mode);
    return table;
}();

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

alignas(64) inline constexpr std::array<Int16, ClassCount> PeriodSlopes = [] {
    std::array<Int16, ClassCount> slopes{};
    for (std::size_t mode = 0; mode < ClassCount; ++mode) {
        slopes[mode] = static_cast<Int16>(
            evaluateCoefficients(CoefficientTable[mode], Period)
        );
    }
    return slopes;
}();

alignas(64) inline constexpr
std::array<std::array<UInt8, Period>, ClassCount> PeriodRemainders = [] {
    std::array<std::array<UInt8, Period>, ClassCount> table{};
    for (UInt64 remainder = 0; remainder < Period; ++remainder) {
        Int64 value = 0;
        for (std::size_t inner = 0; inner < ClassCount; ++inner) {
            for (std::size_t outer = 0; outer < ClassCount; ++outer) {
                value += Int64(Mobius[outer]) * Int64(Mobius[inner])
                       * Int64(remainder
                             / (Divisors[outer] * Divisors[inner]));
            }
            table[inner][remainder] = static_cast<UInt8>(
                value + RemainderBias
            );
        }
    }
    return table;
}();

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
    const T blocks = quotient / T(Period);
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * T(Period)
    );
    return T(PeriodSlopes[mode]) * blocks
         + T(PeriodRemainders[mode][remainder]) - T(RemainderBias);
}

template<typename T>
static inline T evaluatePairPeriod(
    std::size_t leftMode,
    std::size_t rightMode,
    T quotient
) {
    const T blocks = quotient / T(Period);
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * T(Period)
    );
    return T(PeriodSlopes[leftMode] + PeriodSlopes[rightMode]) * blocks
         + T(PeriodRemainders[leftMode][remainder])
         + T(PeriodRemainders[rightMode][remainder])
         - T(2 * RemainderBias);
}

static constexpr std::size_t innerClass(UInt64 value, UInt64 commonNu) {
    if (value <= commonNu / 30) return 7;
    if (value <= commonNu / 15) return 6;
    if (value <= commonNu / 10) return 5;
    if (value <= commonNu / 6)  return 4;
    if (value <= commonNu / 5)  return 3;
    if (value <= commonNu / 3)  return 2;
    if (value <= commonNu / 2)  return 1;
    if (value <= commonNu)      return 0;
    return ClassCount;
}

static constexpr bool verifyGeneratedTables() {
    constexpr std::array<Int16, ClassCount> expectedSlopes = {
        240, 120, 40, -8, 32, 56, 72, 64
    };
    for (UInt64 denominator : Denominators) {
        if (Period % denominator != 0) return false;
    }
    for (UInt64 left : Divisors) {
        for (UInt64 right : Divisors) {
            if (basisIndex(left * right) >= BasisCount) return false;
        }
    }
    for (std::size_t mode = 0; mode < ClassCount; ++mode) {
        if (PeriodSlopes[mode] != expectedSlopes[mode]) return false;
        for (std::size_t basis = 0; basis < BasisCount; ++basis) {
            const Int64 coefficient = CoefficientTable[mode][basis];
            if (coefficient < -8 || coefficient > 4) return false;
        }
    }
    for (UInt64 remainder = 0; remainder < Period; ++remainder) {
        Int64 value = 0;
        for (std::size_t inner = 0; inner < ClassCount; ++inner) {
            for (std::size_t outer = 0; outer < ClassCount; ++outer) {
                value += Int64(Mobius[outer]) * Int64(Mobius[inner])
                       * Int64(remainder
                             / (Divisors[outer] * Divisors[inner]));
            }
            if (value < -10 || value > 240) return false;
            if (Int64(PeriodRemainders[inner][remainder])
                    - RemainderBias != value)
                return false;
        }
    }
    return true;
}
static_assert(verifyGeneratedTables(),
              "coherent Q=30 tables must match direct divisor sums");
static_assert(sizeof(PeriodSlopes) == 16,
              "coherent Q=30 slopes must occupy 16 bytes");
static_assert(sizeof(PeriodRemainders) == 7200,
              "coherent Q=30 remainder tables must occupy 7200 bytes");

static constexpr bool innerClassesAreExact() {
    for (UInt64 commonNu = 0; commonNu < 120; ++commonNu) {
        for (UInt64 value = 1; value <= commonNu + 1; ++value) {
            UInt8 activeMask = 0;
            for (std::size_t divisor = 0; divisor < Divisors.size(); ++divisor) {
                if (value <= commonNu / Divisors[divisor])
                    activeMask |= UInt8(1) << divisor;
            }
            const std::size_t activeClass = innerClass(value, commonNu);
            if (activeClass == ClassCount) {
                if (activeMask != 0) return false;
            } else {
                const UInt8 expected = static_cast<UInt8>(
                    (UInt16(1) << (activeClass + 1)) - 1
                );
                if (activeMask != expected) return false;
            }
        }
    }
    return true;
}
static_assert(innerClassesAreExact(),
              "coherent Q=30 cutoff classes must preserve every endpoint");

template<Mode M>
struct Evaluator {
    template<typename T>
    static inline T eval(T quotient) {
        return evaluatePeriod(static_cast<std::size_t>(M), quotient);
    }
};

template<typename Callback>
static inline void dispatchMode(Mode mode, Callback&& callback) {
    switch (mode) {
        case Mode::P1:  callback(std::integral_constant<Mode, Mode::P1>{}); return;
        case Mode::P2:  callback(std::integral_constant<Mode, Mode::P2>{}); return;
        case Mode::P3:  callback(std::integral_constant<Mode, Mode::P3>{}); return;
        case Mode::P5:  callback(std::integral_constant<Mode, Mode::P5>{}); return;
        case Mode::P6:  callback(std::integral_constant<Mode, Mode::P6>{}); return;
        case Mode::P10: callback(std::integral_constant<Mode, Mode::P10>{}); return;
        case Mode::P15: callback(std::integral_constant<Mode, Mode::P15>{}); return;
        case Mode::P30: callback(std::integral_constant<Mode, Mode::P30>{}); return;
    }
}

} // namespace CoherentS2Q30

struct S2Q30Spec {
    using Mode = CoherentS2Q30::Mode;
    inline static constexpr std::size_t ModeCount =
        CoherentS2Q30::ClassCount;
    inline static constexpr UInt64 Period = CoherentS2Q30::Period;
    inline static constexpr Int64 RemainderBias =
        CoherentS2Q30::RemainderBias;
    inline static constexpr const auto& Divisors =
        CoherentS2Q30::Divisors;
    inline static constexpr const auto& Signs = CoherentS2Q30::Mobius;
    inline static constexpr const auto& PeriodSlopes =
        CoherentS2Q30::PeriodSlopes;
    inline static constexpr const auto& PeriodRemainders =
        CoherentS2Q30::PeriodRemainders;

    template<typename T>
    using Accumulator = std::conditional_t<
        std::is_same_v<T, UInt128>, Int128, Int64
    >;

    static constexpr Mode modeFor(UInt64 commonNu, UInt64 value) {
        return static_cast<Mode>(CoherentS2Q30::innerClass(value, commonNu));
    }

    template<typename T>
    static inline Accumulator<T> evalRuntimeDirect(Mode mode, T quotient) {
        Int128 result = 0;
        const std::size_t index = static_cast<std::size_t>(mode);
        for (std::size_t basis = 0;
             basis < CoherentS2Q30::BasisCount;
             ++basis) {
            result += Int128(CoherentS2Q30::CoefficientTable[index][basis])
                    * Int128(quotient / T(
                        CoherentS2Q30::Denominators[basis]
                    ));
        }
        return static_cast<Accumulator<T>>(result);
    }

    template<typename T>
    static inline Accumulator<T> evalRuntimePeriod900(Mode mode, T quotient) {
        const T blocks = quotient / T(Period);
        const std::size_t remainder = static_cast<std::size_t>(
            quotient - blocks * T(Period)
        );
        const std::size_t index = static_cast<std::size_t>(mode);
        const Int128 result = Int128(PeriodSlopes[index]) * Int128(blocks)
                            + Int128(PeriodRemainders[index][remainder])
                            - Int128(RemainderBias);
        return static_cast<Accumulator<T>>(result);
    }

    template<typename T>
    static inline Accumulator<T> evalRuntimePeriod900Pair(
        Mode left,
        Mode right,
        T quotient
    ) {
        const T blocks = quotient / T(Period);
        const std::size_t remainder = static_cast<std::size_t>(
            quotient - blocks * T(Period)
        );
        const std::size_t leftIndex = static_cast<std::size_t>(left);
        const std::size_t rightIndex = static_cast<std::size_t>(right);
        const Int128 result = Int128(PeriodSlopes[leftIndex]
                                      + PeriodSlopes[rightIndex])
                            * Int128(blocks)
                            + Int128(PeriodRemainders[leftIndex][remainder])
                            + Int128(PeriodRemainders[rightIndex][remainder])
                            - Int128(2 * RemainderBias);
        return static_cast<Accumulator<T>>(result);
    }
};

namespace S2Q30Detail {

static inline bool wheelAccepts(UInt64 value) {
    return (value & 1ULL) != 0 && value % 3 != 0 && value % 5 != 0;
}

template<typename Evaluator>
static inline Int64 updateMode64(
    UInt64 n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    if (lo > hi) return 0;
    Int64 result = 0;

    auto add = [&](UInt64 denominator) {
        const UInt64 quotient = UseDivisionFree && denominator <= dCAP
            ? qCache.quotient(n, denominator)
            : n / denominator;
        result += Int64(Mu[denominator - L1])
                * Evaluator::template eval<Int64>(Int64(quotient));
    };

    UInt64 block = lo - lo % 30;
    UInt64 fullBlock = block;
    if (fullBlock + 1 < lo) fullBlock += 30;
    const UInt64 prefixHi = fullBlock == 0
        ? 0
        : std::min(hi, fullBlock - 1);
    for (UInt64 value = lo; value <= prefixHi; ++value) {
        if (wheelAccepts(value)) add(value);
    }

    block = fullBlock;
    while (block <= hi && hi - block >= 29) {
        add(block + 1);  add(block + 7);  add(block + 11); add(block + 13);
        add(block + 17); add(block + 19); add(block + 23); add(block + 29);
        if (std::numeric_limits<UInt64>::max() - block < 30) return result;
        block += 30;
    }
    if (block <= hi) {
        for (UInt64 value = block; value <= hi; ++value) {
            if (wheelAccepts(value)) add(value);
        }
    }
    return result;
}

template<typename Evaluator>
static inline void updateMode128Direct(
    const UInt128& n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    Int128& result
) {
    if (lo > hi) return;
    for (UInt64 value = lo;; ++value) {
        if (wheelAccepts(value)) {
            const Int8 mu = Mu[value - L1];
            if (mu != 0) {
                const Int128 quotient = static_cast<Int128>(n / value);
                result += Int128(mu) * Evaluator::template eval<Int128>(
                    quotient
                );
            }
        }
        if (value == hi) break;
    }
}

template<typename Evaluator, UInt64 Residue>
static inline void updateMode128PredictedResidue(
    const UInt128& n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    Int128& result
) {
    static_assert(Residue < 30, "invalid wheel-30 residue");
    UInt64 value = lo + (Residue + 30 - lo % 30) % 30;
    if (value > hi) return;

    UInt64 qPrev = static_cast<UInt64>(n / (value - 30));
    UInt64 qCur = static_cast<UInt64>(n / value);
    UInt64 qEst = 0;
    const Int8 firstMu = Mu[value - L1];
    if (firstMu != 0) {
        result += Int64(firstMu) * Evaluator::template eval<Int64>(
            static_cast<Int64>(qCur)
        );
    }

    while (hi - value >= 30) {
        value += 30;
        update_quotients_fixed_stride<30, false>(
            n, value, qCur, qPrev, qEst
        );
        const Int8 mu = Mu[value - L1];
        if (mu != 0) {
            result += Int64(mu) * Evaluator::template eval<Int64>(
                static_cast<Int64>(qEst)
            );
        }
    }
}

template<typename Evaluator>
static inline Int128 updateMode128(
    const UInt128& n,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    UInt64 step30Boundary
) {
    if (lo > hi) return 0;
    Int128 result = 0;
    const UInt64 directHi = std::min(hi, step30Boundary);
    if (lo <= directHi)
        updateMode128Direct<Evaluator>(n, L1, lo, directHi, Mu, result);

    const UInt64 predictedLo = std::max(lo, step30Boundary + 1);
    if (predictedLo <= hi) {
        updateMode128PredictedResidue<Evaluator, 1>(
            n, L1, predictedLo, hi, Mu, result
        );
        updateMode128PredictedResidue<Evaluator, 7>(
            n, L1, predictedLo, hi, Mu, result
        );
        updateMode128PredictedResidue<Evaluator, 11>(
            n, L1, predictedLo, hi, Mu, result
        );
        updateMode128PredictedResidue<Evaluator, 13>(
            n, L1, predictedLo, hi, Mu, result
        );
        updateMode128PredictedResidue<Evaluator, 17>(
            n, L1, predictedLo, hi, Mu, result
        );
        updateMode128PredictedResidue<Evaluator, 19>(
            n, L1, predictedLo, hi, Mu, result
        );
        updateMode128PredictedResidue<Evaluator, 23>(
            n, L1, predictedLo, hi, Mu, result
        );
        updateMode128PredictedResidue<Evaluator, 29>(
            n, L1, predictedLo, hi, Mu, result
        );
    }
    return result;
}

template<typename Callback>
static inline void dispatch(
    UInt64 x1,
    UInt64 x2,
    UInt64 commonNu,
    Callback&& callback
) {
    if (x1 > x2) return;
    const std::array<UInt64, CoherentS2Q30::ClassCount> cutoffs = {
        commonNu / 30, commonNu / 15, commonNu / 10, commonNu / 6,
        commonNu / 5, commonNu / 3, commonNu / 2, commonNu
    };
    UInt64 previous = 0;
    for (std::size_t cell = 0; cell < cutoffs.size(); ++cell) {
        const UInt64 lo = std::max(x1, previous + 1);
        const UInt64 hi = std::min(x2, cutoffs[cell]);
        if (lo <= hi) {
            const CoherentS2Q30::Mode mode = static_cast<CoherentS2Q30::Mode>(
                CoherentS2Q30::ClassCount - 1 - cell
            );
            CoherentS2Q30::dispatchMode(mode, [&](auto modeTag) {
                callback(modeTag, lo, hi);
            });
        }
        previous = cutoffs[cell];
    }
}

template<typename Callback>
static inline void dispatchCoherent(
    UInt64 x1,
    UInt64 x2,
    UInt64 commonNu,
    Callback&& callback
) {
    dispatch(x1, x2, commonNu, std::forward<Callback>(callback));
}

} // namespace S2Q30Detail

static inline Int64 update_S2_coherent_q30(
    UInt64 n,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt64 commonNu,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    Int64 result = 0;
    S2Q30Detail::dispatch(
        x1, x2, commonNu,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr CoherentS2Q30::Mode Mode = decltype(modeTag)::value;
            result += S2Q30Detail::updateMode64<
                CoherentS2Q30::Evaluator<Mode>
            >(n, L1, lo, hi, Mu, qCache, dCAP);
        }
    );
    return result;
}

static inline Int128 update_S2_coherent_q30_128(
    const UInt128& n,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt64 commonNu
) {
    Int128 result = 0;
    const UInt64 step30Boundary =
        quotient_predictor_first_unit_curvature<30>(n);
    S2Q30Detail::dispatch(
        x1, x2, commonNu,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr CoherentS2Q30::Mode Mode = decltype(modeTag)::value;
            result += S2Q30Detail::updateMode128<
                CoherentS2Q30::Evaluator<Mode>
            >(n, L1, lo, hi, Mu, step30Boundary);
        }
    );
    return result;
}
