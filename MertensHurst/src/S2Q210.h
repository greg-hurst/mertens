#pragma once

// ============================================================================
// S2Q210.h — completed outer/inner-Q=210 S2 kernels.
//
// The sixteen inclusive inner prefixes share a period-44,100 signed table.
// Each row is generated from the exact first-difference recurrence, then used
// by both the unordered hot square and the strict b>N segmented tail.
// ============================================================================

#include "S2.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

namespace CoherentS2Q210 {

enum class Mode : UInt8 {
    P1,
    P2,
    P3,
    P5,
    P6,
    P7,
    P10,
    P14,
    P15,
    P21,
    P30,
    P35,
    P42,
    P70,
    P105,
    P210,
};

inline constexpr std::size_t ClassCount = 16;
inline constexpr UInt64 Period = 44100;
inline constexpr std::array<UInt64, ClassCount> Divisors = {
      1,   2,   3,   5,   6,   7,  10,  14,
     15,  21,  30,  35,  42,  70, 105, 210
};
inline constexpr std::array<Int8, ClassCount> Mobius = {
     1, -1, -1, -1,  1, -1,  1,  1,
     1,  1, -1,  1, -1, -1, -1,  1
};
inline constexpr std::array<Int16, ClassCount> ExpectedSlopes = {
    10080, 5040, 1680, -336, 1344,  -96,  912, 1632,
     2304, 2784, 2448, 2736, 2496, 2352, 2256, 2304
};
inline constexpr std::array<UInt8, 48> Residues = {
      1,  11,  13,  17,  19,  23,  29,  31,
     37,  41,  43,  47,  53,  59,  61,  67,
     71,  73,  79,  83,  89,  97, 101, 103,
    107, 109, 113, 121, 127, 131, 137, 139,
    143, 149, 151, 157, 163, 167, 169, 173,
    179, 181, 187, 191, 193, 197, 199, 209
};

static constexpr bool verifyConstants() {
    if (Divisors.front() != 1 || Divisors.back() != 210) return false;
    for (std::size_t index = 1; index < ClassCount; ++index) {
        if (Divisors[index - 1] >= Divisors[index]) return false;
    }
    for (UInt64 left : Divisors) {
        for (UInt64 right : Divisors) {
            if (Period % (left * right) != 0) return false;
        }
    }
    std::size_t residue = 0;
    for (UInt64 value = 1; value < 210; ++value) {
        const bool accepted = (value & 1ULL) != 0
                           && value % 3 != 0
                           && value % 5 != 0
                           && value % 7 != 0;
        if (!accepted) continue;
        if (residue >= Residues.size() || Residues[residue] != value)
            return false;
        ++residue;
    }
    return residue == Residues.size();
}
static_assert(verifyConstants(),
              "coherent Q=210 constants must describe the full wheel");

struct DensePeriodTable {
    alignas(64) std::array<Int16, ClassCount> slopes{};
    alignas(64) std::array<std::array<Int16, Period>, ClassCount> remainders{};

    DensePeriodTable() {
        std::array<Int32, ClassCount> current{};
        constexpr std::array<UInt64, 4> primes = {2, 3, 5, 7};

        for (UInt64 remainder = 0; remainder < Period; ++remainder) {
            for (std::size_t mode = 0; mode < ClassCount; ++mode) {
                if (current[mode] < std::numeric_limits<Int16>::min()
                    || current[mode] > std::numeric_limits<Int16>::max())
                    std::abort();
                remainders[mode][remainder] = static_cast<Int16>(
                    current[mode]
                );
            }

            const UInt64 value = remainder + 1;
            UInt64 support = 1;
            Int32 sign = 1;
            bool squarefree = true;
            for (UInt64 prime : primes) {
                if (value % (prime * prime) == 0) {
                    squarefree = false;
                    break;
                }
                if (value % prime == 0) {
                    support *= prime;
                    sign = -sign;
                }
            }
            if (!squarefree) continue;

            const auto first = std::lower_bound(
                Divisors.begin(), Divisors.end(), support
            );
            if (first == Divisors.end() || *first != support)
                std::abort();
            for (std::size_t mode = static_cast<std::size_t>(
                     first - Divisors.begin()
                 ); mode < ClassCount; ++mode) {
                current[mode] += sign;
            }
        }

        for (std::size_t mode = 0; mode < ClassCount; ++mode) {
            if (current[mode] != ExpectedSlopes[mode]) std::abort();
            slopes[mode] = static_cast<Int16>(current[mode]);
        }
    }
};

static_assert(sizeof(DensePeriodTable::slopes)
              == ClassCount * sizeof(Int16));
static_assert(sizeof(DensePeriodTable::remainders)
              == ClassCount * Period * sizeof(Int16));
static_assert(sizeof(DensePeriodTable) == 1411264,
              "coherent Q=210 dense table layout changed");

template<typename T>
using Accumulator = Int128;

template<typename T>
static inline Accumulator<T> evaluatePeriod(
    const DensePeriodTable& table,
    std::size_t mode,
    T quotient
) {
    static_assert(std::is_same_v<T, UInt64> || std::is_same_v<T, UInt128>,
                  "Q=210 quotient must be UInt64 or UInt128");
    const T blocks = quotient / T(Period);
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * T(Period)
    );
    return Int128(table.slopes[mode]) * Int128(blocks)
         + Int128(table.remainders[mode][remainder]);
}

static inline Int64 evaluatePeriod64(
    const DensePeriodTable& table,
    std::size_t mode,
    UInt64 quotient
) {
    const UInt64 blocks = quotient / Period;
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * Period
    );
    return Int64(table.slopes[mode]) * static_cast<Int64>(blocks)
         + Int64(table.remainders[mode][remainder]);
}

template<typename T>
static inline Accumulator<T> evaluatePairPeriod(
    const DensePeriodTable& table,
    std::size_t leftMode,
    std::size_t rightMode,
    T quotient
) {
    static_assert(std::is_same_v<T, UInt64> || std::is_same_v<T, UInt128>,
                  "Q=210 quotient must be UInt64 or UInt128");
    const T blocks = quotient / T(Period);
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * T(Period)
    );
    return Int128(Int64(table.slopes[leftMode])
                + Int64(table.slopes[rightMode])) * Int128(blocks)
         + Int128(Int64(table.remainders[leftMode][remainder])
                + Int64(table.remainders[rightMode][remainder]));
}

static inline Int64 evaluatePairPeriod64(
    const DensePeriodTable& table,
    std::size_t leftMode,
    std::size_t rightMode,
    UInt64 quotient
) {
    const UInt64 blocks = quotient / Period;
    const std::size_t remainder = static_cast<std::size_t>(
        quotient - blocks * Period
    );
    return (Int64(table.slopes[leftMode])
          + Int64(table.slopes[rightMode])) * static_cast<Int64>(blocks)
         + Int64(table.remainders[leftMode][remainder])
         + Int64(table.remainders[rightMode][remainder]);
}

template<typename T>
static inline Accumulator<T> evaluateDirect(std::size_t mode, T quotient) {
    static_assert(std::is_same_v<T, UInt64> || std::is_same_v<T, UInt128>,
                  "Q=210 quotient must be UInt64 or UInt128");
    Int128 result = 0;
    for (std::size_t outer = 0; outer < ClassCount; ++outer) {
        for (std::size_t inner = 0; inner <= mode; ++inner) {
            result += Int128(Mobius[outer]) * Int128(Mobius[inner])
                    * Int128(quotient / T(
                        Divisors[outer] * Divisors[inner]
                    ));
        }
    }
    return result;
}

static constexpr std::size_t innerClass(UInt64 value, UInt64 commonNu) {
    if (value <= commonNu / 210) return 15;
    if (value <= commonNu / 105) return 14;
    if (value <= commonNu / 70)  return 13;
    if (value <= commonNu / 42)  return 12;
    if (value <= commonNu / 35)  return 11;
    if (value <= commonNu / 30)  return 10;
    if (value <= commonNu / 21)  return 9;
    if (value <= commonNu / 15)  return 8;
    if (value <= commonNu / 14)  return 7;
    if (value <= commonNu / 10)  return 6;
    if (value <= commonNu / 7)   return 5;
    if (value <= commonNu / 6)   return 4;
    if (value <= commonNu / 5)   return 3;
    if (value <= commonNu / 3)   return 2;
    if (value <= commonNu / 2)   return 1;
    if (value <= commonNu)       return 0;
    return ClassCount;
}

static constexpr bool innerClassesAreExact() {
    for (UInt64 commonNu = 0; commonNu < 840; ++commonNu) {
        if (innerClass(commonNu + 1, commonNu) != ClassCount) return false;
        for (std::size_t mode = 0; mode < ClassCount; ++mode) {
            const UInt64 hi = commonNu / Divisors[mode];
            const UInt64 lo = mode + 1 == ClassCount
                ? 1
                : commonNu / Divisors[mode + 1] + 1;
            if (lo > hi) continue;
            if (innerClass(lo, commonNu) != mode) return false;
            if (innerClass(hi, commonNu) != mode) return false;
        }
    }
    return true;
}
static_assert(innerClassesAreExact(),
              "coherent Q=210 cutoff classes must preserve every endpoint");

template<Mode M>
struct Evaluator {
    static inline Int64 eval64(
        const DensePeriodTable& table,
        UInt64 quotient
    ) {
        return evaluatePeriod64(
            table, static_cast<std::size_t>(M), quotient
        );
    }

    static inline Int128 eval128(
        const DensePeriodTable& table,
        const UInt128& quotient
    ) {
        return evaluatePeriod(
            table, static_cast<std::size_t>(M), quotient
        );
    }
};

template<typename Callback>
static inline void dispatchMode(Mode mode, Callback&& callback) {
    switch (mode) {
        case Mode::P1:   callback(std::integral_constant<Mode, Mode::P1>{}); return;
        case Mode::P2:   callback(std::integral_constant<Mode, Mode::P2>{}); return;
        case Mode::P3:   callback(std::integral_constant<Mode, Mode::P3>{}); return;
        case Mode::P5:   callback(std::integral_constant<Mode, Mode::P5>{}); return;
        case Mode::P6:   callback(std::integral_constant<Mode, Mode::P6>{}); return;
        case Mode::P7:   callback(std::integral_constant<Mode, Mode::P7>{}); return;
        case Mode::P10:  callback(std::integral_constant<Mode, Mode::P10>{}); return;
        case Mode::P14:  callback(std::integral_constant<Mode, Mode::P14>{}); return;
        case Mode::P15:  callback(std::integral_constant<Mode, Mode::P15>{}); return;
        case Mode::P21:  callback(std::integral_constant<Mode, Mode::P21>{}); return;
        case Mode::P30:  callback(std::integral_constant<Mode, Mode::P30>{}); return;
        case Mode::P35:  callback(std::integral_constant<Mode, Mode::P35>{}); return;
        case Mode::P42:  callback(std::integral_constant<Mode, Mode::P42>{}); return;
        case Mode::P70:  callback(std::integral_constant<Mode, Mode::P70>{}); return;
        case Mode::P105: callback(std::integral_constant<Mode, Mode::P105>{}); return;
        case Mode::P210: callback(std::integral_constant<Mode, Mode::P210>{}); return;
    }
    std::abort();
}

} // namespace CoherentS2Q210

struct S2Q210Spec {
    using Mode = CoherentS2Q210::Mode;
    inline static constexpr std::size_t ModeCount =
        CoherentS2Q210::ClassCount;
    inline static constexpr UInt64 Period = CoherentS2Q210::Period;
    inline static constexpr const auto& Divisors =
        CoherentS2Q210::Divisors;
    inline static constexpr const auto& Signs =
        CoherentS2Q210::Mobius;

    template<typename T>
    using Accumulator = CoherentS2Q210::Accumulator<T>;

    static constexpr Mode modeFor(UInt64 commonNu, UInt64 value) {
        const std::size_t mode = CoherentS2Q210::innerClass(
            value, commonNu
        );
        if (mode == ModeCount) std::abort();
        return static_cast<Mode>(mode);
    }

    template<typename T>
    static inline Accumulator<T> evalRuntimeDirect(Mode mode, T quotient) {
        return CoherentS2Q210::evaluateDirect(
            static_cast<std::size_t>(mode), quotient
        );
    }

    template<typename T>
    static inline Accumulator<T> evalRuntimePeriod44100(
        const CoherentS2Q210::DensePeriodTable& table,
        Mode mode,
        T quotient
    ) {
        return CoherentS2Q210::evaluatePeriod(
            table, static_cast<std::size_t>(mode), quotient
        );
    }

    template<typename T>
    static inline Accumulator<T> evalRuntimePeriod44100Pair(
        const CoherentS2Q210::DensePeriodTable& table,
        Mode left,
        Mode right,
        T quotient
    ) {
        return CoherentS2Q210::evaluatePairPeriod(
            table, static_cast<std::size_t>(left),
            static_cast<std::size_t>(right), quotient
        );
    }
};

namespace S2Q210Detail {

static inline bool firstResidueAtLeast(
    UInt64 lower,
    UInt64 residue,
    UInt64& first
) {
    UInt64 block = lower - lower % 210;
    if (block <= std::numeric_limits<UInt64>::max() - residue) {
        const UInt64 candidate = block + residue;
        if (candidate >= lower) {
            first = candidate;
            return true;
        }
    }
    if (block > std::numeric_limits<UInt64>::max() - 210) return false;
    block += 210;
    if (block > std::numeric_limits<UInt64>::max() - residue) return false;
    first = block + residue;
    return true;
}

template<typename Callback>
static inline void dispatch(
    UInt64 x1,
    UInt64 x2,
    UInt64 commonNu,
    Callback&& callback
) {
    if (x1 > x2) return;
    constexpr std::array<UInt64, CoherentS2Q210::ClassCount> denominators = {
        210, 105, 70, 42, 35, 30, 21, 15,
         14,  10,  7,  6,  5,  3,  2,  1
    };
    constexpr std::array<CoherentS2Q210::Mode,
                         CoherentS2Q210::ClassCount> modes = {
        CoherentS2Q210::Mode::P210,
        CoherentS2Q210::Mode::P105,
        CoherentS2Q210::Mode::P70,
        CoherentS2Q210::Mode::P42,
        CoherentS2Q210::Mode::P35,
        CoherentS2Q210::Mode::P30,
        CoherentS2Q210::Mode::P21,
        CoherentS2Q210::Mode::P15,
        CoherentS2Q210::Mode::P14,
        CoherentS2Q210::Mode::P10,
        CoherentS2Q210::Mode::P7,
        CoherentS2Q210::Mode::P6,
        CoherentS2Q210::Mode::P5,
        CoherentS2Q210::Mode::P3,
        CoherentS2Q210::Mode::P2,
        CoherentS2Q210::Mode::P1
    };

    UInt64 previous = 0;
    for (std::size_t cell = 0; cell < denominators.size(); ++cell) {
        if (previous == std::numeric_limits<UInt64>::max()) break;
        const UInt64 cutoff = commonNu / denominators[cell];
        const UInt64 lo = std::max(x1, previous + 1);
        const UInt64 hi = std::min(x2, cutoff);
        if (lo <= hi) {
            CoherentS2Q210::dispatchMode(modes[cell], [&](auto modeTag) {
                callback(modeTag, lo, hi);
            });
        }
        previous = cutoff;
    }
}

template<CoherentS2Q210::Mode Mode>
static inline Int64 updateMode64(
    const CoherentS2Q210::DensePeriodTable& table,
    UInt64 numerator,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    if (lo > hi) return 0;
    Int64 result = 0;
    using Evaluator = CoherentS2Q210::Evaluator<Mode>;

    auto addDirect = [&](UInt64 denominator) {
        const UInt64 quotient = numerator / denominator;
        result += Int64(Mu[denominator - L1])
                * Evaluator::eval64(table, quotient);
    };
    auto addCached = [&](UInt64 denominator) {
        const UInt64 quotient = qCache.quotient(numerator, denominator);
        result += Int64(Mu[denominator - L1])
                * Evaluator::eval64(table, quotient);
    };
    auto addClippedBlock = [&](UInt64 block) {
        for (UInt64 residue : CoherentS2Q210::Residues) {
            if (block > std::numeric_limits<UInt64>::max() - residue)
                break;
            const UInt64 denominator = block + residue;
            if (denominator < lo || denominator > hi) continue;
            if constexpr (UseDivisionFree) {
                denominator <= dCAP
                    ? addCached(denominator)
                    : addDirect(denominator);
            } else {
                addDirect(denominator);
            }
        }
    };
    auto addFullDirect = [&](UInt64 block) {
        for (UInt64 residue : CoherentS2Q210::Residues)
            addDirect(block + residue);
    };
    auto addFullCached = [&](UInt64 block) {
        for (UInt64 residue : CoherentS2Q210::Residues)
            addCached(block + residue);
    };
    auto advance = [](UInt64& block) {
        if (block > std::numeric_limits<UInt64>::max() - 210) return false;
        block += 210;
        return true;
    };

    UInt64 block = lo - lo % 210;
    addClippedBlock(block);
    if (!advance(block)) return result;
    if constexpr (UseDivisionFree) {
        while (block <= hi && hi - block >= 209
               && block <= dCAP && dCAP - block >= 209) {
            addFullCached(block);
            if (!advance(block)) return result;
        }
    }
    while (block <= hi && hi - block >= 209) {
        addFullDirect(block);
        if (!advance(block)) return result;
    }
    if (block <= hi) addClippedBlock(block);
    return result;
}

template<CoherentS2Q210::Mode Mode>
static inline void updateMode128Direct(
    const CoherentS2Q210::DensePeriodTable& table,
    const UInt128& numerator,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    Int128& result
) {
    if (lo > hi) return;
    using Evaluator = CoherentS2Q210::Evaluator<Mode>;

    auto add = [&](UInt64 denominator) {
        const Int8 mu = Mu[denominator - L1];
        if (mu == 0) return;
        const UInt128 quotient = numerator / denominator;
        result += Int128(mu) * Evaluator::eval128(table, quotient);
    };
    auto addClippedBlock = [&](UInt64 block) {
        for (UInt64 residue : CoherentS2Q210::Residues) {
            if (block > std::numeric_limits<UInt64>::max() - residue)
                break;
            const UInt64 denominator = block + residue;
            if (denominator >= lo && denominator <= hi)
                add(denominator);
        }
    };
    auto addFullBlock = [&](UInt64 block) {
        for (UInt64 residue : CoherentS2Q210::Residues)
            add(block + residue);
    };
    auto advance = [](UInt64& block) {
        if (block > std::numeric_limits<UInt64>::max() - 210) return false;
        block += 210;
        return true;
    };

    UInt64 block = lo - lo % 210;
    addClippedBlock(block);
    if (!advance(block)) return;
    while (block <= hi && hi - block >= 209) {
        addFullBlock(block);
        if (!advance(block)) return;
    }
    if (block <= hi) addClippedBlock(block);
}

template<CoherentS2Q210::Mode Mode>
static inline void updateMode128PredictedResidue(
    const CoherentS2Q210::DensePeriodTable& table,
    const UInt128& numerator,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    UInt64 residue,
    Int128& result
) {
    UInt64 denominator = 0;
    if (!firstResidueAtLeast(lo, residue, denominator)
        || denominator > hi)
        return;
#ifndef NDEBUG
    assert(denominator > 210);
#endif
    using Evaluator = CoherentS2Q210::Evaluator<Mode>;
    UInt64 qPrev = static_cast<UInt64>(numerator / (denominator - 210));
    UInt64 qCur = static_cast<UInt64>(numerator / denominator);
    const Int8 firstMu = Mu[denominator - L1];
    if (firstMu != 0)
        result += Int128(firstMu) * Evaluator::eval64(table, qCur);

    UInt64 qEst = 0;
    while (hi - denominator >= 210) {
        denominator += 210;
        update_quotients_fixed_stride<210, false>(
            numerator, denominator, qCur, qPrev, qEst
        );
        const Int8 mu = Mu[denominator - L1];
        if (mu != 0)
            result += Int128(mu) * Evaluator::eval64(table, qEst);
    }
}

template<CoherentS2Q210::Mode Mode>
static inline Int128 updateMode128(
    const CoherentS2Q210::DensePeriodTable& table,
    const UInt128& numerator,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const Int8* __restrict Mu,
    UInt64 predictorBoundary
) {
    if (lo > hi) return 0;
    Int128 result = 0;
    const UInt64 directHi = std::min(
        hi, std::max<UInt64>(predictorBoundary, 210)
    );
    updateMode128Direct<Mode>(
        table, numerator, L1, lo, directHi, Mu, result
    );
    if (directHi == hi) return result;

    const UInt64 predictedLo = std::max(lo, directHi + 1);
    for (UInt64 residue : CoherentS2Q210::Residues) {
        updateMode128PredictedResidue<Mode>(
            table, numerator, L1, predictedLo, hi,
            Mu, residue, result
        );
    }
    return result;
}

} // namespace S2Q210Detail

static inline Int64 update_S2_coherent_q210(
    const CoherentS2Q210::DensePeriodTable& table,
    UInt64 numerator,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt64 commonNu,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    Int64 result = 0;
    S2Q210Detail::dispatch(
        x1, x2, commonNu,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr CoherentS2Q210::Mode Mode = decltype(modeTag)::value;
            result += S2Q210Detail::updateMode64<Mode>(
                table, numerator, L1, lo, hi, Mu, qCache, dCAP
            );
        }
    );
    return result;
}

static inline Int128 update_S2_coherent_q210_128(
    const CoherentS2Q210::DensePeriodTable& table,
    const UInt128& numerator,
    UInt64 L1,
    UInt64 x1,
    UInt64 x2,
    const Int8* __restrict Mu,
    UInt64 commonNu
) {
    Int128 result = 0;
    const UInt64 predictorBoundary =
        quotient_predictor_first_unit_curvature<210>(numerator);
    S2Q210Detail::dispatch(
        x1, x2, commonNu,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr CoherentS2Q210::Mode Mode = decltype(modeTag)::value;
            result += S2Q210Detail::updateMode128<Mode>(
                table, numerator, L1, lo, hi, Mu, predictorBoundary
            );
        }
    );
    return result;
}
