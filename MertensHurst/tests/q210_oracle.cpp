#include "../src/S1Q210.h"
#include "../src/S2Q210.h"

#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// ============================================================================
// q210_oracle.cpp — independent arithmetic checks for the Q=210 kernels.
//
// The expected values below come from the literal double divisor sum, not the
// first-difference recurrence used by the production table.
// ============================================================================

namespace {

constexpr std::size_t ClassCount = 16;
constexpr UInt64 Period = 44100;
constexpr std::array<UInt64, ClassCount> Divisors = {
      1,   2,   3,   5,   6,   7,  10,  14,
     15,  21,  30,  35,  42,  70, 105, 210
};
constexpr std::array<Int8, ClassCount> Signs = {
     1, -1, -1, -1,  1, -1,  1,  1,
     1,  1, -1,  1, -1, -1, -1,  1
};
constexpr std::array<Int16, ClassCount> Slopes = {
    10080, 5040, 1680, -336, 1344,  -96,  912, 1632,
     2304, 2784, 2448, 2736, 2496, 2352, 2256, 2304
};

using Table = CoherentS2Q210::DensePeriodTable;
static_assert(std::is_same_v<
    decltype(CoherentS2Q210::evaluateDirect(std::size_t(0), UInt64(0))),
    Int128
>);
static_assert(std::is_same_v<
    decltype(CoherentS2Q210::evaluateDirect(std::size_t(0), UInt128(0))),
    Int128
>);
static_assert(std::is_same_v<
    decltype(CoherentS2Q210::evaluatePeriod(
        std::declval<const Table&>(), std::size_t(0), UInt64(0)
    )),
    Int128
>);
static_assert(std::is_same_v<
    decltype(CoherentS2Q210::evaluatePeriod(
        std::declval<const Table&>(), std::size_t(0), UInt128(0)
    )),
    Int128
>);
static_assert(std::is_same_v<
    decltype(CoherentS2Q210::evaluatePairPeriod(
        std::declval<const Table&>(), std::size_t(0),
        std::size_t(0), UInt64(0)
    )),
    Int128
>);
static_assert(std::is_same_v<
    decltype(CoherentS2Q210::evaluatePairPeriod(
        std::declval<const Table&>(), std::size_t(0),
        std::size_t(0), UInt128(0)
    )),
    Int128
>);

UInt64 checks = 0;

bool require(bool condition, const char* label) {
    ++checks;
    if (condition) return true;
    std::cerr << "Q210 oracle failure: " << label
              << " at check " << checks << std::endl;
    return false;
}

template<typename T>
Int128 literalMode(std::size_t mode, T quotient) {
    Int128 result = 0;
    for (std::size_t outer = 0; outer < ClassCount; ++outer) {
        for (std::size_t inner = 0; inner <= mode; ++inner) {
            const T denominator = T(Divisors[outer] * Divisors[inner]);
            result += Int128(Signs[outer]) * Int128(Signs[inner])
                    * Int128(quotient / denominator);
        }
    }
    return result;
}

UInt128 decimal128(const char* digits) {
    UInt128 value = 0;
    for (const char* digit = digits; *digit != '\0'; ++digit)
        value = value * 10 + UInt128(*digit - '0');
    return value;
}

UInt64 nextRandom(UInt64& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

bool verifyTable(const CoherentS2Q210::DensePeriodTable& table) {
    Int16 minimum = std::numeric_limits<Int16>::max();
    Int16 maximum = std::numeric_limits<Int16>::min();
    for (std::size_t mode = 0; mode < ClassCount; ++mode) {
        if (!require(table.slopes[mode] == Slopes[mode], "table slope"))
            return false;
        if (!require(literalMode(mode, UInt64(Period)) == Slopes[mode],
                     "literal slope"))
            return false;
        for (UInt64 quotient = 0; quotient < Period; ++quotient) {
            const Int128 expected = literalMode(mode, quotient);
            minimum = std::min(minimum, table.remainders[mode][quotient]);
            maximum = std::max(maximum, table.remainders[mode][quotient]);
            if (!require(expected == table.remainders[mode][quotient],
                         "table remainder"))
                return false;
        }
    }
    return require(minimum == -339, "table minimum")
        && require(maximum == 10080, "table maximum")
        && require(sizeof(table) == 1411264, "aligned table size");
}

bool verifyEvaluators(const CoherentS2Q210::DensePeriodTable& table) {
    constexpr UInt64 UInt64Max = std::numeric_limits<UInt64>::max();
    constexpr UInt64 finalPeriod = UInt64Max / Period * Period;
    constexpr std::array<UInt64, 10> edges = {
        0, 1, Period - 1, Period, Period + 1,
        (UInt64(1) << 63) - 1, UInt64(1) << 63,
        finalPeriod - 1, finalPeriod, UInt64Max
    };
    for (UInt64 quotient : edges) {
        for (std::size_t mode = 0; mode < ClassCount; ++mode) {
            const Int128 expected = literalMode(mode, quotient);
            if (!require(CoherentS2Q210::evaluateDirect(mode, quotient)
                         == expected, "UInt64 direct evaluator"))
                return false;
            if (!require(CoherentS2Q210::evaluatePeriod(
                             table, mode, quotient
                         ) == expected,
                         "UInt64 period evaluator"))
                return false;
        }
        for (std::size_t left = 0; left < ClassCount; ++left) {
            for (std::size_t right = 0; right < ClassCount; ++right) {
                const Int128 expected = literalMode(left, quotient)
                                      + literalMode(right, quotient);
                if (!require(CoherentS2Q210::evaluatePairPeriod(
                                 table, left, right, quotient
                             ) == expected,
                             "UInt64 pair evaluator"))
                    return false;
            }
        }
    }

    constexpr UInt64 HotLimit = 1000000000000000000ULL;
    constexpr std::array<UInt64, 9> hotEdges = {
        0, 1, Period - 1, Period, Period + 1,
        HotLimit - Period, HotLimit - 1, HotLimit,
        HotLimit / Period * Period
    };
    for (UInt64 quotient : hotEdges) {
        for (std::size_t mode = 0; mode < ClassCount; ++mode) {
            const Int128 expected = literalMode(mode, quotient);
            if (!require(expected >= std::numeric_limits<Int64>::min()
                         && expected <= std::numeric_limits<Int64>::max(),
                         "hot single representability")
                || !require(CoherentS2Q210::evaluatePeriod64(
                                table, mode, quotient
                            ) == Int64(expected),
                            "bounded UInt64 hot evaluator"))
                return false;
        }
        for (std::size_t left = 0; left < ClassCount; ++left) {
            for (std::size_t right = 0; right < ClassCount; ++right) {
                const Int128 expected = literalMode(left, quotient)
                                      + literalMode(right, quotient);
                if (!require(expected >= std::numeric_limits<Int64>::min()
                             && expected <= std::numeric_limits<Int64>::max(),
                             "hot pair representability")
                    || !require(CoherentS2Q210::evaluatePairPeriod64(
                                    table, left, right, quotient
                                ) == Int64(expected),
                                "bounded UInt64 hot pair evaluator"))
                    return false;
            }
        }
    }

    if (!require(literalMode(0, UInt64(211)) == 49,
                 "orientation witness P1")
        || !require(literalMode(1, UInt64(211)) == 25,
                    "orientation witness P2")
        || !require(CoherentS2Q210::evaluatePairPeriod(
                        table, 0, 1, UInt64(211)
                    ) == 74,
                    "orientation witness pair"))
        return false;

    const std::array<UInt128, 12> wideEdges = {
        UInt128(0), UInt128(1), UInt128(Period - 1), UInt128(Period),
        (UInt128(1) << 64) - 1,
        UInt128(1) << 64,
        (UInt128(1) << 64) + 1,
        decimal128("9999999999999999999999999"),
        decimal128("10000000000000000000000000"),
        decimal128("10000000000000000000000001"),
        (UInt128(1) << 96) + UInt128(Period - 1),
        (UInt128(1) << 112) - 1
    };
    for (UInt128 quotient : wideEdges) {
        for (std::size_t mode = 0; mode < ClassCount; ++mode) {
            const Int128 expected = literalMode(mode, quotient);
            if (!require(CoherentS2Q210::evaluateDirect(mode, quotient)
                         == expected, "UInt128 direct evaluator"))
                return false;
            if (!require(CoherentS2Q210::evaluatePeriod(
                             table, mode, quotient
                         ) == expected,
                         "UInt128 period evaluator"))
                return false;
        }
        for (std::size_t mode = 0; mode < ClassCount; ++mode) {
            const std::size_t other = ClassCount - mode - 1;
            const Int128 expected = literalMode(mode, quotient)
                                  + literalMode(other, quotient);
            if (!require(CoherentS2Q210::evaluatePairPeriod(
                             table, mode, other, quotient
                         ) == expected,
                         "UInt128 pair evaluator"))
                return false;
        }
    }

    UInt64 random = 0x9e3779b97f4a7c15ULL;
    for (UInt64 trial = 0; trial < 4000; ++trial) {
        const UInt64 quotient = nextRandom(random) % (HotLimit + 1);
        const std::size_t left = nextRandom(random) % ClassCount;
        const std::size_t right = nextRandom(random) % ClassCount;
        const Int128 leftExpected = literalMode(left, quotient);
        const Int128 pairExpected = leftExpected
                                  + literalMode(right, quotient);
        if (!require(CoherentS2Q210::evaluatePeriod64(
                         table, left, quotient
                     ) == Int64(leftExpected),
                     "random bounded UInt64 hot evaluator")
            || !require(CoherentS2Q210::evaluatePairPeriod64(
                            table, left, right, quotient
                        ) == Int64(pairExpected),
                        "random bounded UInt64 hot pair evaluator"))
            return false;
    }

    const UInt128 wideMask = (UInt128(1) << 112) - 1;
    for (UInt64 trial = 0; trial < 1000; ++trial) {
        const UInt128 quotient = (
            (UInt128(nextRandom(random)) << 64) | nextRandom(random)
        ) & wideMask;
        const std::size_t left = nextRandom(random) % ClassCount;
        const std::size_t right = nextRandom(random) % ClassCount;
        const Int128 expected = literalMode(left, quotient)
                              + literalMode(right, quotient);
        if (!require(CoherentS2Q210::evaluatePairPeriod(
                         table, left, right, quotient
                     ) == expected,
                     "random bounded UInt128 pair evaluator"))
            return false;
    }
    return true;
}

std::size_t expectedInnerClass(UInt64 value, UInt64 commonNu) {
    std::size_t expected = ClassCount;
    for (std::size_t mode = 0; mode < ClassCount; ++mode) {
        if (value <= commonNu / Divisors[mode]) expected = mode;
    }
    return expected;
}

bool verifyDispatch(UInt64 commonNu, UInt64 x1, UInt64 x2) {
    if (x1 > x2) return true;
    const UInt64 width = x2 - x1 + 1;
    std::vector<UInt8> coverage(static_cast<std::size_t>(width), 0);
    bool valid = true;
    S2Q210Detail::dispatch(
        x1, x2, commonNu,
        [&](auto modeTag, UInt64 lo, UInt64 hi) {
            constexpr auto mode = decltype(modeTag)::value;
            if (lo > hi || lo < x1 || hi > x2) valid = false;
            for (UInt64 value = lo;; ++value) {
                const std::size_t expected = expectedInnerClass(
                    value, commonNu
                );
                if (expected != static_cast<std::size_t>(mode))
                    valid = false;
                ++coverage[static_cast<std::size_t>(value - x1)];
                if (value == hi) break;
            }
        }
    );
    if (!require(valid, "inner dispatcher mode")) return false;
    for (UInt64 value = x1;; ++value) {
        const UInt8 expected = value >= 1 && value <= commonNu ? 1 : 0;
        if (!require(coverage[static_cast<std::size_t>(value - x1)]
                     == expected,
                     "inner dispatcher ownership"))
            return false;
        if (value == x2) break;
    }
    return true;
}

bool verifyInnerClasses() {
    UInt64 ownedChecks = 0;
    UInt64 emptyCells = 0;
    for (UInt64 commonNu = 0; commonNu <= 420; ++commonNu) {
        for (UInt64 value = 1; value <= commonNu; ++value) {
            if (!require(CoherentS2Q210::innerClass(value, commonNu)
                         == expectedInnerClass(value, commonNu),
                         "inner cutoff class"))
                return false;
            ++ownedChecks;
        }
        if (!require(CoherentS2Q210::innerClass(
                         commonNu + 1, commonNu
                     ) == ClassCount,
                     "inactive inner cutoff"))
            return false;
        for (std::size_t mode = 0; mode < ClassCount; ++mode) {
            const UInt64 hi = commonNu / Divisors[mode];
            const UInt64 lo = mode + 1 == ClassCount
                ? 1
                : commonNu / Divisors[mode + 1] + 1;
            emptyCells += lo > hi;
        }
        if (!verifyDispatch(commonNu, 1, commonNu + 1)) return false;
    }
    if (!require(ownedChecks == 88410, "exhaustive owned cutoff count")
        || !require(emptyCells == 928, "tied empty-cell count"))
        return false;

    constexpr std::array<UInt64, 4> largeSplits = {
        1000037,
        (UInt64(1) << 32) + 12345,
        (UInt64(1) << 63) + 54321,
        std::numeric_limits<UInt64>::max() - 100000
    };
    UInt64 largeChecks = 0;
    for (UInt64 commonNu : largeSplits) {
        for (UInt64 divisor : Divisors) {
            const UInt64 cutoff = commonNu / divisor;
            for (UInt64 value : {cutoff - 1, cutoff, cutoff + 1}) {
                if (!require(CoherentS2Q210::innerClass(value, commonNu)
                             == expectedInnerClass(value, commonNu),
                             "large inner endpoint"))
                    return false;
                ++largeChecks;
            }
            if (!verifyDispatch(commonNu, cutoff - 1, cutoff + 1))
                return false;
        }
    }
    if (!require(ownedChecks + largeChecks == 88602,
                 "total cutoff endpoint count"))
        return false;

    constexpr UInt64 UInt64Max = std::numeric_limits<UInt64>::max();
    for (UInt64 divisor : Divisors) {
        const UInt64 cutoff = UInt64Max / divisor;
        const UInt64 lo = cutoff == 0 ? 0 : cutoff - 1;
        const UInt64 hi = cutoff == UInt64Max ? cutoff : cutoff + 1;
        for (UInt64 value = lo;; ++value) {
            if (!require(CoherentS2Q210::innerClass(value, UInt64Max)
                         == expectedInnerClass(value, UInt64Max),
                         "full UInt64 inner endpoint"))
                return false;
            if (value == hi) break;
        }
        if (!verifyDispatch(UInt64Max, lo, hi)) return false;
    }
    return true;
}

UInt64 literalBoundaryFactor(UInt64 value) {
    Int128 result = 0;
    for (std::size_t divisor = 0; divisor < ClassCount; ++divisor)
        result += Int128(Signs[divisor])
                * Int128(value / Divisors[divisor]);
    return static_cast<UInt64>(result);
}

bool verifyBoundaryValue(UInt64 value) {
    const UInt64 expected = literalBoundaryFactor(value);
    return require(coherentS1BoundaryFactorQ210(value) == expected,
                   "S1 literal boundary factor")
        && require(coherentS1BoundaryFactorQ210(value)
                   == coherentS1BoundaryFactorQ30(value)
                    - coherentS1BoundaryFactorQ30(value / 7),
                   "S1 H30 difference");
}

bool verifyBoundaryFactor() {
    UInt64 coprimeCount = 0;
    for (UInt64 value = 0; value <= 100000; ++value) {
        if (value != 0 && std::gcd(value, UInt64(210)) == 1)
            ++coprimeCount;
        if (!verifyBoundaryValue(value)) return false;
        if (value <= 20000
            && !require(coherentS1BoundaryFactorQ210(value) == coprimeCount,
                        "S1 direct coprime count"))
            return false;
    }
    constexpr UInt64 Midpoint = UInt64(1) << 63;
    for (UInt64 offset = 0; offset <= 100; ++offset) {
        if (!verifyBoundaryValue(Midpoint - offset)
            || !verifyBoundaryValue(Midpoint + offset))
            return false;
    }
    constexpr UInt64 UInt64Max = std::numeric_limits<UInt64>::max();
    for (UInt64 offset = 0; offset < 1000; ++offset) {
        if (!verifyBoundaryValue(UInt64Max - offset)) return false;
    }
    return true;
}

template<typename Helper>
bool verifyResidueHelper(const char* label, Helper&& helper) {
    constexpr UInt64 UInt64Max = std::numeric_limits<UInt64>::max();
    for (UInt64 offset = 0; offset < 1000; ++offset) {
        const UInt64 lower = UInt64Max - offset;
        for (UInt64 residue : CoherentS2Q210::Residues) {
            UInt128 expected = UInt128(lower / 210) * 210 + residue;
            if (expected < lower) expected += 210;
            const bool expectedFound = expected <= UInt64Max;
            UInt64 first = 0;
            const bool found = helper(lower, residue, first);
            if (!require(found == expectedFound, label)) return false;
            if (found
                && !require(first == static_cast<UInt64>(expected), label))
                return false;
        }
    }
    return true;
}

bool verifyResidueEdges() {
    return verifyResidueHelper(
        "S1 last-UInt64 residue",
        [](UInt64 lower, UInt64 residue, UInt64& first) {
            return S1Q210Detail::firstResidueAtLeast(
                lower, residue, first
            );
        }
    ) && verifyResidueHelper(
        "S2 last-UInt64 residue",
        [](UInt64 lower, UInt64 residue, UInt64& first) {
            return S2Q210Detail::firstResidueAtLeast(
                lower, residue, first
            );
        }
    );
}

Int8 syntheticMu(UInt64 offset) {
    return static_cast<Int8>(Int64((offset * 17 + 5) % 3) - 1);
}

bool wheel210Accepts(UInt64 value) {
    return std::gcd(value, UInt64(210)) == 1;
}

bool verifyS2Traversals(
    const CoherentS2Q210::DensePeriodTable& table
) {
    constexpr std::array<Int64, 6> offsets = {
        -211, -1, 0, 1, 209, 421
    };
    const std::array<UInt128, 2> wideNumerators = {
        decimal128("100000000000000000000"),
        decimal128("10000000000000000000000000")
    };
    for (UInt128 numerator : wideNumerators) {
        const UInt64 boundary =
            quotient_predictor_first_unit_curvature<210>(numerator);
        for (Int64 offset : offsets) {
            const UInt64 middle = offset < 0
                ? boundary - UInt64(-offset)
                : boundary + UInt64(offset);
            const UInt64 x1 = middle - 421;
            const UInt64 x2 = middle + 631;
            std::vector<Int8> mu(static_cast<std::size_t>(x2 - x1 + 1));
            for (UInt64 index = 0; index < mu.size(); ++index)
                mu[index] = syntheticMu(index);

            for (std::size_t cell = 0; cell < ClassCount; ++cell) {
                const UInt64 commonNu = Divisors[cell] * middle;
                Int128 expected = 0;
                for (UInt64 value = x1;; ++value) {
                    const std::size_t mode = expectedInnerClass(
                        value, commonNu
                    );
                    if (mode != ClassCount && wheel210Accepts(value)) {
                        expected += Int128(mu[value - x1])
                                  * literalMode(mode, numerator / value);
                    }
                    if (value == x2) break;
                }
                const Int128 actual = update_S2_coherent_q210_128(
                    table, numerator, x1, x1, x2,
                    mu.data(), commonNu
                );
                if (!require(actual == expected,
                             "wide S2 stride-210 traversal"))
                    return false;
            }
        }
    }

    constexpr UInt64 numerator = 1000000000000000000ULL;
    const UInt64 boundary =
        quotient_predictor_first_unit_curvature<210>(numerator);
    const UInt64 x1 = boundary - 419;
    const UInt64 x2 = boundary + 637;
    std::vector<Int8> mu(static_cast<std::size_t>(x2 - x1 + 1));
    for (UInt64 index = 0; index < mu.size(); ++index)
        mu[index] = syntheticMu(index + 1);
    QuotientCache qCache;
    for (std::size_t cell = 0; cell < ClassCount; ++cell) {
        const UInt64 commonNu = Divisors[cell] * boundary;
        Int128 expected = 0;
        for (UInt64 value = x1;; ++value) {
            const std::size_t mode = expectedInnerClass(value, commonNu);
            if (mode != ClassCount && wheel210Accepts(value)) {
                expected += Int128(mu[value - x1])
                          * literalMode(mode, numerator / value);
            }
            if (value == x2) break;
        }
        const Int64 actual = update_S2_coherent_q210(
            table, numerator, x1, x1, x2,
            mu.data(), commonNu, qCache, 0
        );
        if (!require(Int128(actual) == expected,
                     "narrow S2 wheel-210 traversal"))
            return false;
    }
    return true;
}

bool verifyS1Traversals() {
    const UInt128 numerator = decimal128("10000000000000000000");
    const UInt64 boundary =
        quotient_predictor_first_unit_curvature<210>(numerator);
    constexpr std::array<Int64, 4> offsets = {-500, 0, 500, 1000};
    QuotientCache qCache;

    for (Int64 offset : offsets) {
        const UInt64 middle = offset < 0
            ? boundary - UInt64(-offset)
            : boundary + UInt64(offset);
        const UInt64 start = middle - 631;
        const UInt64 end = middle + 1891;
        const UInt64 L1 = static_cast<UInt64>(numerator / end);
        const UInt64 L2 = static_cast<UInt64>(numerator / start);
        const UInt64 length = L2 - L1 + 1;
        std::vector<Int8> residual(static_cast<std::size_t>(length));
        for (UInt64 index = 0; index < length; ++index)
            residual[index] = static_cast<Int8>(
                Int64((index * 29 + 11) % 17) - 8
            );
        constexpr UInt64 Stride = UInt64(1)
                                << MertensSieveDetail::STRIDE_LOG;
        std::vector<Int16> coarse(
            static_cast<std::size_t>((length + Stride - 1) / Stride), 0
        );

        Int128 expected = 0;
        for (UInt64 value = start;; ++value) {
            if (wheel210Accepts(value)) {
                const UInt64 quotient = static_cast<UInt64>(
                    numerator / value
                );
                expected += residual[quotient - L1];
            }
            if (value == end) break;
        }

        const Int128 separate = S1Q210Detail::sumCoprime210(
            numerator, L1, L2, start, end,
            coarse.data(), residual.data(), qCache, 0, false
        );
        const Int128 interleaved = S1Q210Detail::sumCoprime210(
            numerator, L1, L2, start, end,
            coarse.data(), residual.data(), qCache, 0, true
        );
        if (!require(separate == expected,
                     "wide S1 separate stride-210 traversal")
            || !require(interleaved == expected,
                        "wide S1 interleaved stride-210 traversal"))
            return false;
    }
    return true;
}

bool dumpPayload(
    const CoherentS2Q210::DensePeriodTable& table,
    const std::string& path
) {
    std::vector<UInt8> payload;
    payload.reserve(1411232);
    auto append = [&](Int16 value) {
        const UInt16 bits = static_cast<UInt16>(value);
        payload.push_back(static_cast<UInt8>(bits));
        payload.push_back(static_cast<UInt8>(bits >> 8));
    };
    for (Int16 slope : table.slopes) append(slope);
    for (const auto& row : table.remainders) {
        for (Int16 remainder : row) append(remainder);
    }
    if (!require(payload.size() == 1411232, "logical payload size"))
        return false;

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return require(false, "open payload output");
    output.write(
        reinterpret_cast<const char*>(payload.data()),
        static_cast<std::streamsize>(payload.size())
    );
    return require(output.good(), "write payload output");
}

} // namespace

int main(int argc, char** argv) {
    CoherentS2Q210::DensePeriodTable table;
    if (!verifyTable(table)
        || !verifyEvaluators(table)
        || !verifyInnerClasses()
        || !verifyBoundaryFactor()
        || !verifyResidueEdges()
        || !verifyS2Traversals(table)
        || !verifyS1Traversals())
        return 1;

    if (argc == 3 && std::string(argv[1]) == "--dump") {
        if (!dumpPayload(table, argv[2])) return 1;
    } else if (argc != 1) {
        std::cerr << "usage: q210_oracle [--dump <path>]" << std::endl;
        return 2;
    }

    std::cout << "Q210 oracle passed " << checks << " checks" << std::endl;
    return 0;
}
