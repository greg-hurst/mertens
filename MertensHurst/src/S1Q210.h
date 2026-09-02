#pragma once

// ============================================================================
// S1Q210.h — completed outer-Q=210 S1 kernels.
//
// Completing the divisor group through 2*3*5*7 leaves the 48 residue classes
// coprime to 210. Exact ranges use an eight-lane quotient stepper in
// denominator order. Predictor tails retain separate streams in Loop 0/1 and
// interleave those streams in Loop 2.
// ============================================================================

#include "S1Q30.h"
#include "QuotientStepper.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <type_traits>

namespace S1Q210Detail {

inline constexpr std::size_t StepperLanes = 8;
using Stepper = QuotientStepper<StepperLanes>;

inline constexpr std::array<UInt8, 48> Residues = {
      1,  11,  13,  17,  19,  23,  29,  31,
     37,  41,  43,  47,  53,  59,  61,  67,
     71,  73,  79,  83,  89,  97, 101, 103,
    107, 109, 113, 121, 127, 131, 137, 139,
    143, 149, 151, 157, 163, 167, 169, 173,
    179, 181, 187, 191, 193, 197, 199, 209
};
static_assert(Residues.size() % Stepper::Width == 0);

inline constexpr std::size_t StepperGroups =
    Residues.size() / Stepper::Width;

inline constexpr std::array<
    Stepper::Batch, StepperGroups
> StepperDistances = [] {
    std::array<Stepper::Batch, StepperGroups> distances{};
    for (std::size_t group = 0; group < StepperGroups; ++group) {
        const std::size_t previousGroup = group == 0
            ? StepperGroups - 1
            : group - 1;
        for (std::size_t lane = 0; lane < Stepper::Width; ++lane) {
            const UInt64 next = Residues[Stepper::Width * group + lane]
                              + (group == 0 ? 210 : 0);
            const UInt64 previous =
                Residues[Stepper::Width * previousGroup + lane];
            distances[group][lane] = next - previous;
        }
    }
    return distances;
}();

static constexpr UInt64 stepperMaximumDistance() {
    UInt64 maximum = 0;
    for (const Stepper::Batch& distances : StepperDistances) {
        for (UInt64 distance : distances)
            maximum = std::max(maximum, distance);
    }
    return maximum;
}

inline constexpr UInt64 StepperMaximumDistance = stepperMaximumDistance();
static_assert(StepperMaximumDistance == 40);

template<typename TArg>
static inline bool stepperSupportsRange(
    const TArg& y,
    UInt64 start,
    UInt64 end
) {
    if (start > end) return false;
    UInt64 fullBlock = start - start % 210;
    if (fullBlock + 1 < start) {
        if (fullBlock > std::numeric_limits<UInt64>::max() - 210)
            return false;
        fullBlock += 210;
    }
    return fullBlock <= end && end - fullBlock >= 209
        && Stepper::supports(
            y, fullBlock + Residues[0], StepperMaximumDistance
        );
}

static constexpr bool verifyResidues() {
    std::size_t count = 0;
    for (UInt64 value = 1; value < 210; ++value) {
        const bool accepted = (value & 1ULL) != 0
                           && value % 3 != 0
                           && value % 5 != 0
                           && value % 7 != 0;
        if (!accepted) continue;
        if (count >= Residues.size() || Residues[count] != value)
            return false;
        ++count;
    }
    return count == Residues.size();
}
static_assert(verifyResidues(),
              "wheel-210 residues must contain every coprime class");

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

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumDirectScalar(
    const TArg& y,
    UInt64 L1,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    Acc result = 0;
    if (start > end) return result;

    auto add = [&](UInt64 denominator) {
        const UInt64 quotient = S1Q6Detail::exactSparseQuotient(
            y, denominator, qCache, dCAP
        );
        result += static_cast<Acc>(GET_M(M, R, L1, quotient));
    };
    auto addClippedBlock = [&](UInt64 block) {
        for (UInt64 residue : Residues) {
            if (block > std::numeric_limits<UInt64>::max() - residue)
                break;
            const UInt64 denominator = block + residue;
            if (denominator >= start && denominator <= end)
                add(denominator);
        }
    };
    auto addFullBlock = [&](UInt64 block) {
        for (UInt64 residue : Residues)
            add(block + residue);
    };
    auto advance = [](UInt64& block) {
        if (block > std::numeric_limits<UInt64>::max() - 210) return false;
        block += 210;
        return true;
    };

    UInt64 block = start - start % 210;
    addClippedBlock(block);
    if (!advance(block)) return result;
    while (block <= end && end - block >= 209) {
        addFullBlock(block);
        if (!advance(block)) return result;
    }
    if (block <= end) addClippedBlock(block);
    return result;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumDirectStepped(
    const TArg& y,
    UInt64 L1,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    Acc result = 0;
    if (start > end) return result;

    if (!stepperSupportsRange(y, start, end)) {
        return sumDirectScalar(
            y, L1, start, end, M, R, qCache, dCAP
        );
    }

    UInt64 block = start - start % 210;
    UInt64 fullBlock = block;
    if (fullBlock + 1 < start) fullBlock += 210;

    if (start < fullBlock) {
        result += sumDirectScalar(
            y, L1, start, std::min(end, fullBlock - 1),
            M, R, qCache, dCAP
        );
    }

    auto denominators = [](UInt64 base, std::size_t group) {
        Stepper::Batch values;
        for (std::size_t lane = 0; lane < Stepper::Width; ++lane)
            values[lane] = base + Residues[Stepper::Width * group + lane];
        return values;
    };
    auto add = [&](const Stepper::Batch& quotients) {
        for (UInt64 quotient : quotients)
            result += static_cast<Acc>(GET_M(M, R, L1, quotient));
    };

    Stepper stepper;
    add(stepper.initialize(y, denominators(fullBlock, 0)));
    for (std::size_t group = 1; group < StepperGroups; ++group) {
        add(stepper.step(
            denominators(fullBlock, group), StepperDistances[group]
        ));
    }

    block = fullBlock;
    for (;;) {
        if (std::numeric_limits<UInt64>::max() - block < 210)
            return result;
        block += 210;
        if (block > end || end - block < 209) break;
        for (std::size_t group = 0; group < StepperGroups; ++group) {
            add(stepper.step(
                denominators(block, group), StepperDistances[group]
            ));
        }
    }

    if (block <= end) {
        result += sumDirectScalar(
            y, L1, block, end, M, R, qCache, dCAP
        );
    }
    return result;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumDirect(
    const TArg& y,
    UInt64 L1,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;

    if constexpr (std::is_same_v<TArg, UInt64> && !UseDivisionFree) {
        return sumDirectScalar(
            y, L1, start, end, M, R, qCache, dCAP
        );
    } else if constexpr (std::is_same_v<TArg, UInt64>) {
        Acc result = 0;
        const UInt64 cacheEnd = std::min(end, dCAP);
        if (start <= cacheEnd) {
            result += sumDirectScalar(
                y, L1, start, cacheEnd, M, R, qCache, dCAP
            );
        }
        if (cacheEnd == end) return result;

        const UInt64 steppedStart = std::max(start, cacheEnd + 1);
        result += sumDirectStepped(
            y, L1, steppedStart, end, M, R, qCache, dCAP
        );
        return result;
    } else {
        return sumDirectStepped(
            y, L1, start, end, M, R, qCache, dCAP
        );
    }
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumSeparatePredicted(
    const TArg& y,
    UInt64 L1,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    Acc result = 0;

    for (UInt64 residue : Residues) {
        UInt64 denominator = 0;
        if (!firstResidueAtLeast(start, residue, denominator)
            || denominator > end)
            continue;
#ifndef NDEBUG
        assert(denominator > 210);
#endif
        UInt64 qPrev = S1Q6Detail::exactSparseQuotient(
            y, denominator - 210, qCache, dCAP
        );
        UInt64 qCur = S1Q6Detail::exactSparseQuotient(
            y, denominator, qCache, dCAP
        );
        result += static_cast<Acc>(GET_M(M, R, L1, qCur));

        UInt64 qEst = 0;
        while (end - denominator >= 210) {
            denominator += 210;
            update_quotients_fixed_stride<210, false>(
                y, denominator, qCur, qPrev, qEst
            );
            result += static_cast<Acc>(GET_M(M, R, L1, qEst));
        }
    }
    return result;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumInterleavedPredicted(
    const TArg& y,
    UInt64 L1,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    struct State {
        UInt64 qPrev = 0;
        UInt64 qCur = 0;
        bool started = false;
    };
    std::array<State, Residues.size()> states{};

    for (std::size_t state = 0; state < Residues.size(); ++state) {
        UInt64 denominator = 0;
        if (!firstResidueAtLeast(start, Residues[state], denominator)
            || denominator > end)
            continue;
#ifndef NDEBUG
        assert(denominator > 210);
#endif
        states[state].qPrev = S1Q6Detail::exactSparseQuotient(
            y, denominator - 210, qCache, dCAP
        );
        states[state].qCur = S1Q6Detail::exactSparseQuotient(
            y, denominator, qCache, dCAP
        );
    }

    Acc result = 0;
    auto add = [&](std::size_t state, UInt64 denominator) {
        UInt64 quotient = states[state].qCur;
        if (states[state].started) {
            update_quotients_fixed_stride<210, false>(
                y, denominator, states[state].qCur,
                states[state].qPrev, quotient
            );
        } else {
            states[state].started = true;
        }
        result += static_cast<Acc>(GET_M(M, R, L1, quotient));
    };
    auto addClippedBlock = [&](UInt64 block) {
        for (std::size_t state = 0; state < Residues.size(); ++state) {
            const UInt64 residue = Residues[state];
            if (block > std::numeric_limits<UInt64>::max() - residue)
                break;
            const UInt64 denominator = block + residue;
            if (denominator >= start && denominator <= end)
                add(state, denominator);
        }
    };
    auto addFullBlock = [&](UInt64 block) {
        for (std::size_t state = 0; state < Residues.size(); ++state)
            add(state, block + Residues[state]);
    };
    auto advance = [](UInt64& block) {
        if (block > std::numeric_limits<UInt64>::max() - 210) return false;
        block += 210;
        return true;
    };

    UInt64 block = start - start % 210;
    addClippedBlock(block);
    if (!advance(block)) return result;
    while (block <= end && end - block >= 209) {
        addFullBlock(block);
        if (!advance(block)) return result;
    }
    if (block <= end) addClippedBlock(block);
    return result;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumCoprime210(
    const TArg& y,
    UInt64 L1,
    UInt64 L2,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP,
    bool interleavePredictors
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    if (L1 == 0 || L1 > L2 || start > end) return Acc(0);

    const TArg loBySegment = L2 == std::numeric_limits<UInt64>::max()
        ? TArg(1)
        : y / TArg(L2 + 1) + TArg(1);
    const TArg hiBySegment = y / TArg(L1);
    if (loBySegment > TArg(end) || hiBySegment < TArg(start)) return Acc(0);
    const UInt64 lo = std::max(start, static_cast<UInt64>(loBySegment));
    const UInt64 hi = hiBySegment >= TArg(end)
        ? end
        : static_cast<UInt64>(hiBySegment);
    if (lo > hi) return Acc(0);

    if constexpr (std::is_same_v<TArg, UInt64> && !UseDivisionFree) {
        return sumDirect(y, L1, lo, hi, M, R, qCache, dCAP);
    }

    if constexpr (std::is_same_v<TArg, UInt128> && !UseDivisionFree) {
        if (stepperSupportsRange(y, lo, hi))
            return sumDirect(y, L1, lo, hi, M, R, qCache, dCAP);
    }

    UInt64 exactThrough = S1Q6Detail::sparsePredictorBoundary<210>(
        y, lo, hi, dCAP
    );
    if (exactThrough == 0)
        return sumDirect(y, L1, lo, hi, M, R, qCache, dCAP);
    if constexpr (std::is_same_v<TArg, UInt64> && UseDivisionFree)
        exactThrough = std::max(exactThrough, dCAP);
    exactThrough = std::min(exactThrough, hi);

    Acc result = sumDirect(
        y, L1, lo, exactThrough, M, R, qCache, dCAP
    );
    if (exactThrough == hi) return result;

    const UInt64 predictedLo = exactThrough + 1;
    if (interleavePredictors) {
        result += sumInterleavedPredicted(
            y, L1, predictedLo, hi, M, R, qCache, dCAP
        );
    } else {
        result += sumSeparatePredicted(
            y, L1, predictedLo, hi, M, R, qCache, dCAP
        );
    }
    return result;
}

} // namespace S1Q210Detail

static inline UInt64 coherentS1BoundaryFactorQ210(UInt64 commonKappa) {
    return coherentS1BoundaryFactorQ30(commonKappa)
         - coherentS1BoundaryFactorQ30(commonKappa / 7);
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg>
evaluateS1OuterQ210ZeroComplete(
    const TArg& y,
    UInt64 lowerExclusive,
    UInt64 commonKappa,
    UInt64 L1,
    UInt64 L2,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP,
    bool loop2
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    if (lowerExclusive == std::numeric_limits<UInt64>::max()) return Acc(0);
    return S1Q210Detail::sumCoprime210(
        y, L1, L2, lowerExclusive + 1, commonKappa,
        M, R, qCache, dCAP, loop2
    );
}
