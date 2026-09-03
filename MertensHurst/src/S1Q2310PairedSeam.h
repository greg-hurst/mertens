#pragma once

// ============================================================================
// S1Q2310PairedSeam.h — exact Q210-to-Q2310 paired-row S1 kernel.
//
// Pairing the existing Q210 rows a and 11a removes the multiples of eleven
// from the parent row.  The parent's native split is retained; the child's
// different split leaves one short Q210 interval in child coordinates.
// ============================================================================

#include "S1Q210.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <type_traits>

namespace S1Q2310PairedSeamDetail {

inline constexpr UInt64 Wheel = 2310;
inline constexpr std::size_t ResidueCount = 480;
inline constexpr std::size_t PhaseCount = 11;
inline constexpr std::size_t StepperLanes = 8;
using Stepper = QuotientStepper<StepperLanes>;

static_assert(Wheel == 210 * PhaseCount);
static_assert(Wheel <= UInt64(std::numeric_limits<UInt16>::max()) + 1);
static_assert(PhaseCount != 0 && ResidueCount != 0);

static constexpr bool wheelAccepts(UInt64 value) {
    return (value & 1ULL) != 0
        && value % 3 != 0
        && value % 5 != 0
        && value % 7 != 0
        && value % 11 != 0;
}

inline constexpr std::array<UInt16, ResidueCount> Residues = [] {
    std::array<UInt16, ResidueCount> residues{};
    std::size_t count = 0;
    for (UInt64 value = 1; value < Wheel; ++value) {
        if (wheelAccepts(value))
            residues[count++] = static_cast<UInt16>(value);
    }
    return residues;
}();

static constexpr bool verifyResidues() {
    std::size_t count = 0;
    for (UInt64 value = 1; value < Wheel; ++value) {
        if (!wheelAccepts(value)) continue;
        if (count >= Residues.size() || Residues[count] != value)
            return false;
        ++count;
    }
    return count == Residues.size();
}
static_assert(verifyResidues(),
              "wheel-2310 residues must contain every coprime class");
static_assert(Residues.size() % Stepper::Width == 0);

inline constexpr std::array<
    std::array<UInt8, S1Q210Detail::Residues.size()>, PhaseCount
> PhaseResidues = [] {
    std::array<
        std::array<UInt8, S1Q210Detail::Residues.size()>, PhaseCount
    > residues{};
    for (UInt64 phase = 0; phase < PhaseCount; ++phase) {
        std::size_t count = 0;
        for (UInt64 residue : S1Q210Detail::Residues) {
            if ((210 * phase + residue) % 11 != 0)
                residues[phase][count++] = static_cast<UInt8>(residue);
        }
    }
    return residues;
}();

inline constexpr std::array<UInt8, PhaseCount> PhaseCounts = [] {
    std::array<UInt8, PhaseCount> counts{};
    for (UInt64 phase = 0; phase < PhaseCount; ++phase) {
        for (UInt64 residue : S1Q210Detail::Residues) {
            counts[phase] += (210 * phase + residue) % 11 != 0;
        }
    }
    return counts;
}();

static constexpr bool verifyPhaseResidues() {
    std::size_t total = 0;
    for (UInt64 phase = 0; phase < PhaseCount; ++phase) {
        if (PhaseCounts[phase] == 0
            || PhaseCounts[phase] > PhaseResidues[phase].size()) {
            return false;
        }
        for (UInt64 index = 0; index < PhaseCounts[phase]; ++index) {
            const UInt64 denominator =
                210 * phase + PhaseResidues[phase][index];
            if (!wheelAccepts(denominator)) return false;
            ++total;
        }
    }
    return total == ResidueCount;
}
static_assert(verifyPhaseResidues(),
              "eleven Q210 phases must partition wheel-2310 residues");

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
            const UInt64 next =
                Residues[Stepper::Width * group + lane]
                + (group == 0 ? Wheel : 0);
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
static_assert(StepperMaximumDistance == 50);

static inline bool firstFullBlock(UInt64 start, UInt64& fullBlock) {
    fullBlock = start - start % Wheel;
    if (start - fullBlock <= Residues[0]) return true;
    if (fullBlock > std::numeric_limits<UInt64>::max() - Wheel)
        return false;
    fullBlock += Wheel;
    return true;
}

template<typename TArg>
static inline bool stepperSupportsRange(
    const TArg& y,
    UInt64 start,
    UInt64 end
) {
    if (start > end) return false;
    UInt64 fullBlock = 0;
    if (!firstFullBlock(start, fullBlock)) return false;
    if (fullBlock > std::numeric_limits<UInt64>::max() - Residues[0])
        return false;
    return fullBlock <= end && end - fullBlock >= Wheel - 1
        && Stepper::supports(
            y, fullBlock + Residues[0], StepperMaximumDistance
        );
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
    auto addClippedBlock = [&](UInt64 block, UInt64 phase) {
        for (UInt64 index = 0; index < PhaseCounts[phase]; ++index) {
            const UInt64 residue = PhaseResidues[phase][index];
            if (block > std::numeric_limits<UInt64>::max() - residue)
                break;
            const UInt64 denominator = block + residue;
            if (denominator >= start && denominator <= end)
                add(denominator);
        }
    };
    auto addFullBlock = [&](UInt64 block, UInt64 phase) {
        for (UInt64 index = 0; index < PhaseCounts[phase]; ++index)
            add(block + PhaseResidues[phase][index]);
    };
    auto advance = [](UInt64& block, UInt64& phase) {
        if (block > std::numeric_limits<UInt64>::max() - 210) return false;
        block += 210;
        phase = phase == 10 ? 0 : phase + 1;
        return true;
    };

    UInt64 block = start - start % 210;
    UInt64 phase = (block / 210) % 11;
    addClippedBlock(block, phase);
    if (!advance(block, phase)) return result;
    while (block <= end && end - block >= 209) {
        addFullBlock(block, phase);
        if (!advance(block, phase)) return result;
    }
    if (block <= end) addClippedBlock(block, phase);
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

    UInt64 fullBlock = 0;
    if (!firstFullBlock(start, fullBlock)) {
        return sumDirectScalar(
            y, L1, start, end, M, R, qCache, dCAP
        );
    }

    if (start < fullBlock) {
        result += sumDirectScalar(
            y, L1, start, std::min(end, fullBlock - 1),
            M, R, qCache, dCAP
        );
    }

    auto denominators = [](UInt64 base, std::size_t group) {
        Stepper::Batch values;
        for (std::size_t lane = 0; lane < Stepper::Width; ++lane) {
            values[lane] = base
                         + Residues[Stepper::Width * group + lane];
        }
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

    UInt64 block = fullBlock;
    for (;;) {
        if (std::numeric_limits<UInt64>::max() - block < Wheel)
            return result;
        block += Wheel;
        if (block > end || end - block < Wheel - 1) break;
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

    // Native UInt64 division is the fastest ARM path.  Enumerating the 480
    // accepted residues means multiples of eleven incur neither a quotient nor
    // a Mertens lookup.
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

    // Keep the 48 mature stride-210 streams.  Advancing across their one
    // multiple of eleven per cycle avoids 480 wide predictor states and seeds.
    for (UInt64 residue : S1Q210Detail::Residues) {
        UInt64 denominator = 0;
        if (!S1Q210Detail::firstResidueAtLeast(
                start, residue, denominator
            ) || denominator > end) {
            continue;
        }
#ifndef NDEBUG
        assert(denominator > 210);
#endif
        UInt64 qPrev = S1Q6Detail::exactSparseQuotient(
            y, denominator - 210, qCache, dCAP
        );
        UInt64 qCur = S1Q6Detail::exactSparseQuotient(
            y, denominator, qCache, dCAP
        );
        UInt8 denominatorMod11 = static_cast<UInt8>(denominator % 11);
        if (denominatorMod11 != 0)
            result += static_cast<Acc>(GET_M(M, R, L1, qCur));

        UInt64 qEstimate = 0;
        while (end - denominator >= 210) {
            denominator += 210;
            update_quotients_fixed_stride<210, false>(
                y, denominator, qCur, qPrev, qEstimate
            );
            denominatorMod11 = denominatorMod11 == 10
                ? 0
                : static_cast<UInt8>(denominatorMod11 + 1);
            if (denominatorMod11 != 0) {
                result += static_cast<Acc>(
                    GET_M(M, R, L1, qEstimate)
                );
            }
        }
    }
    return result;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumCoprime2310(
    const TArg& y,
    UInt64 L1,
    UInt64 L2,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    static_assert(std::is_same_v<TArg, UInt64>
                  || std::is_same_v<TArg, UInt128>);
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

    // The tail deliberately retains stride 210, so its proved curvature
    // boundary is also the stride-210 boundary.
    UInt64 exactThrough = S1Q6Detail::sparsePredictorBoundary<210>(
        y, lo, hi, dCAP
    );
    if (exactThrough == 0)
        return sumDirect(y, L1, lo, hi, M, R, qCache, dCAP);
    if constexpr (std::is_same_v<TArg, UInt64> && UseDivisionFree)
        exactThrough = std::max(exactThrough, dCAP);
    exactThrough = std::max(exactThrough, UInt64(210));
    exactThrough = std::min(exactThrough, hi);

    Acc result = sumDirect(
        y, L1, lo, exactThrough, M, R, qCache, dCAP
    );
    if (exactThrough == hi) return result;

    result += sumSeparatePredicted(
        y, L1, exactThrough + 1, hi, M, R, qCache, dCAP
    );
    return result;
}

} // namespace S1Q2310PairedSeamDetail

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg>
evaluateS1OuterQ2310PairedParent(
    const TArg& y,
    UInt64 lowerExclusive,
    UInt64 commonKappa,
    UInt64 L1,
    UInt64 L2,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    if (lowerExclusive == std::numeric_limits<UInt64>::max()) return Acc(0);
    return S1Q2310PairedSeamDetail::sumCoprime2310(
        y, L1, L2, lowerExclusive + 1, commonKappa,
        M, R, qCache, dCAP
    );
}
