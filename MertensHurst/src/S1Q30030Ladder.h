#pragma once

// ============================================================================
// S1Q30030Ladder.h — exact Q210 -> Q2310 -> Q30030 S1 ladder kernel.
//
// For a square-free root r coprime to 11*13, the full quartet
// r, 11r, 13r, 143r is evaluated once at r. Q30030 deletes both new prime
// classes from the parent; exact Q210/Q2310 seams preserve each native split.
// A truncated triple uses the independent Q210 ghost for y/143.
// ============================================================================

#include "S1Q210.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <type_traits>

namespace S1Q30030LadderDetail {

inline constexpr UInt64 Q210 = 210;
inline constexpr UInt64 Q30030 = 30030;

static constexpr bool coprimeTo210(UInt64 value) {
    return (value & 1ULL) != 0
        && value % 3 != 0
        && value % 5 != 0
        && value % 7 != 0;
}

inline constexpr std::array<UInt8, Q210> NextCoprime210Delta = [] {
    std::array<UInt8, Q210> deltas{};
    for (UInt64 residue = 0; residue < Q210; ++residue) {
        UInt64 delta = 0;
        while (true) {
            const UInt64 candidate = (residue + delta) % Q210;
            if ((candidate & 1ULL) != 0
                && candidate % 3 != 0
                && candidate % 5 != 0
                && candidate % 7 != 0) {
                deltas[residue] = static_cast<UInt8>(delta);
                break;
            }
            ++delta;
        }
    }
    return deltas;
}();

static constexpr bool verifyNextCoprime210Delta() {
    for (UInt64 residue = 0; residue < Q210; ++residue) {
        const UInt64 delta = NextCoprime210Delta[residue];
        if (!coprimeTo210((residue + delta) % Q210)) return false;
        for (UInt64 earlier = 0; earlier < delta; ++earlier) {
            if (coprimeTo210((residue + earlier) % Q210)) return false;
        }
    }
    return true;
}

static_assert(Q210 <= UInt64(std::numeric_limits<UInt8>::max()) + 1);
static_assert(verifyNextCoprime210Delta(),
              "next-coprime-Q210 table must give the first accepted offset");

template<UInt64 ExtraPrime1, UInt64 ExtraPrime2,
         UInt64 WheelValue, std::size_t ResidueCountValue,
         std::size_t PhaseCountValue>
struct ExtendedWheel {
    static constexpr UInt64 Wheel = WheelValue;
    static constexpr std::size_t ResidueCount = ResidueCountValue;
    static constexpr std::size_t PhaseCount = PhaseCountValue;
    static constexpr std::size_t StepperLanes = 8;
    using Stepper = QuotientStepper<StepperLanes>;

    static_assert(PhaseCount != 0 && ResidueCount != 0);
    static_assert(Wheel == Q210 * PhaseCount);
    static_assert(Wheel <= UInt64(std::numeric_limits<UInt16>::max()) + 1);

    static constexpr bool extraAccepts(UInt64 value) {
        return value % ExtraPrime1 != 0
            && (ExtraPrime2 == 1 || value % ExtraPrime2 != 0);
    }

    static constexpr bool accepts(UInt64 value) {
        return (value & 1ULL) != 0
            && value % 3 != 0
            && value % 5 != 0
            && value % 7 != 0
            && extraAccepts(value);
    }

    inline static constexpr std::array<UInt16, ResidueCount> Residues = [] {
        std::array<UInt16, ResidueCount> residues{};
        std::size_t count = 0;
        for (UInt64 value = 1; value < Wheel; ++value) {
            if (accepts(value))
                residues[count++] = static_cast<UInt16>(value);
        }
        return residues;
    }();

    static constexpr bool verifyResidues() {
        std::size_t count = 0;
        for (UInt64 value = 1; value < Wheel; ++value) {
            if (!accepts(value)) continue;
            if (count >= Residues.size() || Residues[count] != value)
                return false;
            ++count;
        }
        return count == Residues.size();
    }
    static_assert(verifyResidues(), "extended-wheel residue table mismatch");
    static_assert(ResidueCount % Stepper::Width == 0);

    inline static constexpr std::array<
        std::array<UInt8, S1Q210Detail::Residues.size()>, PhaseCount
    > PhaseResidues = [] {
        std::array<
            std::array<UInt8, S1Q210Detail::Residues.size()>, PhaseCount
        > residues{};
        for (UInt64 phase = 0; phase < PhaseCount; ++phase) {
            std::size_t count = 0;
            for (UInt64 residue : S1Q210Detail::Residues) {
                if (extraAccepts(Q210 * phase + residue))
                    residues[phase][count++] = static_cast<UInt8>(residue);
            }
        }
        return residues;
    }();

    inline static constexpr std::array<UInt8, PhaseCount> PhaseCounts = [] {
        std::array<UInt8, PhaseCount> counts{};
        for (UInt64 phase = 0; phase < PhaseCount; ++phase) {
            for (UInt64 residue : S1Q210Detail::Residues)
                counts[phase] += extraAccepts(Q210 * phase + residue);
        }
        return counts;
    }();

    static constexpr bool verifyPhases() {
        std::size_t total = 0;
        for (UInt64 phase = 0; phase < PhaseCount; ++phase) {
            if (PhaseCounts[phase] == 0
                || PhaseCounts[phase] > PhaseResidues[phase].size()) {
                return false;
            }
            for (UInt64 index = 0; index < PhaseCounts[phase]; ++index) {
                const UInt64 denominator =
                    Q210 * phase + PhaseResidues[phase][index];
                if (!accepts(denominator)) return false;
                ++total;
            }
        }
        return total == ResidueCount;
    }
    static_assert(verifyPhases(), "Q210 phases do not partition wheel");

    static constexpr std::size_t StepperGroups =
        ResidueCount / Stepper::Width;

    inline static constexpr std::array<
        typename Stepper::Batch, StepperGroups
    > StepperDistances = [] {
        std::array<typename Stepper::Batch, StepperGroups> distances{};
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

    static constexpr UInt64 maximumDistance() {
        UInt64 maximum = 0;
        for (const typename Stepper::Batch& distances : StepperDistances) {
            for (UInt64 distance : distances)
                maximum = std::max(maximum, distance);
        }
        return maximum;
    }

    inline static constexpr UInt64 StepperMaximumDistance =
        maximumDistance();

    static inline bool firstFullBlock(
        UInt64 start,
        UInt64& fullBlock
    ) {
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
        if (fullBlock
            > std::numeric_limits<UInt64>::max() - Residues[0]) {
            return false;
        }
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
                if (block
                    > std::numeric_limits<UInt64>::max() - residue) {
                    break;
                }
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
            if (block > std::numeric_limits<UInt64>::max() - Q210)
                return false;
            block += Q210;
            phase = phase + 1 == PhaseCount ? 0 : phase + 1;
            return true;
        };

        UInt64 block = start - start % Q210;
        UInt64 phase = (block / Q210) % PhaseCount;
        addClippedBlock(block, phase);
        if (!advance(block, phase)) return result;
        while (block <= end && end - block >= Q210 - 1) {
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
            typename Stepper::Batch values{};
            for (std::size_t lane = 0; lane < Stepper::Width; ++lane) {
                values[lane] = base
                    + Residues[Stepper::Width * group + lane];
            }
            return values;
        };
        auto add = [&](const typename Stepper::Batch& quotients) {
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
            if (block > std::numeric_limits<UInt64>::max() - Wheel)
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
            result += sumDirectStepped(
                y, L1, std::max(start, cacheEnd + 1), end,
                M, R, qCache, dCAP
            );
            return result;
        } else {
            return sumDirectStepped(
                y, L1, start, end, M, R, qCache, dCAP
            );
        }
    }

    template<typename TArg, typename MIntT>
    static inline S1Q6Detail::Accumulator<TArg> sumPredicted(
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
        static_assert(std::is_same_v<TArg, UInt64>
                      || std::is_same_v<TArg, UInt128>);
        Acc result = 0;
        for (UInt64 residue : S1Q210Detail::Residues) {
            UInt64 denominator = 0;
            if (!S1Q210Detail::firstResidueAtLeast(
                    start, residue, denominator
                ) || denominator > end) {
                continue;
            }
#ifndef NDEBUG
            assert(denominator > Q210);
#endif
            const UInt64 previous = denominator - Q210;
            UInt64 qPrev = S1Q6Detail::exactSparseQuotient(
                y, previous, qCache, dCAP
            );
            UInt64 qCur = S1Q6Detail::exactSparseQuotient(
                y, denominator, qCache, dCAP
            );
            if (extraAccepts(denominator))
                result += static_cast<Acc>(GET_M(M, R, L1, qCur));

            UInt64 qEstimate = 0;
            while (end - denominator >= Q210) {
                denominator += Q210;
                update_quotients_fixed_stride<Q210, false>(
                    y, denominator, qCur, qPrev, qEstimate
                );
                if (extraAccepts(denominator)) {
                    result += static_cast<Acc>(
                        GET_M(M, R, L1, qEstimate)
                    );
                }
            }
        }
        return result;
    }

    template<typename TArg, typename MIntT>
    static inline S1Q6Detail::Accumulator<TArg> sumCoprime(
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
        if (loBySegment > TArg(end) || hiBySegment < TArg(start))
            return Acc(0);
        const UInt64 lo = std::max(start, static_cast<UInt64>(loBySegment));
        const UInt64 hi = hiBySegment >= TArg(end)
            ? end
            : static_cast<UInt64>(hiBySegment);
        if (lo > hi) return Acc(0);

        if constexpr (std::is_same_v<TArg, UInt64> && !UseDivisionFree) {
            return sumDirectScalar(y, L1, lo, hi, M, R, qCache, dCAP);
        }
        if constexpr (std::is_same_v<TArg, UInt128> && !UseDivisionFree) {
            if (stepperSupportsRange(y, lo, hi)) {
                return sumDirectStepped(
                    y, L1, lo, hi, M, R, qCache, dCAP
                );
            }
        }

        UInt64 exactThrough = S1Q6Detail::sparsePredictorBoundary<Q210>(
            y, lo, hi, dCAP
        );
        if (exactThrough == 0) {
            return sumDirect(
                y, L1, lo, hi, M, R, qCache, dCAP
            );
        }
        if constexpr (std::is_same_v<TArg, UInt64> && UseDivisionFree)
            exactThrough = std::max(exactThrough, dCAP);
        exactThrough = std::max(exactThrough, Q210);
        exactThrough = std::min(exactThrough, hi);
        Acc result = sumDirect(
            y, L1, lo, exactThrough, M, R, qCache, dCAP
        );
        if (exactThrough < hi) {
            result += sumPredicted(
                y, L1, exactThrough + 1, hi, M, R, qCache, dCAP
            );
        }
        return result;
    }
};

using Wheel30030 = ExtendedWheel<11, 13, Q30030, 5760, 143>;
static_assert(Wheel30030::StepperMaximumDistance == 58);

} // namespace S1Q30030LadderDetail

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg>
evaluateS1OuterQ30030LadderParent(
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
    if (lowerExclusive == std::numeric_limits<UInt64>::max())
        return Acc(0);
    return S1Q30030LadderDetail::Wheel30030::sumCoprime(
        y, L1, L2, lowerExclusive + 1, commonKappa,
        M, R, qCache, dCAP
    );
}
