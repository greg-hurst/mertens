#pragma once

// ============================================================================
// S1Q30.h — completed outer-Q=30 S1 kernels.
//
// With one common split, a completed divisor group leaves exactly the eight
// residue classes coprime to 30. Exact ranges use an eight-lane quotient
// stepper in denominator order. Predictor tails retain independent streams in
// Loop 0/1 and interleave those streams in Loop 2.
// ============================================================================

#include "S1Q6.h"
#include "QuotientStepper.h"

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>

namespace S1Q30Detail {

inline constexpr std::size_t StepperLanes = 8;
using Stepper = QuotientStepper<StepperLanes>;
inline constexpr UInt64 StepperSpan = 30;

inline constexpr std::array<UInt64, 8> Residues = {
    1, 7, 11, 13, 17, 19, 23, 29
};
static_assert(Stepper::Width == Residues.size());

static inline bool wheelAccepts(UInt64 value) {
    return (value & 1ULL) != 0 && value % 3 != 0 && value % 5 != 0;
}

template<typename TArg>
static inline bool stepperSupportsRange(
    const TArg& y,
    UInt64 start,
    UInt64 end
) {
    if (start > end) return false;
    UInt64 fullBlock = start - start % 30;
    if (fullBlock + 1 < start) {
        if (fullBlock > std::numeric_limits<UInt64>::max() - 30)
            return false;
        fullBlock += 30;
    }
    return fullBlock <= end
        && end - fullBlock >= 2 * StepperSpan - 1
        && Stepper::supports(
            y, fullBlock + Residues[0], StepperSpan
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

    UInt64 block = start - start % 30;
    UInt64 fullBlock = block;
    if (fullBlock + 1 < start) fullBlock += 30;
    const UInt64 prefixHi = fullBlock == 0
        ? 0
        : std::min(end, fullBlock - 1);
    for (UInt64 value = start; value <= prefixHi; ++value) {
        if (wheelAccepts(value)) add(value);
    }

    block = fullBlock;
    while (block <= end && end - block >= 29) {
        add(block + 1);  add(block + 7);  add(block + 11); add(block + 13);
        add(block + 17); add(block + 19); add(block + 23); add(block + 29);
        if (std::numeric_limits<UInt64>::max() - block < 30) return result;
        block += 30;
    }
    if (block <= end) {
        for (UInt64 value = block; value <= end; ++value) {
            if (wheelAccepts(value)) add(value);
        }
    }
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

    UInt64 block = start - start % 30;
    UInt64 fullBlock = block;
    if (fullBlock + 1 < start) fullBlock += 30;

    if (start < fullBlock) {
        result += sumDirectScalar(
            y, L1, start, std::min(end, fullBlock - 1),
            M, R, qCache, dCAP
        );
    }

    auto denominators = [](UInt64 base) {
        Stepper::Batch values;
        for (std::size_t lane = 0; lane < Stepper::Width; ++lane)
            values[lane] = base + Residues[lane];
        return values;
    };
    auto add = [&](const Stepper::Batch& quotients) {
        for (UInt64 quotient : quotients)
            result += static_cast<Acc>(GET_M(M, R, L1, quotient));
    };

    Stepper stepper;
    add(stepper.initialize(y, denominators(fullBlock)));
    block = fullBlock + StepperSpan;
    while (block <= end && end - block >= StepperSpan - 1) {
        add(stepper.step<StepperSpan>(denominators(block)));
        if (std::numeric_limits<UInt64>::max() - block < StepperSpan)
            return result;
        block += StepperSpan;
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
static inline S1Q6Detail::Accumulator<TArg> sumSequentialPredicted(
    const TArg& y,
    UInt64 L1,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP,
    UInt64 firstUnitCurvature
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    Acc result = 0;
    result += S1Q6Detail::sumPredictedResidueStream<30, 1>(
        y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
    );
    result += S1Q6Detail::sumPredictedResidueStream<30, 7>(
        y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
    );
    result += S1Q6Detail::sumPredictedResidueStream<30, 11>(
        y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
    );
    result += S1Q6Detail::sumPredictedResidueStream<30, 13>(
        y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
    );
    result += S1Q6Detail::sumPredictedResidueStream<30, 17>(
        y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
    );
    result += S1Q6Detail::sumPredictedResidueStream<30, 19>(
        y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
    );
    result += S1Q6Detail::sumPredictedResidueStream<30, 23>(
        y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
    );
    result += S1Q6Detail::sumPredictedResidueStream<30, 29>(
        y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
    );
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
    Acc result = 0;
    if (start > end) return result;

    std::array<UInt64, Residues.size()> qPrev{};
    std::array<UInt64, Residues.size()> qCur{};
    std::array<bool, Residues.size()> initialized{};

    auto add = [&](std::size_t state, UInt64 denominator) {
        UInt64 quotient = 0;
        if (!initialized[state]) {
            qPrev[state] = S1Q6Detail::exactSparseQuotient(
                y, denominator - 30, qCache, dCAP
            );
            qCur[state] = S1Q6Detail::exactSparseQuotient(
                y, denominator, qCache, dCAP
            );
            quotient = qCur[state];
            initialized[state] = true;
        } else {
            update_quotients_fixed_stride<30, false>(
                y, denominator, qCur[state], qPrev[state], quotient
            );
        }
        result += static_cast<Acc>(GET_M(M, R, L1, quotient));
    };

    UInt64 block = start - start % 30;
    for (;;) {
        for (std::size_t state = 0; state < Residues.size(); ++state) {
            const UInt64 denominator = block + Residues[state];
            if (denominator >= start && denominator <= end)
                add(state, denominator);
        }
        if (std::numeric_limits<UInt64>::max() - block < 30
            || block + 30 > end)
            break;
        block += 30;
    }
    return result;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumCoprime30(
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

    const UInt64 firstUnitCurvature =
        S1Q6Detail::sparsePredictorBoundary<30>(y, lo, hi, dCAP);
    if (firstUnitCurvature == 0)
        return sumDirect(y, L1, lo, hi, M, R, qCache, dCAP);

    UInt64 exactThrough = firstUnitCurvature;
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
        result += sumSequentialPredicted(
            y, L1, predictedLo, hi,
            M, R, qCache, dCAP, firstUnitCurvature
        );
    }
    return result;
}

} // namespace S1Q30Detail

static inline UInt64 coherentS1BoundaryFactorQ30(UInt64 commonKappa) {
    return commonKappa
         - commonKappa / 2
         - commonKappa / 3
         + commonKappa / 6
         - commonKappa / 5
         + commonKappa / 10
         + commonKappa / 15
         - commonKappa / 30;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg>
evaluateS1OuterQ30ZeroComplete(
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
    return S1Q30Detail::sumCoprime30(
        y, L1, L2, lowerExclusive + 1, commonKappa,
        M, R, qCache, dCAP, loop2
    );
}
