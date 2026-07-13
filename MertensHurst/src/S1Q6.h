#pragma once

// ============================================================================
// S1Q6.h — exact outer-Q=6 S1 transform kernels.
//
// The Q=2 kernel remains available as an exact compile-time reference path;
// recovery still retains the complete all-squarefree partial-value state.
//
// For y = floor(x/k), ell = floor(y/u), and
//
//   A =       kappa(y)
//   B = 2 *   kappa(floor(y/2))
//   C = 3 *   kappa(floor(y/3))
//   D = 6 *   kappa(floor(y/6)),
//
// the four modes evaluate the active subsets of
//
//   S1(y) - S1(y/2) - S1(y/3) + S1(y/6).
//
// Normal inputs have A <= B <= C <= D. Optimized disjoint residue streams are
// used in that case. Exact unsimplified stream combinations are the fallback
// for ties/order anomalies, so correctness never relies on asymptotics.
// ============================================================================

#include "S1.h"

#include <algorithm>
#include <type_traits>
#include <vector>

enum class S1OuterQ6Mode : UInt8 {
    Single,
    Minus2,
    Minus2Minus3,
    Full6,
};

struct S1Q6WorkItem {
    UInt32 compactIndex;
    UInt32 outerK;
    S1OuterQ6Mode mode;
};

namespace S1Q6Detail {

// Bit r selects the residue r mod 6.
static constexpr UInt8 MASK_R0          = UInt8(1u << 0);
static constexpr UInt8 MASK_R3          = UInt8(1u << 3);
static constexpr UInt8 MASK_ALL         = UInt8(0x3f);
static constexpr UInt8 MASK_EVEN        = UInt8((1u << 0) | (1u << 2) | (1u << 4));
static constexpr UInt8 MASK_ODD         = UInt8((1u << 1) | (1u << 3) | (1u << 5));
static constexpr UInt8 MASK_COPRIME6    = UInt8((1u << 1) | (1u << 5));
static constexpr UInt8 MASK_MULTIPLE3   = UInt8((1u << 0) | (1u << 3));
static constexpr UInt8 MASK_R2_R3_R4    = UInt8((1u << 2) | (1u << 3) | (1u << 4));

template<typename TArg>
using Accumulator = std::conditional_t<std::is_same_v<TArg, UInt128>, Int128, Int64>;

template<typename TArg>
static inline UInt64 exactSparseQuotient(
    const TArg& y,
    UInt64 denominator,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    if constexpr (std::is_same_v<TArg, UInt64> && UseDivisionFree) {
        if (dCAP != 0 && denominator <= dCAP)
            return qCache.quotient(y, denominator);
    }
    return static_cast<UInt64>(y / denominator);
}

template<UInt8 Mask, typename TArg, typename MIntT>
static inline Accumulator<TArg> sumSparseResiduesDirect(
    const TArg& y,
    UInt64 L1,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    static_assert((Mask & ~MASK_ALL) == 0, "invalid residue mask");
    Accumulator<TArg> result = 0;
    if (start > end) return result;

    auto addOne = [&](UInt64 denominator) {
        const UInt64 quotient = exactSparseQuotient(y, denominator, qCache, dCAP);
        result += static_cast<Accumulator<TArg>>(GET_M(M, R, L1, quotient));
    };

    auto addBoundary = [&](UInt64 denominator) {
        switch (denominator % 6) {
            case 0: if constexpr (Mask & (1u << 0)) addOne(denominator); break;
            case 1: if constexpr (Mask & (1u << 1)) addOne(denominator); break;
            case 2: if constexpr (Mask & (1u << 2)) addOne(denominator); break;
            case 3: if constexpr (Mask & (1u << 3)) addOne(denominator); break;
            case 4: if constexpr (Mask & (1u << 4)) addOne(denominator); break;
            case 5: if constexpr (Mask & (1u << 5)) addOne(denominator); break;
        }
    };

    UInt64 base = start + ((6 - start % 6) % 6);
    for (UInt64 denominator = start; denominator < base && denominator <= end; ++denominator)
        addBoundary(denominator);

    while (base <= end && end - base >= 5) {
        if constexpr (Mask & (1u << 0)) addOne(base + 0);
        if constexpr (Mask & (1u << 1)) addOne(base + 1);
        if constexpr (Mask & (1u << 2)) addOne(base + 2);
        if constexpr (Mask & (1u << 3)) addOne(base + 3);
        if constexpr (Mask & (1u << 4)) addOne(base + 4);
        if constexpr (Mask & (1u << 5)) addOne(base + 5);
        base += 6;
    }

    for (UInt64 denominator = base; denominator <= end; ++denominator)
        addBoundary(denominator);

    return result;
}

template<UInt64 Step, UInt64 Residue, typename TArg, typename MIntT>
static inline Accumulator<TArg> sumPredictedResidueStream(
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
    static_assert(Step > 0, "invalid residue-stream stride");
    static_assert(Residue < Step, "invalid residue-stream class");

    Accumulator<TArg> result = 0;
    UInt64 denominator = start + (Residue + Step - start % Step) % Step;
    if (denominator > end) return result;

    UInt64 exactThrough = firstUnitCurvature;
    if constexpr (std::is_same_v<TArg, UInt64> && UseDivisionFree)
        exactThrough = std::max(exactThrough, dCAP);

    UInt64 qPrev = 0;
    bool havePrevious = false;
    while (denominator <= end && denominator <= exactThrough) {
        qPrev = exactSparseQuotient(y, denominator, qCache, dCAP);
        result += static_cast<Accumulator<TArg>>(GET_M(M, R, L1, qPrev));
        havePrevious = true;
        if (end - denominator < Step) return result;
        denominator += Step;
    }

    if (denominator > end) return result;
    if (!havePrevious)
        qPrev = exactSparseQuotient(y, denominator - Step, qCache, dCAP);

    UInt64 qCur = exactSparseQuotient(y, denominator, qCache, dCAP);
    result += static_cast<Accumulator<TArg>>(GET_M(M, R, L1, qCur));

    UInt64 qEst = 0;
    while (end - denominator >= Step) {
        denominator += Step;
        update_quotients_fixed_stride<Step, false>(
            y, denominator, qCur, qPrev, qEst
        );
        result += static_cast<Accumulator<TArg>>(GET_M(M, R, L1, qEst));
    }

    return result;
}

template<UInt64 Step, typename TArg>
static inline UInt64 sparsePredictorBoundary(
    const TArg& y,
    UInt64 start,
    UInt64 end,
    UInt64 dCAP
) {
    // Short streams do not repay predictor setup, and the division-free
    // UInt64 backend should consume its already-built quotient cache first.
    if (end - start < 8 * Step) return 0;
    if constexpr (std::is_same_v<TArg, UInt64> && UseDivisionFree) {
        if (end <= dCAP) return 0;
    }

    // Most active segments lie wholly on one side of the boundary.  These
    // exact endpoint tests avoid a cube root in both common cases.
    if (quotient_predictor_has_unit_curvature<Step>(y, start))
        return start;
    if (!quotient_predictor_has_unit_curvature<Step>(y, end))
        return 0;

    return quotient_predictor_first_unit_curvature<Step>(y);
}

template<UInt8 Mask, typename TArg, typename MIntT>
static inline Accumulator<TArg> sumSparseResidues(
    const TArg& y,
    UInt64 L1,
    UInt64 start,
    UInt64 end,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    static_assert((Mask & ~MASK_ALL) == 0, "invalid residue mask");

    // Direct UInt64 division remains faster on ARM.  UInt128 division and the
    // x86 division-free backend switch to proved fixed-stride predictors once
    // the corresponding stream reaches its own unit-curvature boundary.
    if constexpr (std::is_same_v<TArg, UInt64> && !UseDivisionFree) {
        return sumSparseResiduesDirect<Mask>(
            y, L1, start, end, M, R, qCache, dCAP
        );
    } else if constexpr (Mask == MASK_MULTIPLE3) {
        const UInt64 firstUnitCurvature = sparsePredictorBoundary<3>(
            y, start, end, dCAP
        );
        if (firstUnitCurvature == 0)
            return sumSparseResiduesDirect<Mask>(
                y, L1, start, end, M, R, qCache, dCAP
            );
        return sumPredictedResidueStream<3, 0>(
            y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
        );
    } else {
        const UInt64 firstUnitCurvature = sparsePredictorBoundary<6>(
            y, start, end, dCAP
        );
        if (firstUnitCurvature == 0)
            return sumSparseResiduesDirect<Mask>(
                y, L1, start, end, M, R, qCache, dCAP
            );
        Accumulator<TArg> result = 0;
        if constexpr (Mask & (1u << 0))
            result += sumPredictedResidueStream<6, 0>(
                y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
            );
        if constexpr (Mask & (1u << 1))
            result += sumPredictedResidueStream<6, 1>(
                y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
            );
        if constexpr (Mask & (1u << 2))
            result += sumPredictedResidueStream<6, 2>(
                y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
            );
        if constexpr (Mask & (1u << 3))
            result += sumPredictedResidueStream<6, 3>(
                y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
            );
        if constexpr (Mask & (1u << 4))
            result += sumPredictedResidueStream<6, 4>(
                y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
            );
        if constexpr (Mask & (1u << 5))
            result += sumPredictedResidueStream<6, 5>(
                y, L1, start, end, M, R, qCache, dCAP, firstUnitCurvature
            );
        return result;
    }
}

template<UInt8 Mask, typename TArg, typename MIntT>
static inline Accumulator<TArg> sumClippedStream(
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
    if (L1 == 0 || L1 > L2 || start > end) return 0;

    const TArg loBySegment = (L2 == ~UInt64(0))
        ? TArg(1)
        : y / TArg(L2 + 1) + TArg(1);
    const TArg hiBySegment = y / TArg(L1);

    if (loBySegment > TArg(end) || hiBySegment < TArg(start)) return 0;
    const UInt64 lo = std::max(start, static_cast<UInt64>(loBySegment));
    const UInt64 hi = hiBySegment >= TArg(end)
        ? end
        : static_cast<UInt64>(hiBySegment);
    if (lo > hi) return 0;

    // Reuse the existing tuned all/odd/even backends. They include quotient
    // cache/predictor support. Sparse stride-6 masks use direct division above
    // dCAP initially; this can be split out if a future tuning pass justifies it.
    if constexpr (Mask == MASK_ALL || Mask == MASK_ODD || Mask == MASK_EVEN) {
        constexpr ParityMode parity = Mask == MASK_ALL
            ? ParityMode::All
            : (Mask == MASK_ODD ? ParityMode::Odd : ParityMode::Even);
        if constexpr (std::is_same_v<TArg, UInt128>) {
            UInt64 qPrev = 0;
            UInt64 qCur = 0;
            return update_S1_128<parity>(y, L1, lo, hi, M, R, qPrev, qCur);
        } else {
            return update_S1<parity>(y, L1, lo, hi, M, R, qCache, dCAP);
        }
    } else {
        return sumSparseResidues<Mask>(y, L1, lo, hi, M, R, qCache, dCAP);
    }
}

} // namespace S1Q6Detail

static inline S1OuterQ6Mode classifyS1OuterQ6(UInt64 k, UInt64 N) {
    if (k <= N / 6) return S1OuterQ6Mode::Full6;
    if (k <= N / 3) return S1OuterQ6Mode::Minus2Minus3;
    if (k <= N / 2) return S1OuterQ6Mode::Minus2;
    return S1OuterQ6Mode::Single;
}

// hash2 maps compact square-free indices back to their original k. The result
// contains only k coprime to 6, eliminating repeated hot-loop residue checks.
static inline std::vector<S1Q6WorkItem> buildS1Q6Worklist(
    const UInt32* hash2,
    UInt32 compactCount,
    UInt64 N
) {
    std::vector<S1Q6WorkItem> work;
    work.reserve(compactCount / 2 + 1);
    for (UInt32 index = 1; index <= compactCount; ++index) {
        const UInt32 k = hash2[index];
        if (k == 0 || k > N) break;
        const UInt32 residue = k % 6;
        if (residue == 1 || residue == 5)
            work.push_back(S1Q6WorkItem{index, k, classifyS1OuterQ6(k, N)});
    }
    return work;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> evaluateS1OuterQ6(
    S1OuterQ6Mode mode,
    const TArg& y,
    UInt64 lowerExclusive,
    UInt64 A,
    UInt64 B,
    UInt64 C,
    UInt64 D,
    UInt64 L1,
    UInt64 L2,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using namespace S1Q6Detail;
    using Acc = Accumulator<TArg>;

    if (lowerExclusive == ~UInt64(0)) return Acc(0);
    const UInt64 lo = lowerExclusive + 1;

    auto all = [&](UInt64 from, UInt64 to) {
        return sumClippedStream<MASK_ALL>(y, L1, L2, from, to, M, R, qCache, dCAP);
    };
    auto odd = [&](UInt64 from, UInt64 to) {
        return sumClippedStream<MASK_ODD>(y, L1, L2, from, to, M, R, qCache, dCAP);
    };
    auto even = [&](UInt64 from, UInt64 to) {
        return sumClippedStream<MASK_EVEN>(y, L1, L2, from, to, M, R, qCache, dCAP);
    };
    auto coprime6 = [&](UInt64 from, UInt64 to) {
        return sumClippedStream<MASK_COPRIME6>(y, L1, L2, from, to, M, R, qCache, dCAP);
    };
    auto r0 = [&](UInt64 from, UInt64 to) {
        return sumClippedStream<MASK_R0>(y, L1, L2, from, to, M, R, qCache, dCAP);
    };
    auto r3 = [&](UInt64 from, UInt64 to) {
        return sumClippedStream<MASK_R3>(y, L1, L2, from, to, M, R, qCache, dCAP);
    };
    auto multiple3 = [&](UInt64 from, UInt64 to) {
        return sumClippedStream<MASK_MULTIPLE3>(y, L1, L2, from, to, M, R, qCache, dCAP);
    };
    auto r2r3r4 = [&](UInt64 from, UInt64 to) {
        return sumClippedStream<MASK_R2_R3_R4>(y, L1, L2, from, to, M, R, qCache, dCAP);
    };

    if (mode == S1OuterQ6Mode::Single)
        return all(lo, A);

    if (mode == S1OuterQ6Mode::Minus2) {
        if (A <= B)
            return odd(lo, A) - even(std::max(lo, A + 1), B);
        // Exact fallback: 1_{n<=A} - 1_{2|n}1_{n<=B}.
        return all(lo, A) - even(lo, B);
    }

    if (mode == S1OuterQ6Mode::Minus2Minus3) {
        if (A <= B && B <= C) {
            return coprime6(lo, A)
                 - r0(lo, A)
                 - r2r3r4(std::max(lo, A + 1), B)
                 - Acc(2) * r0(std::max(lo, A + 1), B)
                 - multiple3(std::max(lo, B + 1), C);
        }
        // Exact fallback: 1_{n<=A} - 1_{2|n}1_{n<=B}
        //                                - 1_{3|n}1_{n<=C}.
        return all(lo, A) - even(lo, B) - multiple3(lo, C);
    }

    if (A <= B && B <= C && C <= D) {
        return coprime6(lo, A)
             - even(std::max(lo, A + 1), B)
             - r3(std::max(lo, A + 1), C)
             + r0(std::max(lo, C + 1), D);
    }

    // Exact fallback: full 1 - 2 - 3 + 6 transform.
    return all(lo, A) - even(lo, B) - multiple3(lo, C) + r0(lo, D);
}
