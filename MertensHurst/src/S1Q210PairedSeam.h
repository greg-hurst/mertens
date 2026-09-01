#pragma once

// ============================================================================
// S1Q210PairedSeam.h — exact native-Q30 paired-row S1 correction.
//
// Pairing the existing rows a and 7a removes the multiples of seven from the
// parent Q30 sum. Different native split points leave one short Q30 interval,
// represented here in parent coordinates as eight stride-210 streams.
// ============================================================================

#include "S1Q210.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>

namespace S1Q210PairedSeamDetail {

inline constexpr std::array<UInt8, 8> Residues = {
      7,  49,  77,  91, 119, 133, 161, 203
};

static constexpr bool verifyResidues() {
    std::size_t index = 0;
    for (UInt64 value = 1; value < 210; ++value) {
        bool accepted = value % 7 == 0;
        if (accepted) {
            const UInt64 quotient = value / 7;
            accepted = (quotient & 1ULL) != 0
                    && quotient % 3 != 0
                    && quotient % 5 != 0;
        }
        if (!accepted) continue;
        if (index >= Residues.size() || Residues[index] != value)
            return false;
        ++index;
    }
    return index == Residues.size();
}
static_assert(verifyResidues(),
              "paired Q210 seam residues must equal seven times wheel-30");

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumDirect(
    const TArg& y,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    Acc result = 0;
    if (lo > hi) return result;

    UInt64 block = lo - lo % 210;
    for (;;) {
        for (UInt64 residue : Residues) {
            if (block > std::numeric_limits<UInt64>::max() - residue)
                break;
            const UInt64 denominator = block + residue;
            if (denominator < lo || denominator > hi) continue;
            const UInt64 quotient = S1Q6Detail::exactSparseQuotient(
                y, denominator, qCache, dCAP
            );
            result += static_cast<Acc>(GET_M(M, R, L1, quotient));
        }
        if (block > std::numeric_limits<UInt64>::max() - 210
            || hi - block < 210)
            break;
        block += 210;
    }
    return result;
}

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> sumPredicted(
    const TArg& y,
    UInt64 L1,
    UInt64 lo,
    UInt64 hi,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    Acc result = 0;

    for (UInt64 residue : Residues) {
        UInt64 denominator = 0;
        if (!S1Q210Detail::firstResidueAtLeast(
                lo, residue, denominator
            ) || denominator > hi) {
            continue;
        }

        UInt64 qPrev = 0;
        if (denominator <= 210) {
            qPrev = S1Q6Detail::exactSparseQuotient(
                y, denominator, qCache, dCAP
            );
            result += static_cast<Acc>(GET_M(M, R, L1, qPrev));
            if (hi - denominator < 210) continue;
            denominator += 210;
        } else {
            qPrev = S1Q6Detail::exactSparseQuotient(
                y, denominator - 210, qCache, dCAP
            );
        }

#ifndef NDEBUG
        assert(denominator > 210);
#endif
        UInt64 qCur = S1Q6Detail::exactSparseQuotient(
            y, denominator, qCache, dCAP
        );
        result += static_cast<Acc>(GET_M(M, R, L1, qCur));

        UInt64 qEstimate = 0;
        while (hi - denominator >= 210) {
            denominator += 210;
            update_quotients_fixed_stride<210, false>(
                y, denominator, qCur, qPrev, qEstimate
            );
            result += static_cast<Acc>(GET_M(M, R, L1, qEstimate));
        }
    }
    return result;
}

// qCommon and qNative are child-coordinate quotient endpoints. The seam is
// strict at qCommon and inclusive at qNative; scale only after resolving that
// ordering so qCommon + 1 is known not to overflow.
template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> evaluateJParentStride210(
    const TArg& y,
    UInt64 qCommon,
    UInt64 qNative,
    UInt64 L1,
    UInt64 L2,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    using Acc = S1Q6Detail::Accumulator<TArg>;
    if (L1 == 0 || L1 > L2 || qCommon >= qNative) return Acc(0);

    const UInt128 scaledStart = (UInt128(qCommon) + UInt128(1)) * UInt128(7);
    const UInt128 scaledEnd = UInt128(qNative) * UInt128(7);
    const UInt128 maxUInt64 = std::numeric_limits<UInt64>::max();
    if (scaledStart > maxUInt64 || scaledEnd > maxUInt64
        || scaledStart > scaledEnd) {
        std::cerr << "Internal error: paired Q210 S1 seam endpoint "
                  << "exceeds UInt64." << std::endl;
        std::abort();
    }
    const UInt64 start = static_cast<UInt64>(scaledStart);
    const UInt64 end = static_cast<UInt64>(scaledEnd);

    const TArg segmentStart = L2 == std::numeric_limits<UInt64>::max()
        ? TArg(1)
        : y / TArg(L2 + 1) + TArg(1);
    const TArg segmentEnd = y / TArg(L1);
    if (segmentStart > TArg(end) || segmentEnd < TArg(start)) return Acc(0);
    const UInt64 lo = std::max(start, static_cast<UInt64>(segmentStart));
    const UInt64 hi = segmentEnd >= TArg(end)
        ? end
        : static_cast<UInt64>(segmentEnd);
    if (lo > hi) return Acc(0);

    if constexpr (std::is_same_v<TArg, UInt64> && !UseDivisionFree) {
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
    result += sumPredicted(
        y, L1, exactThrough + 1, hi, M, R, qCache, dCAP
    );
    return result;
}

} // namespace S1Q210PairedSeamDetail

template<typename TArg, typename MIntT>
static inline S1Q6Detail::Accumulator<TArg> evaluateS1Q210PairedSeam(
    const TArg& y,
    UInt64 qCommon,
    UInt64 qNative,
    UInt64 L1,
    UInt64 L2,
    const MIntT* __restrict M,
    const Int8* __restrict R,
    const QuotientCache& qCache,
    UInt64 dCAP
) {
    return S1Q210PairedSeamDetail::evaluateJParentStride210(
        y, qCommon, qNative, L1, L2, M, R, qCache, dCAP
    );
}
