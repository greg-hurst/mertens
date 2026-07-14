#pragma once

// ============================================================================
// OuterRecovery.h — square-free partial-value recovery.
//
// Let P_j be stored for every square-free 1 <= j <= N. Define W_i by
//
//   P_i = sum_{m: i*m <= N, i*m square-free} W_{i*m}.
//
// Decreasing-index back substitution recovers W.
// ============================================================================

#include "types.h"

#include <vector>

// Recover only W_1 from
//
//   P_i = sum_{m: i*m <= N, i*m square-free} W_{i*m}.
//
// Möbius inversion gives W_1 = sum_i mu(i) P_i directly. Negative mu signs
// are packed by compact index; wideValues owns indices [1, wideCount] and
// narrowValues owns the remaining indices.
static inline Int128 recoverSquarefreeFinalValue(
    const std::vector<Int64>& narrowValues,
    const std::vector<Int128>& wideValues,
    UInt32 wideCount,
    const std::vector<UInt64>& negativeSigns,
    UInt32 compactCount
) {
    Int128 result = 0;
    for (UInt32 index = 1; index <= compactCount; ++index) {
        const Int128 value = index <= wideCount
            ? wideValues[index]
            : static_cast<Int128>(narrowValues[index]);
        if ((negativeSigns[index >> 6] >> (index & 63)) & 1ULL)
            result -= value;
        else
            result += value;
    }
    return result;
}

template<typename T>
static inline void recoverSquarefreeInPlace(
    std::vector<T>& values,
    const std::vector<UInt32>& hash,
    UInt32 N
) {
    for (UInt32 offset = 0; offset < N; ++offset) {
        const UInt32 i = N - offset;
        const UInt32 hi = hash[i];
        if (hi == 0) continue;

        T accumulator = values[hi];
        for (UInt64 multiple = UInt64(2) * i; multiple <= N; multiple += i) {
            const UInt32 position = hash[static_cast<UInt32>(multiple)];
            if (position != 0) accumulator -= values[position];
        }
        values[hi] = accumulator;
    }
}
