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
