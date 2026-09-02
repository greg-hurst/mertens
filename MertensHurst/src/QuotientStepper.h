#pragma once

// ============================================================================
// QuotientStepper.h — exact quotients at increasing denominators.
//
// Independent quotient/remainder streams expose enough native divisions for
// the processor to overlap their latency. Callers provide each successive
// batch in increasing-denominator order.
// ============================================================================

#include "types.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

template<std::size_t Lanes>
struct QuotientStepper {
    static_assert(Lanes > 0, "quotient-stepper width must be positive");

    static constexpr std::size_t Width = Lanes;
    using Batch = std::array<UInt64, Width>;

    Batch quotients{};
    Batch remainders{};

    template<typename TArg>
    static inline bool supports(
        const TArg& numerator,
        UInt64 firstDenominator,
        UInt64 maximumDistance
    ) {
        static_assert(
            std::is_same_v<TArg, UInt64> || std::is_same_v<TArg, UInt128>,
            "unsupported quotient-stepper numerator type"
        );
        assert(firstDenominator > 0);
        assert(maximumDistance > 0);
        const UInt64 quotientLimit =
            std::numeric_limits<UInt64>::max() / maximumDistance;
        return static_cast<UInt128>(numerator)
            <= static_cast<UInt128>(quotientLimit) * firstDenominator;
    }

    template<typename TArg>
    __attribute__((always_inline)) inline const Batch& initialize(
        const TArg& numerator,
        const Batch& firstDenominators
    ) {
        static_assert(
            std::is_same_v<TArg, UInt64> || std::is_same_v<TArg, UInt128>,
            "unsupported quotient-stepper numerator type"
        );
        initializeBatch(
            numerator, firstDenominators,
            std::make_index_sequence<Width>{}
        );
        return quotients;
    }

    __attribute__((always_inline)) inline const Batch& step(
        const Batch& nextDenominators,
        const Batch& distances
    ) {
        stepBatch(
            nextDenominators, distances,
            std::make_index_sequence<Width>{}
        );
        return quotients;
    }

    template<UInt64 Distance>
    __attribute__((always_inline)) inline const Batch& step(
        const Batch& nextDenominators
    ) {
        static_assert(Distance > 0, "quotient-stepper distance must be positive");
        stepUniformBatch<Distance>(
            nextDenominators, std::make_index_sequence<Width>{}
        );
        return quotients;
    }

private:
    template<std::size_t Lane, typename TArg>
    __attribute__((always_inline)) inline void initializeOne(
        const TArg& numerator,
        UInt64 denominator
    ) {
        assert(denominator > 0);
        const TArg quotient = numerator / denominator;
        assert(static_cast<UInt128>(quotient)
               <= std::numeric_limits<UInt64>::max());
        quotients[Lane] = static_cast<UInt64>(quotient);
        remainders[Lane] = static_cast<UInt64>(
            numerator - quotient * denominator
        );
    }

    template<typename TArg, std::size_t... Lane>
    __attribute__((always_inline)) inline void initializeBatch(
        const TArg& numerator,
        const Batch& firstDenominators,
        std::index_sequence<Lane...>
    ) {
        (initializeOne<Lane>(numerator, firstDenominators[Lane]), ...);
    }

    template<std::size_t Lane>
    __attribute__((always_inline)) inline void stepOne(
        UInt64 nextDenominator,
        UInt64 distance
    ) {
        assert(distance > 0);
        assert(nextDenominator >= distance);
        assert(quotients[Lane] == 0
               || distance <= std::numeric_limits<UInt64>::max()
                            / quotients[Lane]);

        const UInt64 movement = quotients[Lane] * distance;
        if (movement <= remainders[Lane]) {
            remainders[Lane] -= movement;
        } else {
            const UInt64 deficit = movement - remainders[Lane];
            const UInt64 decrease =
                1 + (deficit - 1) / nextDenominator;

            // The two operations may wrap separately, but their combined
            // modulo-2^64 result is the exact remainder, which is < D.
            UInt64 product = 0;
            (void)__builtin_mul_overflow(
                decrease, nextDenominator, &product
            );
            (void)__builtin_sub_overflow(
                product, deficit, &remainders[Lane]
            );

            assert(decrease <= quotients[Lane]);
            quotients[Lane] -= decrease;
        }
        assert(remainders[Lane] < nextDenominator);
    }

    template<std::size_t... Lane>
    __attribute__((always_inline)) inline void stepBatch(
        const Batch& nextDenominators,
        const Batch& distances,
        std::index_sequence<Lane...>
    ) {
        (stepOne<Lane>(nextDenominators[Lane], distances[Lane]), ...);
    }

    template<UInt64 Distance, std::size_t... Lane>
    __attribute__((always_inline)) inline void stepUniformBatch(
        const Batch& nextDenominators,
        std::index_sequence<Lane...>
    ) {
        (stepOne<Lane>(nextDenominators[Lane], Distance), ...);
    }
};
