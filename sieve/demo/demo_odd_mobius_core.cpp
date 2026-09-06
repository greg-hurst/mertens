// demo_odd_mobius_core.cpp — Validate and benchmark the native packed-odd sieve.
//
// Usage:
//   ./demo_odd_mobius_core --validate
//   ./demo_odd_mobius_core --benchmark <lo> <length> [segment_span] [repetitions]
//   ./demo_odd_mobius_core --all <lo> <length> [segment_span] [repetitions]

#include "SegmentedMobiusSieve.h"
#include "SegmentedOddMobiusSieve.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

struct Window {
    UInt64 lo;
    UInt64 hi;
};

UInt32 primeBound(UInt64 hi) {
    UInt64 root = static_cast<UInt64>(std::sqrt(static_cast<long double>(hi)));
    while (static_cast<__uint128_t>(root + 1) * (root + 1) <= hi) ++root;
    while (static_cast<__uint128_t>(root) * root > hi) --root;
    return static_cast<UInt32>(std::max<UInt64>(
        SegmentedMobiusSieveCore::MIN_PRIMES_BOUND, root));
}

template<bool Finalize>
bool validateWindow(UInt64 lo, UInt64 hi) {
    const UInt64 span = hi - lo + 1;
    const auto primes = SegmentedMobiusSieveCore::primesUpTo(primeBound(hi));

    SegmentedMobiusSieveCore full(span);
    SegmentedOddMobiusSieveCore odd(span);
    full.template sieve<Finalize>(lo, hi, primes);
    odd.template sieve<Finalize>(lo, hi, primes);

    const UInt64 firstOdd = SegmentedOddMobiusSieveCore::firstOddAtOrAbove(lo);
    const UInt64 count = SegmentedOddMobiusSieveCore::countOdds(lo, hi);
    if (odd.lo() != lo || odd.hi() != hi || odd.firstOdd() != firstOdd
        || odd.oddCount() != count || odd.capacity() < count) {
        std::fprintf(stderr,
                     "coordinate contract failed for [%llu, %llu]\n",
                     (unsigned long long)lo, (unsigned long long)hi);
        return false;
    }

    for (UInt64 i = 0; i < count; ++i) {
        const UInt64 n = firstOdd + 2 * i;
        const Int8 oddValue = odd[i];
        const Int8 fullValue = full[n - lo];
        // Square hits may be zeroed before or after other positive log-weight
        // additions, so two raw squareful bytes need only both be nonnegative.
        // Finalized values, and all negative squarefree encodings, are exact.
        const bool valuesMatch = Finalize
            ? oddValue == fullValue
            : ((oddValue >= 0 && fullValue >= 0) || oddValue == fullValue);
        if (odd.originalAt(i) != n || !valuesMatch) {
            std::fprintf(stderr,
                         "%s mismatch at n=%llu in [%llu, %llu]: odd=%d full=%d\n",
                         Finalize ? "mu" : "raw",
                         (unsigned long long)n,
                         (unsigned long long)lo,
                         (unsigned long long)hi,
                         static_cast<int>(oddValue),
                         static_cast<int>(fullValue));
            return false;
        }
    }
    return true;
}

template<bool Finalize>
bool validatePartition(UInt64 lo, UInt64 hi, UInt64 segmentSpan) {
    for (UInt64 segmentLo = lo; segmentLo <= hi;) {
        const UInt64 segmentHi = std::min(hi, segmentLo + segmentSpan - 1);
        if (!validateWindow<Finalize>(segmentLo, segmentHi)) return false;
        if (segmentHi == hi) break;
        segmentLo = segmentHi + 1;
    }
    return true;
}

bool validate() {
    // Short intervals exercise every endpoint parity and the n=1 special case.
    for (UInt64 lo = 1; lo <= 20; ++lo) {
        for (UInt64 len = 1; len <= 20; ++len) {
            const UInt64 hi = lo + len - 1;
            if (!validateWindow<true>(lo, hi)
                || !validateWindow<false>(lo, hi))
                return false;
        }
    }

    // Prime squares, packed-stencil seams, powers of two, medium-prime paths,
    // and an interval just above the default bucket-scheduler transition.
    const UInt64 directCutoff = 800 * SegmentedOddMobiusSieveCore::STENCIL_PERIOD;
    const UInt64 schedulerPoint = (directCutoff + 100) * (directCutoff + 100);
    const UInt64 packedM2 = 256 * SegmentedOddMobiusSieveCore::STENCIL_PERIOD;
    const UInt64 schedulerSpan = 2 * (5 * packedM2 + 17);
    const Window windows[] = {
        {21, 401},
        {3400, 3550},
        {6900, 7010},
        {13820, 13920},
        {(UInt64(1) << 20) - 257, (UInt64(1) << 20) + 257},
        {(UInt64(1) << 32) - 100003, (UInt64(1) << 32) + 100003},
        {(UInt64(1) << 40) - 250003, (UInt64(1) << 40) + 250003},
        {schedulerPoint - 250003,
         schedulerPoint - 250003 + schedulerSpan - 1},
        {schedulerPoint - 250002,
         schedulerPoint - 250002 + schedulerSpan - 1},
    };

    for (const Window& window : windows) {
        if (!validateWindow<true>(window.lo, window.hi)
            || !validateWindow<false>(window.lo, window.hi))
            return false;
    }

    // Repeated arbitrary segment boundaries ensure each fresh packed origin is
    // mapped independently rather than relying on odd-aligned callers.
    if (!validatePartition<true>(1, 2000003, 138601)
        || !validatePartition<false>(1, 2000003, 138600)
        || !validatePartition<true>(1000000000000ULL,
                                    1000002000007ULL, 333337)
        || !validatePartition<false>(1000000000000ULL,
                                     1000002000007ULL, 333336))
        return false;

    std::printf("odd Mobius validation passed\n");
    return true;
}

template<typename SieveT, bool OddOnly>
double timeSieve(UInt64 lo, UInt64 length, UInt64 segmentSpan,
                 const std::vector<UInt32>& primes, long long& checksum) {
    SieveT sieve(segmentSpan);
    checksum = 0;
    const UInt64 hi = lo + length - 1;

    double elapsed = 0.0;
    for (UInt64 segmentLo = lo; segmentLo <= hi;) {
        const UInt64 segmentHi = std::min(hi, segmentLo + segmentSpan - 1);
        const auto t0 = std::chrono::steady_clock::now();
        sieve.sieve(segmentLo, segmentHi, primes);
        const auto t1 = std::chrono::steady_clock::now();
        elapsed += std::chrono::duration<double>(t1 - t0).count();

        if constexpr (OddOnly) {
            for (UInt64 i = 0; i < sieve.oddCount(); ++i)
                checksum += sieve[i];
        } else {
            const UInt64 firstOdd = SegmentedOddMobiusSieveCore::firstOddAtOrAbove(segmentLo);
            for (UInt64 n = firstOdd; n <= segmentHi; n += 2)
                checksum += sieve[n - segmentLo];
        }

        if (segmentHi == hi) break;
        segmentLo = segmentHi + 1;
    }
    return elapsed;
}

bool benchmark(UInt64 lo, UInt64 length, UInt64 segmentSpan, UInt32 repetitions) {
    if (lo == 0 || length == 0 || segmentSpan == 0 || repetitions == 0
        || length - 1 > std::numeric_limits<UInt64>::max() - lo) {
        std::fprintf(stderr, "benchmark bounds must be nonzero and must not overflow\n");
        return false;
    }

    const UInt64 hi = lo + length - 1;
    const auto primes = SegmentedMobiusSieveCore::primesUpTo(primeBound(hi));
    double bestFull = std::numeric_limits<double>::infinity();
    double bestOdd = std::numeric_limits<double>::infinity();
    long long expectedChecksum = 0;

    for (UInt32 rep = 0; rep < repetitions; ++rep) {
        long long fullChecksum = 0;
        long long oddChecksum = 0;
        double fullTime;
        double oddTime;
        if ((rep & 1) == 0) {
            fullTime = timeSieve<SegmentedMobiusSieveCore, false>(
                lo, length, segmentSpan, primes, fullChecksum);
            oddTime = timeSieve<SegmentedOddMobiusSieveCore, true>(
                lo, length, segmentSpan, primes, oddChecksum);
        } else {
            oddTime = timeSieve<SegmentedOddMobiusSieveCore, true>(
                lo, length, segmentSpan, primes, oddChecksum);
            fullTime = timeSieve<SegmentedMobiusSieveCore, false>(
                lo, length, segmentSpan, primes, fullChecksum);
        }

        if (fullChecksum != oddChecksum) {
            std::fprintf(stderr, "benchmark checksum mismatch: odd=%lld full=%lld\n",
                         oddChecksum, fullChecksum);
            return false;
        }
        expectedChecksum = oddChecksum;
        bestFull = std::min(bestFull, fullTime);
        bestOdd = std::min(bestOdd, oddTime);

        std::printf("run %u: full=%.6f s odd=%.6f s speedup=%.4fx\n",
                    rep + 1, fullTime, oddTime, fullTime / oddTime);
    }

    const UInt64 oddCount = SegmentedOddMobiusSieveCore::countOdds(lo, hi);
    std::printf("best: full=%.6f s odd=%.6f s speedup=%.4fx, "
                "odd=%.3f ns/original, %.3f ns/packed, checksum=%lld\n",
                bestFull, bestOdd, bestFull / bestOdd,
                bestOdd * 1e9 / static_cast<double>(length),
                bestOdd * 1e9 / static_cast<double>(oddCount),
                expectedChecksum);
    return true;
}

UInt64 parse(const char* value) {
    return static_cast<UInt64>(std::strtoull(value, nullptr, 10));
}

} // namespace

int main(int argc, char* argv[]) {
    const bool runValidation = argc == 1
        || std::strcmp(argv[1], "--validate") == 0
        || std::strcmp(argv[1], "--all") == 0;
    const bool runBenchmark = argc >= 2
        && (std::strcmp(argv[1], "--benchmark") == 0
            || std::strcmp(argv[1], "--all") == 0);

    if (!runValidation && !runBenchmark) {
        std::fprintf(stderr,
            "Usage: %s --validate\n"
            "       %s --benchmark <lo> <length> [segment_span] [repetitions]\n"
            "       %s --all <lo> <length> [segment_span] [repetitions]\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    if (runValidation && !validate()) return 1;
    if (runBenchmark) {
        if (argc < 4) {
            std::fprintf(stderr, "benchmark mode requires lo and length\n");
            return 1;
        }
        const UInt64 lo = parse(argv[2]);
        const UInt64 length = parse(argv[3]);
        const UInt64 segmentSpan = argc >= 5 ? parse(argv[4]) : 2000000000ULL;
        const UInt32 repetitions = argc >= 6
            ? static_cast<UInt32>(parse(argv[5])) : 1;
        if (!benchmark(lo, length, segmentSpan, repetitions)) return 1;
    }
    return 0;
}
