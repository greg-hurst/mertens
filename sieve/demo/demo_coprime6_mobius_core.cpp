// demo_coprime6_mobius_core.cpp — Validate and benchmark the native wheel-6 sieve.
//
// Usage:
//   ./demo_coprime6_mobius_core --validate
//   ./demo_coprime6_mobius_core --benchmark <lo> <length> [span] [repetitions]
//   ./demo_coprime6_mobius_core --all <lo> <length> [span] [repetitions]

#include "SegmentedMobiusSieve.h"
#include "SegmentedCoprime6MobiusSieve.h"

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

bool validateCoordinates() {
    UInt64 expectedIndex = 0;
    for (UInt64 n = 1; n <= 100000; ++n) {
        const bool represented = n % 2 != 0 && n % 3 != 0;
        const UInt64 count = SegmentedCoprime6MobiusSieveCore::countThrough(n);
        if (represented) {
            if (SegmentedCoprime6MobiusSieveCore::originalAt(expectedIndex) != n
                || count != expectedIndex + 1) {
                std::fprintf(stderr, "global coordinate mismatch at n=%llu\n",
                             (unsigned long long)n);
                return false;
            }
            ++expectedIndex;
        } else if (count != expectedIndex) {
            std::fprintf(stderr, "count-through mismatch at n=%llu\n",
                         (unsigned long long)n);
            return false;
        }
    }

    for (UInt64 length = 0; length <= 100; ++length) {
        UInt64 observedMax = 0;
        if (length != 0) {
            for (UInt64 lo = 1; lo <= 24; ++lo) {
                observedMax = std::max(
                    observedMax,
                    SegmentedCoprime6MobiusSieveCore::countRange(
                        lo, lo + length - 1));
            }
        }
        if (observedMax
            != SegmentedCoprime6MobiusSieveCore::maxCountForSpan(length)) {
            std::fprintf(stderr,
                         "maximum span capacity mismatch at length=%llu: "
                         "observed=%llu formula=%llu\n",
                         (unsigned long long)length,
                         (unsigned long long)observedMax,
                         (unsigned long long)
                             SegmentedCoprime6MobiusSieveCore::maxCountForSpan(length));
            return false;
        }
    }
    return true;
}

template<bool Finalize>
bool validateWindow(UInt64 lo, UInt64 hi) {
    const UInt64 span = hi - lo + 1;
    const auto primes = SegmentedMobiusSieveCore::primesUpTo(primeBound(hi));

    SegmentedMobiusSieveCore full(span);
    SegmentedCoprime6MobiusSieveCore coprime6(span);
    full.template sieve<Finalize>(lo, hi, primes);
    coprime6.template sieve<Finalize>(lo, hi, primes);

    const UInt64 first = SegmentedCoprime6MobiusSieveCore::firstPackedIndex(lo);
    const UInt64 count = SegmentedCoprime6MobiusSieveCore::countRange(lo, hi);
    const UInt64 firstN = count == 0
        ? 0
        : SegmentedCoprime6MobiusSieveCore::originalAt(first);
    if (coprime6.lo() != lo || coprime6.hi() != hi
        || coprime6.firstPackedIndex() != first
        || coprime6.packedCount() != count
        || coprime6.coprime6Count() != count
        || coprime6.firstCoprime6() != firstN
        || coprime6.capacity() < count) {
        std::fprintf(stderr,
                     "coordinate contract failed for [%llu, %llu]\n",
                     (unsigned long long)lo, (unsigned long long)hi);
        return false;
    }

    for (UInt64 i = 0; i < count; ++i) {
        const UInt64 n = SegmentedCoprime6MobiusSieveCore::originalAt(first + i);
        const Int8 packedValue = coprime6[i];
        const Int8 fullValue = full[n - lo];
        const bool valuesMatch = Finalize
            ? packedValue == fullValue
            : ((packedValue >= 0 && fullValue >= 0)
               || packedValue == fullValue);
        if (coprime6.originalAtLocal(i) != n || !valuesMatch) {
            std::fprintf(stderr,
                         "%s mismatch at n=%llu in [%llu, %llu]: "
                         "coprime6=%d full=%d\n",
                         Finalize ? "mu" : "raw",
                         (unsigned long long)n,
                         (unsigned long long)lo,
                         (unsigned long long)hi,
                         static_cast<int>(packedValue),
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
    if (!validateCoordinates()) return false;

    // Every pair of short endpoint phases, including represented and excluded
    // singleton intervals and n=1.
    for (UInt64 lo = 1; lo <= 30; ++lo) {
        for (UInt64 len = 1; len <= 30; ++len) {
            if (!validateWindow<true>(lo, lo + len - 1)
                || !validateWindow<false>(lo, lo + len - 1))
                return false;
        }
    }

    const UInt64 directCutoff =
        3600 * SegmentedCoprime6MobiusSieveCore::STENCIL_PERIOD;
    const UInt64 schedulerPoint = (directCutoff + 100) * (directCutoff + 100);
    const UInt64 packedM2 =
        1152 * SegmentedCoprime6MobiusSieveCore::STENCIL_PERIOD;
    // Twelve packed subsegments also wrap an LP_SIZE=8 diagnostic build.
    const UInt64 schedulerSpan = 3 * (12 * packedM2 + 23);
    const Window windows[] = {
        {31, 601},
        {740, 810},
        {2290, 2340},
        {4610, 4660},
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

    // Deterministic arbitrary bounds exercise every wheel phase at nontrivial
    // coordinates without making correctness depend on a fixed partition.
    UInt64 state = 0x9E3779B97F4A7C15ULL;
    for (UInt32 i = 0; i < 48; ++i) {
        state ^= state << 7;
        state ^= state >> 9;
        const UInt64 lo = 1000000000000ULL + state % 1000000000ULL;
        const UInt64 len = 1 + (state >> 32) % 200003;
        if (!validateWindow<true>(lo, lo + len - 1)
            || !validateWindow<false>(lo, lo + len - 1))
            return false;
    }

    if (!validatePartition<true>(1, 2000003, 138601)
        || !validatePartition<false>(1, 2000003, 138600)
        || !validatePartition<true>(1000000000000ULL,
                                    1000002000007ULL, 333337)
        || !validatePartition<false>(1000000000000ULL,
                                     1000002000007ULL, 333336))
        return false;

    std::printf("coprime-6 Mobius validation passed\n");
    return true;
}

template<typename SieveT, bool Coprime6Only>
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

        if constexpr (Coprime6Only) {
            for (UInt64 i = 0; i < sieve.packedCount(); ++i)
                checksum += sieve[i];
        } else {
            const UInt64 first =
                SegmentedCoprime6MobiusSieveCore::firstPackedIndex(segmentLo);
            const UInt64 count =
                SegmentedCoprime6MobiusSieveCore::countRange(segmentLo, segmentHi);
            for (UInt64 i = 0; i < count; ++i) {
                const UInt64 n = SegmentedCoprime6MobiusSieveCore::originalAt(first + i);
                checksum += sieve[n - segmentLo];
            }
        }

        if (segmentHi == hi) break;
        segmentLo = segmentHi + 1;
    }
    return elapsed;
}

bool benchmark(UInt64 lo, UInt64 length, UInt64 segmentSpan, UInt32 repetitions) {
    if (lo == 0 || length == 0 || segmentSpan == 0 || repetitions == 0
        || length - 1 > std::numeric_limits<UInt64>::max() - lo) {
        std::fprintf(stderr,
                     "benchmark bounds must be nonzero and must not overflow\n");
        return false;
    }

    const UInt64 hi = lo + length - 1;
    const auto primes = SegmentedMobiusSieveCore::primesUpTo(primeBound(hi));
    double bestFull = std::numeric_limits<double>::infinity();
    double bestCoprime6 = std::numeric_limits<double>::infinity();
    long long expectedChecksum = 0;

    for (UInt32 rep = 0; rep < repetitions; ++rep) {
        long long fullChecksum = 0;
        long long coprime6Checksum = 0;
        double fullTime;
        double coprime6Time;
        if ((rep & 1) == 0) {
            fullTime = timeSieve<SegmentedMobiusSieveCore, false>(
                lo, length, segmentSpan, primes, fullChecksum);
            coprime6Time = timeSieve<SegmentedCoprime6MobiusSieveCore, true>(
                lo, length, segmentSpan, primes, coprime6Checksum);
        } else {
            coprime6Time = timeSieve<SegmentedCoprime6MobiusSieveCore, true>(
                lo, length, segmentSpan, primes, coprime6Checksum);
            fullTime = timeSieve<SegmentedMobiusSieveCore, false>(
                lo, length, segmentSpan, primes, fullChecksum);
        }

        if (fullChecksum != coprime6Checksum) {
            std::fprintf(stderr,
                         "benchmark checksum mismatch: coprime6=%lld full=%lld\n",
                         coprime6Checksum, fullChecksum);
            return false;
        }
        expectedChecksum = coprime6Checksum;
        bestFull = std::min(bestFull, fullTime);
        bestCoprime6 = std::min(bestCoprime6, coprime6Time);

        std::printf("run %u: full=%.6f s coprime6=%.6f s speedup=%.4fx\n",
                    rep + 1, fullTime, coprime6Time, fullTime / coprime6Time);
    }

    const UInt64 count = SegmentedCoprime6MobiusSieveCore::countRange(lo, hi);
    std::printf("best: full=%.6f s coprime6=%.6f s speedup=%.4fx, "
                "coprime6=%.3f ns/original, %.3f ns/packed, checksum=%lld\n",
                bestFull, bestCoprime6, bestFull / bestCoprime6,
                bestCoprime6 * 1e9 / static_cast<double>(length),
                bestCoprime6 * 1e9 / static_cast<double>(count),
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
            "       %s --benchmark <lo> <length> [span] [repetitions]\n"
            "       %s --all <lo> <length> [span] [repetitions]\n",
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
