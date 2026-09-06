// demo_coprime6_mertens_core.cpp — Validate packed coprime-6 Mertens forms.

#include "SegmentedCoprime6MertensSieve.h"
#include "SegmentedMertensSieve.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>
#include <omp.h>

static void fail(const char* what, UInt64 lo, UInt64 hi, UInt64 pos,
                 Int64 expected, Int64 actual) {
    std::fprintf(stderr,
                 "FAIL: %s on [%llu, %llu] at %llu: expected %lld, got %lld\n",
                 what,
                 (unsigned long long)lo, (unsigned long long)hi,
                 (unsigned long long)pos,
                 (long long)expected, (long long)actual);
    std::exit(1);
}

struct ResidualExtrema {
    int lo = 127;
    int hi = -128;
};

static void verifyCoordinates() {
    UInt64 count = 0;
    for (UInt64 n = 0; n <= 100000; ++n) {
        if (n != 0 && n % 2 != 0 && n % 3 != 0) ++count;
        const UInt64 got =
            SegmentedCoprime6MobiusSieveCore::countThrough(n);
        if (got != count)
            fail("countThrough", 0, 100000, n, count, got);
    }

    for (UInt64 k = 0; k < count; ++k) {
        const UInt64 n =
            SegmentedCoprime6MobiusSieveCore::originalAt(k);
        if (n % 2 == 0 || n % 3 == 0)
            fail("originalAt residue", 0, 100000, k, 1, 0);
        if (SegmentedCoprime6MobiusSieveCore::countThrough(n) != k + 1)
            fail("originalAt rank", 0, 100000, k, k + 1,
                 SegmentedCoprime6MobiusSieveCore::countThrough(n));
    }

    for (UInt64 length = 0; length <= 256; ++length) {
        UInt64 observedMax = 0;
        for (UInt64 lo = 1; lo <= 48; ++lo) {
            const UInt64 hi = length == 0 ? lo - 1 : lo + length - 1;
            observedMax = std::max(
                observedMax,
                SegmentedCoprime6MobiusSieveCore::countRange(lo, hi)
            );
        }
        const UInt64 expected =
            SegmentedCoprime6MobiusSieveCore::maxCountForSpan(length);
        if (observedMax != expected)
            fail("maxCountForSpan", 1, length, length,
                 expected, observedMax);
        if (Coprime6MertensSieveDetail::packedCapacity(length) != expected)
            fail("M6 packed capacity", 1, length, length, expected,
                 Coprime6MertensSieveDetail::packedCapacity(length));
    }

    const UInt64 max = std::numeric_limits<UInt64>::max();
    const UInt64 maxCount =
        SegmentedCoprime6MobiusSieveCore::countThrough(max);
    const UInt64 last =
        SegmentedCoprime6MobiusSieveCore::originalAt(maxCount - 1);
    if (last % 2 == 0 || last % 3 == 0
        || SegmentedCoprime6MobiusSieveCore::countThrough(last) != maxCount)
        fail("maximum coordinate", 0, max, maxCount - 1, 1, 0);
}

static void verifyInterval(
    UInt64 lo, UInt64 hi, Int64 mertens6Prev,
    const std::vector<Int8>& ordinaryMu,
    const std::vector<UInt32>& primes,
    SegmentedCoprime6MertensSieveCore& compressed,
    SegmentedCoprime6MertensSieveCore& fused,
    SegmentedCoprime6MertensSieveCoreT<
        Coprime6MertensStorage::Direct>& direct,
    ResidualExtrema& extrema) {
    const UInt64 span = hi < lo ? 0 : hi - lo + 1;
    const UInt64 packedCapacity =
        SegmentedCoprime6MobiusSieveCore::maxCountForSpan(span);
    const UInt64 coarseCapacity =
        Coprime6MertensSieveDetail::coarseCount(packedCapacity);

    std::vector<Int64> coarse(std::max<UInt64>(coarseCapacity, 1));
    std::vector<Int64> fusedCoarse(std::max<UInt64>(coarseCapacity, 1));
    std::vector<Int64> directValues(std::max<UInt64>(packedCapacity, 1));
    std::vector<Int8> residual(std::max<UInt64>(packedCapacity, 1));

    const UInt64 firstPacked =
        SegmentedCoprime6MobiusSieveCore::firstPackedIndex(lo);
    const UInt64 packedCount =
        SegmentedCoprime6MobiusSieveCore::countRange(lo, hi);
    std::vector<Int8> expectedMu(packedCount);
    std::vector<Int64> expectedM(packedCount);

    Int64 running = mertens6Prev;
    for (UInt64 i = 0; i < packedCount; ++i) {
        const UInt64 n =
            SegmentedCoprime6MobiusSieveCore::originalAt(firstPacked + i);
        expectedMu[i] = ordinaryMu[n];
        running += expectedMu[i];
        expectedM[i] = running;
    }

    compressed.sieve(
        lo, hi, mertens6Prev, coarse.data(), residual.data(), primes
    );
    if (compressed.firstPackedIndex() != firstPacked)
        fail("firstPackedIndex", lo, hi, lo, firstPacked,
             compressed.firstPackedIndex());
    if (compressed.packedCount() != packedCount)
        fail("packedCount", lo, hi, lo, packedCount,
             compressed.packedCount());
    if (compressed.firstCoprime6()
        != (packedCount == 0
            ? 0
            : SegmentedCoprime6MobiusSieveCore::originalAt(firstPacked)))
        fail("firstCoprime6", lo, hi, lo,
             packedCount == 0
                ? 0
                : SegmentedCoprime6MobiusSieveCore::originalAt(firstPacked),
             compressed.firstCoprime6());

    for (UInt64 i = 0; i < packedCount; ++i) {
        if (compressed.mobiusSieve()[i] != expectedMu[i])
            fail("retained coprime-6 mu", lo, hi,
                 SegmentedCoprime6MobiusSieveCore::originalAt(
                     firstPacked + i
                 ), expectedMu[i], compressed.mobiusSieve()[i]);
    }

    fused.sieveInPlace(
        lo, hi, mertens6Prev, fusedCoarse.data(), primes
    );
    direct.sieveInPlace(
        lo, hi, mertens6Prev, directValues.data(), primes
    );

    if (hi >= lo) {
        for (UInt64 pos = lo; pos <= hi; ++pos) {
            const UInt64 through =
                SegmentedCoprime6MobiusSieveCore::countThrough(pos);
            const Int64 expected = through <= firstPacked
                                 ? mertens6Prev
                                 : expectedM[through - firstPacked - 1];
            const Int64 gotCompressed =
                compressed.getCoprime6Mertens(coarse.data(), pos);
            const Int64 gotFused =
                fused.getCoprime6Mertens(fusedCoarse.data(), pos);
            const Int64 gotDirect =
                direct.getCoprime6Mertens(directValues.data(), pos);
            const Int64 gotMacro = GET_COPRIME6_MERTENS(
                coarse.data(), residual.data(), firstPacked,
                mertens6Prev, pos
            );

            if (gotCompressed != expected)
                fail("compressed lookup", lo, hi, pos,
                     expected, gotCompressed);
            if (gotFused != expected)
                fail("fused lookup", lo, hi, pos, expected, gotFused);
            if (gotDirect != expected)
                fail("direct lookup", lo, hi, pos, expected, gotDirect);
            if (gotMacro != expected)
                fail("GET_COPRIME6_MERTENS", lo, hi, pos,
                     expected, gotMacro);

            if (through > firstPacked) {
                const Int64 gotInRange = GET_COPRIME6_MERTENS_IN_RANGE(
                    coarse.data(), residual.data(), firstPacked, pos
                );
                if (gotInRange != expected)
                    fail("GET_COPRIME6_MERTENS_IN_RANGE", lo, hi, pos,
                         expected, gotInRange);
            }

            if (pos == hi) break;
        }
    }

    for (UInt64 start = 0; start < packedCount; start += 256) {
        const UInt64 end = std::min(start + UInt64(256), packedCount);
        const Int64 before = start ? expectedM[start - 1] : mertens6Prev;
        Int64 localMin = std::numeric_limits<Int64>::max();
        Int64 localMax = std::numeric_limits<Int64>::min();
        for (UInt64 i = start; i < end; ++i) {
            const Int64 local = expectedM[i] - before;
            localMin = std::min(localMin, local);
            localMax = std::max(localMax, local);
        }
        if (localMax - localMin > 255)
            fail("stride-256 residual span", lo, hi, start,
                 255, localMax - localMin);

        const UInt64 block = start >> 8;
        const Int64 expectedCoarse = before + localMin + 128;
        if (coarse[block] != expectedCoarse)
            fail("shifted coarse", lo, hi, start,
                 expectedCoarse, coarse[block]);

        for (UInt64 i = start; i < end; ++i) {
            const Int64 local = expectedM[i] - before;
            const Int64 expectedResidual = local - localMin - 128;
            extrema.lo = std::min(extrema.lo, (int)expectedResidual);
            extrema.hi = std::max(extrema.hi, (int)expectedResidual);
            if (expectedResidual < -128 || expectedResidual > 127)
                fail("residual range", lo, hi, i, 0, expectedResidual);
            if (residual[i] != expectedResidual)
                fail("residual", lo, hi, i,
                     expectedResidual, residual[i]);
        }
    }
}

static void verifySequential(
    UInt64 limit, UInt64 segmentSpan,
    const std::vector<Int64>& referenceM6,
    const std::vector<Int8>& ordinaryMu,
    const std::vector<UInt32>& primes,
    ResidualExtrema& extrema) {
    SegmentedCoprime6MertensSieveCore compressed(segmentSpan);
    SegmentedCoprime6MertensSieveCore fused(segmentSpan);
    SegmentedCoprime6MertensSieveCoreT<
        Coprime6MertensStorage::Direct> direct(segmentSpan);

    Int64 mertens6Prev = 0;
    UInt64 lo = 1;
    while (lo <= limit) {
        const UInt64 span = std::min(segmentSpan, limit - lo + 1);
        const UInt64 hi = lo + span - 1;
        verifyInterval(
            lo, hi, mertens6Prev, ordinaryMu, primes,
            compressed, fused, direct, extrema
        );
        mertens6Prev = referenceM6[hi];
        if (hi == limit) break;
        lo = hi + 1;
    }
}

static void verifyIterators(const std::vector<Int64>& referenceM6) {
    static constexpr UInt64 segmentSpan = 769;
    static constexpr int segmentCount = 16;
    SegmentedCoprime6MertensSieve compressed(segmentSpan);
    SegmentedCoprime6MertensSieveT<
        Coprime6MertensStorage::Direct> direct(segmentSpan);
    std::vector<Int32> compressedValues;
    std::vector<Int32> directValues;

    UInt64 expectedLo = 1;
    for (int segment = 0; segment < segmentCount; ++segment) {
        const UInt64 expectedHi = expectedLo + segmentSpan - 1;
        const auto compressedNext = compressed.nextSegment();
        const auto directNext = direct.nextSegment();
        if (compressedNext.first != expectedLo
            || compressedNext.second != expectedHi)
            fail("compressed nextSegment", expectedLo, expectedHi,
                 expectedLo, expectedHi, compressedNext.second);
        if (directNext.first != expectedLo || directNext.second != expectedHi)
            fail("direct nextSegment", expectedLo, expectedHi,
                 expectedLo, expectedHi, directNext.second);

        if (!compressed.next() || !direct.next())
            fail("iterator next", expectedLo, expectedHi, expectedLo, 1, 0);

        const UInt64 firstPacked =
            SegmentedCoprime6MobiusSieveCore::firstPackedIndex(expectedLo);
        const UInt64 count =
            SegmentedCoprime6MobiusSieveCore::countRange(
                expectedLo, expectedHi
            );
        if (compressed.firstPackedIndex() != firstPacked
            || direct.firstPackedIndex() != firstPacked)
            fail("iterator firstPackedIndex", expectedLo, expectedHi,
                 expectedLo, firstPacked, compressed.firstPackedIndex());
        if (compressed.packedCount() != count
            || direct.packedCount() != count)
            fail("iterator packedCount", expectedLo, expectedHi,
                 expectedLo, count, compressed.packedCount());

        compressed.getSegmentValues(compressedValues);
        direct.getSegmentValues(directValues);
        if (compressedValues.size() != count || directValues.size() != count)
            fail("iterator values size", expectedLo, expectedHi,
                 expectedLo, count, compressedValues.size());

        const Int32* coarse = compressed.getCoarseData();
        const Int8* residual = compressed.getResidualData();
        const Int32* directData = direct.getSegmentData();
        for (UInt64 i = 0; i < count; ++i) {
            const UInt64 n =
                SegmentedCoprime6MobiusSieveCore::originalAt(
                    firstPacked + i
                );
            const Int64 expected = referenceM6[n];
            const Int64 unpacked = coarse[i >> 8] + residual[i];
            if (compressedValues[i] != expected)
                fail("compressed iterator values", expectedLo, expectedHi,
                     n, expected, compressedValues[i]);
            if (directValues[i] != expected)
                fail("direct iterator values", expectedLo, expectedHi,
                     n, expected, directValues[i]);
            if (unpacked != expected)
                fail("compressed iterator accessors", expectedLo,
                     expectedHi, n, expected, unpacked);
            if (directData[i] != expected)
                fail("direct iterator accessor", expectedLo, expectedHi,
                     n, expected, directData[i]);
        }

        for (UInt64 pos = expectedLo; pos <= expectedHi; ++pos) {
            const Int64 expected = referenceM6[pos];
            if (compressed.getCoprime6Mertens(pos) != expected)
                fail("compressed iterator lookup", expectedLo, expectedHi,
                     pos, expected, compressed.getCoprime6Mertens(pos));
            if (direct.getCoprime6Mertens(pos) != expected)
                fail("direct iterator lookup", expectedLo, expectedHi,
                     pos, expected, direct.getCoprime6Mertens(pos));
        }

        expectedLo = expectedHi + 1;
    }
}

int main() {
    static constexpr UInt64 N = 2000000;
    const UInt32 primeBound = (UInt32)std::max(
        (Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
        (Int64)std::round(std::sqrt((double)N))
    );
    const auto primes = SegmentedMobiusSieveCore::primesUpTo(primeBound);

    SegmentedMobiusSieveCore reference(N);
    reference.sieve(1, N, primes);
    std::vector<Int8> ordinaryMu(N + 1);
    std::vector<Int64> ordinaryMertens(N + 1, 0);
    std::vector<Int64> mertens6(N + 1, 0);
    for (UInt64 n = 1; n <= N; ++n) {
        ordinaryMu[n] = reference[n - 1];
        ordinaryMertens[n] = ordinaryMertens[n - 1] + ordinaryMu[n];
        mertens6[n] = mertens6[n - 1]
                    + ((n % 2 != 0 && n % 3 != 0) ? ordinaryMu[n] : 0);

        const Int64 fromM6 = mertens6[n] - mertens6[n / 2]
                           - mertens6[n / 3] + mertens6[n / 6];
        if (ordinaryMertens[n] != fromM6)
            fail("M/M6 identity", 1, N, n,
                 ordinaryMertens[n], fromM6);
    }

    verifyCoordinates();

    ResidualExtrema extrema;
    const UInt64 spans[] = {
        1, 2, 3, 4, 5, 6, 7, 127, 255, 256, 257,
        769, 770, 771, 2309, 65537, 131071
    };
    for (UInt64 span : spans) {
        const UInt64 limit = span <= 7 ? 96
                           : span <= 771 ? 65536
                           : N;
        verifySequential(
            limit, span, mertens6, ordinaryMu, primes, extrema
        );
    }

    // Every pair of endpoint residues, including intervals with an excluded
    // prefix and final excluded tail.
    for (UInt64 loResidue = 0; loResidue < 6; ++loResidue) {
        for (UInt64 hiResidue = 0; hiResidue < 6; ++hiResidue) {
            const UInt64 lo = 600 + loResidue;
            const UInt64 hi = 1200 + hiResidue;
            const UInt64 span = hi - lo + 1;
            SegmentedCoprime6MertensSieveCore compressed(span);
            SegmentedCoprime6MertensSieveCore fused(span);
            SegmentedCoprime6MertensSieveCoreT<
                Coprime6MertensStorage::Direct> direct(span);
            verifyInterval(
                lo, hi, mertens6[lo - 1], ordinaryMu, primes,
                compressed, fused, direct, extrema
            );
        }
    }

    // Exact packed-stride boundaries from both wheel phases.
    const UInt64 packedLengths[] = {255, 256, 257, 511, 512, 513};
    for (UInt64 phase = 0; phase < 2; ++phase) {
        const UInt64 firstPacked = 73 + phase;
        for (UInt64 count : packedLengths) {
            const UInt64 lo =
                SegmentedCoprime6MobiusSieveCore::originalAt(firstPacked);
            const UInt64 hi =
                SegmentedCoprime6MobiusSieveCore::originalAt(
                    firstPacked + count - 1
                );
            const UInt64 span = hi - lo + 1;
            SegmentedCoprime6MertensSieveCore compressed(span);
            SegmentedCoprime6MertensSieveCore fused(span);
            SegmentedCoprime6MertensSieveCoreT<
                Coprime6MertensStorage::Direct> direct(span);
            verifyInterval(
                lo, hi, mertens6[lo - 1], ordinaryMu, primes,
                compressed, fused, direct, extrema
            );
        }
    }

    const std::pair<UInt64, UInt64> ranges[] = {
        {2, 4}, {4, 5}, {8, 10}, {24, 26}, {48, 50},
        {255, 1025}, {769, 4097}, {3464, 10401},
        {65535, 131329}, {999983, 1131053},
        {1048575, 1179651}
    };
    for (const auto& range : ranges) {
        const UInt64 lo = range.first;
        const UInt64 hi = range.second;
        const UInt64 span = hi - lo + 1;
        SegmentedCoprime6MertensSieveCore compressed(span);
        SegmentedCoprime6MertensSieveCore fused(span);
        SegmentedCoprime6MertensSieveCoreT<
            Coprime6MertensStorage::Direct> direct(span);
        verifyInterval(
            lo, hi, mertens6[lo - 1], ordinaryMu, primes,
            compressed, fused, direct, extrema
        );
    }

    UInt64 randomState = 0x9e3779b97f4a7c15ULL;
    for (int trial = 0; trial < 64; ++trial) {
        randomState = randomState * 6364136223846793005ULL
                    + 1442695040888963407ULL;
        const UInt64 lo = 1 + randomState % (N - 8192);
        randomState = randomState * 6364136223846793005ULL
                    + 1442695040888963407ULL;
        const UInt64 span = 1 + randomState % 8192;
        const UInt64 hi = lo + span - 1;
        SegmentedCoprime6MertensSieveCore compressed(span);
        SegmentedCoprime6MertensSieveCore fused(span);
        SegmentedCoprime6MertensSieveCoreT<
            Coprime6MertensStorage::Direct> direct(span);
        verifyInterval(
            lo, hi, mertens6[lo - 1], ordinaryMu, primes,
            compressed, fused, direct, extrema
        );
    }

    {
        static constexpr UInt64 lo = 123456;
        static constexpr UInt64 hi = 312345;
        static constexpr Int64 arbitraryCarry = -1234567890123LL;
        static constexpr UInt64 span = hi - lo + 1;
        SegmentedCoprime6MertensSieveCore compressed(span);
        SegmentedCoprime6MertensSieveCore fused(span);
        SegmentedCoprime6MertensSieveCoreT<
            Coprime6MertensStorage::Direct> direct(span);
        verifyInterval(
            lo, hi, arbitraryCarry, ordinaryMu, primes,
            compressed, fused, direct, extrema
        );
    }

    // Exercise the explicit empty-range contract without touching storage.
    {
        SegmentedCoprime6MertensSieveCore compressed(1);
        SegmentedCoprime6MertensSieveCore fused(1);
        SegmentedCoprime6MertensSieveCoreT<
            Coprime6MertensStorage::Direct> direct(1);
        verifyInterval(
            2, 1, 99, ordinaryMu, primes,
            compressed, fused, direct, extrema
        );
    }

    const auto values = Coprime6MertensSieveValues(10000, 777);
    const UInt64 expectedSize =
        SegmentedCoprime6MobiusSieveCore::countThrough(10000);
    if (values.size() != expectedSize)
        fail("Coprime6MertensSieveValues size", 1, 10000, 10000,
             expectedSize, values.size());
    for (UInt64 k = 0; k < values.size(); ++k) {
        const UInt64 n =
            SegmentedCoprime6MobiusSieveCore::originalAt(k);
        if (values[k] != mertens6[n])
            fail("Coprime6MertensSieveValues", 1, 10000, n,
                 mertens6[n], values[k]);
    }
    const Int32 value = Coprime6MertensSieve(10000, 777);
    if (value != mertens6[10000])
        fail("Coprime6MertensSieve", 1, 10000, 10000,
             mertens6[10000], value);

    verifyIterators(mertens6);

    // Re-run representative boundaries under several OpenMP team sizes.
    const int originalThreads = omp_get_max_threads();
    const int threadCounts[] = {1, 2, 8, 32};
    for (int threads : threadCounts) {
        omp_set_num_threads(threads);
        static constexpr UInt64 lo = 65535;
        static constexpr UInt64 hi = 196607;
        static constexpr UInt64 span = hi - lo + 1;
        SegmentedCoprime6MertensSieveCore compressed(span);
        SegmentedCoprime6MertensSieveCore fused(span);
        SegmentedCoprime6MertensSieveCoreT<
            Coprime6MertensStorage::Direct> direct(span);
        verifyInterval(
            lo, hi, mertens6[lo - 1], ordinaryMu, primes,
            compressed, fused, direct, extrema
        );
    }
    omp_set_num_threads(originalThreads);

    std::printf(
        "PASS: coprime-6 Mertens compressed/direct/fused validation "
        "through %llu\n",
        (unsigned long long)N
    );
    std::printf("Observed stride-256 residual range: [%d, %d]\n",
                extrema.lo, extrema.hi);
    return 0;
}
