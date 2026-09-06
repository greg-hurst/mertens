// demo_odd_mertens_core.cpp — Validate packed odd Mertens representations.

#include "SegmentedMertensSieve.h"
#include "SegmentedOddMertensSieve.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

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

static void verifyInterval(
    UInt64 lo, UInt64 hi, Int64 oddMertensPrev,
    const std::vector<UInt32>& primes,
    SegmentedMobiusSieveCore& full,
    SegmentedOddMertensSieveCore& compressed,
    SegmentedOddMertensSieveCore& fused,
    SegmentedOddMertensSieveCoreT<OddMertensStorage::Direct>& direct,
    ResidualExtrema& extrema) {
    const UInt64 span = hi - lo + 1;
    const UInt64 packedCapacity = span / 2 + (span & 1);
    const UInt64 coarseCapacity =
        (packedCapacity + OddMertensSieveDetail::STRIDE - 1)
        >> OddMertensSieveDetail::STRIDE_LOG;

    std::vector<Int64> coarse(std::max<UInt64>(coarseCapacity, 1));
    std::vector<Int64> fusedCoarse(std::max<UInt64>(coarseCapacity, 1));
    std::vector<Int64> directValues(std::max<UInt64>(packedCapacity, 1));
    std::vector<Int8> residual(std::max<UInt64>(packedCapacity, 1));

    full.sieve(lo, hi, primes);
    const Int8* fullMu = full.data();
    const UInt64 firstOdd = SegmentedOddMobiusSieveCore::firstOddAtOrAbove(lo);
    const UInt64 oddCount = SegmentedOddMobiusSieveCore::countOdds(lo, hi);
    std::vector<Int8> expectedMu(oddCount);
    std::vector<Int64> expectedM(oddCount);

    Int64 running = oddMertensPrev;
    for (UInt64 i = 0; i < oddCount; ++i) {
        const UInt64 n = firstOdd + 2 * i;
        expectedMu[i] = fullMu[n - lo];
        running += expectedMu[i];
        expectedM[i] = running;
    }

    compressed.sieve(lo, hi, oddMertensPrev, coarse.data(), residual.data(), primes);
    if (compressed.firstOdd() != firstOdd)
        fail("firstOdd", lo, hi, lo, firstOdd, compressed.firstOdd());
    if (compressed.oddCount() != oddCount)
        fail("oddCount", lo, hi, lo, oddCount, compressed.oddCount());

    for (UInt64 i = 0; i < oddCount; ++i) {
        if (compressed.mobiusSieve()[i] != expectedMu[i])
            fail("retained odd mu", lo, hi, firstOdd + 2 * i,
                 expectedMu[i], compressed.mobiusSieve()[i]);
    }

    fused.sieveInPlace(lo, hi, oddMertensPrev, fusedCoarse.data(), primes);
    direct.sieveInPlace(lo, hi, oddMertensPrev, directValues.data(), primes);

    for (UInt64 pos = lo; pos <= hi; ++pos) {
        const Int64 expected = pos < firstOdd || oddCount == 0
                             ? oddMertensPrev
                             : expectedM[(pos - firstOdd) >> 1];
        const Int64 gotCompressed = compressed.getOddMertens(coarse.data(), pos);
        const Int64 gotFused = fused.getOddMertens(fusedCoarse.data(), pos);
        const Int64 gotDirect = direct.getOddMertens(directValues.data(), pos);
        const Int64 gotMacro = GET_ODD_MERTENS(
            coarse.data(), residual.data(), firstOdd, oddMertensPrev, pos);

        if (gotCompressed != expected)
            fail("compressed lookup", lo, hi, pos, expected, gotCompressed);
        if (gotFused != expected)
            fail("fused lookup", lo, hi, pos, expected, gotFused);
        if (gotDirect != expected)
            fail("direct lookup", lo, hi, pos, expected, gotDirect);
        if (gotMacro != expected)
            fail("GET_ODD_MERTENS", lo, hi, pos, expected, gotMacro);
    }

    for (UInt64 start = 0; start < oddCount;
         start += OddMertensSieveDetail::STRIDE) {
        const UInt64 end = std::min(start + OddMertensSieveDetail::STRIDE,
                                    oddCount);
        const Int64 before = start ? expectedM[start - 1] : oddMertensPrev;
        Int64 localMin = std::numeric_limits<Int64>::max();
        Int64 localMax = std::numeric_limits<Int64>::min();
        for (UInt64 i = start; i < end; ++i) {
            const Int64 local = expectedM[i] - before;
            localMin = std::min(localMin, local);
            localMax = std::max(localMax, local);
        }
        if (localMax - localMin > 255)
            fail("stride-256 residual span", lo, hi, firstOdd + 2 * start,
                 255, localMax - localMin);

        const UInt64 block = start >> OddMertensSieveDetail::STRIDE_LOG;
        const Int64 expectedCoarse = before + localMin + 128;
        if (coarse[block] != expectedCoarse)
            fail("shifted coarse", lo, hi, firstOdd + 2 * start,
                 expectedCoarse, coarse[block]);

        for (UInt64 i = start; i < end; ++i) {
            const Int64 local = expectedM[i] - before;
            const Int64 expectedResidual = local - localMin - 128;
            extrema.lo = std::min(extrema.lo, (int)expectedResidual);
            extrema.hi = std::max(extrema.hi, (int)expectedResidual);
            if (expectedResidual < -128 || expectedResidual > 127)
                fail("stride-256 residual range", lo, hi, firstOdd + 2 * i,
                     0, expectedResidual);
            if (residual[i] != expectedResidual)
                fail("residual", lo, hi, firstOdd + 2 * i,
                     expectedResidual, residual[i]);
        }
    }
}

static void verifySequential(
    UInt64 limit, UInt64 segmentSpan,
    const std::vector<Int64>& referenceOddMertens,
    const std::vector<UInt32>& primes,
    ResidualExtrema& extrema) {
    SegmentedMobiusSieveCore full(segmentSpan);
    SegmentedOddMertensSieveCore compressed(segmentSpan);
    SegmentedOddMertensSieveCore fused(segmentSpan);
    SegmentedOddMertensSieveCoreT<OddMertensStorage::Direct> direct(segmentSpan);

    Int64 oddMertensPrev = 0;
    for (UInt64 lo = 1; lo <= limit; lo += segmentSpan) {
        const UInt64 hi = std::min(lo + segmentSpan - 1, limit);
        verifyInterval(lo, hi, oddMertensPrev, primes,
                       full, compressed, fused, direct, extrema);
        oddMertensPrev = referenceOddMertens[hi];
    }
}

static void verifyIterators(const std::vector<Int64>& referenceOddMertens) {
    static constexpr UInt64 segmentSpan = 513;
    static constexpr int segmentCount = 16;
    SegmentedOddMertensSieve compressed(segmentSpan);
    SegmentedOddMertensSieveT<OddMertensStorage::Direct> direct(segmentSpan);
    std::vector<Int32> compressedValues;
    std::vector<Int32> directValues;

    UInt64 expectedLo = 1;
    for (int segment = 0; segment < segmentCount; ++segment) {
        const UInt64 expectedHi = expectedLo + segmentSpan - 1;
        const auto compressedNext = compressed.nextSegment();
        const auto directNext = direct.nextSegment();
        if (compressedNext.first != expectedLo || compressedNext.second != expectedHi)
            fail("compressed nextSegment", expectedLo, expectedHi, expectedLo,
                 expectedLo, compressedNext.first);
        if (directNext.first != expectedLo || directNext.second != expectedHi)
            fail("direct nextSegment", expectedLo, expectedHi, expectedLo,
                 expectedLo, directNext.first);

        if (!compressed.next() || !direct.next())
            fail("iterator next", expectedLo, expectedHi, expectedLo, 1, 0);

        const UInt64 firstOdd =
            SegmentedOddMobiusSieveCore::firstOddAtOrAbove(expectedLo);
        const UInt64 oddCount =
            SegmentedOddMobiusSieveCore::countOdds(expectedLo, expectedHi);

        if (compressed.lo() != expectedLo || compressed.hi() != expectedHi)
            fail("compressed bounds", expectedLo, expectedHi, expectedLo,
                 expectedHi, compressed.hi());
        if (direct.lo() != expectedLo || direct.hi() != expectedHi)
            fail("direct bounds", expectedLo, expectedHi, expectedLo,
                 expectedHi, direct.hi());
        if (compressed.firstOdd() != firstOdd || direct.firstOdd() != firstOdd)
            fail("iterator firstOdd", expectedLo, expectedHi, expectedLo,
                 firstOdd, compressed.firstOdd());
        if (compressed.oddCount() != oddCount || direct.oddCount() != oddCount)
            fail("iterator oddCount", expectedLo, expectedHi, expectedLo,
                 oddCount, compressed.oddCount());

        compressed.getSegmentValues(compressedValues);
        direct.getSegmentValues(directValues);
        if (compressedValues.size() != oddCount || directValues.size() != oddCount)
            fail("iterator packed value count", expectedLo, expectedHi,
                 expectedLo, oddCount, compressedValues.size());

        const Int32* coarse = compressed.getCoarseData();
        const Int8* residual = compressed.getResidualData();
        const Int32* directData = direct.getSegmentData();
        for (UInt64 i = 0; i < oddCount; ++i) {
            const UInt64 n = firstOdd + 2 * i;
            const Int64 expected = referenceOddMertens[n];
            const Int64 unpacked =
                coarse[i >> OddMertensSieveDetail::STRIDE_LOG] + residual[i];
            if (compressedValues[i] != expected)
                fail("compressed iterator values", expectedLo, expectedHi,
                     n, expected, compressedValues[i]);
            if (directValues[i] != expected)
                fail("direct iterator values", expectedLo, expectedHi,
                     n, expected, directValues[i]);
            if (unpacked != expected)
                fail("compressed iterator accessors", expectedLo, expectedHi,
                     n, expected, unpacked);
            if (directData[i] != expected)
                fail("direct iterator accessor", expectedLo, expectedHi,
                     n, expected, directData[i]);
        }

        for (UInt64 pos = expectedLo; pos <= expectedHi; ++pos) {
            const Int64 expected = referenceOddMertens[pos];
            if (compressed.getOddMertens(pos) != expected)
                fail("compressed iterator lookup", expectedLo, expectedHi,
                     pos, expected, compressed.getOddMertens(pos));
            if (direct.getOddMertens(pos) != expected)
                fail("direct iterator lookup", expectedLo, expectedHi,
                     pos, expected, direct.getOddMertens(pos));
        }

        const SegmentedOddMertensSieve& constCompressed = compressed;
        const SegmentedOddMertensSieveT<OddMertensStorage::Direct>& constDirect = direct;
        if (constCompressed.core().oddCount() != oddCount
            || constDirect.core().oddCount() != oddCount)
            fail("const iterator core", expectedLo, expectedHi, expectedLo,
                 oddCount, constCompressed.core().oddCount());

        expectedLo = expectedHi + 1;
    }
}

int main() {
    static constexpr UInt64 N = 2000000;
    const UInt32 primeBound = (UInt32)std::max(
        (Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
        (Int64)std::round(std::sqrt((double)N)));
    const auto primes = SegmentedMobiusSieveCore::primesUpTo(primeBound);

    SegmentedMobiusSieveCore reference(N);
    reference.sieve(1, N, primes);
    std::vector<Int64> ordinaryMertens(N + 1, 0);
    std::vector<Int64> oddMertens(N + 1, 0);
    for (UInt64 n = 1; n <= N; ++n) {
        ordinaryMertens[n] = ordinaryMertens[n - 1] + reference[n - 1];
        oddMertens[n] = oddMertens[n - 1] + ((n & 1) ? reference[n - 1] : 0);
        const Int64 fromOdd = oddMertens[n] - oddMertens[n >> 1];
        if (ordinaryMertens[n] != fromOdd)
            fail("M(x) = M_2(x) - M_2(x/2)", 1, N, n,
                 ordinaryMertens[n], fromOdd);
    }

    ResidualExtrema extrema;
    verifySequential(32, 1, oddMertens, primes, extrema);
    verifySequential(64, 2, oddMertens, primes, extrema);
    verifySequential(96, 3, oddMertens, primes, extrema);
    verifySequential(8192, 127, oddMertens, primes, extrema);
    verifySequential(16384, 256, oddMertens, primes, extrema);
    verifySequential(65536, 3465, oddMertens, primes, extrema);
    verifySequential(N, 131071, oddMertens, primes, extrema);

    // Deterministic adversarial interval alignments around powers of two,
    // stencil boundaries, squares, and the packed coarse-stride boundary.
    const std::pair<UInt64, UInt64> ranges[] = {
        {2, 2}, {255, 769}, {256, 1024}, {511, 1537},
        {3464, 10401}, {13859, 27721}, {65535, 131329},
        {999983, 1131053}, {1048575, 1179651}
    };
    for (const auto& range : ranges) {
        const UInt64 lo = range.first;
        const UInt64 hi = range.second;
        const UInt64 span = hi - lo + 1;
        SegmentedMobiusSieveCore full(span);
        SegmentedOddMertensSieveCore compressed(span);
        SegmentedOddMertensSieveCore fused(span);
        SegmentedOddMertensSieveCoreT<OddMertensStorage::Direct> direct(span);
        verifyInterval(lo, hi, oddMertens[lo - 1], primes,
                       full, compressed, fused, direct, extrema);
    }

    UInt64 randomState = 0x9e3779b97f4a7c15ULL;
    for (int trial = 0; trial < 64; ++trial) {
        randomState = randomState * 6364136223846793005ULL + 1442695040888963407ULL;
        const UInt64 lo = 1 + randomState % (N - 8192);
        randomState = randomState * 6364136223846793005ULL + 1442695040888963407ULL;
        const UInt64 span = 1 + randomState % 8192;
        const UInt64 hi = lo + span - 1;
        SegmentedMobiusSieveCore full(span);
        SegmentedOddMertensSieveCore compressed(span);
        SegmentedOddMertensSieveCore fused(span);
        SegmentedOddMertensSieveCoreT<OddMertensStorage::Direct> direct(span);
        verifyInterval(lo, hi, oddMertens[lo - 1], primes,
                       full, compressed, fused, direct, extrema);
    }

    {
        static constexpr UInt64 lo = 123456;
        static constexpr UInt64 hi = 312345;
        static constexpr Int64 arbitraryCarry = -1234567890123LL;
        static constexpr UInt64 span = hi - lo + 1;
        SegmentedMobiusSieveCore full(span);
        SegmentedOddMertensSieveCore compressed(span);
        SegmentedOddMertensSieveCore fused(span);
        SegmentedOddMertensSieveCoreT<OddMertensStorage::Direct> direct(span);
        verifyInterval(lo, hi, arbitraryCarry, primes,
                       full, compressed, fused, direct, extrema);
    }

    const auto values = OddMertensSieveValues(10000, 777);
    if (values.size() != 5000)
        fail("OddMertensSieveValues size", 1, 10000, 10000, 5000, values.size());
    for (UInt64 i = 0; i < values.size(); ++i) {
        const UInt64 n = 2 * i + 1;
        if (values[i] != oddMertens[n])
            fail("OddMertensSieveValues", 1, 10000, n, oddMertens[n], values[i]);
    }
    const Int32 value = OddMertensSieve(10000, 777);
    if (value != oddMertens[10000])
        fail("OddMertensSieve", 1, 10000, 10000, oddMertens[10000], value);

    verifyIterators(oddMertens);

    std::printf("PASS: odd Mertens compressed/direct/fused validation through %llu\n",
                (unsigned long long)N);
    std::printf("Observed stride-256 residual range: [%d, %d]\n",
                extrema.lo, extrema.hi);
    return 0;
}
