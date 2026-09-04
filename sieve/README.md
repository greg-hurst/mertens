# Segmented Mertens & Mobius Sieve

A high-performance segmented sieve of the Mobius function $\mu(k)$ and the Mertens function $M(n) = \sum_{k=1}^{n} \mu(k)$, using SIMD intrinsics (ARM NEON, SSE2, AVX2, AVX-512, SVE2) and OpenMP parallelism.

The Mobius sieve uses the following processing tiers:

1. **Stencil primes** $p \le 11$ use a pre-sieved pattern of period $13860 = \text{lcm}(4, 9, 5, 7, 11)$.
2. **Small primes** $13 \le p \le 353$ use hardcoded constant-stride loops.
3. **M1 medium primes** $359 \le p \le 32000$ use constant-prime tables and a four-stream M1 walk.
4. **Tiled medium primes** above 32000 are sieved directly in wider tiles.
5. **Large primes** use a circular bucket scheduler; `BUCKET_SIEVE=0` falls back to tiled direct iteration.

The exact handoff from tiled medium to large primes depends on the finalization path and its direct-sieve cutoff. See [Performance & technical details](PERFORMANCE.md) for the current defaults.

The Mertens sieve wraps the Mobius sieve and adds a parallel prefix sum to convert $\mu$ values into cumulative $M(x)$ values with a compressed storage scheme: $M(x) = \text{coarse}[x / 256] + \text{residual}[x]$. The stride (256) is controlled by the `SIEVE_STRIDE_LOG` build flag (default 8, i.e. $2^8 = 256$). Smaller values increase coarse array size but reduce residual range; larger values do the opposite. See [Int8 residual overflow](#range-limits) for the tradeoffs.

## Platform

Cross-platform. The SIMD finalization step auto-detects the best available instruction set at compile time (AVX-512 > AVX2 > SSE2 on x86, SVE2 > NEON on ARM), falling back to a scalar path when no SIMD is available. On x86 the Makefile also enables the division-free quotient path by default (`DIVISION_FREE`), since hardware division is slower there. Developed and tuned on Apple Silicon; Linux builds with the same `make`, and Windows works through WSL (see the top-level README).

## Dependencies

- **C++17 compiler** (clang++ or g++)
- **OpenMP** — macOS: `brew install libomp`; Linux / WSL: comes with g++ (`sudo apt install build-essential`)

## Building

```
make                  # build all demos
make BUCKET_SIEVE=0   # build without the large-prime bucket scheduler
make clean            # remove build artifacts
```

Binaries are placed in `build/`.

## API

Both the Mobius and Mertens sieves offer three levels of abstraction, from simplest to most control.

All of the low-level performance work lives in the core classes — they are what `MertensHurst` and the record computations use directly. The extra functions and iterators are convenience wrappers over the cores. They are still fast (the iteration itself measures about 1% over a raw core loop), but for maximum speed use a core class and read results in place through the zero-copy accessors instead of the copy-out methods, which cost a full extra pass over each segment.

### Mobius sieve

#### MobiusSieveValues

For one-off computations where you just want all $\mu$ values.

```cpp
#include "SegmentedMobiusSieve.h"

// Compute mu(1), mu(2), ..., mu(N) — returns a vector of size N
std::vector<Int8> mu = MobiusSieveValues(1000000);
// mu[0] = mu(1) = 1, mu[1] = mu(2) = -1, ..., mu[3] = mu(4) = 0

// Optional second argument sets the internal segment size
std::vector<Int8> mu2 = MobiusSieveValues(1000000000, 110880000);
```

#### SegmentedMobiusSieve (iterator)

An iterator for streaming through segments. Construct with any segment size, then call `next()` to advance. Handles prime generation internally. Useful when you need $\mu$ values in order but don't want to hold the entire range in memory.

```cpp
#include "SegmentedMobiusSieve.h"

SegmentedMobiusSieve sieve(10000000);  // 10M per segment

auto [nlo, nhi] = sieve.nextSegment();  // bounds the next next() call will sieve

while (sieve.next()) {
    // Point lookup
    Int8 muK = sieve.getMobius(k);  // k in [sieve.lo(), sieve.hi()]

    // Zero-copy pointer to the segment's values (preferred in hot loops)
    const Int8* mu = sieve.getSegmentData();
    // mu[0] = mu(sieve.lo()), mu[1] = mu(sieve.lo()+1), ...

    // Or get an owned copy as a flat Int8 vector — costs a full
    // pass over the segment
    std::vector<Int8> vals;
    sieve.getSegmentValues(vals);

    if (sieve.hi() >= N) break;
}
```

#### SegmentedMobiusSieveCore (manual control)

Full manual control. You manage primes, buffers, and segment iteration yourself. Arbitrary intervals are supported — no alignment constraints.

```cpp
#include "SegmentedMobiusSieve.h"

UInt64 lo = 100, hi = 200000;

SegmentedMobiusSieveCore sieve(hi - lo + 1);
auto primes = SegmentedMobiusSieveCore::primesUpTo(std::max(360, (Int32)std::sqrt((double)hi)));

sieve.sieve(lo, hi, primes);

Int8 muK = sieve[k - lo];    // mu(k) for k in [lo, hi]
```

### Mertens sieve

#### MertensSieve / MertensSieveValues

For one-off computations where you just want the answer.

```cpp
#include "SegmentedMertensSieve.h"

// Compute M(N) — returns a single value
Int32 m = MertensSieve(1000000);    // m = 212

// Compute M(1), M(2), ..., M(N) — returns a vector of size N
std::vector<Int32> M = MertensSieveValues(1000000);
// M[0] = M(1) = 1, M[1] = M(2) = 0, ..., M[999999] = M(10^6) = 212
```

#### SegmentedMertensSieve (iterator)

An iterator for streaming through segments. Construct with any segment size, then call `next()` to advance. Handles prime generation internally.

The iterator is templated on `MertensStorage`:
- `Compressed` (default): coarse + Int8 residual. Best for point lookups.
- `Direct`: full Int32 array. Best when you need all values.

```cpp
#include "SegmentedMertensSieve.h"

// Compressed mode (default) — good for point lookups
SegmentedMertensSieve sieve(10000000);

auto [nlo, nhi] = sieve.nextSegment();  // bounds the next next() call will sieve

while (sieve.next()) {
    Int32 m = sieve.getMertens(pos);  // pos in [sieve.lo(), sieve.hi()]

    // Zero-copy pointers to the compressed representation:
    // M(pos) = C[i >> STRIDE_LOG] + R[i] with i = pos - sieve.lo()
    const Int32* C = sieve.getCoarseData();
    const Int8*  R = sieve.getResidualData();

    if (sieve.hi() >= N) break;
}

// Direct mode — good for bulk value retrieval
SegmentedMertensSieveT<MertensStorage::Direct> sieve2(10000000);

while (sieve2.next()) {
    const Int32* data = sieve2.getSegmentData();  // zero-copy access
    // data[0] = M(sieve2.lo()), data[1] = M(sieve2.lo()+1), ...
    if (sieve2.hi() >= N) break;
}
```

#### SegmentedMertensSieveCore (manual control)

Full manual control. You manage primes, buffers, and segment iteration yourself. Arbitrary intervals and segment sizes are supported. This is what `MertensHurst` uses internally.

```cpp
#include "SegmentedMertensSieve.h"

UInt64 segSize = 10000000;

UInt32 sqrtN = (UInt32)std::round(std::sqrt((double)N));
auto primes = SegmentedMobiusSieveCore::primesUpTo(std::max(360, sqrtN));

// Compressed mode (default) — point lookups via getM macro
SegmentedMertensSieveCore sieve(segSize);
UInt64 numIntervals =
    (segSize + SegmentedMertensSieveCore::STRIDE - 1)
    >> SegmentedMertensSieveCore::STRIDE_LOG;
std::vector<Int64> M(numIntervals);
Int64 MPrev = 0;

for (UInt64 slo = 1; slo <= N; slo += segSize) {
    UInt64 shi = std::min(slo + segSize - 1, N);
    sieve.sieveInPlace(slo, shi, MPrev, M.data(), primes);
    MPrev = sieve.getMertens(M.data(), shi);
}

// Direct mode — full M values, no compression overhead
SegmentedMertensSieveCoreT<MertensStorage::Direct> sieveDirect(segSize);
std::vector<Int32> Mfull(segSize);
Int32 MPrev2 = 0;

for (UInt64 slo = 1; slo <= N; slo += segSize) {
    UInt64 shi = std::min(slo + segSize - 1, N);
    sieveDirect.sieveInPlace(slo, shi, MPrev2, Mfull.data(), primes);
    // Mfull[k] = M(slo + k) for k in [0, shi - slo]
    MPrev2 = sieveDirect.getMertens(Mfull.data(), shi);
}
```

Compressed mode has two sieve methods:
- `sieve()` — separate residual buffer (preserves $\mu$ data)
- `sieveInPlace()` — residual aliases the $\mu$ buffer (saves memory)

## Demos

```
build/demo_mobius_values <N> [seg]       # MobiusSieveValues(N) — prints mu(1)..mu(20) and M(N)
build/demo_mobius_segmented <N> [seg]    # SegmentedMobiusSieve iterator
build/demo_mobius_core <N> [seg]         # SegmentedMobiusSieveCore manual loop
build/demo_mertens_value <N>             # MertensSieve(N) — just prints M(N)
build/demo_mertens_values <N>            # MertensSieveValues(N) — prints M(1)..M(20) and M(N)
build/demo_mertens_segmented <N> [seg]   # SegmentedMertensSieve iterator
build/demo_mertens_core <N> [seg]        # SegmentedMertensSieveCore manual loop
```

```
$ ./build/demo_mobius_values 20
MobiusSieveValues: N = 20
mu(1) = 1
mu(2) = -1
mu(3) = -1
mu(4) = 0
...
mu(20) = 0

$ ./build/demo_mertens_value 1000000
MertensSieve: N = 1000000
M(1000000) = 212

$ ./build/demo_mertens_values 20
MertensSieveValues: N = 20
M(1) = 1
M(2) = 0
M(3) = -1
...
M(20) = -3
```

## Range limits

With default settings, the effective sieve limit is $2.05 \times 10^{17}$ (bucket scheduler constraint). With `BUCKET_SIEVE=0`, the limit is $2^{60}-2^{32}$ when `DIVISION_FREE=1`, or roughly $1.8 \times 10^{19}$ when direct division is used. See [PERFORMANCE.md](PERFORMANCE.md) for the full analysis of each constraint:

- **Log-prime encoding** — the uniform ceil-log2 scheme is collision-free; the only cap is the 7-bit field overflowing, at $N < 2^{64} \approx 1.8 \times 10^{19}$ (§5)
- **Bucket scheduler** — `LP_SIZE = 512` limits range to $\sim 2.05 \times 10^{17}$ (§6)
- **UInt32 primes** — caps sieve endpoint at $\sim 1.8 \times 10^{19}$ (§7)
- **Division-free quotient cache** — requires the endpoint below $2^{60}-2^{32}$ (§3)
- **Int8 residual overflow** — default `STRIDE_LOG=8` has a heuristic range near $1.9 \times 10^{25}$; stride 7 has a proof for every range (§8)

## Files

```
types.h                    Type aliases (Int128, Int64, Int8, etc.)
simd_defs.h                SIMD platform detection (NEON/SSE/AVX2/AVX-512/SVE2)
SegmentedMobiusSieve.h     Mobius sieve: core class, iterator, and MobiusSieveValues
SegmentedMobiusSieve.cpp   Mobius sieve implementation (SIMD finalization, bucket scheduler)
SegmentedMertensSieve.h    Mertens sieve: core class, iterator, standalone functions, and getM macro
stencil_data.h             Precomputed 13860-element stencil array
demo/                      Demo programs (one per API tier, for both Mobius and Mertens)
PERFORMANCE.md             Runtime, memory layout, and range constraint analysis
```
