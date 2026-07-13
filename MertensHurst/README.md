# MertensHurst

A high-performance implementation of the Mertens function $M(n) = \sum_{k=1}^{n} \mu(k)$, using the classical $O(n^{2/3})$ combinatorial algorithm ([Hurst 2026](https://arxiv.org/abs/2607.07566)). Computes $M(n)$ for values up to $10^{25}$.

This code was used to set the record for computing the Mertens function.

## Platform

Developed and tuned on Apple Silicon; also builds and runs on x86-64 (the paper's cross-machine comparisons used a 28-core Xeon W). SIMD kernels are auto-selected at compile time: NEON/SVE2 on ARM, SSE2/AVX2/AVX-512 on x86.

On Apple Silicon the code uses direct hardware integer division, which is fast on ARM. On x86 the Makefile defaults to the division-free codepath (Granlund-Montgomery quotient cache plus quotient predictor) — see the `DIVISION_FREE` flag below and `COMPILE_FLAGS.md`.

Linux builds with the same `make`. On Windows use WSL (see the top-level README).

## Dependencies

- **C++17 compiler** (clang++ or g++)
- **OpenMP** — macOS: `brew install libomp`; Linux / WSL: comes with g++ (`sudo apt install build-essential`)

## Building

```
make        # build the binary
make q2     # build the original all-Q2 reference binary
make clean  # remove build artifacts
```

The normal binary, `build/mertens`, uses exact outer-$Q=6$ transforms for both
$S_1$ and $S_2$. The reference target produces `build/mertens_q2`; it is slower
but computes the same result through the original all-$Q=2$ path.

### Division-free mode

The Makefile auto-detects the CPU architecture and selects the optimal integer division strategy:

- **ARM** (`DIVISION_FREE=0`, default on Apple Silicon): uses fast hardware division directly.
- **x86_64** (`DIVISION_FREE=1`, default on x86): uses division-free methods (Granlund-Montgomery quotient cache + quotient predictor) to avoid expensive hardware `div` instructions in hot loops.

To override the auto-detected default:

```
make DIVISION_FREE=1   # force division-free quotient methods
make DIVISION_FREE=0   # force direct hardware division
make BUCKET_SIEVE=0    # disable the large-prime bucket scheduler
```

Both `DIVISION_FREE` and `BUCKET_SIEVE` produce identical numerical results. The choices affect only performance.

## Usage

```
./build/mertens <n> [options]
```

where `n` is an integer with $10^8 \le n \le 10^{25}$, in plain decimal or scientific notation (`1e22`, `2.5e21`).

Options:

- `--profile` (or `-p`): print a timing breakdown by computation phase, along with the parameter values used.
- `--segment-cap <len>`: cap on the sieve segment length in the large-segment phase, in integers (default: 12000000000, about 12 GB — the sieve costs roughly one byte per integer). Larger segments mean fewer sieve passes but more memory; the value is rounded up to a multiple of the stencil period (13860). Raise it for very large inputs (the $10^{25}$ record run used $4 \times 10^{11}$, about 400 GB) if you have the RAM.
- `--u <value>`: set the sieve truncation point $u$ directly, bypassing the default formula. Must satisfy $0 < u < n$. Hard caps are enforced at runtime per build: $u \le 2.06 \times 10^{17}$ with the bucket scheduler (the default), and $u \lesssim 1.8 \times 10^{19}$ always (UInt32 primes / byte encoding). On `DIVISION_FREE=1` builds also keep $u < 2^{60} - 2^{32}$ (see `INPUT_BOUNDS.md` constraints 3-5). Larger $u$ shifts work from S1/S2 summation into sieving; smaller $u$ does the opposite.
- `--u-factor <value>`: override the scaling factor in the $u$ formula: $u = \lceil \text{factor} \cdot (n / \ln \ln n)^{2/3} \rceil$. Must be positive. The default factor is computed as $\text{clamp}(1.85 - 0.05 \log_{10} n,\ 0.75,\ 1.05)$. Mutually exclusive with `--u`.
- `--nu-ratio <value>`: S1/S2 split ratio (default: $1.5$). Controls the boundary between the S1 (Mertens sum) and S2 (Möbius sum) ranges via $\nu(x) = \lfloor \text{ratio} \cdot \sqrt{x} \rfloor$. Must be positive. Affects only performance, not correctness.

Examples:

```
$ ./build/mertens 10000000000
M(10000000000) = -33722 in 0.011 seconds

$ ./build/mertens 10000000000000000 --profile
M(10000000000000000) = -3195437 in 7.1 seconds

--------------- Loop 1 16-bit ---------------
...

$ ./build/mertens 10000000000000000000000000 --segment-cap 50000000000
M(10000000000000000000000000) = ... 
```

To use as a library in your own code:

```cpp
#include "MertensHurst.h"

Int64 result = MertensHurst(n);                          // 10^8 <= n <= 10^25
Int64 result = MertensHurst(n, true);                    // with profiling output
Int64 result = MertensHurst(n, false, 50000000000ULL);   // custom Loop 2 segment cap
Int64 result = MertensHurst(n, false, 12000000000ULL,
                            0, 0.85, 1.5);               // custom u-factor and nu-ratio
```

## File structure

```
INPUT_BOUNDS.md             Analysis of bounds on n
COMPILE_FLAGS.md            What each build flag does
src/
  MertensHurst.h            Public API: Int64 MertensHurst(UInt128 n)
  MertensHurst.cpp          Algorithm orchestration (loops, back substitution)
  S1.h                      S1 summation functions (64-bit and 128-bit)
  S1Q6.h                    Exact outer-Q6 S1 transform kernels
  S2.h                      S2 summation functions (64-bit and 128-bit)
  S2Q6.h                    Exact outer-Q6 S2 dispatch
  S2Q6Modes.h               Static outer-Q6 S2 coefficient modes
  OuterRecovery.h           Square-free back substitution
  QuotientPredictor.h       Division-free quotient estimation
  main.cpp                  Driver program
../sieve/                   Shared segmented Mobius sieve (see sieve/README.md)
../sieve/SegmentedMertensSieve.h  Mertens sieve (prefix sum over Mobius values)
../sieve/QuotientCache.h    Granlund-Montgomery quotient cache (compile-time optional)
build/                      Compiled binary (gitignored)
```
