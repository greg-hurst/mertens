# MertensHT

An implementation of the Mertens function $M(n) = \sum_{k=1}^{n} \mu(k)$ using the Helfgott-Thompson $O(n^{3/5})$ combinatorial algorithm ([arXiv:2101.08773](https://arxiv.org/abs/2101.08773)). Tested through $10^{23}$; inputs beyond that are enabled by 256-bit fallbacks (projected timings for $10^{24}$ and $10^{25}$ appear in [Hurst 2026](https://arxiv.org/abs/2607.07566)).

This implementation includes [30 performance optimizations](OPTIMIZATIONS.md) over the [original author code](https://arxiv.org/src/2101.08773v4/anc) (the ancillary files of the arXiv submission), running **3.4x-4.5x faster** across $10^{16}$ to $10^{23}$ on the record machine.

## Platform

Developed and tuned on Apple Silicon; also builds and runs on x86-64 (the paper's comparisons used a 28-core Xeon W). The shared sieve auto-selects its SIMD backend at compile time, and on x86 defaults to its division-free quotient path.

Linux builds with the same `make`. On Windows use WSL (see the top-level README) — this code in particular assumes a 64-bit `long`, which native Windows toolchains do not provide.

## Dependencies

- **C++17 compiler** (clang++ or g++)
- **OpenMP** — macOS: `brew install libomp`; Linux / WSL: comes with g++ (`sudo apt install build-essential`)
- **Boost** (header-only, for 256-bit fallbacks) — macOS: `brew install boost`; Linux / WSL: `sudo apt install libboost-dev`

Two dependency changes relative to the original author code. GMP is no longer
needed: the original used it for exact integer roots of 128-bit quantities,
all of which are now native integer routines. Boost is new: the
256-bit fallbacks that enable inputs beyond $10^{23}$ need exact arithmetic
wider than `__int128` (see [OPTIMIZATIONS.md](OPTIMIZATIONS.md), item 30).

## Building

```
make                  # build the binary
make BUCKET_SIEVE=0   # build without the large-prime bucket scheduler
make clean            # remove build artifacts
```

The binary is placed in `build/`.

## Usage

```
./build/mertens_ht <n> [--profile] [--vdiv <v>] [--C <c>] [--D <d>] [--ntasks <n>] [--sarr-cap <bytes>]
```

where `n` is the value: computes $M(n)$ for $n \geq 10^6$, in plain decimal or scientific notation (`1e22`).

Options:

- `--profile` (or `-p`): print a timing breakdown by computation phase.
- `--vdiv <v>`: divisor for $v = x^{2/5} / \text{vdiv}$ (default: 2.0).
- `--C <c>`: crossover parameter controlling thresholds in LargeFree (default: 42).
- `--D <d>`: tile size for DDSum (default: 16).
- `--ntasks <n>`: task multiplier for parallelism granularity (default: 32).
- `--sarr-cap <bytes>`: memory cap for the SArr working arrays (default: 12 GB). Raise it on big machines — the record runs used 400 GB — or lower it to fit small ones; the large-non-free phase trades parallel width for memory.

Examples:

```
$ ./build/mertens_ht 1e12
M(1000000000000) = 62366 in 0.11 seconds

$ ./build/mertens_ht 1e16 --profile
M(10000000000000000) = -3195437 in 43 seconds

$ ./build/mertens_ht 1e14 --ntasks 128
M(100000000000000) = -875575 in 2.2 seconds
```

To use as a library in your own code:

```cpp
#include "MertensHT.h"

Int64 result = MertensHT(x);                      // default parameters
Int64 result = MertensHT(x, true);                 // with profiling
Int64 result = MertensHT(x, false, 2.0, 42, 16, 32,
                         12000000000LL);               // all params explicit
```

## File structure

```
OPTIMIZATIONS.md        30 performance optimizations over the original code
COMPILE_FLAGS.md        What each build flag does
src/
  MertensHT.h           Public API: Int64 MertensHT(Int128 x)
  MertensHT.cpp         Algorithm orchestration (BruteM, LargeFree, LargeNonFree)
  IntegerMath.h         Floor division, modular arithmetic, integer roots
  Factorization.h       Factorization stencil, FacToSumMu
  SumByLin.h            Diophantine approximation and double-sum hot paths
  main.cpp              Driver program
../sieve/               Shared segmented Mobius sieve (see sieve/README.md)
build/                  Compiled binary (gitignored)
```
