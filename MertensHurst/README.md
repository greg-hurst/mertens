# MertensHurst

A high-performance implementation of the Mertens function $M(n) = \sum_{k=1}^{n} \mu(k)$, using the classical $O(n^{2/3})$ combinatorial algorithm ([Hurst 2026](https://arxiv.org/abs/2607.07566)). Accepts inputs through $10^{26}$; completed results have been independently verified through $10^{25}$.

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
make s2-unordered  # build the coherent period-36/unordered-S2 profile
make q30-coupled   # build the complete coupled Q=30 profile
make q210-coupled  # build the complete full-M Loop-2 Q=210 profile
make q210-coupled-loop2-p1  # explicit compile-time full-M Loop 2
make q210-coupled-loop2-p2  # explicit compile-time packed-odd Loop 2
make q210-coupled-loop2-p6  # explicit compile-time packed-coprime-6 Loop 2
make q210-coupled-odd-loop2  # compatibility alias for the P2 binary
make q210-coupled-record  # Q=210 with rigorously bounded Int8 residuals
make q210-coupled-native  # build the pre-ladder Q=210 baseline
make clean  # remove build artifacts
```

The normal binary, `build/mertens`, uses exact outer-$Q=6$ transforms for both
$S_1$ and $S_2$. The reference target produces `build/mertens_q2`; it is slower
but computes the same result through the original all-$Q=2$ path.

The opt-in `s2-unordered` target produces `build/mertens_s2_unordered`. It is
a contract-closed final-value profile: coherent outer-$Q=6$ splits, generated
period-36 kernels, and exact unordered hot-$S_2$ pairing are enabled together,
while full recovery and the expensive ordered-square validator are disabled.
Its runtime interface and exact result are the same as `build/mertens`.

The opt-in `q30-coupled` and `q210-coupled` targets extend that same exact
stack through the square-free divisors of 30 and 210. Q=210 is selected only
when one shared split is legal for the entire run; otherwise the whole run
falls back to Q=30. It never mixes Q=30 and Q=210 rows.

The `q210-coupled` profile also groups eligible $S_1$ rows through 11 and 13.
It uses $Q=2310$ and $Q=30030$ parent kernels with exact $Q=210$ seams, while
preserving the original dynamic-per-row scheduler. Loop 2 stays native;
wide Loop-1 roots use only the profitable factor-11 pairing. Two membership
masks and two monotone child maps are temporary and released before Loop 2
(about 249 MB at $10^{25}$). The Q=210 coefficient table occupies about
1.35 MiB and is also released before Loop 2. Use `q210-coupled-native` for
the exact pre-ladder baseline.

The compile-time `LOOP2_SIEVE_P` selector changes only the S1-only Loop 2.
The `q210-coupled-loop2-p1` target retains full $M$,
`q210-coupled-loop2-p2` uses the exact identity

$$M(x)=M_2(x)-M_2(\lfloor x/2\rfloor),\qquad
M_2(x)=\sum_{\substack{k\le x\\k\text{ odd}}}\mu(k),$$

and sieves $M_2$ natively in packed odd coordinates. The two signed terms are
applied at their respective segment visits to the same existing S1 row and
accumulator. A short full-M bridge handles the phase seam exactly; Loop 0/1,
S2, unordered S2, recovery, and the factor-11/factor-13 ladder are unchanged.
The bridge uses smaller temporary chunks so allocator-retained bridge pages do
not overlap a second full-size allocation. The main odd phase still consumes
the full stored-entry budget. A P2 binary whose runtime Q210 guard fails uses
the compiled full-$M$ fallback. The historical `q210-coupled-odd-loop2`
targets and binary paths remain aliases for compatibility.

The `q210-coupled-loop2-p6` target instead uses

$$M(x)=M_6(x)-M_6(\lfloor x/2\rfloor)-M_6(\lfloor x/3\rfloor)
       +M_6(\lfloor x/6\rfloor),\qquad
M_6(x)=\sum_{\substack{k\le x\\(k,6)=1}}\mu(k),$$

with a native packed coprime-to-6 sieve. Its M6 stream starts at one with
zero carry. Each Q210 row keeps one common ownership domain from the original
unscaled numerator and clips up to four scaled live bands inside it. Wide rows
partition the union of those bands and visit each denominator once, deriving
the exact $q/2$, $q/3$, and $q/6$ arguments from one $q=\lfloor y/d\rfloor$.
Narrow rows retain the faster cache-friendly four-stream traversal. The Q210
quotient stepper remains a compile-time width of eight; full narrow fusion and
a width of sixteen were both slower in local measurements. There is no full-M
bridge or retained M6 prefix history. At the same stored-entry cap, a P6
segment covers three times the original-coordinate span of P1. A failed
runtime Q210 guard still activates the full-M backend with the P1 scheduler
domain. The stricter P6 scheduler reach is enforced only when P6 is active.

Fixed-parameter measurements on the 32-thread M3 Ultra (`nuRatio=0.9`,
`u-factor=0.5`, segment cap $4\times10^{11}$) gave:

| Input | Full-M total | Odd total | Speedup | Peak RSS change |
|---:|---:|---:|---:|---:|
| $10^{16}$ | 1.0084 s | 0.8814 s | 1.144x | about +22% |
| $10^{19}$ | 89.2612 s | 77.5044 s | 1.152x | +23.1% |

The $10^{16}$ times are medians of 12 alternating runs per binary; the
$10^{19}$ row uses isolated representative runs. At $10^{19}$, Loop-2 sieve
time fell from 29.4047 s to 16.6669 s. The second signed S1 visit added 1.0804
s, only 8.5% of the 12.7378 s sieve saving. The odd path uses the same primary
byte-entry budget as the full path; its additional prefix metadata explains
the measured RSS increase.

For rigorously bounded compressed residuals on a large run,
`q210-coupled-record` builds the same Q=210 algorithm with
`SIEVE_STRIDE_LOG=7`. The ordinary profile retains the faster stride 8; its
byte residual has a very conservative statistical margin but does not have
the stride-7 worst-case proof.

Loop 0 groups every eligible factor-13 root; Loop 1 does so only when the
root argument is at most $10^{18}$, leaving wider roots as two factor-11
pairs. Within a full quartet, the hierarchical seam is selected only when
the duplicated Q=210 interval
$(\lfloor\kappa/143\rfloor,\lfloor\kappa_{13}/11\rfloor]$ contains a
coprime-to-210 denominator in the current sieve segment. Otherwise the
algebraically equivalent independent-Q210 seams avoid unnecessary traversal.

The Q=210 profile has dedicated validation builds:

```
make q210-coupled-validate      # ordered-square comparison enabled
make q210-coupled-sanitize      # ASan and UBSan executable
make q210-coupled-fallback      # validate a forced Q210-to-Q30 fallback
make q210-coupled-loop2-p2-validate  # full-M and direct-odd row checks
make q210-coupled-loop2-p2-sanitize  # P2 Loop 2 under ASan and UBSan
make q210-coupled-loop2-p2-fallback  # force and validate P2-to-P1 fallback
make q210-coupled-loop2-p6-validate  # full-M/direct-M6/stream/row checks
make q210-coupled-loop2-p6-sanitize  # P6 Loop 2 under ASan and UBSan
make q210-coupled-loop2-p6-fallback  # force and validate P6-to-P1 fallback
```

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

where `n` is an integer with $10^8 \le n \le 10^{26}$, in plain decimal or scientific notation (`1e22`, `2.5e21`). The selected split must also satisfy $\lfloor\text{nuRatio}\sqrt n\rfloor > 13860$; with the default ratio 0.9 this requires approximately $n \ge 2.372 \times 10^8$.

Options:

- `--profile` (or `-p`): print a timing breakdown by computation phase, along with the parameter values used.
- `--segment-cap <len>`: cap on stored sieve entries in the large-segment phase (default: 12000000000, about 12 GB — plus compressed-prefix state). The P1, packed-P2, and packed-P6 Loop 2 sieves cover one, two, and three original integers per stored entry, respectively. Larger caps mean fewer sieve passes but more memory; the value is rounded up to a multiple of the stencil period (13860). Raise it for very large inputs (the $10^{25}$ record run used $4 \times 10^{11}$) if you have the RAM; budget the full memory model in `INPUT_BOUNDS.md` before a $10^{26}$ run.
- `--u <value>`: set the sieve truncation point $u$ directly, bypassing the default formula. Must satisfy $0 < u < n$. Hard caps are enforced at runtime per build: P1/P2 permit $u \le 2.05 \times 10^{17}$ with the bucket scheduler, while active P6 permits $u \le 115{,}571{,}495{,}477{,}370{,}241$; all modes require $u \lesssim 1.8 \times 10^{19}$ from UInt32 primes and byte encoding. On `DIVISION_FREE=1` builds also keep $u < 2^{60} - 2^{32}$ (see `INPUT_BOUNDS.md` constraints 3-5). Larger $u$ shifts work from S1/S2 summation into sieving; smaller $u$ does the opposite.
- `--u-factor <value>`: override the scaling factor in the $u$ formula: $u = \lceil \text{factor} \cdot (n / \ln \ln n)^{2/3} \rceil$. Must be positive. The default factor is $\max(0.30, \min(0.55, 0.55 - 0.025(\log_{10} n - 16)))$: $0.55$ through $10^{16}$, decreasing by $0.025$ per decade to $0.30$ at $10^{26}$. Mutually exclusive with `--u`.
- `--nu-ratio <value>`: S1/S2 split ratio (default: $0.9$). Controls the boundary between the S1 (Mertens sum) and S2 (Möbius sum) ranges via $\nu(x) = \lfloor \text{ratio} \cdot \sqrt{x} \rfloor$. Must be positive. Affects only performance, not correctness.

Examples:

```
$ ./build/mertens 10000000000
M(10000000000) = -33722 in 0.011 seconds

$ ./build/mertens 10000000000000000 --profile
M(10000000000000000) = -3195437 in 2.0 seconds

--------------- Loop 1 16-bit ---------------
...

$ ./build/mertens 10000000000000000000000000 --segment-cap 50000000000
M(10000000000000000000000000) = ... 
```

To use as a library in your own code:

```cpp
#include "MertensHurst.h"

Int64 result = MertensHurst(n);                          // default split: 2.372e8 <= n <= 1e26
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
  MertensHurst.cpp          Algorithm orchestration (loops, final recovery)
  S1.h                      S1 summation functions (64-bit and 128-bit)
  S1Q6.h                    Exact outer-Q6 S1 transform kernels
  S1Q30.h                   Completed outer-Q30 S1 kernels
  S1Q210.h                  Completed outer-Q210 S1 kernels
  S1Q2310PairedSeam.h       Exact factor-11 paired S1 kernel
  S1Q30030Ladder.h          Exact factor-11/factor-13 S1 ladder
  S2.h                      S2 summation functions (64-bit and 128-bit)
  S2Q6.h                    Exact outer-Q6 S2 dispatch
  S2Q6Modes.h               Static outer-Q6 S2 coefficient modes
  S2Q30.h                   Coupled Q30 S2 kernels
  S2Q210.h                  Coupled Q210 S2 kernels and period table
  OuterRecovery.h           Final-value inversion and optional full recovery
  QuotientStepper.h         Exact batched quotient/remainder transport
  QuotientPredictor.h       Division-free quotient estimation
  main.cpp                  Driver program
../sieve/                   Shared segmented Mobius sieve (see sieve/README.md)
../sieve/SegmentedMertensSieve.h  Mertens sieve (prefix sum over Mobius values)
../sieve/SegmentedOddMobiusSieve.h  Native packed odd-only Mobius sieve
../sieve/SegmentedOddMertensSieve.h Odd-prefix sieve for M_2(x)
../sieve/SegmentedCoprime6MobiusSieve.h Native packed coprime-to-6 Mobius sieve
../sieve/SegmentedCoprime6MertensSieve.h Prefix sieve for M_6(x)
../sieve/QuotientCache.h    Granlund-Montgomery quotient cache (compile-time optional)
build/                      Compiled binary (gitignored)
```
