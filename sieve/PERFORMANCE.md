# Sieve Performance & Technical Details

This document catalogs the performance characteristics, memory layout, and architectural constraints of the segmented Mobius and Mertens sieve. It complements the API-focused [README](README.md) with deeper technical detail.

---

## 1. Runtime complexity

Sieving $\mu(k)$ over $[1, N]$ requires visiting each integer once per prime factor below $\sqrt{N}$. The total work across all primes is governed by the prime harmonic series:

$$T(N) = O\!\left(\sum_{p \le \sqrt{N}} \frac{N}{p}\right) = O(N \log \log N)$$

by Mertens' second theorem. The Mertens prefix sum adds a linear pass, so the overall standalone complexity is $O(N \log \log N)$ time and $O(B)$ space, where $B$ is the segment size.

### As a subroutine in MertensHurst

When used inside the $O(n^{2/3})$ MertensHurst algorithm, the sieve range is $[1, u]$ where

$$u = \left\lceil \texttt{fac} \cdot \left(\frac{n}{\ln \ln n}\right)^{\!2/3} \right\rceil, \qquad \texttt{fac} = \text{clamp}\!\left(1.85 - 0.05 \log_{10} n,\ 0.75,\ 1.05\right).$$

The factor $\texttt{fac}$ is $1.05$ for $n \le 10^{16}$ and decreases linearly in $\log_{10} n$ to $0.75$ at $n = 10^{22}$ and above, balancing sieve cost $O(u \log \log u)$ against the $S_1$/$S_2$ summation cost $O(n / \sqrt{u})$. The $\ln \ln n$ term in the denominator reflects that sieving becomes relatively cheaper at larger $n$ because each prime eliminates a $1/p$ fraction of positions, and the sum $\sum 1/p$ grows only as $\log \log n$.

The sieve's contribution to total MertensHurst runtime is therefore $O\!\left((n / \ln \ln n)^{2/3} \cdot \log \log n\right) = O\!\left(n^{2/3} (\log \log n)^{1/3}\right)$.

### Empirical throughput

The following timings were measured on an Apple M3 Ultra with `BUCKET_SIEVE=1` and OpenMP enabled.

Cumulative standalone times through the listed endpoint (Table 5 of the paper; segment sizes as listed there):

| $N$ | $\mu$ sieve | $\mu + M$ sieve |
|-----|-------------|---------------|
| $10^{9}$ | 0.038 s | 0.067 s |
| $10^{11}$ | 2.66 s | 3.17 s |
| $10^{13}$ | 433.7 s | 469.3 s |
| $10^{16}$ | 7.00 d | 7.39 d |

The Mertens sieve costs slightly more than the Mobius sieve alone, since it adds the segmented prefix sum into the compressed representation.

Throughput scales linearly with $N$. The constant factor depends on the segment size (larger segments amortize prime-list setup, with diminishing returns past $\sim 10^7$), whether the bucket scheduler is enabled (faster when large primes are present), whether division-free mode is active (beneficial on x86, not used on ARM where hardware division is fast), and the SIMD backend.

---

## 2. Memory layout

### Mobius sieve buffers

The `SegmentedMobiusSieveCore` maintains three buffers:

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| `mMu` | `Int8[]` | $B$ bytes | Main sieve buffer; holds packed values during sieving, then $\mu \in \{-1, 0, +1\}$ after finalization |
| `mPreMu` | `Int8[]` | $M_1 + P$ bytes | Tiled stencil copy for Phase 1 (read-only after init) |
| `mStencilData` | `Int8[]` | $P = 13{,}860$ bytes | Base stencil pattern (copied from `stencil_data.h`) |

where $B$ is the user-specified segment size and $M_1 = 4P = 55{,}440$.

When the bucket scheduler is enabled, each OpenMP thread maintains its own array of `LP_SIZE = 512` bucket vectors. In the default narrow format (`SIEVE_NARROW_ENTRY=1`) each entry is a 4-byte prime and the hit offset is recomputed per hit; the wide format packs (offset, $p \bmod M_2$, $p / M_2$, log weight) into 8 bytes and is divide-free per hit (see §6). The total bucket memory depends on how many large primes land in each sub-segment, but is typically small relative to the sieve buffer.

### Mertens compressed storage

In `Compressed` mode (the default), $M(x)$ is stored as

$$M(x) = \texttt{coarse}[x \gg 8] + \texttt{residual}[x],$$

where the coarse array stores a full Mertens value once every $H = 256$ positions, and the residual array stores the signed byte offset from the nearest coarse sample. A lookup of $M(x)$ therefore requires one access into the coarse array and one byte access into the residual array.

| Array | Type | Size per segment | Purpose |
|-------|------|-----------------|---------|
| `coarse` | `Int64[]` | $\lceil B / 256 \rceil \times 8$ bytes | $M$ at every 256th position |
| `residual` | `Int8[]` | $B$ bytes | Signed offset from nearest coarse sample |

The purpose of this representation is to trade a slightly more complicated lookup for a substantially smaller working set and less memory traffic. If every sieved value of $M$ were stored as a 32-bit integer, the storage cost would be four bytes per entry. With stride $H = 256$, the coarse array contributes slightly more than one byte per $H$ entries and the residual array contributes one byte per entry, for a total of about one quarter of the 32-bit baseline. Relative to a 64-bit baseline, the reduction is about a factor of eight. A lookup now requires one access into the coarse array and one byte access into the residual array. In the MertensHurst algorithm, `getM` is called billions of times in the $S_1$ inner loop, so the compressed layout's cache footprint matters more than the extra addition.

In `Direct` mode, $M(x)$ is stored as a flat `Int32[]` array ($4B$ bytes). This is appropriate when the consumer reads all values sequentially rather than performing random-access lookups.

For a $10^9$-element segment, the compressed mode uses approximately 1.03 GB (31 MB coarse + 1 GB residual) compared to 4 GB in direct mode.

### In-place sieving

`sieveInPlace()` aliases the residual array onto the $\mu$ buffer, eliminating one $B$-byte allocation. This is used in MertensHurst's Loop 2, where the $\mu$ values are not needed after the prefix sum, saving approximately 12 GB at large $n$.

---

## 3. Stencil pre-sieve

The first primes are handled by a stencil of period $P = 13{,}860 = \text{lcm}(4, 9, 5, 7, 11)$. A stencil of this period, pre-sieved by 2, 3, $2^2$, 5, 7, $3^2$, and 11, is computed once and then copied into each new segment. After the stencil has been copied, the small primes through 353 are applied by hardcoded unrolled loops. Medium primes begin at 359: M1 medium primes through `M1_CONSTANT_PRIME_CAP` use fully unrolled constant-stride tables, the remainder through `M1_PRIME_CAP` use a four-stream walk on M1 chunks, and tiled medium primes use wider direct-sieve tiles. Large primes are handled by the bucket scheduler below.

The five smallest primes account for the densest sieving work. Their combined period is small enough to fit in L1 cache (13,860 bytes) while eliminating the most expensive per-segment iterations. Adding 13 would increase the period to $180{,}180$, which is still feasible but offers diminishing returns since prime 13 only touches $1/13 \approx 7.7\%$ of positions.

---

## 4. Sieve phases and sub-segment hierarchy

The paper's small, medium, and large categories are implemented with the following processing tiers:

| Tier | Primes | Unit size | Method |
|------|--------|-----------|--------|
| Stencil | $p \le 11$ | $P = 13{,}860$ | Copy the pre-sieved stencil |
| Small | $13 \le p \le 353$ | $M_1 = 8P = 110{,}880$ | Hardcoded constant-stride loops |
| M1 medium | $359 \le p \le 32{,}000$ | $M_1$ | Constant-stride tables through 1,021, then four adjacent streams |
| Tiled medium | $32{,}000 < p \le X$ | $M_2$ or a larger direct tile | Four-stream direct iteration |
| Large | $p > X$ | $M_2 = 64P = 887{,}040$ | Circular bucket scheduler |

The M1-stage unit $M_1$ keeps its working set in L1 cache. $M_2$ balances per-sub-segment overhead against cache pressure for medium primes. $M_3 = 90P = 1{,}247{,}400$ is the crossover point where a prime's stride exceeds the sub-segment length, making direct iteration wasteful since each prime hits at most one position per sub-segment.

Primes 13 through 353 (indices 5 through 70 in a typical prime list) are sieved with manually unrolled constant-stride loops. By default, M1 medium primes 359 through 1,021 come from compact constant-prime tables that the compiler fully unrolls; primes 1,031 through `M1_PRIME_CAP` use the general log-prime encoding while walking four adjacent streams together. Override `SIEVE_M1_CONSTANT_PRIME_CAP` at compile time to retune this handoff from 353 through 2,039, for example with `make EXTRA_CXXFLAGS="-DSIEVE_M1_CONSTANT_PRIME_CAP=1523"`.

### Bucket scheduler

Large primes require additional treatment. For a sub-segment being processed, most primes larger than its length do not hit that sub-segment at all. With fixed-size sub-segments, iterating through all such primes for every sub-segment wastes time and can raise the sieving cost from essentially linear to about $O(u^{3/2})$. The bucket scheduler avoids rediscovering these sparse hits by carrying each large prime forward from one contributing sub-segment to the next.

The scheduler uses a circular buffer of `LP_SIZE = 512` buckets indexed by `subSegIndex & 511`. Each large prime is stored in the bucket corresponding to the next sub-segment that contains a multiple of that prime. When the sub-segment is processed, the prime contributes to that one location, and is then pushed forward into the bucket for its next hit. The power-of-two choice keeps the bucket index arithmetic simple, since wraparound can be handled by masking rather than by an integer division.

The direct-sieve cutoff $X$ is $2{,}772{,}000$ when finalization is fused into the prefix sum and $1{,}247{,}400$ when the sieve finalizes $\mu$ separately. When the bucket scheduler is disabled (`BUCKET_SIEVE=0`), all primes above `M1_PRIME_CAP` fall back to tiled direct iteration.

---

## 5. Log-prime encoding (ceil-log2 scheme) and the byte-overflow bound

The Mobius sieve stores one signed byte per integer during the sieving phase. Rather than storing the product of prime factors, it stores a compact approximation to the sum of logarithms of prime factors, together with parity information. As in [Hurst 2016](https://arxiv.org/pdf/1610.08551) and [Kuznetsov 2011](https://arxiv.org/pdf/1108.0135), a per-prime weight is added to each multiple of $p$, multiples of $p^2$ are zeroed, and at the end of the segment the accumulated sum $S$ is compared against the size of the integer to decide whether a remaining prime cofactor is present. Each byte holds `0x80 | S`: bit 7 is the squarefree flag and the low 7 bits are $S$.

The weight is $\lceil \log_2 p \rceil \mid 1$ (uniformly, for every prime including 2, 3, 5, 7, 11 in the stencil), where $\mid$ denotes bitwise OR. The OR-with-1 keeps every weight odd, so bit 0 of $S$ is the factor-count parity for free. Using the **ceiling** rather than the floor makes finalization an exact comparison against $\lfloor \log_2 N \rfloor$ (see §9) and, more importantly, makes the encoding collision-free:

- **Fully factored** squarefree $N$: $S = \sum_{p \mid N} (\lceil \log_2 p \rceil \mid 1) \ge \sum \lceil \log_2 p \rceil \ge \lceil \log_2 N \rceil > \lfloor \log_2 N \rfloor$ (for $N$ not a power of two), so it is always recognized as fully factored.
- **Leftover prime** $N = m \cdot q$ with $q > \sqrt N$ prime and unsieved: every prime satisfies $(\lceil \log_2 p \rceil \mid 1) < 2\log_2 p$, so $S(m) = \sum_{p \mid m}(\lceil\log_2 p\rceil\mid 1) < 2\log_2 m < \log_2 m + \log_2 q = \log_2 N$ (using $q > m$). Hence $S(m) \le \lfloor \log_2 N \rfloor$ and the "fully factored" test can never falsely fire.

So there is **no collision bound** in the sense of the earlier floor/`numbits` scheme — the only limit is the 7-bit field overflowing. Since $S < 2\log_2 N$, the sum stays below 128 for all $N < 2^{64}$, i.e. the encoding is exact up to $\sim 1.8 \times 10^{19}$ (the same order as the UInt32 prime cap, §7). The earlier mixed scheme (floor weights on the stencil primes, `numbits` elsewhere) needed a small slack margin in finalization and had its first misclassification near $1.157 \times 10^{18}$; the uniform ceil scheme removes both the margin and that bound. An exhaustive worst-case search (each $N$ tested with sieve limit $\sqrt N$) confirms zero misclassifications below $10^9$, consistent with the proof above.

---

## 6. Bucket scheduler capacity and setup cost

### Capacity

The bucket scheduler uses `LP_SIZE = 512` buckets with sub-segment length $M_2 = 887{,}040$. The scheduler must satisfy $\sqrt{u} < \texttt{LP\_SIZE} \times M_2$, since $\sqrt{u}$ is the largest prime that can appear in the segmented sieve. In the record implementation these constants were `LP_SIZE` $= 512$ and $M_2 = 887{,}040$; changing the sub-segment size or scheduler capacity changes the supported range accordingly.

$$\sqrt{N} < 512 \times 887{,}040 = 454{,}164{,}480 \implies N < 2.06 \times 10^{17}$$

For larger ranges, build with `-DSIEVE_LP_SIZE=1024`, which raises the limit to $\sim 8.25 \times 10^{17}$ (still below the encoding overflow cap of §5). Alternatively, building with `BUCKET_SIEVE=0` disables the scheduler entirely; all primes use direct iteration, which removes this constraint but is slower.

When the sieve is used inside MertensHurst with the default $u(n)$ formula (`fac` $= 0.75$ at scale), this supports inputs to roughly $n = 5.9 \times 10^{26}$ (the figure quoted in the paper), comfortably above the tested $10^{25}$.

### Setup cost

The bucket scheduler removes the main large-prime slowdown by carrying each large prime from one hit to the next within a sieve segment. It is nevertheless rebuilt for each segment. During construction, each thread scans the large-prime list up to roughly $\sqrt{u}$ to find the first hit in its thread-local subinterval. Thus there is a per-segment setup term of size roughly

$$\pi(\sqrt{u}) \sim \frac{\sqrt{u}}{\log \sqrt{u}}.$$

For the present computations this is not a bottleneck: the final sieve segments are large enough that this setup cost is amortized over the actual sieving work. It can become relevant if the segment size is forced much smaller, or if the supported endpoint is pushed far enough that rebuilding the scheduler starts to dominate the large-prime hits.

A possible future optimization would be to carry some bucket state across segment boundaries. That is not done in the current implementation because it is not just a local tweak. The current data structure partitions each segment into thread-local subintervals and builds the scheduler state for those subintervals independently, so a later segment cannot naturally know the previous bucket state without a more fundamental retooling of the scheduler ownership.

### Entry layout: narrow vs wide

The bucket entry layout is build-selectable (`NARROW_ENTRY`, default 1). The
narrow entry is just the prime (4 B): the hit offset is recomputed each
sub-segment by one divide and the log weight by CLZ. The wide entry (8 B)
packs offset, $p \bmod M_2$, $p / M_2$, and the log weight, so forwarding is a
branchless Bresenham step with no per-hit divide — but the doubled entry
stream is what saturates memory bandwidth at scale. Measured head-to-head on
the 32-thread M3 Ultra: narrow wins by up to ~22% at $10^{16}$ with the gap
growing, the crossover is near $3 \times 10^{14}$, and narrow is at most ~2%
slower on the cheap decades below that. Sub-buckets band on the stored
offset, so they exist only in wide builds. The trade-off is machine-specific:
on x86 the per-hit divide is costlier, so a wide build with `SUB_BUCKETS=0`
may win — re-measure before trusting the ARM-tuned default.

---

## 7. UInt32 prime storage

Primes are stored as `UInt32`, which caps them at $\sim 4.29 \times 10^9$. Since the largest prime sieved is $\sqrt{N}$, this limits the sieve endpoint to $N < (2^{32} - 1)^2 \approx 1.8 \times 10^{19}$. This is well above the bucket scheduler bound ($2.06 \times 10^{17}$), and with the ceil encoding of §5 there is no separate collision cap below $2^{64}$, so it is never the binding constraint in practice.

---

## 8. Int8 residual overflow (compressed Mertens only)

The compressed Mertens representation stores residuals as `Int8` over intervals of length $H = 2^{\texttt{STRIDE\_LOG}}$ (default $H = 256$). If the prefix sum of $\mu$ within any interval exceeds $[-128, 127]$, this overflows silently.

A heuristic from Ng, ["The distribution of the summatory function of the Mobius function"](https://www.cs.uleth.ca/~nathanng/RESEARCH/mobiusshort.pdf), treats $\mu$ as independent random variables with $\Pr(\mu(k) \ne 0) = 6/\pi^2$ and estimates the expected maximum short-interval fluctuation over $[1, X]$ as:

$$\max |M(x+H) - M(x)| \sim \sqrt{\frac{12H}{\pi^2}\cdot\log\!\left(\frac{X}{H}\right)}$$

Setting this equal to 128 gives expected overflow at:

| `STRIDE_LOG` | $H$  | Max sieve range       |
|--------------|------|-----------------------|
| 7            | 128  | $\infty$ (see below)  |
| 8            | 256  | $1.9 \times 10^{25}$  |
| 9            | 512  | $1.4 \times 10^{14}$  |
| 10           | 1024 | $5.3 \times 10^{8}$   |

At `STRIDE_LOG = 7` ($H = 128$), overflow is mathematically impossible: every interval of 128 consecutive integers contains at least $\lfloor 128/4 \rfloor = 32$ multiples of 4, which are non-squarefree ($\mu = 0$). At most 96 values can be non-zero, so the prefix sum is bounded by $|\text{residual}| \le 96 < 128$.

For smaller inputs, `STRIDE_LOG` can be raised to 9 or 10 (`-DSIEVE_STRIDE_LOG=9`) to shrink the coarse array and improve cache friendliness. For large inputs, `STRIDE_LOG = 8` (the default) is safe well beyond any practical sieve range.

---

## 9. SIMD finalization

After the three sieve phases, each position holds a packed `Int8` (`0x80 | S` if squarefree, else 0). Finalization decodes each byte to $\mu \in \{-1,0,1\}$:

- not squarefree (bit 7 clear / byte $\ge 0$) $\Rightarrow \mu = 0$;
- $N \le 2$ (only the first segment) is hardcoded: $\mu(1)=1,\ \mu(2)=-1$ (the $S$-vs-floor test would otherwise misread these);
- otherwise, with $S = \texttt{byte} \mathbin{\&} \texttt{0x7F}$ and parity $\texttt{par} = \texttt{byte} \mathbin{\&} 1$: $\mu = 1 - 2\,\texttt{par}$ if $S > \lfloor \log_2 N \rfloor$ (fully factored), else $\mu = 2\,\texttt{par} - 1$ (an unsieved prime cofactor remains).

This is a single flag mask, a cheap guard, and one comparison — the ceil-log2 weighting (§5) is what makes the comparison exact, with no slack margin. Backend-specific SIMD kernels are provided for NEON, SVE2, SSE2, AVX2, and AVX-512, with a scalar fallback; detection is at compile time via preprocessor macros, with no runtime dispatch. Both operands of the $S > \lfloor \log_2 N \rfloor$ test are in $[0,127]$, so it is a plain signed-byte compare (no unsigned-bias trick).

The threshold $\lfloor \log_2 N \rfloor$ is constant between consecutive powers of two, so a segment is finalized in constant-threshold runs and the kernel broadcasts one threshold per run. Powers of two are spaced far wider than the sub-segment length $M_2$ at scale, so in the common case a sub-segment is a single run; only sub-segments that straddle a power of two are split (into two runs), at negligible cost.

---

## 10. Division-free mode (x86 only)

Integer division is relatively cheap on the tested Apple Silicon machines, but on many x86 machines it is more important to avoid divisions or reuse auxiliary data. The `DIVISION_FREE` build flag (default on for x86, off for ARM) enables the `QuotientCache`, which precomputes multiplier data for a range of denominators, replacing a quotient by a 128-bit multiply and shifts using the invariant-division method of Granlund and Montgomery.

For a divisor $d$, the cache precomputes a multiplier $m$ and shift $s$ such that:

$$\lfloor n / d \rfloor = \lfloor (n + \lfloor m \cdot n \cdot 2^{-60} \rfloor) \cdot 2^{-s} \rfloor$$

Two cache variants are used: `QuotientCache` for a contiguous divisor range $[1, B]$, used in $S_1$/$S_2$ summation, and `SieveQuotientCache`, indexed by prime, used in the Mobius sieve for computing $\lceil \text{lo} / p \rceil$. The choice between direct division, the Quotient Predictor, and the Quotient Cache changes performance but not correctness.

---

## Summary of range constraints

The binding constraint depends on configuration:

| Constraint | Limit | Binding when |
|------------|-------|-------------|
| Log-encoding byte overflow | $\sim 1.8 \times 10^{19}$ ($N < 2^{64}$) | Fundamental encoding limit (ceil scheme, §5) |
| Bucket scheduler (`LP_SIZE=512`) | $2.06 \times 10^{17}$ | `BUCKET_SIEVE=1` (default) |
| UInt32 primes | $1.8 \times 10^{19}$ | `BUCKET_SIEVE=0` (co-binding with the encoding) |
| Int8 residual (`STRIDE_LOG=8`) | $1.9 \times 10^{25}$ | Compressed Mertens only |

With default settings, the bucket scheduler is the most restrictive at $2.06 \times 10^{17}$. Building with `BUCKET_SIEVE=0` raises the effective limit to $\sim 1.8 \times 10^{19}$ (the encoding and UInt32 prime caps, which coincide in order of magnitude). The uniform ceil-log2 encoding (§5) is collision-free, so unlike the earlier mixed floor/`numbits` scheme there is no $1.157 \times 10^{18}$ cap.
