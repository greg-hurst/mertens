# Bounds on $n$ for `MertensHurst(n)`

This document catalogs the constraints that limit the supported input `UInt128 n`. It aims to be exhaustive, but there may be additional constraints not yet identified. Each entry notes which source file(s) the constraint originates from, and what would need to change to relax it.

The imposed bounds are $10^8 \le n \le 10^{25}$. The lower bound is discussed in constraint 0; the upper bounds (constraints 1-10) are ordered from most restrictive to least restrictive.

---

## 0. Lower bound: $10^8 \leq n$

**Source:** `MertensHurst.cpp`, `SegmentedMobiusSieve.h`

The algorithm produces incorrect results for small $n$. The primary cause is structural: the minimum viable sieve segment size is `B == BF == STENCIL_PERIOD == 13860`, and the main sieve loop `while (L2 < nu_max)` only executes when the combined S2 and S1 ranges exceed one segment. Since `nu_max == 1.5*sqrt(n)` by default, the loop requires `B < nu_max`, which implies $n > (13860 / 1.5)^2 \approx 8.5 \times 10^7$.

More generally, if the split is changed to `nu_max = c*sqrt(n)`, then this structural threshold becomes roughly $(13860/c)^2$. Decreasing the split constant therefore raises the lower input bound, and the enforced $10^8$ lower limit should be revisited whenever this tuning is changed.

Below this threshold, neither S2 nor S1 updates are performed: `partial_values` remains at its initial value of $1$ for every entry, and the final recovery produces garbage.

Additional reasons the algorithm is not designed for small $n$:

- **Segment-size and chunk-size formulas are tuned for large inputs.** `getSegmentSize`, the S2 chunking parameter `CHUNK_LEN`, and the Loop 2 segment size all use heuristics (e.g., $\sqrt{n}$, $n^{1/3}$, $\sqrt{2u}$) that become degenerate when their values fall below `STENCIL_PERIOD`. For `n < BF^2` $\approx 1.92 \times 10^8$, `getSegmentSize` returns the bare minimum of `BF`, leaving no room for the sieve range to span multiple segments.

- **The S2 mode-splitting assumes non-trivial sub-ranges.** The multi-mode S2 dispatch divides the summation range $[1, \nu]$ at boundaries like $\nu/6$, $\nu/3$, $\nu/2$. For small $n$, the per-argument `nus[i] == get_nu(n/i)` shrinks rapidly with `i`, and some mode sub-ranges may become empty. While empty ranges are handled gracefully, the algorithm has not been validated in this regime.

- **Asymptotic tuning constants.** The `fac` formula, the `M16BITMAX` transition point, and the Loop 2 segment-size formula all embed constants calibrated for $10^8 \le n$. These constants are not wrong for smaller $n$, but they have not been verified to produce correct segment layouts or sieve coverage below $10^8$.

---

## 1. Tested up to $10^{25}$

This is the largest input for which correctness has been verified. Higher values may trigger unknown edge cases.

---

## 2. Floating-point precision

**Source:** `MertensHurst.cpp`

$n$ is cast to `double` (53-bit mantissa) for `sqrt`, `cbrt`, and `log`. Precision loss is corrected by Newton iteration in `isqrt_u128`, but if auxiliary calculations (segment sizes, iteration bounds, etc.) are computed inconsistently across the codebase, results may be wrong. Verified correct up to $10^{25}$.

---

## 3. `LP_SIZE == 512`

**Source:** `SegmentedMobiusSieve.h`

The bucket scheduler for large primes uses a circular buffer of `LP_SIZE` buckets. Its exact largest schedulable prime is `(LP_SIZE - 1) * M2`, so the runtime conservatively enforces $u \le (511 \times 887{,}040)^2 = 205{,}460{,}437{,}612{,}953{,}600 \approx 2.05 \times 10^{17}$. (This is the configuration used for the record runs and described in Section 7 of the paper: `LP_SIZE = 512`, `M2 = 887,040`.)

With `fac == 0.75`: **max $n \approx 5.9 \times 10^{26}$** (the figure quoted in the paper). Build with `make EXTRA_CXXFLAGS=-DSIEVE_LP_SIZE=1024` to go higher (the default narrow entries store the prime as `UInt32`, and the wide packed payload supports `LP_SIZE` up to exactly 1024, so capacity is not payload-limited); the runtime cap on $u$ scales with the flag automatically. Alternatively, build with `make BUCKET_SIEVE=0` to disable the bucket scheduler entirely (all primes are sieved as medium primes via direct iteration), which removes this constraint.

**Note:** This bound is enforced at runtime in `MertensHurst.cpp` for every source of $u$ (default formula, `--u`, or `--u-factor`), via `SegmentedMobiusSieveCore::schedulerReach()`. Builds with `USE_BUCKET_SIEVE=0` are subject only to constraints 4, 5, and 8.

---

## 4. Quotient cache domain (`DIVISION_FREE=1` builds only)

**Source:** `QuotientCache.h`, `SegmentedMobiusSieve.cpp`

When built with `DIVISION_FREE=1` (the default on x86; never used on ARM), the Mobius sieve computes $\lceil x/p \rceil$ via the Granlund-Montgomery multiply-shift with `SHIFT == 60`, which is exact only for arguments below $2^{60}$. `ceilDiv` forms $\text{val} = x + p - 1$ with $x \le u$ and $p < 2^{32}$, so the sieve range must satisfy

$$u < 2^{60} - 2^{32} = 1\,152\,921\,500\,311\,879\,680 \approx 1.1529 \times 10^{18}.$$

Inverting the default $u(n)$ formula (`fac == 0.75` at this scale), the smallest input whose sieve range reaches this bound is **max $n \approx 7.9 \times 10^{27}$** — far tighter than the encoding/prime caps of constraints 5 and 8, so on `DIVISION_FREE=1` builds with the bucket scheduler disabled this constraint binds first.

`SegmentedMobiusSieveCore::sieve()` asserts the bound at entry (active only in non-`NDEBUG` builds); it is not otherwise enforced at runtime, so on a `DIVISION_FREE=1` release build a manual $u$ above $2^{60} - 2^{32}$ computes incorrect quotients. In the default configuration this window is unreachable, since the bucket-scheduler cap of constraint 3 ($2.05 \times 10^{17}$) is enforced first. Build with `DIVISION_FREE=0` to remove this constraint entirely.

---

## 5. Byte-encoding cap: $u < 2^{64}$

**Source:** `SegmentedMobiusSieve.cpp`

The sieve stores the accumulated ceil-log2 weights $\lceil \log_2 p \rceil \mid 1$ in a 7-bit byte field. This encoding is collision-free (finalization is an exact comparison — see `sieve/PERFORMANCE.md` §5 and Section 5 of the paper), and the accumulated sum stays below 128 for every $u < 2^{64}$. The requirement $u < 2^{64}$ is therefore the only intrinsic encoding limit.

In practice it coincides with the `UInt32` prime cap of constraint 8, since $\sqrt{u} < 2^{32}$ is the same bound: both give $u \lesssim 1.8 \times 10^{19}$ and **max $n \approx 5 \times 10^{29}$** under the default $u(n)$ formula. This combined cap is enforced at runtime in `MertensHurst.cpp`, together with the bucket-scheduler cap of constraint 3 on `USE_BUCKET_SIEVE=1` builds.

---

## 6. `UInt64` quotients in 128-bit `S2`/`S1` paths

**Source:** `S2.h`, `S1.h`, `QuotientPredictor.h`

The 128-bit $S_2$ and $S_1$ code paths compute $\lfloor n/x \rfloor$ as `UInt128` but then store the result in `UInt64` variables: the quotient predictor values `q_est`/`q_cur`/`q_prev`, `S2_term` arguments, and `getM` positions. This means every quotient encountered must fit in a signed 64-bit integer.

The largest quotients arise in two sub-ranges of the 128-bit loop. In the "fast" range ($x > \sqrt[3]{2n}$), the quotient is $\lfloor n/x \rfloor < n^{2/3} / \sqrt[3]{2}$. In the "small" range, the quotient is bounded by $u \approx n^{2/3}$ because it must index into the sieve segment. In both cases the quotient is $O(n^{2/3})$, so the requirement is $n^{2/3} < 2^{63}$, i.e. **$n < 2.8 \times 10^{28}$**.

---

## 7. `UInt32` hash indices

**Source:** `MertensHurst.cpp`

The `hash` and `hash2` lookups use `UInt32` indices up to $\nu \sim n^{1/3}$. This overflows when $\nu > 2^{32} \approx 4.29 \times 10^9$, which implies **$n < 7.9 \times 10^{28}$**.

---

## 8. `UInt32` primes

**Source:** `SegmentedMobiusSieve.h`

Primes are stored as `UInt32`, which caps them at $\sim 4.29 \times 10^9$. Since the largest prime is $\sqrt{u}$, we must have $u < 1.8 \times 10^{19}$ and therefore **$n < 7.9 \times 10^{28}$**.

---

## 9. `Int8` residual array `R` with `STRIDE_LOG == 8`

**Source:** `SegmentedMertensSieve.h`

The compressed Mertens representation stores residuals as `Int8` over intervals of `STRIDE == 1 << STRIDE_LOG`. If the prefix sum of $\mu$ within any interval exceeds $[-128, 127]$, this overflows silently.

A heuristic from Ng in, ["The distribution of the summatory function of the Mobius function"](https://www.cs.uleth.ca/~nathanng/RESEARCH/mobiusshort.pdf) treats $\mu$ as independent random variables with $\Pr(\mu(k) \ne 0) = 6/\pi^2$. The expected maximum short sum over $[1, X]$ in intervals of length $H$ is:

$$\max |M(x+H) - M(x)| \sim \sqrt{\frac{12H}{\pi^2}\cdot\log\left(\frac{X}{H}\right)}.$$

Setting this equal to 128 (`Int8` limit) gives expected overflow at:

| `STRIDE_LOG` | $H$  | max $u$       | max $n$         |
|--------------|------|---------------|--------------------------|
| 7            | 128  | $\infty$      | $\infty$                 |
| 8            | 256  | $1.9 \times 10^{25}$ | $6 \times 10^{38}$ |
| 9            | 512  | $1.4 \times 10^{14}$ | $1.1 \times 10^{22}$ |
| 10           | 1024 | $5.3 \times 10^{8}$  | $4.2 \times 10^{13}$ |

At `STRIDE_LOG = 7` ($H = 128$), overflow is mathematically impossible: every interval of 128 consecutive integers contains at least $\lfloor 128/4 \rfloor = 32$ multiples of 4, which are non-squarefree and have $\mu = 0$. At most 96 values can be non-zero, so the prefix sum is bounded by $|\text{residual}| \le 96 < 128$.

Verified: no overflow observed up to $u = 13\\,694\\,622\\,981\\,236\\,974$ (the value of $u$ used at $n = 10^{25}$). For smaller inputs, `STRIDE_LOG` can be increased to `9` (for roughly $n < 1.1 \times 10^{22}$) or `10` (for roughly $n < 4.2 \times 10^{13}$), or decreased to `7` for a hard guarantee of no overflow at the cost of doubling the coarse `M` array. This shrinks (or grows) the coarse `M` array and affects cache friendliness.

---

## 10. Intermediate accumulation overflow in 64-bit paths

**Source:** `S2.h`, `S2Q6.h`, `S2Q6Modes.h`, `S1.h`, `S1Q6.h`, `MertensHurst.cpp`

The final $S_2$ and $S_1$ values are each $O(x^{2/3})$ per argument $x$ and cancel to give $M(x) = O(\sqrt{x})$, but the **running** partial sums during accumulation can be larger than the final values due to incomplete cancellation. We must verify that no accumulator overflows at any intermediate step. Throughout this section, $x$ denotes the per-entry argument `partial_args[i]` $= \lfloor n/i \rfloor$; the 64-bit path handles $x < 10^{18}$.

### $S_2$ running partial sum

The production path uses the exact outer-$Q=6$ transform in `S2Q6.h`; the original all-$Q=2$ formulation remains available through `make q2`. Both have $O(x)$ worst-case absolute accumulation, but their coefficients and cutoff regions differ.

The earlier $\approx 3.95x$ estimate based on `S2_term<9>` through `S2_term<0>` applies specifically to the all-$Q=2$ path and should not be quoted as a proof for the transformed production path. The $Q=6$ implementation uses static signed coefficient modes with denominators dividing 36, so its accumulator bound must be evaluated from `S2Q6Modes.h`. Correctness has been stress-tested against the Q2 oracle through the supported range, but a separate tight analytic UInt64-path bound for the Q6 modes remains to be written.

### $S_1$ running partial sum

The production path likewise evaluates an exact outer-$Q=6$ combination of the $S_1$ terms at $y$, $\lfloor y/2 \rfloor$, $\lfloor y/3 \rfloor$, and $\lfloor y/6 \rfloor$; `make q2` retains the original single-stream form.

For the underlying Q2 stream, each $|M(q)| \le 0.571 \sqrt{q}$ empirically for $q \le 10^{16}$ ([Table 6.1, Hurst 2016](https://arxiv.org/pdf/1610.08551)), giving the familiar estimate $S_1(x) \lesssim 1.14x$. The Q6 combination has only a fixed number of these exact streams and therefore remains $O(x)$, but its tight constant should be derived from the four active residue masks in `S1Q6.h` rather than inherited unchanged from the Q2 estimate.

Note: Table 6.1 of Hurst 2016 verifies $|M(q)/\sqrt{q}| \le 0.571$ only up to $q = 10^{16}$. Extending beyond $n = 10^{25}$ pushes $u$ past $10^{16}$, so the table no longer directly covers all queried $M(q)$ values. However, the bounds above are already loose by orders of magnitude (they assume zero cancellation from $\mu$ signs), so even a modest increase in the empirical ratio should not threaten `Int64` overflow.

### `partial_values[i]` running total

Each `partial_values[i]` accumulates the transformed $S_2$ and $S_1$ contributions plus its initialization term across all sieve segments. The former $\approx 5.1x$ estimate combines Q2-only constants and is therefore not a proof for the current Q6 production path. In practice the intermediate values remain far below `Int64` limits in the supported-range stress tests; a rigorous production bound should combine the Q6 constants described above.

### Final recovery

Production computes only $M(n)$ by a signed Möbius sum of the stored square-free partial values. Its accumulator therefore has the same scale as the partial-value state discussed above and is held in `Int128` whenever the leading partial values require it. The optional `FULL_RECOVERY=1` build instead performs decreasing-index back substitution of every square-free value as a correctness oracle; its recovery sum is negligible relative to the earlier S1/S2 accumulations.
