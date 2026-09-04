# Bounds on $n$ for `MertensHurst(n)`

This document catalogs the constraints that limit the supported input `UInt128 n`. It aims to be exhaustive, but there may be additional constraints not yet identified. Each entry notes which source file(s) the constraint originates from, and what would need to change to relax it.

The imposed absolute bounds are $10^8 \le n \le 10^{26}$. Correctness has been verified by completed computations through $10^{25}$; $10^{26}$ is accepted after the runtime and structural checks below but remains unverified until that computation is completed independently.

---

## 0. Lower bound and split guard

**Source:** `MertensHurst.cpp`, `SegmentedMobiusSieve.h`

The minimum viable sieve segment size is `B == BF == STENCIL_PERIOD == 13860`, and the main sieve loop requires `B < nu_max`. The implementation reduces its initial segment to the largest stencil multiple below `nu_max` and rejects the configuration if `nu_max <= BF`; it no longer continues with an empty main loop.

Since `nu_max == floor(0.9*sqrt(n))` by default, this requires approximately $n \ge 2.372 \times 10^8$. Inputs between the absolute $10^8$ floor and this default threshold require a larger explicit `nuRatio`.

More generally, if the split is changed to `nu_max = c*sqrt(n)`, then this structural threshold becomes roughly $(13860/c)^2$. Decreasing the split constant therefore raises the parameter-dependent lower input bound.

Additional reasons the algorithm is not designed for small $n$:

- **Segment-size and chunk-size formulas are tuned for large inputs.** `getSegmentSize`, the S2 chunking parameter `CHUNK_LEN`, and the Loop 2 segment size all use heuristics (e.g., $\sqrt{n}$, $n^{1/3}$, $\sqrt{2u}$). The initial size is clamped against the actual split before allocation.

- **The S2 mode-splitting assumes non-trivial sub-ranges.** The multi-mode S2 dispatch divides the summation range $[1, \nu]$ at boundaries like $\nu/6$, $\nu/3$, $\nu/2$. For small $n$, the per-argument `nus[i] == get_nu(n/i)` shrinks rapidly with `i`, and some mode sub-ranges may become empty. While empty ranges are handled gracefully, the algorithm has not been validated in this regime.

- **Asymptotic tuning constants.** The default `fac`, the `M16BITMAX` transition point, and the Loop 2 segment-size formula all embed constants calibrated for $10^8 \le n$. These constants are not wrong for smaller $n$, but they have not been verified to produce correct segment layouts or sieve coverage below $10^8$.

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

With the tuned default factor (which reaches its $0.30$ floor at $10^{26}$): **max $n \approx 2.3 \times 10^{27}$**. Build with `make EXTRA_CXXFLAGS=-DSIEVE_LP_SIZE=1024` to go higher (the default narrow entries store the prime as `UInt32`, and the wide packed payload supports `LP_SIZE` up to exactly 1024, so capacity is not payload-limited); the runtime cap on $u$ scales with the flag automatically. Alternatively, build with `make BUCKET_SIEVE=0` to disable the bucket scheduler entirely (all primes are sieved as medium primes via direct iteration), which removes this constraint.

**Note:** This bound is enforced at runtime in `MertensHurst.cpp` for every source of $u$ (default formula, `--u`, or `--u-factor`), via `SegmentedMobiusSieveCore::schedulerReach()`. Builds with `USE_BUCKET_SIEVE=0` are subject only to constraints 4, 5, and 8.

At $n=10^{26}$ with the default factor, $\sqrt u \approx 158.9$ million.
This is below both the full 512-bucket reach (453,277,440) and the
226,638,720 half-reach used by the optional two-$p$ mod-4 skip. Therefore
neither `LP_SIZE=1024` nor a mod-4-path change is needed at this input.

---

## 4. Quotient cache domain (`DIVISION_FREE=1` builds only)

**Source:** `QuotientCache.h`, `SegmentedMobiusSieve.cpp`

When built with `DIVISION_FREE=1` (the default on x86; never used on ARM), the Mobius sieve computes $\lceil x/p \rceil$ via the Granlund-Montgomery multiply-shift with `SHIFT == 60`, which is exact only for arguments below $2^{60}$. `ceilDiv` forms $\text{val} = x + p - 1$ with $x \le u$ and $p < 2^{32}$, so the sieve range must satisfy

$$u < 2^{60} - 2^{32} = 1\,152\,921\,500\,311\,879\,680 \approx 1.1529 \times 10^{18}.$$

Inverting the tuned default $u(n)$ formula (`fac == 0.30` at this scale), the smallest input whose sieve range reaches this bound is **max $n \approx 3.2 \times 10^{28}$**. This remains below the encoding/prime caps of constraints 5 and 8, although other algorithmic limits may bind first.

`SegmentedMobiusSieveCore::sieve()` asserts the bound at entry, and `MertensHurst` folds the exact inclusive maximum $(2^{60}-2^{32})-1$ into its build-aware runtime cap. In the default configuration the bucket-scheduler cap of constraint 3 ($2.05 \times 10^{17}$) binds first. Build with `DIVISION_FREE=0` to remove this constraint entirely.

---

## 5. Byte-encoding cap: $u < 2^{64}$

**Source:** `SegmentedMobiusSieve.cpp`

The sieve stores the accumulated ceil-log2 weights $\lceil \log_2 p \rceil \mid 1$ in a 7-bit byte field. This encoding is collision-free (finalization is an exact comparison — see `sieve/PERFORMANCE.md` §5 and Section 5 of the paper), and the accumulated sum stays below 128 for every $u < 2^{64}$. The requirement $u < 2^{64}$ is therefore the only intrinsic encoding limit.

In practice it coincides with the `UInt32` prime cap of constraint 8, since $\sqrt{u} < 2^{32}$ is the same bound: both give $u \lesssim 1.8 \times 10^{19}$ and **max $n \approx 2.0 \times 10^{30}$** under the tuned default $u(n)$ formula. This combined cap is enforced at runtime in `MertensHurst.cpp`, together with the bucket-scheduler cap of constraint 3 on `USE_BUCKET_SIEVE=1` builds.

---

## 6. `UInt64` quotients in 128-bit `S2`/`S1` paths

**Source:** `S2.h`, `S1.h`, `QuotientPredictor.h`

The 128-bit $S_2$ and $S_1$ code paths compute $\lfloor n/x \rfloor$ as `UInt128` but then store the result in `UInt64` variables: the quotient predictor values `q_est`/`q_cur`/`q_prev`, `S2_term` arguments, and `getM` positions. This means every quotient encountered must fit in a signed 64-bit integer.

The largest quotients arise in two sub-ranges of the 128-bit loop. In the "fast" range ($x > \sqrt[3]{2n}$), the quotient is $\lfloor n/x \rfloor < n^{2/3} / \sqrt[3]{2}$. In the "small" range, the quotient is bounded by $u \approx n^{2/3}$ because it must index into the sieve segment. In both cases the quotient is $O(n^{2/3})$, so the requirement is $n^{2/3} < 2^{63}$, i.e. **$n < 2.8 \times 10^{28}$**.

---

## 7. `UInt32` outer and compact indices

**Source:** `MertensHurst.cpp`

The initial outer extent $\lfloor n/u\rfloor$, the compact square-free count, and the `hash`/`hash2` indices use `UInt32`. The runtime requires $n/u < 2^{32}-1$ before narrowing. Under the tuned default factor this becomes the first structural ceiling at approximately **$n = 1.275 \times 10^{26}$**, above the imposed $10^{26}$ cap.

At $n=10^{26}$, the current floating-point formula gives $u \approx 2.52632 \times 10^{16}$ (the last digits are libm-dependent), $\lfloor n/u\rfloor=3\,958\,326\,568$, and exactly $2\,406\,374\,010$ square-free outer entries. All fit in `UInt32`; the square-free count does not fit in `Int32`, which is why the outer counters are unsigned.

---

## 8. `UInt32` primes

**Source:** `SegmentedMobiusSieve.h`

Primes are stored as `UInt32`, which caps them at $\sim 4.29 \times 10^9$. Since the largest prime is $\sqrt{u}$, we must have $u < 1.8 \times 10^{19}$ and therefore **$n \lesssim 2.0 \times 10^{30}$** under the tuned default formula.

---

## 9. `Int8` residual array `R` with `STRIDE_LOG == 8`

**Source:** `SegmentedMertensSieve.h`

The compressed Mertens representation stores residuals as `Int8` over intervals of `STRIDE == 1 << STRIDE_LOG`. If the prefix sum of $\mu$ within any interval exceeds $[-128, 127]$, this overflows silently.

A heuristic from Ng in, ["The distribution of the summatory function of the Mobius function"](https://www.cs.uleth.ca/~nathanng/RESEARCH/mobiusshort.pdf) treats $\mu$ as independent random variables with $\Pr(\mu(k) \ne 0) = 6/\pi^2$. The expected maximum short sum over $[1, X]$ in intervals of length $H$ is:

$$\max |M(x+H) - M(x)| \sim \sqrt{\frac{12H}{\pi^2}\cdot\log\left(\frac{X}{H}\right)}.$$

Setting this equal to 128 (`Int8` limit) gives expected overflow at:

| `STRIDE_LOG` | $H$  | heuristic max $u$ |
|--------------|------|--------------------|
| 7            | 128  | $\infty$ (proved)  |
| 8            | 256  | $1.9 \times 10^{25}$ |
| 9            | 512  | $1.4 \times 10^{14}$ |
| 10           | 1024 | $5.3 \times 10^{8}$  |

At `STRIDE_LOG = 7` ($H = 128$), overflow is mathematically impossible: every interval of 128 consecutive integers contains at least $\lfloor 128/4 \rfloor = 32$ multiples of 4, which are non-squarefree and have $\mu = 0$. At most 96 values can be non-zero, so the prefix sum is bounded by $|\text{residual}| \le 96 < 128$.

Verified: no overflow observed up to $u = 13\,694\,622\,981\,236\,974$ in the completed $10^{25}$ computation. The default stride 8 remains heuristic rather than a proof at $10^{26}$. `make q210-coupled-record` selects `STRIDE_LOG=7`, giving the hard bound above; the ordinary Q210 target retains stride 8 for maximum speed. Strides 9 and 10 are appropriate only for smaller ranges covered by the table.

---

## Practical memory at $10^{26}$

With the tuned default $u$ and a segment cap of $4 \times 10^{11}$, the
largest Loop 2 segment alone needs about 414 GB in the ordinary stride-8
build: 400 GB for the in-place sieve, plus the coarse Mertens array and its
prefix helpers. Retained outer state brings the estimated process peak to
roughly 452--454 GB before allocator and runtime overhead. The rigorous
stride-7 record build doubles the coarse and prefix-helper counts, raising
the estimate to roughly 466--468 GB. Choose the segment cap from available
RAM; lowering it changes performance, not the answer.

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

The coupled-Q210 profile's selective factor-11/factor-13 ladder combines at
most four completed-Q210 components in one narrow kernel result. If $K$ is
the largest narrow common-$\kappa$, a conservative absolute bound is

$$
4 \cdot 48 \left\lceil\frac{K}{210}\right\rceil
    (2^{31}+128).
$$

The implementation evaluates this bound in `UInt128` before allocating the
ladder maps and falls back to native Q210 $S_1$ if it exceeds
`INT64_MAX`. At `nuRatio=0.9`, $K \le 1\,111\,111\,111$ throughout
the narrow $y \le 10^{18}$ path, so the bound is about
$2.182 \times 10^{18}$, or 23.7% of `INT64_MAX`. The combined narrow
destination update is independently formed in `Int128` and range-checked
before storage.

### `partial_values[i]` running total

Each `partial_values[i]` accumulates the transformed $S_2$ and $S_1$ contributions plus its initialization term across all sieve segments. The former $\approx 5.1x$ estimate combines Q2-only constants and is therefore not a proof for the current Q6 production path. In practice the intermediate values remain far below `Int64` limits in the supported-range stress tests; a rigorous production bound should combine the Q6 constants described above.

### Final recovery

Production computes only $M(n)$ by a signed Möbius sum of the stored square-free partial values. Its accumulator therefore has the same scale as the partial-value state discussed above and is held in `Int128` whenever the leading partial values require it. The optional `FULL_RECOVERY=1` build instead performs decreasing-index back substitution of every square-free value as a correctness oracle; its recovery sum is negligible relative to the earlier S1/S2 accumulations.
