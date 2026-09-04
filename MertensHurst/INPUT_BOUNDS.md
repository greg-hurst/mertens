# Bounds on $n$ for `MertensHurst(n)`

This document catalogs the constraints that limit the supported input `UInt128 n`. It aims to be exhaustive, but there may be additional constraints not yet identified. Each entry notes which source file(s) the constraint originates from, what kind of limit it is, and what would need to change to relax it.

The sections are ordered by the approximate magnitude at which a constraint first becomes relevant. Unless a section says otherwise, conversions from a bound on $u$ to a bound on $n$ use the current default

$$
u(n)=\left\lceil f(n)\left(\sqrt[3]{\frac{n}{\log\log n}}\right)^2\right\rceil,
\qquad
f(n)=\operatorname{clamp}\bigl(0.55-0.025(\log_{10}n-16),0.30,0.55\bigr),
$$

and `nuRatio == 0.9`. The resulting $n$ thresholds are approximate and their last digits may depend on the platform's floating-point library.

The imposed absolute bounds are $10^8 \le n \le 10^{26}$. Correctness has been verified by completed computations through $10^{25}$; $10^{26}$ is accepted after the runtime and structural checks below but remains unverified until that computation is completed independently.

## Roadmap by magnitude

| Approximate $n$ | Event | Kind |
|---:|---|---|
| $2.372\times10^8$ | Default split first clears one stencil period | Parameter-dependent lower bound |
| $2^{64}\approx1.845\times10^{19}$ | Legacy all-Q2 S1 bound narrowing ceases to be generally safe | Profile-specific hard bound |
| $10^{25}$ | Largest completed and independently checked computation | Validation frontier |
| $10^{26}$ | Current imposed cap; accepted but not yet independently verified | Software frontier |
| $1.275\times10^{26}$ | `UInt32` outer-state extent is exhausted | Hard correctness bound |
| $2.91\times10^{26}$ | Two-$p$ mod-4 skip starts disabling with `LP_SIZE=512` | Performance transition only |
| $2.35\times10^{27}$ | `LP_SIZE=512` bucket reach is exhausted | Build-specific hard bound |
| $1.90\times10^{28}$ | `LP_SIZE=1024` bucket reach is exhausted | Build-specific hard bound |
| $3.15\times10^{28}$ | Prime-indexed division-free sieve quotient domain is exhausted | Build-specific hard bound |
| $3.96\times10^{28}$ | Contiguous quotient-cache count and the ordered Q2/Q6 predictor path first reach width limits | Build/profile-specific hard bounds |
| Around $1.7\times10^{29}$ | Coarse `Int32` prefix helpers become an empirical concern | Unproved representation bound |
| $7.21\times10^{29}$ | Q6 S1 predictor reaches its signed-quotient limit | Profile-specific hard bound |
| $1.19\times10^{30}$ | Step-30 Q30 predictor reaches its signed-quotient limit | Profile-specific hard bound |
| $2.05\times10^{30}$ | `UInt32` primes, `UInt64` sieve coordinates, and byte encoding are exhausted | Coordinated hard bound |
| $8.11\times10^{31}$ | Integer square roots no longer convert exactly to `double` | Floating-point transition |
| $2^{128}-1\approx3.40\times10^{38}$ | `UInt128` input domain is exhausted | Fundamental type bound |

The table is a guide, not a substitute for the detailed conditions below. In particular, several entries depend on compile-time options, and unchecked accumulator widths do not have a known safe decade.

---

## 0. Lower bound and split guard

**Source:** `src/MertensHurst.cpp`, `../sieve/SegmentedMobiusSieve.h`

The minimum viable sieve segment size is `B == BF == STENCIL_PERIOD == 13860`, and the main sieve loop requires `B < nu_max`. The implementation reduces its initial segment to the largest stencil multiple below `nu_max` and rejects the configuration if `nu_max <= BF`; it no longer continues with an empty main loop.

The implementation evaluates `floor(0.9*floor(sqrt(n)))` by default, so this requires approximately $n \ge 2.372 \times 10^8$. Inputs between the absolute $10^8$ floor and this default threshold require a larger explicit `nuRatio`.

More generally, if the split is changed to `nu_max = c*sqrt(n)`, then this structural threshold becomes roughly $(13860/c)^2$. Decreasing the split constant therefore raises the parameter-dependent lower input bound.

Additional reasons the algorithm is not designed for small $n$:

- **Segment-size and chunk-size formulas are tuned for large inputs.** `getSegmentSize`, the S2 chunking parameter `CHUNK_LEN`, and the Loop 2 segment size all use heuristics such as $\sqrt n$, $n^{1/3}$, and $\sqrt{2u}$. The initial size is clamped against the actual split before allocation.

- **The S2 mode splitting assumes non-trivial sub-ranges.** The multi-mode S2 dispatch divides the summation range $[1,\nu]$ at boundaries such as $\nu/6$, $\nu/3$, and $\nu/2$. For small $n$, the per-argument `nus[i] == get_nu(n/i)` shrinks rapidly with `i`, and some mode sub-ranges may become empty. Empty ranges are handled gracefully, but the algorithm has not been validated in this regime.

- **Asymptotic tuning constants.** The default $u$ factor, the `M16BITMAX` transition point, and the Loop 2 segment-size formula all embed constants calibrated for $n\ge10^8$. These constants are not inherently wrong for smaller $n$, but their segment layouts and sieve coverage have not been verified below $10^8$.

---

## 1. Legacy all-Q2 S1 bound at $2^{64}$

**Source:** `src/S1.h`

The reference `make q2` profile's S1 wrapper computes the segment-dependent bounds

```cpp
const UInt64 loBase = UInt64(partialArg / (L2 + 1)) + 1;
const UInt64 hiBase = UInt64(partialArg / L1);
```

before clamping them to the `UInt64` S1 range. For the leading `UInt128` argument and `L1 == 1`, `hiBase` can therefore narrow before the clamp once $n\ge2^{64}$. At $n=2^{64}$ it wraps to zero. The all-Q2 build cannot claim general correctness beyond this point even though many individual inputs may narrow to a value that happens to exceed the later clamp.

The Q210 record profile does not use this wrapper. To extend the Q2 reference build, compute and clamp `loBase` and `hiBase` in `UInt128`, verify that the clamped values fit, and only then convert to `UInt64`.

---

## 2. Tested through $10^{25}$

This is the largest input for which correctness has been verified by a completed computation. Higher values may trigger unknown edge cases. The software accepts $10^{26}$, but the claimed validation frontier should not move until that result is completed and checked independently.

---

## 3. `UInt32` outer and compact indices

**Source:** `src/MertensHurst.cpp`, `src/S1Q6.h`, `src/S1Q30030Ladder.h`, `src/S2Q210.h`, `src/OuterRecovery.h`

The initial outer extent $\lfloor n/u\rfloor$ is represented by `UInt32`, and the runtime requires

$$
\left\lfloor\frac{n}{u}\right\rfloor < 2^{32}-1
$$

before narrowing. Under the current default formula, the first failure occurs at approximately **$n=1.27492819\times10^{26}$**, making this the first structural ceiling beyond the imposed $10^{26}$ cap.

At $n=10^{26}$, the current floating-point formula gives $u\approx2.52632\times10^{16}$, $\lfloor n/u\rfloor=3\,958\,326\,568$, and exactly $2\,406\,374\,010$ square-free outer entries. The minimum outer-safe $u$ is about $2.32831\times10^{16}$, so the default has only about 8.5% headroom. The square-free count already exceeds `Int32`, which is why these counters are unsigned.

Extending this bound is a coordinated representation change, not a one-variable promotion. The `UInt32` assumption appears in:

- outer counts, loop and chunk indices, `mx`, `cnt128`, and `Chunk::i`;
- `hash`, `hash2`, `q6Bases`, compact work maps, and factor-11/factor-13 child maps;
- `S1Q6WorkItem::compactIndex` and its worklist construction;
- unordered-S2 row indices, cutoffs, and endpoints; and
- both final-only and full-recovery interfaces.

These structures should be widened and their memory layout retuned together. Later compact counts and work-map encodings do not all overflow at precisely the same $n$, but widening only `outerCount` would merely expose the next narrowing.

Lowering $u$ helps sieve memory and scheduler capacity but makes this limit worse. Any `UInt32` outer build must satisfy

$$
\frac{n}{2^{32}-1}<u\le u_{\max}.
$$

Ignoring all other constraints, combining the largest legal $u$ for several sieve configurations with the outer-index requirement gives these absolute joint ceilings:

| Sieve configuration | Approximate joint ceiling on $n$ |
|---|---:|
| `LP_SIZE=512` | $8.82\times10^{26}$ |
| `LP_SIZE=1024` | $3.54\times10^{27}$ |
| Buckets off, division-free on | $4.95\times10^{27}$ |
| Buckets and division-free off, `UInt32` primes | $7.92\times10^{28}$ |

The current default formula reaches the outer bound much earlier than these best-case combinations.

---

## 4. Optional two-$p$ mod-4 skip

**Source:** `../sieve/SegmentedMobiusSieve.cpp`

Multiples of four are already zero in the stencil, so the large-prime scheduler may skip such a hit by advancing an additional $p$. This two-$p$ forwarding is used only when the second jump fits inside the bucket ring. For larger primes the implementation automatically falls back to ordinary one-$p$ forwarding; therefore this is a **performance transition, not a correctness bound**.

| `LP_SIZE` | Largest prime with the two-$p$ skip | Approximate default-formula $n$ where the transition begins |
|---:|---:|---:|
| 512 | 226,638,720 | $2.91\times10^{26}$ |
| 1024 | 453,720,960 | $2.36\times10^{27}$ |

At $n=10^{26}$, $\sqrt u\approx158.9$ million, so all scheduled primes remain below the 512-bucket half-reach. No mod-4-path change is needed for this input. Above the threshold, an increasing top band of primes merely loses the optimization.

---

## 5. Bucket-scheduler reach

**Source:** `../sieve/SegmentedMobiusSieve.h`, `../sieve/SegmentedMobiusSieve.cpp`, `src/MertensHurst.cpp`

The large-prime scheduler uses a circular buffer of `LP_SIZE` buckets. Its exact largest schedulable prime is

$$
(\texttt{LP\_SIZE}-1)M_2,
\qquad M_2=887\,040,
$$

so it requires $\sqrt u$ not to exceed that reach.

| `LP_SIZE` | Prime reach | Maximum $u$ | Approximate default-formula $n$ |
|---:|---:|---:|---:|
| 512 | 453,277,440 | 205,460,437,612,953,600 | $2.35\times10^{27}$ |
| 1024 | 907,441,920 | 823,450,838,173,286,400 | $1.90\times10^{28}$ |

The 512-bucket configuration is used for record runs and is described in Section 7 of the paper. Build with `make EXTRA_CXXFLAGS=-DSIEVE_LP_SIZE=1024` for the second row. The runtime derives its cap from `SegmentedMobiusSieveCore::schedulerReach()` for every source of $u$: the default formula, `--u`, or `--u-factor`.

Exactly 1024 buckets fit the wide entry's 10-bit stride field. The current unconditional `static_assert` applies to the common scheduler layout, so going beyond 1024 requires widening or repacking that field, or separating the narrow prime-only representation from the wide layout. For sufficiently large future values, the runtime calculation of `reach * reach` must also be performed and clamped in `UInt128` rather than `UInt64`.

Increasing $M_2$ is another possible way to increase reach, but it is constrained by the 21-bit offset field, the required ordering of the sieve thresholds, and cache/performance effects; it is not a drop-in capacity change.

Building with `make BUCKET_SIEVE=0` sends all large primes through direct iteration and removes **this scheduler constraint only**. Every whole-algorithm limit in the remaining sections still applies.

---

## 6. Prime-indexed quotient domain (`DIVISION_FREE=1` only)

**Source:** `../sieve/QuotientCache.h`, `../sieve/SegmentedMobiusSieve.cpp`, `src/MertensHurst.cpp`

This section concerns `SieveQuotientCache`, the prime-indexed cache used by the Möbius sieve. It is distinct from the contiguous S1/S2 `QuotientCache` in the next section.

When built with `DIVISION_FREE=1`—the default on x86 and an available override on ARM—the sieve computes $\lceil x/p\rceil$ via the Granlund--Montgomery multiply-shift with `SHIFT == 60`. This is exact only for arguments below $2^{60}$. Since `ceilDiv` forms $\text{val}=x+p-1$ with $x\le u$ and $p<2^{32}$, the sieve range must satisfy

$$
u<2^{60}-2^{32}=1\,152\,921\,500\,311\,879\,680.
$$

The largest inclusive $u$ is therefore $(2^{60}-2^{32})-1$. Inverting the default formula gives approximately **$n=3.15\times10^{28}$**. The sieve enforces this with an unconditional runtime abort, and `MertensHurst` folds the same bound into its build-aware cap.

The bucket-scheduler cap binds first when buckets are enabled. Build with `DIVISION_FREE=0` to remove this particular domain constraint; this is the default on ARM.

---

## 7. Contiguous quotient cache and predictor widths

**Source:** `../sieve/QuotientCache.h`, `src/MertensHurst.cpp`, `src/QuotientPredictor.h`, `src/S1.h`, `src/S1Q6.h`, `src/S1Q30.h`, `src/S1Q210.h`, `src/S1Q30030Ladder.h`, `src/S2.h`, `src/S2Q30.h`, `src/S2Q210.h`

### Contiguous division-free cache

With `DIVISION_FREE=1`, S1/S2 precompute one `UInt8` shift and one `UInt64` multiplier for every denominator through

$$
d_{\max}=\left\lceil\sqrt[3]{2n}\right\rceil.
$$

`dCAP` is calculated as `UInt64`, but `QuotientCache::count` and `QuotientCache::init` use `UInt32`. The cache first stops fitting when $d_{\max}>2^{32}-1$, at approximately

$$
n=\frac{(2^{32}-1)^3}{2}=3.9614081\times10^{28}.
$$

Earlier constraints always bind first in the current code. This count becomes relevant after the outer representation and the applicable sieve limits have been redesigned. The cache payload itself is nine bytes per denominator, or about 38.65 GB at the count boundary. Extending the range requires a wider count and an explicit memory decision, not just a cast.

### Predictor paths

The historical blanket requirement $n^{2/3}<2^{63}$ is not an algorithm-wide bound. Quotient width depends on the wheel and on whether the ordered or unordered path is active:

| Path | Approximate first signed-quotient limit |
|---|---:|
| Ordered Q2/Q6 S2 predictor | $3.96\times10^{28}$ |
| Q6 S1 predictor through $u$ | $7.21\times10^{29}$ |
| Step-30 Q30 predictor | $1.19\times10^{30}$ |
| Step-210 Q210 predictor | $8.32\times10^{30}$ |

The current unordered-Q210 wide path preserves its early quotients as `UInt128`; when its step-210 predictor begins, the quotient has already fallen into the supported `UInt64` range. Consequently, the old $2.8\times10^{28}$ statement is **not** a ceiling on the Q210 profile, and the sieve/prime limits below bind first. Retained Q2 and ordered fallback builds still have their own earlier predictor contract.

Some unordered Q6/Q30 wide entry points convert individual wide quotients to `Int128`. Those casts become another path-specific audit point near $n=2^{127}$, but they do not by themselves establish a complete profile-wide ceiling: the applicable direct/predictor transitions and every downstream evaluator must be checked together. Validation/direct-oracle paths may also be narrower than production and require a separate audit when extending a particular build.

---

## 8. Coarse `Int32` Mertens state

**Source:** `src/MertensHurst.cpp`, `src/S1Q30030Ladder.h`, `../sieve/SegmentedMertensSieve.h`

The compressed Mertens representation stores its coarse samples in `Int32`. `M32`, `MPrev`, interval sums, thread offsets, and the prefix helpers therefore require every represented $M(x)$ for $x\le u$, including intermediate segment-prefix totals, to fit in signed 32 bits. This is separate from the `Int8` residual constraint below, and there is currently no release-mode range guard.

There is no known analytic $n$ threshold. If the empirical envelope from [Table 6.1 of Hurst 2016](https://arxiv.org/pdf/1610.08551)

$$
|M(x)|\le0.571\sqrt{x}
$$

is extrapolated far beyond the $10^{16}$ range for which it was checked, an arbitrary range sum can be bounded conservatively by

$$
|M(b)-M(a)|\le |M(b)|+|M(a)|
$$

This makes the prefix-helper storage a concern around $u\approx3.54\times10^{18}$ or **$n=1.7\times10^{29}$**. A single global coarse sample $M(x)$ reaches `Int32` scale later, around $u\approx1.41\times10^{19}$ or **$n=1.4\times10^{30}$**. Both figures are planning estimates, not proofs.

Before approaching that scale, run a checked `Int64`-coarse validation build and then promote `M32`, `MPrev`, interval sums, thread offsets, and their helper interfaces together. The factor-11/factor-13 ladder's conservative bound, which currently embeds `2^31+128`, must be updated at the same time. The promotion approximately doubles the coarse and prefix-helper memory.

---

## 9. `UInt64` sieve domain, `UInt32` primes, and byte encoding

**Source:** `src/MertensHurst.cpp`, `../sieve/SegmentedMobiusSieve.h`, `../sieve/SegmentedMobiusSieve.cpp`

Three representation limits meet at essentially the same scale:

1. Primes are stored as `UInt32`, so the largest sieve prime $\sqrt u$ must not exceed $2^{32}-1$.
2. The sieve stores its range, segment endpoints, hit positions, and many related coordinates in `UInt64`.
3. The Möbius sieve stores accumulated odd ceil-log2 weights in a 7-bit byte field. The encoding is collision-free for every $u<2^{64}$; finalization uses an exact comparison.

The current runtime cap is

$$
u\le(2^{32}-1)^2=18\,446\,744\,065\,119\,617\,025,
$$

which also keeps $u<2^{64}$. Under the default formula this is reached at approximately **$n=2.05\times10^{30}$**.

Relaxing this is a coordinated redesign: widen the prime representation, widen sieve coordinates and APIs beyond `UInt64`, redesign the byte encoding, and audit every endpoint expression such as `L1 + B - 1` for overflow or saturation. Near the present theoretical cap there is not enough `UInt64` headroom to assume that a large segment length can always be added before taking a minimum.

The prime generator should also stop converting a floating-point square root directly to `UInt32`; a segmented or exact-integer generator would address that conversion and reduce its transient memory cost.

The default $u$ is presently calculated as `double` and converted to `UInt64` before the hard-cap check. At this extension point, finiteness and representability must be checked before that conversion so an out-of-range floating value cannot be narrowed first.

---

## 10. Floating-point transitions and the `UInt128` input ceiling

**Source:** `src/MertensHurst.cpp`

The main integer square root is seeded with floating point and then corrected by exact adjustment loops. It is therefore exact over the currently supported range. Not all later uses preserve that exact integer value, however:

- `get_nu` converts the corrected root back to `double`. Once $\sqrt n>2^{53}$, equivalently $n>2^{106}\approx8.11\times10^{31}$, unit-by-unit representation is lost and `floor(nuRatio*sqrt(n))` is no longer guaranteed to match the mathematical expression bit for bit.
- The default $u$, segment sizes, and several cutoffs are selected with `double`, `sqrt`, `cbrt`, and `log`. Most of these values choose a legal performance decomposition rather than the mathematical answer, but all callers must compute compatible boundaries.

Formal extension should replace correctness-sensitive floating boundaries with exact or conservatively corrected integer calculations while leaving purely tuning-driven calculations floating point where harmless.

The input type itself ends at

$$
n\le2^{128}-1\approx3.4028\times10^{38}.
$$

Before approaching that endpoint, `isqrt_u128` also needs an end-of-domain audit. Its floating seed can round to $2^{64}$ before conversion to `UInt64`, and its correction expression increments a `UInt64` candidate before widening it to `UInt128`. The seed must be range-checked and the increment and square must be widened or special-cased.

---

## 11. Default `Int8` residual array `R`

**Source:** `../sieve/SegmentedMertensSieve.h`

The compressed Mertens representation stores residuals as `Int8` over intervals of `STRIDE == 1 << STRIDE_LOG`. If the prefix sum of $\mu$ within any interval exceeds $[-128,127]$, this overflows silently.

A heuristic from Ng in ["The distribution of the summatory function of the Mobius function"](https://www.cs.uleth.ca/~nathanng/RESEARCH/mobiusshort.pdf) treats $\mu$ as independent random variables with $\Pr(\mu(k)\ne0)=6/\pi^2$. The expected maximum short sum over $[1,X]$ in intervals of length $H$ is

$$
\max |M(x+H)-M(x)|\sim
\sqrt{\frac{12H}{\pi^2}\log\left(\frac{X}{H}\right)}.
$$

Setting this equal to 128 gives:

| `STRIDE_LOG` | $H$ | Heuristic maximum $u$ |
|---:|---:|---:|
| 7 | 128 | $\infty$ (proved) |
| 8 | 256 | $1.9\times10^{25}$ |
| 9 | 512 | $1.4\times10^{14}$ |
| 10 | 1024 | $5.3\times10^8$ |

At `STRIDE_LOG=7`, overflow is mathematically impossible: every interval of 128 consecutive integers contains at least $\lfloor128/4\rfloor=32$ multiples of four, which are non-squarefree and have $\mu=0$. At most 96 values can be non-zero, so $|\text{residual}|\le96<128$.

No overflow was observed through $u=13\,694\,622\,981\,236\,974$ in the completed $10^{25}$ computation. The ordinary Q210 target retains stride 8 for speed and therefore remains heuristic; `make q210-coupled-record` selects stride 7 and supplies the hard proof. The stride-8 heuristic ceiling is beyond every currently reachable sieve representation bound, while strides 9 and 10 are appropriate only for the smaller ranges shown above.

---

## 12. Intermediate accumulators and final result

**Source:** `src/MertensHurst.cpp`, `src/OuterRecovery.h`, `src/S1.h`, `src/S1Q6.h`, `src/S1Q30.h`, `src/S1Q210.h`, `src/S1Q30030Ladder.h`, `src/S2.h`, `src/S2Q6.h`, `src/S2Q6Modes.h`, `src/S2Q30.h`, `src/S2Q210.h`

The final S1 and S2 values are $O(x^{2/3})$ per argument $x$ and largely cancel to produce $M(x)$, but running partial sums can be substantially larger than their final values. The $10^{18}$ narrow/wide dispatch is an implementation policy, not by itself a proof that every narrow accumulator fits `Int64`.

The original all-Q2 estimates do not prove the current transformed paths. The Q210 record profile combines native Q210 S2, Q210 S1, and a selective factor-11/factor-13 S1 ladder. A useful extension proof must bound the actual Q210 mode coefficients, boundary updates, compact atomics, per-thread reductions, and their incomplete cancellation. Individual period kernels fitting their destination type does not prove that the sum of many periods fits.

The ladder currently combines at most four completed-Q210 components in one narrow result. If $K$ is its largest narrow common $\kappa$, the implementation evaluates the conservative bound

$$
4\cdot48\left\lceil\frac{K}{210}\right\rceil(2^{31}+128)
$$

in `UInt128` before allocating the ladder maps and falls back to native Q210 S1 if it exceeds `INT64_MAX`. At `nuRatio=0.9`, $K\le1\,111\,111\,111$ throughout the narrow $y\le10^{18}$ path, so this bound is approximately $2.182\times10^{18}$, or 23.7% of `INT64_MAX`. The combined narrow destination update is independently formed in `Int128` and range-checked before storage. Other narrow accumulation sites do not all have equivalent checked bounds.

Wide partial values, unordered-S2 totals, thread reductions, and final recovery use `Int128`, but their additions are not generally checked for overflow. This is not a practical concern near $10^{26}$; it remains a proof obligation for a much larger extension.

Production recovery computes $M(n)$ in `Int128`, but the public API returns `Int64`. The production narrowing is protected only by debug assertions, and the optional `FULL_RECOVERY=1` path also returns `Int64`. Thus the public contract additionally assumes

$$
\texttt{INT64\_MIN}\le M(n)\le\texttt{INT64\_MAX},
$$

for which this code supplies no useful fixed input threshold. Before that assumption becomes questionable, add unconditional release-mode checks to both recovery paths or widen the API and its formatter to `Int128`.
