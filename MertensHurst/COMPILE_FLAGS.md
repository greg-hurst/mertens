# Build flags

Within their documented input domains, every configuration computes the same $M(n)$. The Makefile passes the sieve controls through to the shared library, whose bounds and knobs are documented in [`../sieve/COMPILE_FLAGS.md`](../sieve/COMPILE_FLAGS.md).

| Flag | Default | What it does |
|---|---|---|
| `DIVISION_FREE` | 1 on x86, 0 on ARM | Quotient strategy for the hot $S_1$/$S_2$ loops and the sieve. On ARM, hardware division is fast and the direct path wins. On x86 the division-free path (Granlund-Montgomery cache + quotient predictor) wins. Auto-detected; override with `make DIVISION_FREE=0/1`. |
| `BUCKET_SIEVE` | 1 | Large-prime bucket scheduler in the sieve. P1/P2 use the ordinary $u \le 2.05 \times 10^{17}$ cap; active P6 uses its independently proven $u \le 115{,}571{,}495{,}477{,}370{,}241$ cap. `make BUCKET_SIEVE=0` removes this scheduler constraint at a large speed cost. The next cap is $2^{60}-2^{32}$ when `DIVISION_FREE=1`, or the roughly $1.8 \times 10^{19}$ encoding/prime limit when it is off. |
| `SIEVE_BUCKET_NARROW_ENTRY` | 1 | Full/P1 and packed-odd/P2 bucket entry format. Narrow (prime-only) is fastest on the ARM record machines; wide may win on x86. It does not override native P6. |
| `COPRIME6_NARROW_ENTRY` | 0 | Native P6 bucket entry format. Its independently measured default is the 8-byte wide, divide-free layout; full/P1 and P2 remain narrow. |
| `COPRIME6_DIRECT_CUTOFF_MULT` | 2200 | Native P6 direct-sieve cutoff in 770-entry stencil periods. Values at or below 1728 are rejected by the scheduler's forwarding invariant. |
| `FUSED_FINALIZE` | 1 | Fold Möbius finalization into the Mertens prefix scan in Loop 2, avoiding a separate pass over the sieve buffer. Drives both `-DSIEVE_FUSED_FINALIZE` and `-DSIEVE_COPRIME6_FUSED_FINALIZE`. |
| `S1_OUTER_Q6` | 1 | Exact outer $Q=6$ transform for $S_1$. Set this and `S2_OUTER_Q6` to `0`, or use `make q2`, to build the original all-$Q=2$ reference path. |
| `S2_OUTER_Q6` | 1 | Exact outer $Q=6$ for $S_2$. It requires `S1_OUTER_Q6=1`. Together with inner Q6 this is the normal `build/mertens` path; `make q2` preserves the all-Q2 oracle as `build/mertens_q2`. |
| `Q30_COUPLED` | 0 | Complete outer/inner $Q=30$ promotion. Use `make q30-coupled`; the named target fixes the compatible compact unordered stack and emits `build/mertens_q30_coupled`. |
| `Q210_COUPLED` | 0 | Complete outer/inner $Q=210$ promotion with whole-run fallback to Q30. Use `make q210-coupled`; the named target fixes the complete Q30/Q210 contract and emits `build/mertens_q210_coupled`. |
| `LOOP2_SIEVE_P` | 1 | Compile-time Loop-2 sieve and default-parameter selector. `1` retains full $M$ with `(nuRatio,u-anchor)=(0.90,0.55@10^16)`, `2` uses the packed odd-prefix identity with `(0.95,0.70@10^18)`, and `6` uses the packed coprime-to-6 identity with `(1.00,0.75@10^18)`; every u-factor falls by 0.025 per decade with a 0.30 floor. Use the contract-closed `q210-coupled-loop2-p1`, `-p2`, and `-p6` targets; a failed runtime Q210 guard in either restricted binary retains the full-$M$ backend with the configured mode's parameters. |
| `LOOP2_SIEVE_VALIDATE` | 0 | Validate the selected restricted sieve against its direct prefix oracle and every completed row against full-$M$ Loop 2. Enabled by the named validation and sanitizer targets. |
| `FULL_RECOVERY` | 0 | Recover every square-free partial value by decreasing-index back substitution. Production instead obtains only the requested final value by direct Möbius inversion. The full path is retained as a correctness oracle. Becomes `-DMERTENSHURST_FULL_RECOVERY`; enable it with `make FULL_RECOVERY=1`. |

`make s2-unordered` builds the opt-in `build/mertens_s2_unordered` profile.
The target fixes the complete compatible contract rather than exposing partial
combinations: final-value recovery, both outer-$Q=6$ transforms, coherent split
ownership, generated period-36 kernels, and the unordered hot-$S_2$ square.
The ordered-square validation comparator is compile-time disabled in that
release profile. Incompatible partial combinations are rejected by static
assertions in `MertensHurst.cpp`.

`make q30-coupled` and `make q210-coupled` are likewise contract-closed named
profiles. Q210 requires the complete Q30 profile beneath it. Its shared guard
is evaluated before Q210 worklist filtering or table allocation; a failed
guard retains one coherent Q30 invocation.

The `q210-coupled` profile enables the exact factor-11/factor-13 $S_1$
hierarchy in Loops 0 and 1. Its temporary masks and monotone child maps retain
$O(n/u)$ space and use about 249 MB at $10^{25}$; they are released before
Loop 2. `make q210-coupled-native` emits the pre-hierarchy baseline.
`make q210-coupled-validate` enables both the ladder's per-segment native
comparison and the ordered-square comparator, while
`make q210-coupled-sanitize` builds that validation profile with
AddressSanitizer and UndefinedBehaviorSanitizer.
`make q210-coupled-fallback` forces the Q210-to-Q30 runtime fallback in the
same validation profile.

`make q210-coupled-loop2-p1`, `make q210-coupled-loop2-p2`, and
`make q210-coupled-loop2-p6` select the full-$M$, packed-odd, and native
packed-coprime-to-6 Loop-2 implementations at compile time. Each P2/P6
`-validate` target retains both a direct restricted-prefix comparison for
every signed segment visit and a full-$M$ comparison for every completed row;
the corresponding `-sanitize` target runs the same checks under
AddressSanitizer and
UndefinedBehaviorSanitizer. The corresponding `-fallback` target forces the
Q210 guard failure and verifies that the same binary activates P1. The P6
validation additionally compares every compressed M6 prefix and all four
signed streams with the direct M6 oracle. The historical
`q210-coupled-odd-loop2` target and binary names remain aliases of P2 for
compatibility.

`make q210-coupled-record` builds the production Q210 contract with
`SIEVE_STRIDE_LOG=7`. This gives the compressed `Int8` Mertens residual a
worst-case proof for record-scale runs. The ordinary target retains the
slightly faster stride 8, whose much larger practical range is heuristic.

The MertensHurst Makefile names the full/P2 narrow-entry setting
`SIEVE_BUCKET_NARROW_ENTRY`; the standalone sieve Makefile calls it
`NARROW_ENTRY`. Both produce `SIEVE_NARROW_ENTRY`. Native P6 instead uses
`COPRIME6_NARROW_ENTRY` and `COPRIME6_DIRECT_CUTOFF_MULT`, which produce the
corresponding `COPRIME6_SIEVE_*` defines only on P6 targets.

Any sieve define can be passed through the hook, e.g. `make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024"`.

The runtime cap on $u$ is build-aware: it is computed from `SIEVE_LP_SIZE`, the prime/encoding domain, and the division-free quotient domain when enabled. An out-of-range `--u` or `--u-factor` fails fast with a pointer to `INPUT_BOUNDS.md` instead of silently corrupting the sieve.
