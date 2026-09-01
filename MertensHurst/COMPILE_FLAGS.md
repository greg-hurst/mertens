# Build flags

All performance-only: every configuration computes the same $M(n)$. The Makefile passes these through to the shared sieve, whose own knobs are documented in [`../sieve/COMPILE_FLAGS.md`](../sieve/COMPILE_FLAGS.md).

| Flag | Default | What it does |
|---|---|---|
| `DIVISION_FREE` | 1 on x86, 0 on ARM | Quotient strategy for the hot $S_1$/$S_2$ loops and the sieve. On ARM, hardware division is fast and the direct path wins. On x86 the division-free path (Granlund-Montgomery cache + quotient predictor) wins. Auto-detected; override with `make DIVISION_FREE=0/1`. |
| `BUCKET_SIEVE` | 1 | Large-prime bucket scheduler in the sieve. `make BUCKET_SIEVE=0` removes the enforced $u \le 2.05 \times 10^{17}$ cap (the binding constraint becomes $\sim 1.8 \times 10^{19}$, see `INPUT_BOUNDS.md`), at a large speed cost on long sieve ranges. |
| `SIEVE_BUCKET_NARROW_ENTRY` | 1 | Bucket entry format, passed to the sieve. Narrow (prime-only) is fastest on the ARM record machines; wide may win on x86. Details in the sieve flags doc. |
| `FUSED_FINALIZE` | 1 | Fold Möbius finalization into the Mertens prefix scan in Loop 2, avoiding a separate pass over the sieve buffer. Becomes `-DSIEVE_FUSED_FINALIZE`. |
| `S1_OUTER_Q6` | 1 | Exact outer $Q=6$ transform for $S_1$. Set this and `S2_OUTER_Q6` to `0`, or use `make q2`, to build the original all-$Q=2$ reference path. |
| `S2_OUTER_Q6` | 1 | Exact outer $Q=6$ for $S_2$. It requires `S1_OUTER_Q6=1`. Together with inner Q6 this is the normal `build/mertens` path; `make q2` preserves the all-Q2 oracle as `build/mertens_q2`. |
| `Q30_COUPLED` | 0 | Complete outer/inner $Q=30$ promotion. Use `make q30-coupled`; the named target fixes the compatible compact unordered stack and emits `build/mertens_q30_coupled`. |
| `Q210_COUPLED` | 0 | Complete outer/inner $Q=210$ promotion with whole-run fallback to Q30. Use `make q210-coupled`; the named target fixes the complete Q30/Q210 contract and emits `build/mertens_q210_coupled`. |
| `S1_Q210_PAIRED_SEAM` | 0 | Reindex only paired native-Q30 Loop-0/1 S1 rows through coprime-210 streams and an exact Q30 seam. Use `make q210-s1-paired-seam`; boundary, S2, Loop 2, row state, and recovery remain Q30. |
| `S1_Q210_PAIRED_SEAM_VALIDATE` | 0 | Compare the parent-coordinate seam, child-coordinate Q30 correction, and original native-Q30 pair in every transformed segment. Enabled by `make q210-s1-paired-seam-validate`. |
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
guard retains one coherent Q30 invocation. `make q210-coupled-validate` enables
the ordered-square comparator, while `make q210-coupled-sanitize` builds the
same profile with AddressSanitizer and UndefinedBehaviorSanitizer.

`make q210-s1-paired-seam` is a separate contract-closed native-Q30 profile.
It forces coupled Q210 off, applies the paired finite-difference identity only
in Loop 0/1, and leaves every other Q30 domain unchanged. Its validation and
sanitizer targets compile the pair identity checks into the executable. Every
other named profile filters both seam controls out, and full recovery remains
the unchanged Q6 family-producing path.

The MertensHurst Makefile names its narrow-entry setting `SIEVE_BUCKET_NARROW_ENTRY`; the standalone sieve Makefile names the corresponding setting `NARROW_ENTRY`. Both produce the compiler define `SIEVE_NARROW_ENTRY`.

Any sieve define can be passed through the hook, e.g. `make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024"`.

The runtime cap on $u$ is build-aware: it is computed from the actual compiled constants (including `SIEVE_LP_SIZE`), so an out-of-range `--u` or `--u-factor` fails fast with a pointer to `INPUT_BOUNDS.md` instead of silently corrupting the sieve.
