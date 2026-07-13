# Build flags

All performance-only: every configuration computes the same $M(n)$. The Makefile passes these through to the shared sieve, whose own knobs are documented in [`../sieve/COMPILE_FLAGS.md`](../sieve/COMPILE_FLAGS.md).

| Flag | Default | What it does |
|---|---|---|
| `DIVISION_FREE` | 1 on x86, 0 on ARM | Quotient strategy for the hot $S_1$/$S_2$ loops and the sieve. On ARM, hardware division is fast and the direct path wins. On x86 the division-free path (Granlund-Montgomery cache + quotient predictor) wins. Auto-detected; override with `make DIVISION_FREE=0/1`. |
| `BUCKET_SIEVE` | 1 | Large-prime bucket scheduler in the sieve. `make BUCKET_SIEVE=0` removes the enforced $u \le 2.06 \times 10^{17}$ cap (the binding constraint becomes $\sim 1.8 \times 10^{19}$, see `INPUT_BOUNDS.md`), at a large speed cost on long sieve ranges. |
| `SIEVE_BUCKET_NARROW_ENTRY` | 1 | Bucket entry format, passed to the sieve. Narrow (prime-only) is fastest on the ARM record machines; wide may win on x86. Details in the sieve flags doc. |
| `S1_OUTER_Q6` | 1 | Exact outer $Q=6$ transform for $S_1$. Set this and `S2_OUTER_Q6` to `0`, or use `make q2`, to build the original all-$Q=2$ reference path. |
| `S2_OUTER_Q6` | 1 | Exact outer $Q=6$ for $S_2$. It requires `S1_OUTER_Q6=1`. Together with inner Q6 this is the normal `build/mertens` path; `make q2` preserves the all-Q2 oracle as `build/mertens_q2`. |

Any sieve define can be passed through the hook, e.g. `make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024"`.

The runtime cap on $u$ is build-aware: it is computed from the actual compiled constants (including `SIEVE_LP_SIZE`), so an out-of-range `--u` or `--u-factor` fails fast with a pointer to `INPUT_BOUNDS.md` instead of silently corrupting the sieve.
