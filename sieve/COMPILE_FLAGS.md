# Build flags

Every flag here is a performance knob. None of them change results: all configurations produce identical $\mu$ and $M$ values, and the sieve's checksum tests verify that.

Flags marked *(make)* are Makefile variables, e.g. `make BUCKET_SIEVE=0`. The rest are plain compiler defines, passed via the hook in every Makefile:

```
make EXTRA_CXXFLAGS="-DSIEVE_LP_SIZE=1024 -DSIEVE_STRIDE_LOG=7"
```

| Flag | Default | What it does |
|---|---|---|
| `BUCKET_SIEVE` *(make)* | 1 | The large-prime bucket scheduler. Turning it off sieves every prime by direct iteration: no $2.06 \times 10^{17}$ range cap, but much slower on long ranges. Becomes `-DUSE_BUCKET_SIEVE`. |
| `DIVISION_FREE` *(make)* | 1 on x86, 0 on ARM | Granlund-Montgomery multiply-shift instead of hardware division in the sieve's quotient arithmetic. Wins on x86, loses on Apple Silicon. When on, the sieve endpoint must stay below $2^{60} - 2^{32}$. Becomes `-DUSE_DIVISION_FREE`. |
| `SIEVE_BUCKET_NARROW_ENTRY` *(make)* | 1 | Bucket entry format. Narrow entries are 4-byte prime-only, with one divide per hit; fastest on many-core ARM. Set to 0 for wide 8-byte packed entries (divide-free per hit), which may win on x86 where the divide is costlier. Becomes `-DSIEVE_NARROW_ENTRY`. |
| `SIEVE_SUB_BUCKETS` *(make)* | 1 | Bands bucket hits by offset so the $\mu$ read-modify-writes stay cache-local. Wide entries are persistently banded in the ring; narrow entries retain their 4-byte persistent format and transiently band only the current decoded hits. |
| `SIEVE_SUB_SHIFT` *(make)* | 17 | Wide-entry persistent band size as log2 bytes. 17 = 128 KB = Apple L1D. On x86 (32 KB L1D) try 14-16. |
| `SIEVE_NARROW_SUB_SHIFT` *(make: `NARROW_SUB_SHIFT`)* | 16 | Narrow-entry transient band size as log2 bytes. 16 = 64 KB, the measured optimum on the M3 Ultra. |
| `SIEVE_TWO_PASS` | 0 | Experimental split of the bucket loop into a scatter pass and a forwarding pass. Measured 3-5% slower than the fused loop; kept for reference. |
| `SIEVE_M2_MULT` | 64 | Sub-segment length in stencil periods ($64 \times 13860 = 887{,}040$). Swept 8-128; 64 is optimal on the tested machines. |
| `SIEVE_M3_MULT` | 90 | Direct-sieve cutoff used by the optional finalized block walk, in stencil periods. `-DSIEVE_M2_MULT=8 -DSIEVE_M3_MULT=12` gives an L1-sized configuration for small machines. |
| `SIEVE_FINALIZE_BLOCK_WALK` | 0 | Use the older rolling block walk for separately finalized Mobius sieves. The newer tiled path is 1.5-8.9% faster on the M3 Ultra across tested segment sizes and sieve heights; retain this switch for retesting on other hardware. |
| `SIEVE_FUSED_FINALIZE` | 1 | Folds the byte-log finalization into the Mertens prefix scan instead of running a separate pass over the segment. |
| `LP_ROUTE_THRESHOLD` | 0 on ARM, 15000 on x86 | Scheduler-construction routing: when `numSegments * LP_ROUTE_THRESHOLD >= numLargePrimes`, the first-hit scan is done in a single pass. See the comment in `SegmentedMobiusSieve.cpp`. |
| `SIEVE_M1_MULT` | 8 | M1-stage unit size in stencil periods ($8 \times 13860 = 110{,}880$ bytes, sized for the tested 128 KB L1D). |
| `SIEVE_M1_PRIME_CAP` | 32000 | Upper bound for M1 medium primes processed by the four-stream M1 walk. This is hardware-dependent and can be retuned independently of the stencil period. |
| `SIEVE_LP_SIZE` | 512 | Bucket ring size (power of two, up to 1024). The largest schedulable prime is `LP_SIZE * M2`, so this sets the sieve's range cap: 512 gives $2.06 \times 10^{17}$, 1024 gives $8.25 \times 10^{17}$. MertensHurst's runtime cap on $u$ picks this up automatically. |
| `SIEVE_STRIDE_LOG` | 8 | Compressed-Mertens stride as log2 ($2^8 = 256$). 7 makes residual overflow impossible at the cost of doubling the coarse array; 9-10 shrink it further for small ranges. See `PERFORMANCE.md` §8. |

The one true constant is `STENCIL_PERIOD` (13860): the pre-sieved stencil data is generated for exactly that period, so it is not a knob.
