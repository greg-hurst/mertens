# Möbius μ and Mertens M Computations

This repository was used to first compute

$$M(10^{24}) = 7{,}189{,}337{,}839$$
$$M(10^{25}) = -258{,}560{,}632{,}948,$$

the current records for the Mertens function $M(n) = \sum_{k=1}^{n} \mu(k)$. The code is high-performance, tuned primarily for Apple Silicon, and accompanies the paper [*Practical computations of the Mertens function: $M(10^{24})$ and $M(10^{25})$*](https://arxiv.org/abs/XXXX.XXXXX).

The repository contains two independent implementations of the Mertens function and a shared segmented Möbius/Mertens sieve library. The `sieve/` directory is usable independently of the record computations and provides a standalone segmented Möbius/Mertens sieve for computing $\mu(k)$ and $M(n)$ over intervals.

| Component | Algorithm | Complexity | Known Range | Reference |
|-----------|-----------|------------|-------|-----------|
| [MertensHurst](MertensHurst/) | Hurst | $O(n^{2/3})$ | $10^8 \le n \le 10^{25}$ | |
| [MertensHT](MertensHT/) | Helfgott-Thompson | $O(n^{3/5})$ | tested up to $10^{23}$ | [Helfgott-Thompson 2021](https://arxiv.org/abs/2101.08773) |
| [sieve](sieve/) | Segmented Möbius sieve | $O(N \log \log N)$ | $2^{64}$ | |

## Which tool should I use?

- Use [MertensHurst](MertensHurst/) for the classical isolated-value implementation used for the record computations.
- Use [MertensHT](MertensHT/) for the optimized Helfgott-Thompson implementation and comparisons.
- Use [sieve](sieve/) if you want a standalone segmented Möbius/Mertens sieve for computing $\mu(k)$ or $M(n)$ over intervals.
- Use [GPU_experimental](GPU_experimental/) only for experimental Metal GPU variants.

## Quick start

Each component builds on its own — install only what it needs. Plain `make` builds both implementations and needs everything.

**MertensHurst** — OpenMP only:

```bash
brew install libomp        # Linux / WSL: sudo apt install build-essential
make hurst

./MertensHurst/build/mertens 1e12
# M(1000000000000) = 62366 in 0.03 seconds
```

**MertensHT** — OpenMP and Boost (header-only):

```bash
brew install libomp boost   # Linux / WSL: sudo apt install build-essential libboost-dev
make ht

./MertensHT/build/mertens_ht 1e12
# M(1000000000000) = 62366 in 0.12 seconds
```

**Sieve demos** — OpenMP only:

```bash
brew install libomp        # Linux / WSL: sudo apt install build-essential
make sieve

./sieve/build/demo_mertens_value 1000000000
# M(1000000000) = -222
```

## Documentation map

For general use:

- [README.md](README.md) — project overview and quick start.
- [sieve/README.md](sieve/README.md) — standalone Möbius/Mertens sieve library.

For build and tuning options:

- [MertensHurst/COMPILE_FLAGS.md](MertensHurst/COMPILE_FLAGS.md)
- [MertensHT/COMPILE_FLAGS.md](MertensHT/COMPILE_FLAGS.md)
- [sieve/COMPILE_FLAGS.md](sieve/COMPILE_FLAGS.md)

For implementation details:

- [MertensHurst/INPUT_BOUNDS.md](MertensHurst/INPUT_BOUNDS.md)
- [MertensHT/OPTIMIZATIONS.md](MertensHT/OPTIMIZATIONS.md)
- [sieve/PERFORMANCE.md](sieve/PERFORMANCE.md)

Experimental:

- [GPU_experimental/*/README.md](GPU_experimental/)

## Platform

Developed and tuned on Apple Silicon (the record runs used an M3 Ultra). x86-64 works too: the sieve's SIMD finalization auto-selects NEON/SVE2 on ARM and SSE2/AVX2/AVX-512 on x86, and the Makefiles default to the division-free quotient path on x86, where hardware division is slow. The x86 configuration was exercised on a 28-core Xeon W for the paper's cross-machine comparisons.

macOS and Linux build with the same `make` command. On Windows, use WSL — MSVC is not supported (the code relies on `__int128` and other GCC/Clang extensions), and native MinGW builds are not recommended because MertensHT assumes a 64-bit `long`, which Windows toolchains do not provide. The GPU experiments are macOS-only (Metal).

## Building

```bash
make          # build both MertensHurst and MertensHT
make hurst    # build MertensHurst only
make ht       # build MertensHT only
make sieve    # build sieve demo programs
make clean    # remove all build artifacts
```

Binaries are placed in each component's `build/` directory. Both implementations require a **C++17 compiler** (clang++ or g++) and **OpenMP**; MertensHT additionally requires **Boost** (header-only). Build-time knobs are documented in each component's `COMPILE_FLAGS.md`.

## Sieve demos

The [sieve library](sieve/) can also be used independently for computing $\mu(k)$ and $M(n)$ over arbitrary intervals. It includes standalone demo programs:

```bash
make sieve
./sieve/build/demo_mertens_value 1000000       # M(10^6) = 212
./sieve/build/demo_mobius_values 20            # prints mu(1)..mu(20)
```

See [sieve/README.md](sieve/README.md) for the full API and [sieve/PERFORMANCE.md](sieve/PERFORMANCE.md) for runtime characteristics and internal limits.

## GPU experiments

[GPU_experimental](GPU_experimental/) contains GPU ports of both implementations, behind the GPU results in the paper's Appendix D. This code is experimental: it has not been scrutinized to the level of the CPU implementations, and it is Metal only (AMD GPUs — no CUDA). It exists for exploration, and maybe a real implementation down the road.

The GPU implementations are not required for the main record computations.

## Citing

If you use the mathematical results, algorithmic description, or validation from this project, please cite the paper:

> Greg Hurst, *Practical computations of the Mertens function: $M(10^{24})$ and $M(10^{25})$*, arXiv:XXXX.XXXXX, 2026.

If you use the code or standalone sieve library, please also cite this repository using the repository's [`CITATION.cff`](CITATION.cff) file.

## References

- H. A. Helfgott and L. Thompson, [*Summing $\mu(n)$: a faster elementary algorithm*](https://arxiv.org/abs/2101.08773), 2021. The algorithm implemented in [MertensHT](MertensHT/).
- G. Hurst, [*Practical computations of the Mertens function: $M(10^{24})$ and $M(10^{25})$*](https://arxiv.org/abs/XXXX.XXXXX), 2026. — the paper this repository accompanies.

## License

[MIT](LICENSE)
