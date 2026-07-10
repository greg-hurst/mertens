# Roadmap

This repository contains the code used for the computations reported in

Greg Hurst, *Practical computations of the Mertens function: $M(10^{24})$ and $M(10^{25})$*, arXiv:2607.07566.

The `v1.0` release is the code snapshot associated with the paper. The items below are possible future improvements, not scheduled promises.

## On my radar

These are the main things I am actively thinking about.

### Long-run control

- Add interrupt/cancellation support for long computations.
- Add low-overhead progress callbacks for long sieve and isolated-value runs.

### Performance and input ranges

- Add better support for small arguments, including arguments below $10^8$.
- Improve runtimes for small and medium inputs, roughly below $10^{16}$.
- Give x86-64 performance more attention, including AVX2 and AVX-512 systems.
- Scrutinize MertensHT on inputs beyond $10^{23}$ more, and make sure all needed safe-guards are in place.
- Tune architecture-dependent defaults, such as quotient strategy, segment sizes, bucket sizes, prefetch distances, and scheduling policies.
- Improve true single-threaded performance, beyond simply running the OpenMP code with one thread.

### Reusable components

- Spin out the quotient predictor as a standalone documented primitive.
- Spin out the large-prime bucket scheduler as a reusable component or example.

### Algorithmic experiments

- Explore deeper inclusion-exclusion reductions.
- Add an option, and possibly better defaults, for using deeper inclusion-exclusion when it is practically beneficial.

### GPU experiments

- Extend the experimental GPU work beyond Metal to CUDA.
- Determine which parts of the computation, if any, are worth maintaining as GPU-supported paths.

## Nice to have

These are lower-priority improvements that would make the project easier to use, test, or extend.

- Add quick validation tests using known values of $M(x)$.
- Add benchmark scripts with standardized output.
- Add basic CI build checks for Linux and macOS.
- Add clearer examples for using the standalone Möbius/Mertens sieve as a library.
- Add a short architecture overview for the full codebase.
- Add issue templates for bug reports, performance reports, and validation reports.

## Open questions

- How far can deeper inclusion-exclusion be pushed before summand complexity outweighs summand reduction?
- Can quotient prediction become a useful general primitive for exact floor-quotient streams?
- What bucket representation is best on modern x86, Apple Silicon, and GPU-like memory systems?
- Can automatic parameter selection get close to hand-tuned performance?
