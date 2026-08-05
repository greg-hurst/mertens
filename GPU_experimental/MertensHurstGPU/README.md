# MertensHurstGPU (experimental)

A GPU-offload experiment for MertensHurst: the S2 phase runs as Metal compute
kernels while the CPU keeps the sieve and S1. This is the code behind the GPU
results in Appendix D of the paper.

Experimental means experimental — it has not been scrutinized to the level of
the CPU implementations. It exists for exploration, and maybe a real
implementation down the road.

This directory keeps its own copy of the MertensHurst sources (`src/`) so the
CPU tree stays GPU-free. It is based on the `v1.0` tag and uses the original
all-$Q=2$ S1/S2 path, not the current CPU implementation's outer-$Q=6$
transforms. Only the sieve is shared, so current sieve changes also apply here.

## Platform

Metal only, so macOS only: Apple Silicon, and the dual AMD GPUs of a 2019
Mac Pro (multi-GPU is supported). No CUDA.

## Building and running

```
brew install libomp
make

./build/mertens_gpu 1e16 --profile
```

The CLI is the same as the CPU `mertens` (see
[MertensHurst/README.md](../../MertensHurst/README.md)). The GPU path is
controlled at runtime with environment variables:

- `MERTENS_GPU_S2=0` — disable the GPU path (falls back to CPU S2)
- `MERTENS_GPU_DEVICES=0,1` — select which Metal devices to use
- `MERTENS_GPU_S2_CPUFRAC=<f>` — keep a fraction of the S2 work on the CPU
- `MERTENS_GPU_PROF=1` — print per-kernel GPU timings
