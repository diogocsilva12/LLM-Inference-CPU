# SLURM scripts for llama.cpp workflows

This folder provides both direct sbatch files and a profile-based wrapper.

## Script overview

### Existing direct scripts

- `build-llama.sh`
  - Build TurboQuant fork on ARM CPU nodes.
- `run-llama.sh`
  - Run TurboQuant fork server on ARM CPU nodes.
- `build-llama-a100.sh`
  - Build TurboQuant fork with CUDA on A100 nodes.
- `run-llama-a100.sh`
  - Run TurboQuant fork server on A100 nodes.
- `build-llama-x86.sh`
  - Build TurboQuant fork on x86 CPU nodes.
- `run-llama-x86.sh`
  - Run TurboQuant fork server on x86 CPU nodes.

### Matrix + wrapper scripts

- `build-llama-matrix.sh`
  - Incremental matrix build (arm-cpu, x86-cpu, a100-cuda).
- `run-llama-matrix.sh`
  - Matrix run script for the same targets.
- `submit-llama.sh`
  - Wrapper that submits matrix build/run with the right sbatch options.

### Vanilla (upstream) scripts

- `clone-vanilla-llama.sh`
  - Clone upstream `ggml-org/llama.cpp` into `llamacpp-vanilla/llama.cpp`.
- `build-vanilla-llama.sh`
  - Build upstream clone.
- `run-vanilla-llama.sh`
  - Run upstream clone server.

## Quick start

From the repository root, first make scripts executable once:

```bash
chmod +x slurm/*.sh
```

## Build for CUDA (A100) and run for CUDA

### Option A: direct A100 scripts

Build:

```bash
sbatch slurm/build-llama-a100.sh
```

Run:

```bash
sbatch slurm/run-llama-a100.sh
```

### Option B: wrapper + matrix (recommended)

Build:

```bash
slurm/submit-llama.sh build a100-cuda
```

Run:

```bash
slurm/submit-llama.sh run a100-cuda
```

You can pass a model path directly:

```bash
slurm/submit-llama.sh run a100-cuda /absolute/path/model.gguf
```

## x86 build/run (explicit sbatch files)

Build:

```bash
sbatch slurm/build-llama-x86.sh
```

Run:

```bash
sbatch slurm/run-llama-x86.sh
```

If your x86 partition/account differ, override at submission time:

```bash
sbatch -A <account> -p <x86-partition> slurm/build-llama-x86.sh
sbatch -A <account> -p <x86-partition> slurm/run-llama-x86.sh
```

## ARM CPU build/run

```bash
sbatch slurm/build-llama.sh
sbatch slurm/run-llama.sh
```

## Unified wrapper usage

Build:

```bash
slurm/submit-llama.sh build arm-cpu
slurm/submit-llama.sh build x86-cpu
slurm/submit-llama.sh build a100-cuda
```

Run:

```bash
slurm/submit-llama.sh run arm-cpu
slurm/submit-llama.sh run x86-cpu
slurm/submit-llama.sh run a100-cuda
```

Prebuild common targets:

```bash
slurm/submit-llama.sh prebuild
```

## Incremental builds and prebuilt binaries

To avoid rebuilding everything every time:

- Use matrix builds and keep build directories:
  - `build-matrix/arm64-cpu`
  - `build-matrix/x86_64-cpu`
  - `build-matrix/x86_64-cuda`
- Re-run build commands after code changes; CMake only recompiles changed files.
- Do not delete build directories unless you need a clean rebuild.
- `ccache` is auto-used by matrix build when available.

## One-hop GUI tunnel

Run logs print a ready-to-copy command:

```bash
ssh deucalion -L <local_port>:<node>:<remote_port>
```

Then open:

```text
http://localhost:<local_port>
```

## Notes

- CUDA build is required to use A100 effectively.
- CPU-only runs can reuse a CUDA binary by setting `N_GPU_LAYERS=0`.
- x86 and ARM binaries are architecture-specific, so build once per architecture.
