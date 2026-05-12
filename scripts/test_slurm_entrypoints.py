#!/usr/bin/env python3
"""Static checks for SLURM-only benchmark entrypoints."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def assert_has_sbatch_header(path):
    text = read(path)
    first_lines = "\n".join(text.splitlines()[:12])
    assert "#SBATCH --job-name=" in first_lines, f"{path} is missing a SLURM job header"


def assert_refuses_direct_login_execution(path):
    text = read(path)
    assert "SLURM_JOB_ID" in text, f"{path} does not check whether it is running inside SLURM"
    assert "sbatch" in text, f"{path} does not resubmit itself through sbatch"


def test_server_sweep_is_slurm_job():
    path = "slurm/sweep_benchmark/run-track-a1-server-sweep.sh"
    assert_has_sbatch_header(path)
    assert_refuses_direct_login_execution(path)


def test_x86_server_sweep_wrapper():
    path = "slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh"
    assert_has_sbatch_header(path)
    text = read(path)
    assert "normal-x86" in text
    assert "f202500001hpcvlabepicurex" in text
    assert "#SBATCH --cpus-per-task=128" in text
    assert 'ENGINE_BUILD_DIR_NAME="${ENGINE_BUILD_DIR_NAME:-build-slurm-x86}"' in text
    assert 'THREADS_LIST="${THREADS_LIST:-8 16 24 32 48 64 96 128}"' in text
    assert 'OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-server-sweep-x86}"' in text
    assert 'SLURM_SUBMIT_DIR' in text
    assert 'A1_SERVER_SWEEP_SUBMIT_SCRIPT' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_vanilla_blas_build_scripts():
    openblas = read("slurm/config/build-vanilla-openblas-fujitsu.sh")
    assert "OpenBLAS/0.3.26-GCC-13.3.0-Fujitsu" in openblas
    assert "GGML_BLAS=ON" in openblas
    assert "GGML_BLAS_VENDOR=OpenBLAS" in openblas
    assert "build-blas/openblas-fujitsu" in openblas

    blis = read("slurm/config/build-vanilla-blis.sh")
    assert "BLIS/1.0-GCC-13.3.0" in blis
    assert "GGML_BLAS=ON" in blis
    assert "GGML_BLAS_VENDOR=FLAME" in blis
    assert "build-blas/blis" in blis


def test_vanilla_blas_sweep_wrapper():
    path = "slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh"
    assert_has_sbatch_header(path)
    text = read(path)
    assert "normal-arm" in text
    assert "vanilla-openblas-fujitsu" in text
    assert "vanilla-blis" in text
    assert "build-blas/openblas-fujitsu" in text
    assert "build-blas/blis" in text
    assert "OpenBLAS/0.3.26-GCC-13.3.0-Fujitsu" in text
    assert "BLIS/1.0-GCC-13.3.0" in text
    assert 'OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-vanilla-blas-sweep}"' in text
    assert 'A1_SERVER_SWEEP_SUBMIT_SCRIPT' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_readme_uses_sbatch_for_slurm_entrypoints():
    text = read("README.md")
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh" in text
    assert "MODEL_SPECS=" in text
    assert "mandatory_answer_quality.csv" in text


def test_server_sweep_includes_engine_and_cache_dimensions():
    text = read("slurm/sweep_benchmark/run-track-a1-server-sweep.sh")
    assert "ENGINE_SPECS" in text
    assert "llamacpp-tq/llama-cpp-turboquant" in text
    assert "llamacpp-vanilla/llama.cpp" in text
    assert "ENGINE_CACHE_SWEEPS" in text
    assert "tq=turbo3:turbo3 turbo4:turbo4" in text
    assert "vanilla=f16:f16" in text
    assert "turbo3:turbo3" in text
    assert 'RUN_CACHE_SWEEP="${RUN_CACHE_SWEEP:-0}"' in text
    assert 'LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-3}"' in text
    assert 'MANDATORY_ONLY="${MANDATORY_ONLY:-0}"' in text


def test_server_sweep_supports_slurm_array_mode():
    text = read("slurm/sweep_benchmark/run-track-a1-server-sweep.sh")
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "SLURM_ARRAY_JOB_ID" in text
    assert "LIST_CONFIGS" in text
    assert "SUMMARY_ONLY" in text
    assert "A1_SERVER_SWEEP_SUBMIT_SCRIPT" in text
    assert "build_config_matrix" in text
    assert "run_config_index" in text
    assert "--array=0-" in read("README.md")


def test_server_sweep_disables_fit_and_tails_failed_server_log():
    text = read("slurm/sweep_benchmark/run-track-a1-server-sweep.sh")
    assert 'FIT_PARAMS="${FIT_PARAMS:-off}"' in text
    assert '-fit "$FIT_PARAMS"' in text
    assert "tail -n 80" in text
    assert "server.log tail" in text
    assert "acquire_build_lock" in text
    assert ".build-slurm.lock" in text


def main():
    test_server_sweep_is_slurm_job()
    test_x86_server_sweep_wrapper()
    test_vanilla_blas_build_scripts()
    test_vanilla_blas_sweep_wrapper()
    test_readme_uses_sbatch_for_slurm_entrypoints()
    test_server_sweep_includes_engine_and_cache_dimensions()
    test_server_sweep_supports_slurm_array_mode()
    test_server_sweep_disables_fit_and_tails_failed_server_log()
    print("ok")


if __name__ == "__main__":
    main()
