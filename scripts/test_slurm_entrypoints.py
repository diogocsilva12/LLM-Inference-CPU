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
    path = "slurm/run-track-a1-server-sweep.sh"
    assert_has_sbatch_header(path)
    assert_refuses_direct_login_execution(path)


def test_readme_uses_sbatch_for_slurm_entrypoints():
    text = read("README.md")
    assert "sbatch --export=ALL slurm/run-track-a1-server-sweep.sh" in text
    assert "MODEL_SPECS=" in text
    assert "mandatory_answer_quality.csv" in text


def test_server_sweep_includes_engine_and_cache_dimensions():
    text = read("slurm/run-track-a1-server-sweep.sh")
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


def main():
    test_server_sweep_is_slurm_job()
    test_readme_uses_sbatch_for_slurm_entrypoints()
    test_server_sweep_includes_engine_and_cache_dimensions()
    print("ok")


if __name__ == "__main__":
    main()
