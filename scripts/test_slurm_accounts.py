#!/usr/bin/env python3
"""Static checks for Deucalion ARM/x86 account selection."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARM_ACCOUNT = "f202500001hpcvlabepicurea"
X86_ACCOUNT = "f202500001hpcvlabepicurex"


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_server_sweep_defaults_to_arm_account():
    text = read("slurm/run-track-a1-server-sweep.sh")
    assert f"#SBATCH --account={ARM_ACCOUNT}" in text
    assert "#SBATCH --partition=normal-arm" in text


def test_x86_wrappers_use_x86_account():
    assert f"#SBATCH --account={X86_ACCOUNT}" in read("slurm/build-llama-x86.sh")
    assert f"#SBATCH --account={X86_ACCOUNT}" in read("slurm/run-llama-x86.sh")


def test_submit_helper_defaults():
    text = read("slurm/submit-llama.sh")
    assert f'ARM_ACCOUNT:-{ARM_ACCOUNT}' in text
    assert f'X86_ACCOUNT:-{X86_ACCOUNT}' in text


def main():
    test_server_sweep_defaults_to_arm_account()
    test_x86_wrappers_use_x86_account()
    test_submit_helper_defaults()
    print("ok")


if __name__ == "__main__":
    main()
