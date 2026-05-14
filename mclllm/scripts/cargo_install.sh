#!/bin/bash
#SBATCH --job-name=test-mcl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:10:00
#SBATCH --account=f202500010hpcvlabuminhox
#SBATCH --partition=dev-x86
#SBATCH --output=slurm_logs/job_out_%j
#SBATCH --error=slurm_logs/job_error_%j
#SBATCH --exclusive

set -euo pipefail

PROJECT="/projects/F202500010HPCVLABUMINHO/uminhocp030/heretic-deucalion"

export RUSTUP_HOME="$PROJECT/.rustup"
export CARGO_HOME="$PROJECT/.cargo"

echo ">>> A iniciar instalação do Rust em $CARGO_HOME..."

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

source "$CARGO_HOME/env"

rustup default stable

echo ">>> A criar atalho de ambiente..."

echo "export RUSTUP_HOME=\"$RUSTUP_HOME\"" > "$PROJECT/env_rust.sh"
echo "export CARGO_HOME=\"$CARGO_HOME\"" >> "$PROJECT/env_rust.sh"
echo "source \"\$CARGO_HOME/env\"" >> "$PROJECT/env_rust.sh"

echo ">>> SUCESSO! Versão instalada:"
cargo --version