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
    assert 'ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-vanilla-x86=f16:f16}"' in text
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
    assert 'ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-vanilla-openblas-fujitsu=f16:f16;vanilla-blis=f16:f16}"' in text
    assert "q4_0:q4_0" not in text
    assert 'OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-vanilla-blas-sweep}"' in text
    assert 'A1_SERVER_SWEEP_SUBMIT_SCRIPT' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_turboquant_stress_sweep_wrapper():
    path = "slurm/sweep_benchmark/run-track-a1-turboquant-stress-sweep.sh"
    assert_has_sbatch_header(path)
    text = read(path)
    assert "normal-arm" in text
    assert "llamacpp-tq/llama-cpp-turboquant" in text
    assert 'ENGINE_SPECS="${ENGINE_SPECS:-tq=$PROJECT_DIR/llamacpp-tq/llama-cpp-turboquant}"' in text
    assert 'ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-tq=turbo2:turbo2 turbo3:turbo3 turbo4:turbo4}"' in text
    assert 'THREADS="${THREADS:-46}"' in text
    assert 'THREADS_LIST="${THREADS_LIST:-24 36 46}"' in text
    assert 'CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 8 16 24}"' in text
    assert 'CTX_SIZE_LIST="${CTX_SIZE_LIST:-4096 8192 12288 16384}"' in text
    assert 'MAX_TOKENS_LIST="${MAX_TOKENS_LIST:-128 256 512 768}"' in text
    assert 'RUN_CACHE_SWEEP="${RUN_CACHE_SWEEP:-1}"' in text
    assert 'OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-turboquant-stress-sweep}"' in text
    assert 'A1_SERVER_SWEEP_SUBMIT_SCRIPT' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_focused_server_sweep_wrapper():
    path = "slurm/sweep_benchmark/run-track-a1-server-sweep-focused.sh"
    assert_has_sbatch_header(path)
    text = read(path)
    assert "normal-arm" in text
    assert 'THREADS="${THREADS:-46}"' in text
    assert 'PARALLEL_REQUESTS="${PARALLEL_REQUESTS:-1}"' in text
    assert 'THREADS_LIST="${THREADS_LIST:-24 36 46 48}"' in text
    assert 'CONCURRENCY_LIST="${CONCURRENCY_LIST:-1}"' in text
    assert 'CTX_SIZE_LIST="${CTX_SIZE_LIST:-1024 2048}"' in text
    assert 'MAX_TOKENS_LIST="${MAX_TOKENS_LIST:-128 256}"' in text
    assert 'ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-tq=turbo2:turbo2 turbo4:turbo4;vanilla=f16:f16}"' in text
    assert 'RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-0}"' in text
    assert 'RUN_CONCURRENCY_SWEEP="${RUN_CONCURRENCY_SWEEP:-0}"' in text
    assert 'A1_SERVER_SWEEP_SUBMIT_SCRIPT' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_max_tps_server_sweep_wrapper():
    path = "slurm/sweep_benchmark/run-track-a1-server-sweep-max-tps.sh"
    assert_has_sbatch_header(path)
    text = read(path)
    assert "normal-arm" in text
    assert "hint=nomultithread" in text
    assert 'ENGINE_SPECS="${ENGINE_SPECS:-tq=$TQ_REPO;vanilla-openblas-fujitsu=$VANILLA_REPO}"' in text
    assert 'ENGINE_PRELOAD_SPECS="${ENGINE_PRELOAD_SPECS:-vanilla-openblas-fujitsu=$OPENBLAS_PRELOAD}"' in text
    assert 'ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-tq=turbo2:turbo2 turbo4:turbo4;vanilla-openblas-fujitsu=f16:f16}"' in text
    assert 'THREADS="${THREADS:-46}"' in text
    assert 'CONCURRENCY_LIST="${CONCURRENCY_LIST:-1}"' in text
    assert 'CTX_SIZE_LIST="${CTX_SIZE_LIST:-512 1024 2048}"' in text
    assert 'MANDATORY_ONLY="${MANDATORY_ONLY:-1}"' in text
    assert 'OMP_PLACES="${OMP_PLACES:-cores}"' in text
    assert 'OMP_PROC_BIND="${OMP_PROC_BIND:-close}"' in text
    assert 'GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY:-0-45}"' in text
    assert 'SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:---numa}"' in text
    assert 'A1_SERVER_SWEEP_SUBMIT_SCRIPT' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_vllm_mlc_sweep_wrapper():
    path = "slurm/sweep_benchmark/run-track-a1-vllm-mlc-sweep.sh"
    assert_has_sbatch_header(path)
    text = read(path)
    assert "normal-arm" in text
    assert 'ENGINE_SPECS="${ENGINE_SPECS:-vllm=$PROJECT_DIR/vllm;mlc=$PROJECT_DIR/mclllm}"' in text
    assert 'ENGINE_BIN_SPECS="${ENGINE_BIN_SPECS:-vllm=$VLLM_ADAPTER;mlc=$MLC_ADAPTER}"' in text
    assert 'ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-vllm=f16:f16;mlc=f16:f16}"' in text
    assert 'THREADS="${THREADS:-46}"' in text
    assert 'RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-1}"' in text
    assert 'RUN_THREAD_SWEEP="${RUN_THREAD_SWEEP:-0}"' in text
    assert 'RUN_CONCURRENCY_SWEEP="${RUN_CONCURRENCY_SWEEP:-0}"' in text
    assert 'RUN_CONTEXT_SWEEP="${RUN_CONTEXT_SWEEP:-0}"' in text
    assert 'RUN_DECODE_SWEEP="${RUN_DECODE_SWEEP:-0}"' in text
    assert 'A1_SERVER_SWEEP_SUBMIT_SCRIPT' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_vllm_mlc_sweep_x86_wrapper():
    path = "slurm/sweep_benchmark/run-track-a1-vllm-mlc-sweep-x86.sh"
    assert_has_sbatch_header(path)
    text = read(path)
    assert "normal-x86" in text
    assert "f202500001hpcvlabepicurex" in text
    assert "#SBATCH --cpus-per-task=126" in text
    assert 'THREADS="${THREADS:-126}"' in text
    assert 'ENGINE_SPECS="${ENGINE_SPECS:-vllm=$PROJECT_DIR/vllm;mlc=$PROJECT_DIR/mclllm}"' in text
    assert 'ENGINE_CACHE_SWEEPS="${ENGINE_CACHE_SWEEPS:-vllm=f16:f16;mlc=f16:f16}"' in text
    assert 'RUN_MODEL_SWEEP="${RUN_MODEL_SWEEP:-1}"' in text
    assert 'RUN_THREAD_SWEEP="${RUN_THREAD_SWEEP:-0}"' in text
    assert 'RUN_CONTEXT_SWEEP="${RUN_CONTEXT_SWEEP:-0}"' in text
    assert 'A1_SERVER_SWEEP_SUBMIT_SCRIPT' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_vllm_mlc_adapters_exist():
    vllm_adapter = read("slurm/config/serve-vllm-openai-adapter.sh")
    assert "vllm.entrypoints.openai.api_server" in vllm_adapter
    assert "Cache support (for sweep compatibility): f16" in vllm_adapter
    mlc_adapter = read("slurm/config/serve-mlc-openai-adapter.sh")
    assert "-m mlc_llm serve" in mlc_adapter
    assert "Cache support (for sweep compatibility): f16" in mlc_adapter


def test_node_scaling_sweep_entrypoint():
    path = "slurm/sweep_benchmark/run-track-a1-node-scaling.sh"
    assert_has_sbatch_header(path)
    assert_refuses_direct_login_execution(path)
    text = read(path)
    assert "normal-arm" in text
    assert "#SBATCH --cpus-per-task=46" in text
    assert 'NODE_COUNTS="${NODE_COUNTS:-1 2 4 6 8 16 24 32 64 128 256}"' in text
    assert 'THREADS="${THREADS:-46}"' in text
    assert 'THREADS_BATCH="${THREADS_BATCH:-46}"' in text
    assert 'LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"' in text
    assert 'TRIALS="${TRIALS:-1}"' in text
    assert 'WARMUP_TRIALS="${WARMUP_TRIALS:-1}"' in text
    assert 'N_GPU_LAYERS="${N_GPU_LAYERS:-999}"' in text
    assert 'n_gpu_layers=$N_GPU_LAYERS' in text
    assert 'ENGINE_BUILD_DIR_NAME="${ENGINE_BUILD_DIR_NAME:-build-slurm-rpc}"' in text
    assert 'BUILD_TARGETS="${BUILD_TARGETS:-llama-server rpc-server}"' in text
    assert 'A1_NODE_SCALING_SUBMIT_SCRIPT="${A1_NODE_SCALING_SUBMIT_SCRIPT:-$PROJECT_DIR/slurm/sweep_benchmark/run-track-a1-node-scaling.sh}"' in text
    assert 'A1_NODE_SCALING_SUBMIT_SCRIPT="$A1_NODE_SCALING_SUBMIT_SCRIPT"' in text
    assert '-DGGML_RPC=ON' in text
    assert 'RPC_SERVER_BIN="${RPC_SERVER_BIN:-$LLAMA_REPO/$ENGINE_BUILD_DIR_NAME/bin/rpc-server}"' in text
    assert 'RPC_MAX_SERVERS="${RPC_MAX_SERVERS:-255}"' in text
    assert 'GGML_RPC_MAX_SERVERS=$RPC_MAX_SERVERS' in text
    assert 'run_rpc_worker' in text
    assert 'run_rpc_coordinator' in text
    assert 'rpc_endpoints="$(build_rpc_endpoints)"' in text
    assert '--rpc "$rpc_endpoints"' in text
    assert 'srun --nodes="$NODE_COUNT" --ntasks="$NODE_COUNT"' in text
    assert 'NODE_SWEEP_DEPENDENCY_TYPE="${NODE_SWEEP_DEPENDENCY_TYPE:-afterany}"' in text
    assert 'previous_job_id' in text
    assert '--dependency="${NODE_SWEEP_DEPENDENCY_TYPE}:$previous_job_id"' in text
    assert "Submitting node scaling job with $node_count nodes after job $previous_job_id" in text
    assert "Submitted sequential node-scaling chain" in text
    assert 'rpc_coordinator_${host_safe}' in text
    assert "a1_node_scaling_scaling_summary.csv" in text


def test_gptoss_node_scaling_wrapper():
    path = "slurm/sweep_benchmark/run-track-a1-gptoss-node-scaling.sh"
    assert_has_sbatch_header(path)
    text = read(path)
    assert "a1-gptoss-node-scaling" in text
    assert "/share/chatbot/models/gpt-oss-120b-Q8_0.gguf" in text
    assert 'OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-track-a1-gptoss-120b-node-scaling}"' in text
    assert 'NODE_COUNTS="${NODE_COUNTS:-6 8 12 16 24 32}"' in text
    assert 'SUBMIT_NODE_SWEEP="${SUBMIT_NODE_SWEEP:-1}"' in text
    assert 'CTX_SIZE="${CTX_SIZE:-2048}"' in text
    assert 'MAX_TOKENS="${MAX_TOKENS:-64}"' in text
    assert 'CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"' in text
    assert 'A1_NODE_SCALING_SUBMIT_SCRIPT="${A1_NODE_SCALING_SUBMIT_SCRIPT:-$WRAPPER_SCRIPT}"' in text
    assert 'exec "$BASE_SWEEP_SCRIPT"' in text


def test_readme_uses_sbatch_for_slurm_entrypoints():
    text = read("README.md")
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep-focused.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep-max-tps.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-vllm-mlc-sweep.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-vllm-mlc-sweep-x86.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-server-sweep-x86.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-vanilla-blas-sweep.sh" in text
    assert "SUBMIT_NODE_SWEEP=1 sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-node-scaling.sh" in text
    assert "sbatch --export=ALL slurm/sweep_benchmark/run-track-a1-gptoss-node-scaling.sh" in text
    assert "MODEL_SPECS=" in text
    assert "mandatory_answer_quality.csv" in text


def test_server_sweep_includes_engine_and_cache_dimensions():
    text = read("slurm/sweep_benchmark/run-track-a1-server-sweep.sh")
    assert "ENGINE_SPECS" in text
    assert "llamacpp-tq/llama-cpp-turboquant" in text
    assert "llamacpp-vanilla/llama.cpp" in text
    assert "ENGINE_CACHE_SWEEPS" in text
    assert "tq=turbo2:turbo2 turbo3:turbo3 turbo4:turbo4" in text
    assert "vanilla=f16:f16\"" in text
    assert "turbo3:turbo3" in text
    assert 'RUN_CACHE_SWEEP="${RUN_CACHE_SWEEP:-0}"' in text
    assert 'LIMIT_PER_CATEGORY="${LIMIT_PER_CATEGORY:-5}"' in text
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


def test_server_sweep_collects_node_resources():
    text = read("slurm/sweep_benchmark/run-track-a1-server-sweep.sh")
    assert 'ENABLE_NODE_RESOURCE_MONITOR="${ENABLE_NODE_RESOURCE_MONITOR:-1}"' in text
    assert 'RESOURCE_MONITOR_INTERVAL="${RESOURCE_MONITOR_INTERVAL:-1}"' in text
    assert "start_node_resource_monitor" in text
    assert "node_resources.csv" in text
    assert "dstat --time --cpu --mem --output" in text
    assert "/proc/meminfo" in text


def test_node_scaling_collects_node_resources():
    text = read("slurm/sweep_benchmark/run-track-a1-node-scaling.sh")
    assert 'ENABLE_NODE_RESOURCE_MONITOR="${ENABLE_NODE_RESOURCE_MONITOR:-1}"' in text
    assert 'RESOURCE_MONITOR_INTERVAL="${RESOURCE_MONITOR_INTERVAL:-1}"' in text
    assert "start_node_resource_monitor" in text
    assert "node_resources.csv" in text
    assert "dstat --time --cpu --mem --output" in text
    assert "/proc/meminfo" in text


def main():
    test_server_sweep_is_slurm_job()
    test_x86_server_sweep_wrapper()
    test_vanilla_blas_build_scripts()
    test_vanilla_blas_sweep_wrapper()
    test_turboquant_stress_sweep_wrapper()
    test_focused_server_sweep_wrapper()
    test_max_tps_server_sweep_wrapper()
    test_vllm_mlc_sweep_wrapper()
    test_vllm_mlc_sweep_x86_wrapper()
    test_vllm_mlc_adapters_exist()
    test_node_scaling_sweep_entrypoint()
    test_readme_uses_sbatch_for_slurm_entrypoints()
    test_server_sweep_includes_engine_and_cache_dimensions()
    test_server_sweep_supports_slurm_array_mode()
    test_server_sweep_disables_fit_and_tails_failed_server_log()
    test_server_sweep_collects_node_resources()
    test_node_scaling_collects_node_resources()
    print("ok")


if __name__ == "__main__":
    main()
