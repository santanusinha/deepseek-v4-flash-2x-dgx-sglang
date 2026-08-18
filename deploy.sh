#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DeepSeek-V4-Flash-0731 Deployment on 2x DGX Spark Cluster with SGLang
# =============================================================================
#
# Multi-node Tensor Parallel (TP=2) deployment across two DGX Spark nodes.
# Worker node starts first, then head node.
#
# Usage:
#   ./deploy.sh              # Safe mode (no DSPARK, no CUDA graphs)
#   ./deploy.sh dspark       # DSPARK speculative decoding (with topk=192 patch)
#
# Safe mode disables CUDA graphs for multi-node TP stability on DGX Spark.
# DSPARK mode enables speculative decoding with the topk=192 padding workaround.
#

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HEAD_NAME="ai1"
WORKER_NAME="ai2"

HEAD_MGMT_IP="192.168.3.120"
WORKER_MGMT_IP="192.168.3.121"

HEAD_CLUSTER_IP="192.168.100.10"
WORKER_CLUSTER_IP="192.168.100.11"

SSH_USER="santanu"

MODEL_PATH="/home/santanu/data/models/DeepSeek-V4-Flash-0731"
SGLANG_IMAGE="lmsysorg/sglang:latest-cu130"

WORKER_PORT=8000
DIST_INIT_PORT=20000

MEM_FRACTION="0.85"
CONTEXT_LENGTH="131072"
CHUNKED_PREFILL_SIZE="4096"
MAX_RUNNING_REQUESTS="8"
KV_CACHE_DTYPE="fp8_e4m3"

WORKER_CONTAINER="sglang-worker"
HEAD_CONTAINER="sglang-head"

NCCL_IB_HCA="rocep1s0f1"
NCCL_IB_GID_INDEX="3"
NCCL_SOCKET_IFNAME="enp1s0f1np1"
GLOO_SOCKET_IFNAME="enp1s0f1np1"

# NCCL 2.30.7 upgrade path (fixes DSPARK + multi-node TP rank divergence deadlock)
# SGLang issue #33289: torch-bundled NCCL 2.28.9 wedges the graph/eager mix
# NCCL 2.30.7 library is pre-staged at this path on both nodes
NCCL_UPGRADE_PATH="/tmp/nccl-upgrade"
NCCL_UPGRADE_LIB="${NCCL_UPGRADE_PATH}/libnccl.so.2"

DEPLOY_MODE="${1:-safe}"

DSPARK_PATCH_DIR="/tmp/sglang-dspark-patch"
DSPARK_PATCH_FILE="${DSPARK_PATCH_DIR}/flash_mla_sm120.py"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[ERROR] $*" >&2
    exit 1
}

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------

check_ssh() {
    log "Checking SSH connectivity to worker node..."
    if ssh -o ConnectTimeout=5 "${SSH_USER}@${WORKER_MGMT_IP}" "echo ok" >/dev/null 2>&1; then
        log "SSH to worker node: OK"
    else
        error "Cannot SSH to ${SSH_USER}@${WORKER_MGMT_IP}. Set up passwordless SSH first."
    fi
}

check_docker() {
    log "Checking Docker on head node..."
    ssh "${SSH_USER}@${HEAD_MGMT_IP}" "docker info" >/dev/null 2>&1 || \
        error "Docker not running on head node."

    log "Checking Docker on worker node..."
    ssh "${SSH_USER}@${WORKER_MGMT_IP}" "docker info" >/dev/null 2>&1 || \
        error "Docker not running on worker node."
}

check_model() {
    log "Checking model path on head node..."
    ssh "${SSH_USER}@${HEAD_MGMT_IP}" "[ -d ${MODEL_PATH} ]" || \
        error "Model not found at ${MODEL_PATH} on head node."

    log "Checking model path on worker node..."
    ssh "${SSH_USER}@${WORKER_MGMT_IP}" "[ -d ${MODEL_PATH} ]" || \
        error "Model not found at ${MODEL_PATH} on worker node."
}

check_image() {
    ssh "${SSH_USER}@${HEAD_MGMT_IP}" "docker images ${SGLANG_IMAGE} --format '{{.Repository}}:{{.Tag}}'" | grep -q "." || \
        error "SGLang image not found on head node. Pull it first."

    log "Checking SGLang image on worker node..."
    ssh "${SSH_USER}@${WORKER_MGMT_IP}" \
        "docker images ${SGLANG_IMAGE} --format '{{.Repository}}:{{.Tag}}'" | grep -q "." || \
        error "SGLang image not found on worker node. Pull it first."
}

check_nccl_upgrade() {
    log "Checking NCCL 2.30.7 upgrade library on both nodes..."
    for node_ip in "${HEAD_MGMT_IP}" "${WORKER_MGMT_IP}"; do
        if ! ssh "${SSH_USER}@${node_ip}" "[ -f ${NCCL_UPGRADE_LIB} ]"; then
            error "NCCL upgrade library not found at ${NCCL_UPGRADE_LIB} on ${node_ip}. Stage it first."
        fi
        local ver
        ver=$(ssh "${SSH_USER}@${node_ip}" "strings ${NCCL_UPGRADE_LIB} | grep -oP 'NCCL_VERSION_2\\.\\d+\\.\\d+' | head -1" 2>/dev/null || echo "unknown")
        log "  ${node_ip}: ${ver}"
    done
}

stop_existing() {
    log "Stopping existing SGLang containers on head node..."
    ssh "${SSH_USER}@${HEAD_MGMT_IP}" \
        "docker rm -f ${HEAD_CONTAINER} ${WORKER_CONTAINER} 2>/dev/null" || true

    log "Stopping existing SGLang containers on worker node..."
    ssh "${SSH_USER}@${WORKER_MGMT_IP}" \
        "docker rm -f ${HEAD_CONTAINER} ${WORKER_CONTAINER} 2>/dev/null" || true
}

# -----------------------------------------------------------------------------
# DSPARK Patch Preparation
# -----------------------------------------------------------------------------

prepare_dspark_patch() {
    log "Preparing DSPARK topk=192 patch for SM120/SM121..."

    mkdir -p "${DSPARK_PATCH_DIR}"

    ssh "${SSH_USER}@${HEAD_MGMT_IP}" "docker run --rm ${SGLANG_IMAGE} cat \
        /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py" \
        2>/dev/null | awk '/^"""/{found=1} found' \
        > "${DSPARK_PATCH_FILE}"

    if [ ! -s "${DSPARK_PATCH_FILE}" ]; then
        error "Failed to extract flash_mla_sm120.py from SGLang image."
    fi

    if ! grep -q "topk=192\|_pad = 512" "${DSPARK_PATCH_FILE}"; then
        sed -i '/idx = indices.squeeze(1) if indices.dim() == 3 else indices/a\
\
    # DSPARK topk=192 workaround for SM120/SM121 (SGLang issue #33134)\
    if idx.shape[-1] == 192:\
        _pad = 512 - idx.shape[-1]\
        idx = torch.nn.functional.pad(idx, (0, _pad), value=0)' \
            "${DSPARK_PATCH_FILE}"

        log "Patch applied to ${DSPARK_PATCH_FILE}"
    else
        log "Patch already applied."
    fi

    # Copy patch to both nodes (script may run from a control machine)
    for node_ip in "${HEAD_MGMT_IP}" "${WORKER_MGMT_IP}"; do
        ssh "${SSH_USER}@${node_ip}" "mkdir -p ${DSPARK_PATCH_DIR}"
        # Remove stale directory from a previous botched scp
        ssh "${SSH_USER}@${node_ip}" "[ -d ${DSPARK_PATCH_FILE} ] && rm -rf ${DSPARK_PATCH_FILE} || true"
        scp "${DSPARK_PATCH_FILE}" \
            "${SSH_USER}@${node_ip}:${DSPARK_PATCH_FILE}"
        log "Patch copied to ${node_ip}."
    done
}

# -----------------------------------------------------------------------------
# Deployment: Multi-node TP=2
# -----------------------------------------------------------------------------

get_sglang_args() {
    local mode="${1:-safe}"
    local node_rank="${2}"
    local args="--model-path /models/DeepSeek-V4-Flash-0731 \
        --host 0.0.0.0 \
        --port ${WORKER_PORT} \
        --tp 2 \
        --nnodes 2 \
        --node-rank ${node_rank} \
        --dist-init-addr ${HEAD_CLUSTER_IP}:${DIST_INIT_PORT} \
        --dist-timeout 3600 \
        --trust-remote-code \
        --mem-fraction-static ${MEM_FRACTION} \
        --context-length ${CONTEXT_LENGTH} \
        --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
        --max-running-requests ${MAX_RUNNING_REQUESTS} \
        --moe-runner-backend flashinfer_mxfp4 \
        --moe-a2a-backend none \
        --swa-full-tokens-ratio 0.1 \
        --disable-flashinfer-autotune \
        --kv-cache-dtype ${KV_CACHE_DTYPE} \
        --enable-dp-attention \
        --reasoning-parser deepseek-v4 \
        --tool-call-parser deepseekv4 \
        --skip-server-warmup \
        --watchdog-timeout 600"

    if [ "${mode}" = "safe" ]; then
        args="${args} --disable-cuda-graph"
    elif [ "${mode}" = "dspark" ]; then
        args="${args} --speculative-algorithm DSPARK"
        args="${args} --speculative-dspark-block-size 5"
        args="${args} --speculative-moe-runner-backend flashinfer_mxfp4"
        args="${args} --cuda-graph-max-bs-decode ${MAX_RUNNING_REQUESTS}"
        args="${args} --enable-dp-lm-head"
    fi
    echo "${args}"
}

get_docker_flags() {
    local mode="${1:-safe}"
    local extra_mounts=""

    if [ "${mode}" = "dspark" ]; then
        local patch_path="/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py"
        extra_mounts="-v ${DSPARK_PATCH_FILE}:${patch_path}"
    fi

    echo "--gpus all -d \
        --network host \
        --ipc host \
        --shm-size 32g \
        --privileged \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864 \
        --cap-add IPC_LOCK \
        --device /dev/infiniband \
        -v ${MODEL_PATH}:/models/DeepSeek-V4-Flash-0731 \
        -v /dev/shm:/dev/shm \
        -e NCCL_IB_DISABLE=0 \
        -e NCCL_IB_HCA=${NCCL_IB_HCA} \
        -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} \
        -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
        -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} \
        -e NCCL_DEBUG=INFO \
        -e NCCL_CUMEM_ENABLE=0 \
        -e NCCL_IGNORE_CPU_AFFINITY=1 \
        -e SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
        -v ${NCCL_UPGRADE_LIB}:/opt/nccl-upgrade/libnccl.so.2 \
        -e SGLANG_NCCL_SO_PATH=/opt/nccl-upgrade/libnccl.so.2 \
        ${extra_mounts}"
}

deploy_tp() {
    local mode="${DEPLOY_MODE}"
    log "Deploying DeepSeek-V4-Flash-0731 with TP=2 (mode: ${mode})..."

    local docker_flags
    docker_flags=$(get_docker_flags "${mode}")

    local worker_args
    worker_args=$(get_sglang_args "${mode}" 1)

    local head_args
    head_args=$(get_sglang_args "${mode}" 0)

    # Start worker on Node 2 first
    log "Starting SGLang worker on Node 2 (${WORKER_NAME} / ${WORKER_MGMT_IP})..."
    ssh "${SSH_USER}@${WORKER_MGMT_IP}" "docker run ${docker_flags} \
        --name ${WORKER_CONTAINER} \
        ${SGLANG_IMAGE} \
        python3 -m sglang.launch_server ${worker_args}"

    log "Worker node started. Waiting 10 seconds before starting head..."
    sleep 10

    # Start head on Node 1
    log "Starting SGLang head on Node 1 (${HEAD_NAME} / ${HEAD_MGMT_IP})..."
    ssh "${SSH_USER}@${HEAD_MGMT_IP}" "docker run ${docker_flags} \
        --name ${HEAD_CONTAINER} \
        ${SGLANG_IMAGE} \
        python3 -m sglang.launch_server ${head_args}"

    log "Head node started."
    log "API will be available at http://${HEAD_MGMT_IP}:${WORKER_PORT}"
    log "Model loading may take 5-10 minutes. Monitor with:"
    log "  ssh ${SSH_USER}@${HEAD_MGMT_IP} docker logs -f ${HEAD_CONTAINER}"
    log "  ssh ${SSH_USER}@${WORKER_MGMT_IP} docker logs -f ${WORKER_CONTAINER}"
}

# -----------------------------------------------------------------------------
# Wait for Ready
# -----------------------------------------------------------------------------

wait_for_ready() {
    log "Waiting for head node to become ready (this may take 5-10 minutes)..."
    local elapsed=0
    local max_wait=1800

    while [ ${elapsed} -lt ${max_wait} ]; do
        if curl -sf "http://${HEAD_MGMT_IP}:${WORKER_PORT}/health" >/dev/null 2>&1; then
            log "Head node is ready!"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        log "  Still waiting... (${elapsed}s / ${max_wait}s)"

        if ! ssh "${SSH_USER}@${HEAD_MGMT_IP}" \
            "docker ps --filter name=${HEAD_CONTAINER} --format '{{.Names}}'" | grep -q "."; then
            log "WARNING: Head container is not running. Check logs:"
            log "  ssh ${SSH_USER}@${HEAD_MGMT_IP} docker logs ${HEAD_CONTAINER}"
            error "Head container exited before becoming ready."
        fi
        if ! ssh "${SSH_USER}@${WORKER_MGMT_IP}" \
            "docker ps --filter name=${WORKER_CONTAINER} --format '{{.Names}}'" | grep -q "."; then
            log "WARNING: Worker container is not running. Check logs:"
            log "  ssh ${SSH_USER}@${WORKER_MGMT_IP} docker logs ${WORKER_CONTAINER}"
            error "Worker container exited before becoming ready."
        fi
    done

    error "Server did not become ready within ${max_wait} seconds."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    log "=== DeepSeek-V4-Flash-0731 SGLang Multi-Node Deployment ==="
    log "Head node: ${HEAD_NAME} (${HEAD_MGMT_IP}, cluster: ${HEAD_CLUSTER_IP})"
    log "Worker node: ${WORKER_NAME} (${WORKER_MGMT_IP}, cluster: ${WORKER_CLUSTER_IP})"
    log "Model: ${MODEL_PATH}"
    log "Image: ${SGLANG_IMAGE}"
    log "TP=2, nnodes=2, RDMA via ${NCCL_IB_HCA} (GID ${NCCL_IB_GID_INDEX})"
    log "Mode: ${DEPLOY_MODE}"
    log "Context: ${CONTEXT_LENGTH}, KV cache: ${KV_CACHE_DTYPE}"
    log ""

    check_ssh
    check_docker
    check_model
    check_image
    check_nccl_upgrade
    stop_existing

    if [ "${DEPLOY_MODE}" = "dspark" ]; then
        prepare_dspark_patch
    fi

    deploy_tp
    wait_for_ready

    log ""
    log "=== Deployment Complete ==="
    log "API endpoint: http://${HEAD_MGMT_IP}:${WORKER_PORT}"
    log ""
    log "Test with:"
    log "  curl http://${HEAD_MGMT_IP}:${WORKER_PORT}/health"
    log "  curl http://${HEAD_MGMT_IP}:${WORKER_PORT}/v1/chat/completions \\"
    log "    -H 'Content-Type: application/json' \\"
    log "    -d '{\"model\":\"DeepSeek-V4-Flash-0731\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
}

main "$@"
