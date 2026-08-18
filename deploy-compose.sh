#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Docker Compose Deployment Orchestrator for SGLang DSPARK
# =============================================================================
#
# Runs from the control machine. Builds the Docker image on each DGX Spark
# node independently, then starts the worker first and the head second.
#
# Usage:
#   ./deploy-compose.sh           # Deploy with docker compose
#   ./deploy-compose.sh --build   # Force rebuild the image on each node
#   ./deploy-compose.sh --no-wait # Start containers but do not wait for health
#

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

HEAD_NAME="ai1"
WORKER_NAME="ai2"

HEAD_MGMT_IP="${HEAD_MGMT_IP:-192.168.3.120}"
WORKER_MGMT_IP="${WORKER_MGMT_IP:-192.168.3.121}"

SSH_USER="${SSH_USER:-santanu}"

IMAGE_NAME="sglang-dspark"
DEPLOY_DIR="${DEPLOY_DIR:-/home/${SSH_USER}/sglang-dspark-deploy}"

# Source .env if it exists (for local config on the control machine)
if [ -f "$(dirname "$0")/.env" ]; then
    set -a
    source "$(dirname "$0")/.env"
    set +a
fi

DO_BUILD=false
DO_WAIT=true

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
# Argument Parsing
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            DO_BUILD=true
            shift
            ;;
        --no-wait)
            DO_WAIT=false
            shift
            ;;
        *)
            error "Unknown argument: $1"
            ;;
    esac
done

# Always build if the image does not exist on a node
check_image_on_node() {
    local node_ip="$1"
    if ! ssh "${SSH_USER}@${node_ip}" "docker images ${IMAGE_NAME} --format '{{.Repository}}'" | grep -q "."; then
        return 1
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------

check_ssh() {
    log "Checking SSH connectivity..."
    for node_ip in "${HEAD_MGMT_IP}" "${WORKER_MGMT_IP}"; do
        if ssh -o ConnectTimeout=5 "${SSH_USER}@${node_ip}" "echo ok" >/dev/null 2>&1; then
            log "  SSH to ${node_ip}: OK"
        else
            error "Cannot SSH to ${SSH_USER}@${node_ip}. Set up passwordless SSH first."
        fi
    done
}

check_docker() {
    log "Checking Docker on both nodes..."
    for node_ip in "${HEAD_MGMT_IP}" "${WORKER_MGMT_IP}"; do
        ssh "${SSH_USER}@${node_ip}" "docker info" >/dev/null 2>&1 || \
            error "Docker not running on ${node_ip}."
        ssh "${SSH_USER}@${node_ip}" "docker compose version" >/dev/null 2>&1 || \
            error "Docker Compose plugin not installed on ${node_ip}."
    done
    log "  Docker and Docker Compose: OK on both nodes."
}

check_model() {
    local model_path="${MODEL_PATH:-/home/${SSH_USER}/data/models/DeepSeek-V4-Flash-0731}"
    log "Checking model path on both nodes..."
    for node_ip in "${HEAD_MGMT_IP}" "${WORKER_MGMT_IP}"; do
        ssh "${SSH_USER}@${node_ip}" "[ -d ${model_path} ]" || \
            error "Model not found at ${model_path} on ${node_ip}."
    done
    log "  Model directory: OK on both nodes."
}

# -----------------------------------------------------------------------------
# Sync Files to Nodes
# -----------------------------------------------------------------------------

sync_files() {
    log "Syncing deployment files to both nodes..."

    local script_dir
    script_dir="$(cd "$(dirname "$0")" && pwd)"

    for node_ip in "${HEAD_MGMT_IP}" "${WORKER_MGMT_IP}"; do
        ssh "${SSH_USER}@${node_ip}" "mkdir -p ${DEPLOY_DIR}/patches"

        # Copy Dockerfile
        scp "${script_dir}/Dockerfile" \
            "${SSH_USER}@${node_ip}:${DEPLOY_DIR}/Dockerfile"

        # Copy docker-compose files
        scp "${script_dir}/docker-compose.head.yml" \
            "${SSH_USER}@${node_ip}:${DEPLOY_DIR}/docker-compose.head.yml"
        scp "${script_dir}/docker-compose.worker.yml" \
            "${SSH_USER}@${node_ip}:${DEPLOY_DIR}/docker-compose.worker.yml"

        # Copy patch script
        scp "${script_dir}/patches/apply_dspark_patch.sh" \
            "${SSH_USER}@${node_ip}:${DEPLOY_DIR}/patches/apply_dspark_patch.sh"

        # Copy .env if it exists, otherwise copy .env.example
        if [ -f "${script_dir}/.env" ]; then
            scp "${script_dir}/.env" \
                "${SSH_USER}@${node_ip}:${DEPLOY_DIR}/.env"
        elif [ -f "${script_dir}/.env.example" ]; then
            scp "${script_dir}/.env.example" \
                "${SSH_USER}@${node_ip}:${DEPLOY_DIR}/.env"
        fi

        log "  Files synced to ${node_ip}."
    done
}

# -----------------------------------------------------------------------------
# Build Image on Each Node
# -----------------------------------------------------------------------------

build_image() {
    local node_ip="$1"
    local node_name="$2"

    log "Building ${IMAGE_NAME} on ${node_name} (${node_ip})..."
    ssh "${SSH_USER}@${node_ip}" \
        "cd ${DEPLOY_DIR} && docker build -t ${IMAGE_NAME} ."
    log "  Build complete on ${node_name}."
}

# -----------------------------------------------------------------------------
# Deploy with Docker Compose
# -----------------------------------------------------------------------------

stop_existing() {
    log "Stopping existing containers on both nodes..."

    ssh "${SSH_USER}@${HEAD_MGMT_IP}" \
        "cd ${DEPLOY_DIR} && docker compose -f docker-compose.head.yml down 2>/dev/null" || true
    ssh "${SSH_USER}@${WORKER_MGMT_IP}" \
        "cd ${DEPLOY_DIR} && docker compose -f docker-compose.worker.yml down 2>/dev/null" || true

    # Also remove any leftover containers from deploy.sh
    for node_ip in "${HEAD_MGMT_IP}" "${WORKER_MGMT_IP}"; do
        ssh "${SSH_USER}@${node_ip}" \
            "docker rm -f sglang-head sglang-worker 2>/dev/null" || true
    done

    log "  Existing containers stopped."
}

deploy_compose() {
    # Start worker first
    log "Starting SGLang worker on ${WORKER_NAME} (${WORKER_MGMT_IP})..."
    ssh "${SSH_USER}@${WORKER_MGMT_IP}" \
        "cd ${DEPLOY_DIR} && docker compose -f docker-compose.worker.yml up -d"

    log "Worker started. Waiting 10 seconds before starting head..."
    sleep 10

    # Start head
    log "Starting SGLang head on ${HEAD_NAME} (${HEAD_MGMT_IP})..."
    ssh "${SSH_USER}@${HEAD_MGMT_IP}" \
        "cd ${DEPLOY_DIR} && docker compose -f docker-compose.head.yml up -d"

    log "Head started."
}

# -----------------------------------------------------------------------------
# Wait for Ready
# -----------------------------------------------------------------------------

wait_for_ready() {
    local port="${SGLANG_PORT:-8000}"
    log "Waiting for head node to become ready (this may take 5-10 minutes)..."
    local elapsed=0
    local max_wait=1800

    while [ ${elapsed} -lt ${max_wait} ]; do
        if curl -sf "http://${HEAD_MGMT_IP}:${port}/health" >/dev/null 2>&1; then
            log "Head node is ready!"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        log "  Still waiting... (${elapsed}s / ${max_wait}s)"

        if ! ssh "${SSH_USER}@${HEAD_MGMT_IP}" \
            "docker ps --filter name=sglang-head --format '{{.Names}}'" | grep -q "."; then
            log "WARNING: Head container is not running. Check logs:"
            log "  ssh ${SSH_USER}@${HEAD_MGMT_IP} docker logs sglang-head"
            error "Head container exited before becoming ready."
        fi
        if ! ssh "${SSH_USER}@${WORKER_MGMT_IP}" \
            "docker ps --filter name=sglang-worker --format '{{.Names}}'" | grep -q "."; then
            log "WARNING: Worker container is not running. Check logs:"
            log "  ssh ${SSH_USER}@${WORKER_MGMT_IP} docker logs sglang-worker"
            error "Worker container exited before becoming ready."
        fi
    done

    error "Server did not become ready within ${max_wait} seconds."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    log "=== SGLang DSPARK Docker Compose Deployment ==="
    log "Head node: ${HEAD_NAME} (${HEAD_MGMT_IP})"
    log "Worker node: ${WORKER_NAME} (${WORKER_MGMT_IP})"
    log "Image: ${IMAGE_NAME}"
    log "Deploy dir on nodes: ${DEPLOY_DIR}"
    log "Build: ${DO_BUILD}"
    log ""

    check_ssh
    check_docker
    check_model

    sync_files

    # Build on each node independently
    for node_ip in "${HEAD_MGMT_IP}" "${WORKER_MGMT_IP}"; do
        local node_name
        if [ "${node_ip}" = "${HEAD_MGMT_IP}" ]; then
            node_name="${HEAD_NAME}"
        else
            node_name="${WORKER_NAME}"
        fi

        if [ "${DO_BUILD}" = true ]; then
            build_image "${node_ip}" "${node_name}"
        elif ! check_image_on_node "${node_ip}"; then
            log "Image ${IMAGE_NAME} not found on ${node_name}. Building..."
            build_image "${node_ip}" "${node_name}"
        else
            log "Image ${IMAGE_NAME} already exists on ${node_name}. Skipping build."
        fi
    done

    stop_existing
    deploy_compose

    if [ "${DO_WAIT}" = true ]; then
        wait_for_ready
    fi

    local port="${SGLANG_PORT:-8000}"
    log ""
    log "=== Deployment Complete ==="
    log "API endpoint: http://${HEAD_MGMT_IP}:${port}"
    log ""
    log "Test with:"
    log "  curl http://${HEAD_MGMT_IP}:${port}/health"
    log "  curl http://${HEAD_MGMT_IP}:${port}/v1/chat/completions \\"
    log "    -H 'Content-Type: application/json' \\"
    log "    -d '{\"model\":\"DeepSeek-V4-Flash-0731\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
    log ""
    log "Monitor logs:"
    log "  ssh ${SSH_USER}@${HEAD_MGMT_IP} docker logs -f sglang-head"
    log "  ssh ${SSH_USER}@${WORKER_MGMT_IP} docker logs -f sglang-worker"
    log ""
    log "Stop with: ./stop.sh"
}

main "$@"
