#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Stop all SGLang containers on both DGX Spark nodes
# =============================================================================
#
# Tries docker compose down first. Falls back to docker rm -f for any
# containers that were started with deploy.sh instead of compose.
#

HEAD_MGMT_IP="${HEAD_MGMT_IP:-192.168.3.120}"
WORKER_MGMT_IP="${WORKER_MGMT_IP:-192.168.3.121}"
SSH_USER="${SSH_USER:-santanu}"
HEAD_CONTAINER="sglang-head"
WORKER_CONTAINER="sglang-worker"
DEPLOY_DIR="${DEPLOY_DIR:-/home/${SSH_USER}/sglang-dspark-deploy}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Stopping SGLang containers on head node (${HEAD_MGMT_IP})..."
ssh "${SSH_USER}@${HEAD_MGMT_IP}" \
    "cd ${DEPLOY_DIR} 2>/dev/null && docker compose -f docker-compose.head.yml down 2>/dev/null; \
     docker rm -f ${HEAD_CONTAINER} ${WORKER_CONTAINER} 2>/dev/null" || true
log "Head node containers stopped."

log "Stopping SGLang containers on worker node (${WORKER_MGMT_IP})..."
ssh "${SSH_USER}@${WORKER_MGMT_IP}" \
    "cd ${DEPLOY_DIR} 2>/dev/null && docker compose -f docker-compose.worker.yml down 2>/dev/null; \
     docker rm -f ${HEAD_CONTAINER} ${WORKER_CONTAINER} 2>/dev/null" || true
log "Worker node containers stopped."

log "All SGLang containers stopped."