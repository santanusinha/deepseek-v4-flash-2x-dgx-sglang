#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Verify SGLang deployment on DGX Spark cluster
# =============================================================================

HEAD_MGMT_IP="192.168.3.120"
WORKER_MGMT_IP="192.168.3.121"
SSH_USER="santanu"
WORKER_PORT=8000
HEAD_CONTAINER="sglang-head"
WORKER_CONTAINER="sglang-worker"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

echo ""
log "=== SGLang Deployment Verification ==="
echo ""

# Check head node health
log "1. Checking head node health (http://${HEAD_MGMT_IP}:${WORKER_PORT}/health)..."
if curl -sf "http://${HEAD_MGMT_IP}:${WORKER_PORT}/health" >/dev/null 2>&1; then
    echo "   [OK] Head node is healthy."
else
    echo "   [FAIL] Head node is not responding."
fi

# Check container status on head node
log "2. Checking containers on head node (${HEAD_MGMT_IP})..."
echo "   --- Head node containers ---"
ssh -o ConnectTimeout=5 "${SSH_USER}@${HEAD_MGMT_IP}" \
    "docker ps --filter 'name=sglang' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || \
    echo "   [FAIL] Cannot list containers on head node."

# Check container status on worker node
log "3. Checking containers on worker node (${WORKER_MGMT_IP})..."
echo "   --- Worker node containers ---"
ssh -o ConnectTimeout=5 "${SSH_USER}@${WORKER_MGMT_IP}" \
    "docker ps --filter 'name=sglang' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>/dev/null || \
    echo "   [FAIL] Cannot list containers on worker node."

# Check GPU status on head node
log "4. Checking GPU on head node..."
ssh -o ConnectTimeout=5 "${SSH_USER}@${HEAD_MGMT_IP}" \
    "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null || \
    echo "   [FAIL] Cannot query GPU on head node."

# Check GPU status on worker node
log "5. Checking GPU on worker node..."
ssh -o ConnectTimeout=5 "${SSH_USER}@${WORKER_MGMT_IP}" \
    "nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null || \
    echo "   [FAIL] Cannot query GPU on worker node."

# Test inference
echo ""
log "6. Testing inference via head node..."
echo ""
RESPONSE=$(curl -sf "http://${HEAD_MGMT_IP}:${WORKER_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"Hello, respond with one sentence."}],"max_tokens":50}' 2>&1) || true

if [ -n "${RESPONSE}" ]; then
    echo "   [OK] Inference response received:"
    echo "   ${RESPONSE}" | head -20
else
    echo "   [FAIL] No response from inference endpoint."
fi

echo ""
log "=== Verification Complete ==="
