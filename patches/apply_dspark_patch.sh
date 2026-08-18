#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DSPARK topk=192 padding workaround for SM120/SM121
# =============================================================================
#
# SGLang issue #33134: The SM120 sparse-MLA kernel lacks a topk=192
# instantiation. When DSPARK produces 192 indices, the kernel crashes.
# This patch pads topk from 192 to 512 so the kernel can dispatch.
#
# Reference: flashinfer#3828, SGLang #33134
#

PATCH_FILE="/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py"

if [ ! -f "${PATCH_FILE}" ]; then
    echo "[ERROR] flash_mla_sm120.py not found at ${PATCH_FILE}"
    exit 1
fi

if grep -q "_pad = 512" "${PATCH_FILE}"; then
    echo "[INFO] Patch already applied."
    exit 0
fi

# Apply the topk=192 padding workaround
sed -i '/idx = indices.squeeze(1) if indices.dim() == 3 else indices/a\
\
    # DSPARK topk=192 workaround for SM120/SM121 (SGLang issue #33134)\
    if idx.shape[-1] == 192:\
        _pad = 512 - idx.shape[-1]\
        idx = torch.nn.functional.pad(idx, (0, _pad), value=0)' \
    "${PATCH_FILE}"

# Verify the patch was applied
if grep -q "_pad = 512" "${PATCH_FILE}"; then
    echo "[INFO] Patch applied successfully."
else
    echo "[ERROR] Failed to apply patch."
    exit 1
fi
