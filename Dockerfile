# =============================================================================
# SGLang with NCCL 2.30.7 + DSPARK topk=192 patch for DGX Spark (SM120/SM121)
# =============================================================================
#
# Base: lmsysorg/sglang:latest-cu130 (v0.5.17, CUDA 13.0, aarch64)
#
# Changes:
#   1. Upgrade NCCL to 2.30.7 (fixes multi-node TP rank divergence deadlock)
#   2. Apply DSPARK topk=192 padding patch for SM120/SM121
#
# Build on a DGX Spark node (aarch64):
#   docker build -t sglang-dspark .
#

FROM lmsysorg/sglang:latest-cu130

# -----------------------------------------------------------------------------
# Install NCCL 2.30.7 (fixes SGLang issue #33289: TP rank divergence deadlock)
#
# The base image already has a CUDA keyring configured. Installing a new
# cuda-keyring .deb causes a Signed-By conflict in apt sources. To avoid
# this, download and install the NCCL .deb directly with dpkg.
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/libnccl2_2.30.7-1+cuda13.3_arm64.deb && \
    dpkg -i libnccl2_2.30.7-1+cuda13.3_arm64.deb && \
    rm libnccl2_2.30.7-1+cuda13.3_arm64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Tell SGLang to use the upgraded NCCL library
ENV SGLANG_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/libnccl.so.2

# -----------------------------------------------------------------------------
# Apply DSPARK topk=192 padding patch for SM120/SM121
# -----------------------------------------------------------------------------
COPY patches/apply_dspark_patch.sh /tmp/apply_dspark_patch.sh
RUN chmod +x /tmp/apply_dspark_patch.sh && /tmp/apply_dspark_patch.sh && rm /tmp/apply_dspark_patch.sh
