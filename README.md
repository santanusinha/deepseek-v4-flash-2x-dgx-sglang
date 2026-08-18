# DeepSeek-V4-Flash-0731 on DGX Spark Cluster

Deploy DeepSeek-V4-Flash-0731 across two NVIDIA DGX Spark nodes with SGLang.
Use multi-node Tensor Parallel (TP=2) and DSPARK speculative decoding.

This README gives complete step-by-step instructions to deploy on a DGX Spark
cluster from a control machine.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Cluster Topology](#2-cluster-topology)
3. [Prerequisites](#3-prerequisites)
4. [Step 1 — Set Up Passwordless SSH](#step-1--set-up-passwordless-ssh)
5. [Step 2 — Install Docker and NVIDIA Container Toolkit](#step-2--install-docker-and-nvidia-container-toolkit)
6. [Step 3 — Pull the SGLang Base Image](#step-3--pull-the-sglang-base-image)
7. [Step 4 — Download Model Files](#step-4--download-model-files)
8. [Step 5 — Configure Clustering Network](#step-5--configure-clustering-network)
9. [Step 6 — Configure the Deployment](#step-6--configure-the-deployment)
10. [Step 7 — Deploy](#step-7--deploy)
11. [Step 8 — Verify](#step-8--verify)
12. [Step 9 — Test Inference](#step-9--test-inference)
13. [Step 10 — Monitor Logs](#step-10--monitor-logs)
14. [Step 11 — Stop](#step-11--stop)
15. [Configuration Reference](#configuration-reference)
16. [How the Custom Image Works](#how-the-custom-image-works)
17. [Manual Deployment (Without deploy-compose.sh)](#manual-deployment-without-deploy-composesh)
18. [Legacy deploy.sh](#legacy-deploysh)
19. [Known Issues and Workarounds](#known-issues-and-workarounds)
20. [Model Architecture](#model-architecture)
21. [File Listing](#file-listing)
22. [References](#references)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   Control Machine                     │
│              (your laptop / workstation)              │
│                                                       │
│  deploy-compose.sh  ──SSH──►  Node 1 (Head)          │
│  stop.sh            ──SSH──►  Node 2 (Worker)        │
│  verify.sh                                            │
│  .env                                                │
└──────────────────────────────────────────────────────┘
        │                              │
        │ SSH + scp                    │ SSH + scp
        ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐
│   Node 1 (Head)     │    │   Node 2 (Worker)   │
│   ai1               │    │   ai2               │
│   192.168.3.120     │    │   192.168.3.121     │
│                     │    │                     │
│  ┌───────────────┐  │    │  ┌───────────────┐  │
│  │ sglang-dspark │  │    │  │ sglang-dspark │  │
│  │  (container)  │  │    │  │  (container)  │  │
│  │  rank 0       │  │    │  │  rank 1       │  │
│  └───────┬───────┘  │    │  └───────┬───────┘  │
│          │          │    │          │          │
│  GPU: GB10 (SM120) │    │  GPU: GB10 (SM120) │
│  128 GB unified    │    │  128 GB unified    │
└─────────┬───────────┘    └──────────┬─────────┘
          │                           │
          └─────── QSFP RoCE ────────┘
                  200 Gb/s
                  192.168.100.x
```

**Key design points:**

- The control machine runs the orchestration scripts. It does not need a GPU.
- Each DGX Spark node builds the Docker image independently (native aarch64).
- The worker node (rank 1) starts first. The head node (rank 0) starts second.
- The API endpoint is on the head node.
- NCCL 2.30.7 and the DSPARK patch are baked into the Docker image.

---

## 2. Cluster Topology

| Node | Role | Management IP | Cluster IP | GPU |
|------|------|---------------|------------|-----|
| 1 (ai1) | Head (rank 0) | 192.168.3.120 | 192.168.100.10 | GB10 Grace Blackwell (SM120) |
| 2 (ai2) | Worker (rank 1) | 192.168.3.121 | 192.168.100.11 | GB10 Grace Blackwell (SM120) |

| Component | Details |
|-----------|---------|
| Hardware | 2x DGX Spark (1 GPU per node, 128 GB unified memory) |
| Interconnect | 200 Gb/s QSFP (ConnectX-7, RoCEv2) |
| Model | DeepSeek-V4-Flash-0731 (284B params, 13B active, MoE FP4/FP8) |
| Model size on disk | 156 GB (48 safetensors files) |
| SGLang | v0.5.17 (`lmsysorg/sglang:latest-cu130`) |
| NCCL | 2.30.7 (upgraded from torch-bundled 2.28.9) |

---

## 3. Prerequisites

Before you start, make sure you have:

- Two DGX Spark nodes connected with a QSFP cable.
- A control machine (laptop or workstation) with network access to both nodes.
- The model files (DeepSeek-V4-Flash-0731, 156 GB).
- Root or sudo access on both DGX Spark nodes.

---

## Step 1 — Set Up Passwordless SSH

The control machine must SSH into both DGX Spark nodes without a password.

Run these commands on the control machine:

```bash
# Generate an SSH key if you do not have one
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519

# Copy the key to both nodes
ssh-copy-id santanu@192.168.3.120
ssh-copy-id santanu@192.168.3.121

# Verify passwordless SSH works
ssh santanu@192.168.3.120 "echo ok"
ssh santanu@192.168.3.121 "echo ok"
```

If both commands print `ok`, continue to the next step.

---

## Step 2 — Install Docker and NVIDIA Container Toolkit

Install Docker and the NVIDIA Container Toolkit on both DGX Spark nodes.

SSH into each node and run:

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and log back in for the group change to take effect

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify Docker and GPU access:

```bash
docker info
docker run --rm --gpus all lmsysorg/sglang:latest-cu130 nvidia-smi
```

Verify Docker Compose:

```bash
docker compose version
```

If `docker compose` is not available, install the Compose plugin:

```bash
sudo apt-get install -y docker-compose-plugin
```

---

## Step 3 — Pull the SGLang Base Image

Pull the SGLang base image on both nodes. This image is large. Use a fast
network connection or be patient.

Run on both nodes:

```bash
docker pull lmsysorg/sglang:latest-cu130
```

Verify the image is present:

```bash
docker images lmsysorg/sglang:latest-cu130
```

---

## Step 4 — Download Model Files

Download the DeepSeek-V4-Flash-0731 model files to both nodes. The model is
156 GB. Make sure both nodes have enough disk space.

Run on both nodes:

```bash
mkdir -p /home/santanu/data/models/DeepSeek-V4-Flash-0731
cd /home/santanu/data/models/DeepSeek-V4-Flash-0731

# Download model files from Hugging Face or your preferred source
# Example:
# huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731 \
#   --local-dir /home/santanu/data/models/DeepSeek-V4-Flash-0731
```

Verify the model directory:

```bash
ls -la /home/santanu/data/models/DeepSeek-V4-Flash-0731/
# You should see 48 .safetensors files and config.json
```

---

## Step 5 — Configure Clustering Network

Connect the two DGX Spark nodes with a QSFP cable. Configure the clustering
interface (RoCE) on both nodes.

On Node 1 (ai1):

```bash
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

On Node 2 (ai2):

```bash
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

Verify connectivity:

```bash
# From Node 1
ping -c 3 192.168.100.11

# From Node 2
ping -c 3 192.168.100.10
```

Find the RoCE HCA device name and GID index:

```bash
# On both nodes
ibdev2netdev
# Look for the device mapped to enp1s0f1np1 (e.g., rocep1s0f1)

show_gids
# Look for the GID index with RoCEv2 (IPv4) type
```

For a standard DGX Spark setup, the defaults are:

- HCA: `rocep1s0f1`
- GID index: `3`
- Socket interface: `enp1s0f1np1`

---

## Step 6 — Configure the Deployment

On the control machine, clone or copy this project directory. Then configure
the environment file.

```bash
cp .env.example .env
```

Edit `.env` to match your cluster:

```bash
# --- Model ---
MODEL_PATH=/home/santanu/data/models/DeepSeek-V4-Flash-0731

# --- Network ---
HEAD_CLUSTER_IP=192.168.100.10

# RDMA / NCCL interface configuration
NCCL_IB_HCA=rocep1s0f1
NCCL_IB_GID_INDEX=3
NCCL_SOCKET_IFNAME=enp1s0f1np1
GLOO_SOCKET_IFNAME=enp1s0f1np1

# --- SGLang server ---
SGLANG_PORT=8000
DIST_INIT_PORT=20000

# --- Model serving ---
MEM_FRACTION=0.85
CONTEXT_LENGTH=131072
CHUNKED_PREFILL_SIZE=4096
MAX_RUNNING_REQUESTS=8
KV_CACHE_DTYPE=fp8_e4m3
WATCHDOG_TIMEOUT=600

# --- DSPARK ---
DSPARK_BLOCK_SIZE=5
```

If your DGX Spark cluster uses the default interface names and IPs, you do
not need to change anything.

---

## Step 7 — Deploy

Run the deployment script from the control machine.

### First deployment (builds the image on each node)

```bash
./deploy-compose.sh --build
```

The script does the following:

1. Checks SSH, Docker, and model paths on both nodes.
2. Syncs the Dockerfile, compose files, `.env`, and patch script to both
   nodes at `~/sglang-dspark-deploy/`.
3. Builds the `sglang-dspark` Docker image on each node independently
   (native aarch64 build).
4. Stops any existing SGLang containers on both nodes.
5. Starts the worker container on Node 2 first.
6. Waits 10 seconds.
7. Starts the head container on Node 1.
8. Polls the health endpoint until the server is ready (up to 30 minutes).

### Subsequent deployments (reuse existing image)

```bash
./deploy-compose.sh
```

### Deploy without waiting for health

```bash
./deploy-compose.sh --no-wait
```

### What to expect

The model loading takes 5 to 10 minutes. During this time, the script prints
progress messages:

```
[2026-08-18 03:20:00] Still waiting... (120s / 1800s)
[2026-08-18 03:20:10] Still waiting... (130s / 1800s)
...
[2026-08-18 03:28:00] Head node is ready!
```

When the server is ready, the script prints the API endpoint and test
commands.

---

## Step 8 — Verify

Run the verification script:

```bash
./verify.sh
```

This script checks:

1. Head node health endpoint.
2. Container status on both nodes.
3. GPU status on both nodes.
4. Inference test (sends a test prompt).

Example output:

```
[2026-08-18 03:30:00] === SGLang Deployment Verification ===
[2026-08-18 03:30:00] 1. Checking head node health...
   [OK] Head node is healthy.
[2026-08-18 03:30:00] 2. Checking containers on head node...
   --- Head node containers ---
   NAMES           STATUS
   sglang-head     Up 10 minutes
[2026-08-18 03:30:00] 3. Checking containers on worker node...
   --- Worker node containers ---
   NAMES           STATUS
   sglang-worker   Up 10 minutes
[2026-08-18 03:30:00] 4. Checking GPU on head node...
   NVIDIA GB10, 45C, 85% util, 100GB / 128GB memory
[2026-08-18 03:30:00] 5. Checking GPU on worker node...
   NVIDIA GB10, 43C, 82% util, 98GB / 128GB memory
[2026-08-18 03:30:00] 6. Testing inference via head node...
   [OK] Inference response received:
   {"id":"...","choices":[{"message":{"content":"Hello! How can I help you?"}}]}
```

---

## Step 9 — Test Inference

### Health check

```bash
curl http://192.168.3.120:8000/health
```

### Chat completion

```bash
curl http://192.168.3.120:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4-Flash-0731",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'
```

### Streaming chat completion

```bash
curl http://192.168.3.120:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V4-Flash-0731",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
    "max_tokens": 100,
    "stream": true
  }'
```

### List available models

```bash
curl http://192.168.3.120:8000/v1/models
```

---

## Step 10 — Monitor Logs

### Head node logs

```bash
ssh santanu@192.168.3.120 docker logs -f sglang-head
```

### Worker node logs

```bash
ssh santanu@192.168.3.121 docker logs -f sglang-worker
```

### GPU status

```bash
# Head node
ssh santanu@192.168.3.120 nvidia-smi

# Worker node
ssh santanu@192.168.3.121 nvidia-smi
```

---

## Step 11 — Stop

Stop all containers on both nodes:

```bash
./stop.sh
```

The script tries `docker compose down` first. It falls back to `docker rm -f`
for any containers that were started with the legacy `deploy.sh` script.

---

## Configuration Reference

All configuration is controlled by environment variables. Copy `.env.example`
to `.env` and adjust as needed.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | /home/santanu/data/models/DeepSeek-V4-Flash-0731 | Model path on each node |
| `HEAD_CLUSTER_IP` | 192.168.100.10 | Head node clustering interface IP |
| `NCCL_IB_HCA` | rocep1s0f1 | RoCE HCA device name |
| `NCCL_IB_GID_INDEX` | 3 | RoCEv2 GID index |
| `NCCL_SOCKET_IFNAME` | enp1s0f1np1 | NCCL socket interface name |
| `GLOO_SOCKET_IFNAME` | enp1s0f1np1 | Gloo socket interface name |
| `SGLANG_PORT` | 8000 | SGLang server port |
| `DIST_INIT_PORT` | 20000 | Distributed init port |
| `MEM_FRACTION` | 0.85 | KV cache memory fraction |
| `CONTEXT_LENGTH` | 131072 | Max context length |
| `CHUNKED_PREFILL_SIZE` | 4096 | Chunked prefill size |
| `MAX_RUNNING_REQUESTS` | 8 | Max concurrent requests |
| `KV_CACHE_DTYPE` | fp8_e4m3 | KV cache data type |
| `WATCHDOG_TIMEOUT` | 600 | Watchdog timeout in seconds |
| `DSPARK_BLOCK_SIZE` | 5 | DSPARK speculative block size |

### deploy-compose.sh Variables

These variables control the orchestration script itself. Set them in the
environment or in `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `HEAD_MGMT_IP` | 192.168.3.120 | Head node management IP for SSH |
| `WORKER_MGMT_IP` | 192.168.3.121 | Worker node management IP for SSH |
| `SSH_USER` | santanu | SSH username |
| `DEPLOY_DIR` | /home/santanu/sglang-dspark-deploy | Directory on nodes for deployment files |

### SGLang Server Flags

The Docker Compose files pass these SGLang flags:

| Flag | Value | Purpose |
|------|-------|---------|
| `--tp 2` | 2 | Tensor parallel size (1 GPU per node) |
| `--nnodes 2` | 2 | Number of nodes in the cluster |
| `--node-rank` | 0 (head) / 1 (worker) | Rank of this node |
| `--dist-init-addr` | HEAD_CLUSTER_IP:DIST_INIT_PORT | Address for distributed init |
| `--dist-timeout` | 3600 | Distributed init timeout (seconds) |
| `--mem-fraction-static` | 0.85 | Static memory fraction for KV cache |
| `--context-length` | 131072 | Max context length |
| `--max-running-requests` | 8 | Max concurrent inference requests |
| `--moe-runner-backend` | flashinfer_mxfp4 | MoE kernel backend for FP4 |
| `--kv-cache-dtype` | fp8_e4m3 | KV cache compression format |
| `--enable-dp-attention` | — | Bypasses FlashInfer IPC across nodes |
| `--enable-dp-lm-head` | — | Enables DP for LM head (DSPARK) |
| `--speculative-algorithm` | DSPARK | Speculative decoding algorithm |
| `--speculative-dspark-block-size` | 5 | DSPARK draft block size |
| `--watchdog-timeout` | 600 | Scheduler watchdog timeout |
| `--swa-full-tokens-ratio` | 0.1 | Sliding window attention ratio |
| `--reasoning-parser` | deepseek-v4 | Reasoning content parser |
| `--tool-call-parser` | deepseekv4 | Tool call parser |

### RDMA/NCCL Configuration

DGX Spark uses ConnectX-7 RoCE for inter-node communication:

| Setting | Value | Purpose |
|---------|-------|---------|
| `NCCL_IB_HCA` | rocep1s0f1 | RoCE HCA with 192.168.100.x IPs |
| `NCCL_IB_GID_INDEX` | 3 | RoCEv2/IPv4 GID index |
| `NCCL_SOCKET_IFNAME` | enp1s0f1np1 | Clustering interface |
| `GLOO_SOCKET_IFNAME` | enp1s0f1np1 | Same interface for Gloo |
| `NCCL_CUMEM_ENABLE` | 0 | Disable NCCL CUDA memory management |
| `NCCL_IGNORE_CPU_AFFINITY` | 1 | Ignore CPU affinity issues on aarch64 |

Container flags for RDMA access:

- `--privileged`
- `--device /dev/infiniband`
- `--ulimit memlock=-1:-1`
- `--cap-add IPC_LOCK`
- `--shm-size 32g`
- `--ipc host`
- `--network host`

---

## How the Custom Image Works

The `sglang-dspark` Docker image is built from the Dockerfile. It adds two
things to the base SGLang image:

### 1. NCCL 2.30.7 Upgrade

The base image `lmsysorg/sglang:latest-cu130` bundles NCCL 2.28.9 (torch)
and 2.28.3 (system). These versions cause a TP rank divergence deadlock
under DSPARK multi-node TP (SGLang issues #33289, #33549).

The Dockerfile downloads and installs NCCL 2.30.7 directly from NVIDIA's
repository as a `.deb` package. This avoids the CUDA keyring conflict that
occurs when you try to add the CUDA apt repository to the base image.

The `SGLANG_NCCL_SO_PATH` environment variable tells SGLang to use the
upgraded library.

### 2. DSPARK topk=192 Padding Patch

FlashInfer's SM120 sparse-MLA kernel does not support `topk=192`. DSpark's
draft attention produces `topk=192`, which causes a crash during CUDA graph
capture.

The patch script (`patches/apply_dspark_patch.sh`) modifies
`flash_mla_sm120.py` inside the image. It pads `topk=192` to `topk=512` so
the kernel can dispatch correctly.

### Build the Image Manually

If you want to build the image manually on a node:

```bash
# On a DGX Spark node
cd /path/to/model-deploy
docker build -t sglang-dspark .
```

The build takes about 5 minutes. It downloads a 200 MB NCCL `.deb` and
applies the patch.

---

## Manual Deployment (Without deploy-compose.sh)

If you prefer to run Docker Compose commands manually on each node:

### 1. Copy files to both nodes

```bash
# On the control machine
for node in 192.168.3.120 192.168.3.121; do
  ssh santanu@$node "mkdir -p ~/sglang-dspark-deploy/patches"
  scp Dockerfile docker-compose.head.yml docker-compose.worker.yml .env \
    santanu@$node:~/sglang-dspark-deploy/
  scp patches/apply_dspark_patch.sh \
    santanu@$node:~/sglang-dspark-deploy/patches/
done
```

### 2. Build the image on each node

```bash
# On each node
ssh santanu@192.168.3.120 "cd ~/sglang-dspark-deploy && docker build -t sglang-dspark ."
ssh santanu@192.168.3.121 "cd ~/sglang-dspark-deploy && docker build -t sglang-dspark ."
```

### 3. Start the worker first

```bash
# On Node 2 (worker)
ssh santanu@192.168.3.121 \
  "cd ~/sglang-dspark-deploy && docker compose -f docker-compose.worker.yml up -d"
```

### 4. Wait 10 seconds, then start the head

```bash
sleep 10

# On Node 1 (head)
ssh santanu@192.168.3.120 \
  "cd ~/sglang-dspark-deploy && docker compose -f docker-compose.head.yml up -d"
```

### 5. Wait for health

```bash
# Poll the health endpoint until it responds
while ! curl -sf http://192.168.3.120:8000/health; do
  echo "Waiting..."
  sleep 10
done
echo "Server is ready!"
```

---

## Legacy deploy.sh

The original `deploy.sh` script uses `docker run` directly instead of Docker
Compose. It requires the NCCL 2.30.7 library pre-staged at `/tmp/nccl-upgrade/`
on both nodes.

### Safe mode (no DSPARK, no CUDA graphs)

```bash
./deploy.sh
```

### DSPARK speculative decoding mode

```bash
./deploy.sh dspark
```
### Stage NCCL 2.30.7 for deploy.sh

If you use the legacy `deploy.sh`, stage the NCCL library on both nodes:

```bash
# On the control machine
for node in 192.168.3.120 192.168.3.121; do
  ssh santanu@$node "mkdir -p /tmp/nccl-upgrade"
  scp /tmp/libnccl2_2.30.7_arm64.deb santanu@$node:/tmp/
  ssh santanu@$node \
    "cd /tmp && dpkg-deb -x libnccl2_2.30.7_arm64.deb nccl-extract && \
     cp nccl-extract/usr/lib/aarch64-linux-gnu/libnccl.so.2.30.7 \
        /tmp/nccl-upgrade/libnccl.so.2 && \
     rm -rf /tmp/nccl-extract"
done
```

### deploy.sh Configuration

Edit the variables at the top of `deploy.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HEAD_MGMT_IP` | 192.168.3.120 | Head node management IP (SSH) |
| `WORKER_MGMT_IP` | 192.168.3.121 | Worker node management IP (SSH) |
| `HEAD_CLUSTER_IP` | 192.168.100.10 | Head node clustering interface IP |
| `WORKER_CLUSTER_IP` | 192.168.100.11 | Worker node clustering interface IP |
| `SSH_USER` | santanu | SSH username |
| `MODEL_PATH` | /home/santanu/data/models/DeepSeek-V4-Flash-0731 | Model path on both nodes |
| `SGLANG_IMAGE` | lmsysorg/sglang:latest-cu130 | SGLang Docker image |
| `WORKER_PORT` | 8000 | SGLang server port |
| `MEM_FRACTION` | 0.85 | KV cache memory fraction |
| `CONTEXT_LENGTH` | 131072 | Max context length |
| `KV_CACHE_DTYPE` | fp8_e4m3 | KV cache data type |
| `NCCL_IB_HCA` | rocep1s0f1 | RoCE HCA device name |
| `NCCL_IB_GID_INDEX` | 3 | RoCEv2 GID index |

---

## Known Issues and Workarounds

### FlashInfer IPC Error (Cross-Node TP)

**Symptom:** The container crashes with `CUDART error: invalid device context`
during multi-node TP.

**Cause:** FlashInfer AllReduce Fusion uses CUDA IPC, which does not work across
node boundaries.

**Fix:** The deployment includes `--enable-dp-attention`. This flag bypasses
FlashInfer AllReduce Fusion and uses data-parallel attention instead.

### DSPARK topk=192 Crash (SM120/SM121)

**Symptom:** DSPARK mode fails during CUDA graph capture with `Unsupported
sparse-MLA prefill configuration: topk=192`.

**Cause:** FlashInfer SM120 kernel does not instantiate topk=192 (only
128/512/1024/2048). DSpark draft attention produces topk=192.

**Fix:** The Dockerfile applies a padding workaround that pads topk=192 to
512. See SGLang issue #33134.

**Warning:** This workaround may cause intermittent text corruption (issue
#33985). Use for testing. Monitor for production use.

### TP Rank Divergence Deadlock

**Symptom:** The server hangs under agentic traffic. One rank enters a
collective operation while the peer rank is at request broadcast.

**Cause:** Per-rank scheduler divergence in the DSpark verify path (SGLang
issue #33289).

**Fix:** NCCL 2.30.7 fixes the deadlock. SGLang PR #33614 (DsparkTpSync)
adds sampling decision broadcasts as the complete fix. That PR is not merged
yet.

### Model Loading Takes Long

The 156 GB model loads from disk. This takes 5 to 10 minutes. The scripts
wait up to 30 minutes. If loading takes longer, check disk I/O speed.

---

## Model Architecture

| Property | Value |
|----------|-------|
| Architecture | DeepseekV4ForCausalLM |
| Total parameters | 284B |
| Active parameters per token | 13B (MoE) |
| Routed experts | 256 |
| Experts per token | 6 |
| Shared experts | 1 |
| Expert dtype | FP4 |
| Dense quantization | FP8 (e4m3) |
| Hidden layers | 43 |
| Hidden size | 4096 |
| Attention heads | 64 |
| KV heads | 1 (MLA) |
| Head dimension | 512 |
| MTP (DSPARK) | Block size 5, target layers [40, 41, 42] |
| Max context | 1M positions (YaRN scaling) |
| Disk size | 156 GB (48 safetensors files) |

---

## File Listing

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds `sglang-dspark` image with NCCL 2.30.7 + DSPARK patch |
| `docker-compose.head.yml` | Compose file for head node (rank 0) |
| `docker-compose.worker.yml` | Compose file for worker node (rank 1) |
| `.env.example` | Example environment configuration |
| `patches/apply_dspark_patch.sh` | Patch script for topk=192 padding |
| `deploy-compose.sh` | Orchestration script for compose deployment |
| `deploy.sh` | Legacy deployment script (docker run) |
| `stop.sh` | Stop all containers on both nodes |
| `verify.sh` | Verify deployment health and inference |
| `README.md` | This file |

---

## References

- [SGLang DeepSeek-V4 Cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [SGLang Multi-Node Deployment](https://docs.sglang.ai/references/multi_node_deployment/multi_node.html)
- [DGX Spark Clustering Guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
- [SGLang on DGX Spark (community)](https://github.com/mark-ramsey-ri/sglang-dgx-spark)
- [DSPARK topk=192 Issue](https://github.com/sgl-project/sglang/issues/33134)
- [TP Rank Divergence Issue](https://github.com/sgl-project/sglang/issues/33289)
- [vLLM DSv4-Flash on DGX Spark (reference)](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark)