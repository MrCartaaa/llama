#!/bin/bash
# =============================================================================
# Llama.cpp Runner – Dual RTX Titan (Pascal) Optimized
# SSH with explicit key - Fixed GIT_SSH_COMMAND
# =============================================================================

set -e

# ================== CONFIG ==================
LLAMA_REPO="git@github.com:ikawrakow/ik_llama.cpp.git"
SSH_KEY="/home/john/.ssh/github"

SRC_DIR="$HOME/llama.cpp"
BUILD_DIR="$SRC_DIR/build-ninja"
SRV="$BUILD_DIR/bin/llama-server"

CPU_AFFINITY="0-63"
THR=64

CTX=32768
TEMP=0.7
PORT=8080
TENSOR_SPLIT="0.3,0.7"

echo "=== Llama Runner – ikawrakow fork (SSH) ==="

# ------------------- 1. System deps -------------------
echo "[1/6] Installing build tools..."
sudo apt update -qq
sudo apt install -y -qq cmake ninja-build build-essential git wget curl \
    libcurl4-openssl-dev libssl-dev gnupg2 ca-certificates lsb-release

# ------------------- 2. CUDA -------------------
if ! command -v nvcc &>/dev/null; then
    echo "[2/6] Installing CUDA 12.6..."
    DISTRO_REPO="ubuntu2404"
    KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    KEYRING_FILE="/tmp/cuda-keyring.deb"

    wget --progress=bar:force:noscroll -O "$KEYRING_FILE" "$KEYRING_URL"
    if [ -s "$KEYRING_FILE" ]; then
        sudo dpkg -i "$KEYRING_FILE" && rm "$KEYRING_FILE"
        sudo apt update -qq
        sudo apt install -y -qq cuda-toolkit-12-6
        echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
        source ~/.bashrc
    fi
else
    echo "[2/6] CUDA already installed."
fi
echo "[CUDA] $(nvcc --version | head -n1 || echo 'CPU-only mode')"

# ------------------- 3. Update llama.cpp via SSH -------------------
echo "[3/6] Updating llama.cpp via SSH..."

git config --global --add safe.directory "$SRC_DIR"
mkdir -p ~/.ssh && chmod 700 ~/.ssh
chmod 600 "$SSH_KEY" 2>/dev/null || true
ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null || true
chmod 644 ~/.ssh/known_hosts

mkdir -p "$SRC_DIR"

if [ -d "$SRC_DIR" ]; then
  rm -rf $SRC_DIR
fi

echo "   → Fresh clone via SSH..."
GIT_SSH_COMMAND="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o IdentitiesOnly=yes" \
git clone "$LLAMA_REPO" "$SRC_DIR"

echo "[VERSION] $(cd "$SRC_DIR" && git log -1 --format="%h %as %s" || echo 'unknown')"

# ------------------- 4. CMake (Turing sm_75) -------------------
echo "[4/6] Configuring CMake (Turing sm_75)..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$SRC_DIR" \
    -G Ninja \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_OPENSSL=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DGGML_CUDA_FORCE_MMQ=ON

echo "[5/6] Building..."
ninja -j$THR

# ------------------- 6. Launch -------------------
echo "[LAUNCH] Starting server - Conservative settings..."

taskset -c "$CPU_AFFINITY" "$SRV" \
    --model /home/john/.cache/llama.cpp/unsloth_Qwen3-Coder-Next-GGUF_Qwen3-Coder-Next-UD-Q4_K_S.gguf \
    --chat-template-file /home/john/llama/llama-templates/qwen3_unsloth_chat_template.jinja \
    --tensor-split $TENSOR_SPLIT \
    --split-mode graph \
    --n-gpu-layers -1 \
    --ctx-size 131072 \
    --rope-scaling yarn \
    --rope-scale 4 \
    --yarn-orig-ctx 32768 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --flash-attn on \
    --threads $THR \
    --threads-batch $THR \
    --batch-size 65536 \
    --ubatch-size 32768 \
    --temp $TEMP \
    --top-p 0.95 \
    --top-k 40 \
    --repeat-penalty 1.10 \
    --presence-penalty 0.1 \
    --port $PORT \
    --host 0.0.0.0 \
    --embeddings \
    --jinja \
    --mlock \
    --no-mmap \
    --no-warmup \
    --verbose
