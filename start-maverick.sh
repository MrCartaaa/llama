#!/bin/bash

# =============================================================================

# Llama 4 Maverick – CUDA KEYRING URL FIXED (ubuntu2404)

# 2× RTX Titan | Threadripper 3990X | 128 GB RAM

# =============================================================================

set -e

# ------------------- CONFIG -------------------

#MODEL_NAME="Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf"
#MODEL_NAME="Llama-4-Scout-17B-16E-Instruct-UD-Q2_K_XL.gguf"
# MODEL_NAME="nomic-embed-text-v1.5.Q8_0.gguf"
MODEL_NAME="qwen/Qwen2.5-14B-Instruct-Q4_K_M.gguf"

MODEL_DIR="$HOME/llama-models"

SRC_DIR="$HOME/llama.cpp"

BUILD_DIR="$SRC_DIR/build-ninja"

CLI="$BUILD_DIR/bin/llama-cli"

SRV="$BUILD_DIR/bin/llama-server"

CTX=32768
TEMP=0.1
THR=64
PORT=8080
TENSOR_SPLIT="0.5,0.5"

echo "=== Llama Runner Script – Clean Ninja Build ==="

# ------------------- 1. System deps -------------------

echo "[1/6] Installing build tools... (skipped if up to date)"

sudo apt update -qq

sudo apt install -y -qq cmake ninja-build build-essential git wget curl libcurl4-openssl-dev gnupg2 ca-certificates lsb-release

# ------------------- 2. CUDA -------------------

if ! command -v nvcc &>/dev/null; then

  echo "[2/6] Installing CUDA 12.6 via NVIDIA repo..."

  DISTRO_REPO="ubuntu2404" # For Ubuntu 24.04 (noble); change to ubuntu2204 for 22.04

  KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"

  KEYRING_FILE="/tmp/cuda-keyring.deb"

  # Download with progress

  echo "   Downloading keyring from $KEYRING_URL..."

  wget --progress=bar:force:noscroll -O "$KEYRING_FILE" "$KEYRING_URL"

  # Check if download succeeded (non-empty file)

  if [ -s "$KEYRING_FILE" ]; then

    sudo dpkg -i "$KEYRING_FILE"

    rm "$KEYRING_FILE"

    sudo apt update -qq

    sudo apt install -y -qq cuda-toolkit-12-6

    echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >>~/.bashrc

    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >>~/.bashrc

    source ~/.bashrc

    echo "[+] CUDA 12.6 installed! Reboot recommended."

  else

    echo "Download failed (empty file)—skipping CUDA (CPU mode)."

    rm -f "$KEYRING_FILE"

  fi

else

  echo "[2/6] CUDA already installed."

fi

echo "[CUDA] $(nvcc --version | head -n1 || echo 'CPU-only mode')"

# ------------------- 3. Clone llama.cpp -------------------

# echo "[3/6] Cloning/updating llama.cpp..."

# [ -d "$SRC_DIR" ] || mkdir -p "$SRC_DIR"

# [ -d "$SRC_DIR/.git" ] || git clone https://github.com/ggerganov/llama.cpp "$SRC_DIR"

# (cd "$SRC_DIR" && git pull -q)

# ------------------- 4. Clean + CMake -------------------

echo "[4/6] Cleaning and configuring CMake..."

rm -rf "$BUILD_DIR" "$SRC_DIR/CMakeCache.txt" "$SRC_DIR/CMakeFiles"

mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

cmake "$SRC_DIR" -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_BUILD_TYPE=Release -G Ninja

# ------------------- 5. Build -------------------

echo "[5/6] Building with Ninja (-j$THR)..."

ninja -j$THR

# ------------------- 6. Model -------------------

echo "[6/6] Ensuring model is present..."

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/$MODEL_NAME" ]; then

  echo "   → Model is not present. exiting..."
  exit 1

fi

# ------------------- 7. Launch -------------------

echo "[LAUNCH] Starting Web UI → http://localhost:$PORT"

echo "[LAUNCH] Starting Hybrid Mode: GPU (60 layers) + RAM/CPU Overflow"

# --- 7. Launch: PERFECT 50/50 GPU SPLIT ---

taskset -c 0-63 "$SRV" \
  --model "$MODEL_DIR/$MODEL_NAME" \
  --tensor-split $TENSOR_SPLIT \
  --n-gpu-layers 50 \
  --threads $THR \
  --verbose \
  --ctx-size $CTX \
  --temp $TEMP \
  --port $PORT \
  --host 0.0.0.0 \
  --embeddings \
  --no-warmup
