# llama.cpp Server Tuning

This README documents the **tunable parts** of a `llama-server` launch command and how to adjust them to achieve a preferred result or troubleshoot poor performance.

---

## Baseline Command (Reference)

```bash
taskset -c 0-63 llama-server \
  -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q8_0 \
  --tensor-split 0.48,0.52 \
  --n-gpu-layers -1 \
  --ctx-size 32768 \
  --temp 0.1 \
  --threads 48 \
  --batch-size 1024 \
  --ubatch-size 256 \
  --flash-attn on \
  --port 8080 \
  --host 0.0.0.0 \
  --embeddings \
  --no-warmup \
  --jinja \
  --verbose
```

> Notes:
>
> * Flags vary by llama.cpp build. If a flag errors as *invalid argument*, omit it and use the nearest supported equivalent.

---

## Goals (Pick Your Preferred Result)

### A. Fastest “feels snappy” coding companion

* Prefer **lower context** (8k–16k)
* Ensure **all layers on GPU**
* Use **Flash Attention** if available
* Increase **batch** until stable

### B. Long-context assistant (large files, long chats)

* Use **16k–32k context**
* Keep **Flash Attention** enabled
* Leave **GPU0 headroom** via tensor split bias

### C. Maximum quality (less hallucination, better reasoning)

* Use higher quant: **Q8_0 > Q6_K > Q5_K_XL > Q4_K_M**
* Keep temperature low (0.1–0.2)
* Accept slower tokens/sec

---

## What Each Part Does (and When to Tweak)

### 1) Model Selection (`-hf ...:Q8_0`)

**What it controls**: model + quantization level.

**Tweak when**:

* Responses are weak/inaccurate → move **up** (Q6/Q8)
* Too slow / doesn’t fit VRAM → move **down** (Q5/Q4)

**Preferred result**:

* Best quality: `Q8_0`
* Best balance on 24GB GPUs: `Q5_K_XL` or `Q6_K`

---

### 2) CPU Pinning (`taskset -c 0-63`)

**What it controls**: which CPU cores llama-server may use.

**Tweak when**:

* Tokenization or CPU-side overhead spikes
* You want to reserve cores for other workloads

**Preferred result**:

* Keep pinned for repeatable performance.
* If the server is the only workload, pin a large contiguous range.

---

### 3) GPU Offload (`--n-gpu-layers`)

**What it controls**: how many transformer layers run on GPU.

**Recommended**:

* `--n-gpu-layers -1` (or a very large number) to offload all layers.

**Tweak when**:

* CPU is busy during generation → increase GPU layers
* VRAM OOM during load → reduce GPU layers (last resort)

**Symptoms**:

* Low GPU utilization + slow tokens/sec often means too much on CPU.

---

### 4) Multi-GPU Split (`--tensor-split 0.48,0.52`)

**What it controls**: how model tensors are distributed across GPUs.

**Recommended**:

* Bias *slightly away from GPU0* to leave headroom for cache/overhead.

**Tweak when**:

* GPU0 OOMs but GPU1 has free VRAM → shift more to GPU1 (e.g., `0.45,0.55`)
* GPU1 OOMs first → shift back toward GPU0

**Preferred result**:

* Both GPUs show similar memory usage, with GPU0 slightly lower.

---

### 5) Context Length (`--ctx-size 32768`)

**What it controls**: maximum tokens retained in attention (KV cache).

**Tradeoffs**:

* Higher ctx → better long-context, slower generation, more VRAM.

**Tweak when**:

* OOM at runtime or load → reduce ctx (e.g., 16k)
* Latency feels high → reduce ctx (8k–16k)
* You need more history → increase ctx if VRAM allows

**Preferred result**:

* Coding companion: 8k–16k
* Long-file work: 16k–32k

---

### 6) Flash Attention (`--flash-attn`)

**What it controls**: attention implementation for long contexts.

**Tweak when**:

* Long-context is slow → enable flash attention
* Flag unsupported → rebuild llama.cpp or omit

**Preferred result**:

* Enabled whenever supported, especially at 16k+ ctx.

---

### 7) Threads (`--threads 48`)

**What it controls**: CPU parallelism (tokenization + orchestration + any CPU compute).

**Tweak when**:

* CPU-bound ingestion/tokenization → increase threads
* Diminishing returns or contention → reduce (e.g., 32–48)

**Preferred result**:

* Fully GPU-offloaded: 32–48 often suffices.

---

### 8) Prompt Ingestion Throughput (`--batch-size`, `--ubatch-size`)

**What it controls**: how many prompt tokens are processed per step.

**Tweak when**:

* Loading large prompts is slow → increase batch sizes
* VRAM spikes / instability → reduce batch sizes

**Safe starting points**:

* `--batch-size 512` and `--ubatch-size 128`

**Aggressive (if stable)**:

* `--batch-size 1024` and `--ubatch-size 256`

---

### 9) Sampling (`--temp 0.1`, optional `--top-p`, `--repeat-penalty`)

**What it controls**: creativity vs determinism.

**Tweak when**:

* Outputs are too random → lower temp
* Outputs are too rigid/short → increase temp slightly (0.2)
* Repetition → add `--repeat-penalty 1.05`

**Preferred result for coding**:

* `--temp 0.1` (optionally `--top-p 0.9`)

---

### 10) Tool Calling / Templates (`--jinja`)

**What it controls**: whether server supports OpenAI-style `tools` formatting.

**Tweak when**:

* Client errors like: `tools param requires --jinja` → add `--jinja`
* You don’t want tools/function calling at all → disable tools in the client

---

### 11) Embeddings (`--embeddings`)

**What it controls**: enables embeddings endpoint support.

**Tweak when**:

* You want retrieval/search features → keep enabled
* You do not use embeddings → optional to disable (minor savings)

---

### 12) Warmup (`--no-warmup`)

**What it controls**: whether llama.cpp performs an initial warmup.

**Tweak when**:

* Startup time matters → keep `--no-warmup`
* First-request latency matters → allow warmup (remove the flag)

---

## Troubleshooting Playbook

### Symptom: OOM on GPU0 but GPU1 has room

* Reduce `--ctx-size`
* Bias split away from GPU0: `--tensor-split 0.45,0.55`
* Reduce `--batch-size` / `--ubatch-size`

### Symptom: Slow generation, low GPU utilization

* Ensure `--n-gpu-layers -1` (all on GPU)
* Reduce ctx-size (huge impact)
* Ensure no other process is using the GPUs

### Symptom: Slow prompt ingestion (loading big files)

* Increase `--batch-size` and `--ubatch-size` until stable
* Increase threads modestly (32 → 48)

### Symptom: Client shows “wrong models” or ignores chosen model

* Trust `GET /v1/models` as the server’s source of truth
* Configure the client to use the model id returned by `/v1/models`

### Symptom: `tools param requires --jinja`

* Start the server with `--jinja`
* Or disable tools/function calling in the client

---

## Monitoring Checklist

Run these while testing:

```bash
watch -n 0.5 nvidia-smi
```

Also capture:

* Server logs for tokens/sec
* One “short prompt” test (generation latency)
* One “big prompt” test (ingestion throughput)

---

## Change Strategy

Change **one knob at a time** and measure:

1. tokens/sec during ingestion
2. tokens/sec during generation
3. peak VRAM on each GPU

Keep the best-performing configuration as the new baseline.
