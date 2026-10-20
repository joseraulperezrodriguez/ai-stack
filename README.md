# Building a Custom AI Stack

A tiny GPT built end-to-end — small enough to train in hours on a single consumer GPU, but structurally identical to production LLM systems: same attention, same training loop, same inference tricks, same serving stack. The final service will be **publicly hosted on the web**, not local-only.

## Status at a Glance

| Phase | Status | Notes |
|---|---|---|
| 1 — Data & Tokenizer | Not started | Design ready, saved to the bpe-tokenizer-description file |
| 2 — Model | Not started | 6 layers, 6 heads, 384 dim |
| 3 — Training Loop | Not started | |
| 3.5 — Evaluation | Not started | |
| 4 — Inference | Not started | |
| 5 — Productionization (a/b/c) | Not started | Will be publicly hosted on the web |
| 6 — Stretch goal | Not chosen yet | Pick exactly one: DDP, LoRA, or kernels |

## Key Decisions

- **Data source:** English Wikipedia articles. Start with a manageable subset (a few hundred MB to a few GB); keep a scale-up path to the full 45 GB dump.
- **Model target:** 6 layers, 6 heads, 384 dim; single consumer GPU for training.
- **Hosting:** publicly hosted web service — design model size and quantization for a modest hosted instance.
- **Phase 6 stretch goal:** pick exactly one — DDP, LoRA, or custom kernels.

## Prerequisites / Hardware Reality Check

- A single consumer GPU with ≥8 GB VRAM is sufficient for Phases 1–4 with the 6-layer model on a small corpus.
- CUDA kernel work (Phase 2) and Triton benchmarking (Phase 6) assume an NVIDIA GPU + recent CUDA toolkit.
- Budget 2–3× the optimistic time for Phase 2 and Phase 4 debugging — kernel correctness and KV-cache edge cases are where time goes.
- No GPU? Phases 1, 3 (small scale), and most of Phase 5 still work.
- **Reference implementations:** nanoGPT, llm.c, minbpe (Karpathy); BPE algorithm spec in `high-level-description.txt`.

## The Plan

### Phase 1 — Data Collection & Tokenizer

- English Wikipedia corpus; version raw dump and extracted subset separately; regenerate `.bin` tokens on demand.
- BPE tokenizer from scratch: 256 base byte tokens, iterative pair merging, maps + max heap for counts/positions, overlap handling, tie-break on lowest token id.
- Optional: benchmark compression ratio (bytes/token) vs. GPT-2's tokenizer on the same text.

**Done when:** tokenizer round-trips (encode → decode = identical text); train/val split regenerable from a single command.

### Phase 2 — Model, Written by You

- Transformer decoder in raw PyTorch: embeddings, positional encoding, multi-head self-attention, layernorm, MLP block, causal mask. No `nn.TransformerDecoder` — write attention math yourself.
- Reimplement **one** component as a custom CUDA kernel (attention or layernorm).
- Keep it tiny: 6 layers, 6 heads, 384 dim.

**Done when:** loss decreases steadily on a tiny run with coherent samples; custom kernel matches PyTorch within tolerance (max abs diff < 1e-5 in fp32).

### Phase 3 — Training Loop

- Batching, cross-entropy loss, AdamW, LR warmup + cosine decay, gradient clipping.
- Production habits early: checkpointing, config file (no hardcoded hyperparams), logging (CSV or wandb), fixed seed.
- Optional: implement Adam's update rule by hand once.

**Done when:** kill training mid-run → resume from checkpoint → identical loss trajectory; config alone reproduces a run.

### Phase 3.5 — Evaluation (not optional)

- Validation perplexity at every checkpoint.
- Fixed set of 10–20 generation prompts (fixed seed/params), run at every milestone, diff across checkpoints.
- Results log (CSV/markdown): checkpoint, val perplexity, samples, eval time.

**Done when:** every checkpoint has val perplexity + logged samples; "is checkpoint N better than N−1?" answerable in one sentence with numbers.

### Phase 4 — Inference

- Greedy decoding first, then temperature + top-k/top-p sampling.
- Implement **KV-caching yourself** (easy to get subtly wrong).
- Batch multiple prompts; measure throughput vs. latency.

**Done when:** KV-cached inference is **bit-identical** to non-cached (same seed/params); measured speedup recorded (expect ≥2× on longer prompts).

### Phase 5 — Productionization

Local-first for the learning loop; 5c takes it to the web. Design choices in Phases 2–4 keep the hosting target in mind (small hosted GPU, or CPU with quantization).

- FastAPI server with `/generate` endpoint.
- Quantization (int8, or fp16/bf16) with measured speed/quality tradeoff.
- Docker containerization.

**5a — Serving Realism:** streaming token output (SSE); load test at 1, 4, 16 concurrent requests (wrk, locust, or asyncio script).

**5b — RAG Layer:** embed a few dozen local doc chunks; lightweight local vector index (NumPy matrix + cosine similarity); `/retrieve` endpoint wired into `/generate`; measure effect on fact-style prompts.

**5c — Web Hosting (publicly available):** hosting target & cost model first (cheap cloud VM with quantized model on CPU vs. small hosted/serverless GPU); HTTPS, stable endpoint, API-key auth, rate limiting, request timeouts, `/health` endpoint; same Docker container deploys unchanged; measure cold-start, time-to-first-token, cost per 1k requests.

**Done when:** clean `docker run` serves requests; streaming works; throughput/latency at multiple concurrency levels; quantization numbers recorded; (5c) reachable over HTTPS with auth + rate limiting; monthly running cost known.

### Phase 6 — Stretch Goals (pick ONE)

- Multi-GPU training (DDP) — even 2 GPUs.
- LoRA fine-tuning instead of full training.
- More custom CUDA/Triton kernels benchmarked vs. PyTorch's fused kernels.

**Done when (matching one):** DDP loss curves match single-GPU; or LoRA hits target loss at <1% of full fine-tune memory; or custom kernel beats PyTorch's fused op on a documented benchmark.

## Effort Guide

| Phases | Nature | Effort |
|---|---|---|
| 1–4 | Core loop | A couple of weeks of evenings |
| 3.5 | Continuous discipline | ~1 day to set up |
| 5 | Script → system (incl. hosting) | ~1–2 weeks |
| 6 | Deep dive, one piece only | Open-ended, scoped deliberately |
## Dependent Projects

- [gpu_programming](https://github.com/joseraulperezrodriguez/gpu_programming) — custom CUDA kernels (matmul, attention, layernorm) reused for Phase 2
- [bpe-tokenizer](https://github.com/joseraulperezrodriguez/bpe-tokenizer) — BPE tokenizer implementation used for Phase 1
