# Hierarchical Language Model

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Hierarchical%20Encoders-ee4c2c?logo=pytorch)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20Store-00A98F?logo=pinecone&logoColor=white)](https://www.pinecone.io/)
![CI](https://github.com/CoreyLeath-code/-Hierarchical-Language-Model/actions/workflows/ci.yml/badge.svg)
![Status](https://img.shields.io/badge/Status-Research%20Hardened-brightgreen)

Hierarchical Language Model is a research-oriented PyTorch and FastAPI project for
document representation learning. It demonstrates a token-to-sentence-to-document
pipeline with deterministic tests, API contract validation, benchmark capture, and
deployment hygiene suitable for continued production hardening.

The live Hugging Face generation path is intentionally opt-in. Local tests and CI use
safe deterministic paths so the project remains reproducible without gated model
credentials, GPU hardware, or large model downloads.


## Production Readiness Guide

> This section is the portfolio audit entry point for **-Hierarchical-Language-Model**. It describes an engineering promotion path; it is not a claim that the repository is already production-authorized.

[![CI](https://img.shields.io/github/actions/workflow/status/CoreyLeath-code/-Hierarchical-Language-Model/ci.yml?branch=main&label=CI)](https://github.com/CoreyLeath-code/-Hierarchical-Language-Model/actions) [![License](https://img.shields.io/github/license/CoreyLeath-code/-Hierarchical-Language-Model)](https://github.com/CoreyLeath-code/-Hierarchical-Language-Model/blob/main/LICENSE)

### Architecture flowchart

```mermaid
flowchart LR
    Input --> Validate[Schema + data checks] --> Model[Versioned model] --> Serve[API / dashboard] --> Observe[Metrics + drift]
```

### Quickstart and local validation

The supported local path should be reproducible from a clean checkout. The inferred stack for this repository is **Python/ML**.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest -q
```

If the project uses external services, model artifacts, cloud credentials, or private data, start them through documented local fixtures or mocks. Never place secrets or identifiable records in the repository.

### Research-style metrics and benchmarks

| Evidence | Required record |
|---|---|
| Correctness | Test command, commit SHA, runtime, and pass/fail result |
| Performance | Warm-up, sample count, concurrency, median, p95, p99, throughput, and memory |
| Data/model quality | Dataset version, split strategy, leakage controls, calibration, subgroup results, and uncertainty |
| Runtime | Image digest, health-check latency, resource limits, and rollback target |
| Security | Dependency, secret, SAST, container, and SBOM results |

A benchmark number belongs in a versioned artifact tied to a commit and hardware/runtime description. Engineering benchmarks must not be presented as clinical, financial, safety, or model-quality validation without the appropriate domain evidence.

### Extended Q&A

**What is production-ready for this repository?**  
A reproducible build, tested public contract, controlled configuration, observable runtime, documented security boundary, versioned artifacts, and a tested rollback path.

**What must remain explicit?**  
The intended use, excluded use, data/credential handling, model or algorithm limitations, and which metrics are measured versus aspirational.

**What should be completed next?**  
Use the linked production-readiness issue for this repository as the checklist. Resolve missing tests, deployment instructions, observability, supply-chain controls, and release evidence before attaching a production claim.



## From-Scratch LLM Architecture Research

![LLM Architecture](https://img.shields.io/badge/LLM%20Architecture-Decoder%20Transformer-7c3aed)
![Research Protocol](https://img.shields.io/badge/Research-Reproducible%20Protocol-2563eb)
![Scaling Laws](https://img.shields.io/badge/Scaling%20Laws-Data%20%2B%20Compute-f59e0b)
![Post-Training](https://img.shields.io/badge/Post--Training-SFT%20%2B%20Preference-059669)

This repository now contains a small decoder-only language model built from first
principles alongside the existing hierarchical document model. The goal is to make every
path inspectable: raw-data governance, tokenization, causal attention, scaling experiments,
post-training evaluation, retrieval grounding, serving, and release gates.

The implementation is intentionally small and honest. It is a systems and research
reference that can run on CPU; it does not claim frontier capability or substitute for
large-scale safety evaluation.

### Architecture path

```mermaid
flowchart LR
    Data[Licensed data + manifest] --> Clean[Normalize / deduplicate / PII review]
    Clean --> Tokenize[Tokenizer + frozen vocabulary]
    Tokenize --> Shards[Hashed token shards]
    Shards --> Train[PyTorch decoder LM]
    Train --> Eval[Held-out loss + safety suite]
    Eval --> SFT[SFT]
    SFT --> Pref[Preference optimization]
    Pref --> Registry[Model card + immutable artifact]
    Registry --> Serve[FastAPI / vLLM path]
    Serve --> Observe[p50 p95 p99 + drift + feedback]
```

| Path | Technology | Design decision |
|---|---|---|
| Data | JSONL manifests, SHA-256, deterministic Python filters | Every source has license, provenance, PII, and hash evidence |
| Model | PyTorch, RMSNorm, causal SDPA, SwiGLU, residual blocks | Small modules expose the math and preserve a clean upgrade path |
| Scale | single process → AMP/checkpointing → DDP → FSDP/ZeRO | Promote only when correctness and evaluation remain invariant |
| Post-training | SFT, DPO-style preference experiments, tool/RAG grounding | Compare checkpoints on the same frozen eval suite |
| Serving | FastAPI contract first; vLLM/TGI as the scale path | Local reproducibility precedes continuous batching |
| Retrieval | FAISS default; Pinecone opt-in | CI has no credential/network dependency |
| Evidence | JSON artifacts with seed, config, environment, percentiles | Measured values stay separate from aspirational targets |

### Run the from-scratch model

```bash
python research/llm_from_scratch.py \
  --mode smoke \
  --steps 5 \
  --device cpu \
  --output artifacts/llm-smoke.json
```

The smoke test reports parameter count, initial/final loss, seed, Python/PyTorch versions,
device, and whether the loss decreased. It uses synthetic next-token data only for plumbing
validation; that result is not a language-quality benchmark.

Run the hardware-aware latency harness:

```bash
python research/llm_from_scratch.py \
  --mode benchmark \
  --iterations 30 \
  --batch-size 2 \
  --device cpu \
  --output artifacts/llm-benchmark.json
```

Each artifact records warm-up behavior, model dimensions, parameter count, median/p95/p99
latency, tokens/second, OS, runtime, and seed. Repeat the same command on each hardware
target; do not compare CPU and GPU numbers without labeling the environment.

### Scaling-law research contract

Use the small/medium/large grid below as an experiment plan, not as pre-filled evidence:

| Variant | Layers | Width | Heads | Context | Required evidence |
|---|---:|---:|---:|---:|---|
| S | 2 | 128 | 4 | 128 | held-out loss, tokens, wall time |
| M | 4 | 256 | 8 | 256 | same data protocol and optimizer |
| L | 8 | 512 | 8 | 512 | same eval, memory, and failure record |

A valid study reports parameter count `N`, training tokens `D`, total FLOPs, hardware,
peak memory, final held-out loss, train/validation gap, and fit residuals. Only after several
points exist should a declared function such as
`L(N,D) = E + A/N^alpha + B/D^beta` be fitted with confidence intervals. A single run,
changed tokenizer, changed data mixture, or synthetic fixture cannot support a scaling-law
claim.

### Post-training and release path

1. SFT on curated instruction data with assistant-only loss masking.
2. Preference optimization with documented pair construction, annotator policy, ties, and
   disagreement handling.
3. Tool and retrieval grounding with recall@k, citation precision, and grounded-answer rate.
4. Safety regression tests for refusals, jailbreaks, privacy leakage, and tool misuse.
5. Model-card, SBOM, artifact-integrity, rollback, and ownership approval gates.

### Research-style metrics

| Area | Metrics required before claiming improvement |
|---|---|
| Optimization | train/validation loss, perplexity, seed count, confidence interval |
| Efficiency | step time, tokens/s, peak memory, utilization, batch/context |
| Serving | time-to-first-token, inter-token latency, p50/p95/p99, error rate |
| Data | source hashes, token count, duplicate rate, PII findings, split policy |
| Grounding | recall@k, MRR, citation precision, grounded-answer rate |
| Safety | refusal precision/recall, jailbreak success, privacy probes, regressions |
| Reliability | checkpoint restore, restart recovery, concurrency and load behavior |

No numerical LLM quality score is populated in this README until the dataset, evaluation
method, confidence interval, and reproducible artifact are checked in. The full data,
architecture, scaling-law, post-training, and promotion-gate protocol is in
[docs/llm-systems-research.md](docs/llm-systems-research.md).

## Architecture

```text
Document text
   |
   v
HierarchicalTokenizer
   |
   v
[batch, max_sentences, max_seq_len]
   |
   v
TokenEncoder -> Document GRU -> Classifier logits

FastAPI /generate
   |
   +-- safe fallback by default
   +-- live Hugging Face model when HLM_ENABLE_LIVE_MODEL=true
```

## Repository Layout

```text
api/                    FastAPI request schema and generation gateway
benchmarks/             Deterministic latency benchmark harness
deployment/             Docker Compose deployment blueprint
hierarchical_lm/        Core config, tokenizer, dataset, and model package
src/                    Extended encoder, RAG, provider, and ingestion prototypes
tests/                  Unit, API contract, and tensor-shape regression tests
benchmark-results.json  Recorded benchmark output
metrics.md              Research metrics and quality summary
```

## Pinecone vector retrieval

The RAG layer supports two interchangeable vector-store backends:

| Backend | Use case | Credentials |
|---|---|---|
| `faiss` (default) | Offline development, CI, and reproducible local experiments | None |
| `pinecone` | Hosted retrieval for a deployed service | `PINECONE_API_KEY`, index name, and namespace |

Pinecone is opt-in. Create a Pinecone index whose dimension matches the configured embedding model, then configure the environment without committing secrets:

```bash
cp .env.example .env
# Set these values in .env or your deployment secret store:
VECTORSTORE_BACKEND=pinecone
PINECONE_API_KEY=***
PINECONE_INDEX_NAME=hlm-documents
PINECONE_NAMESPACE=hlm-documents
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The embedding model above produces 384-dimensional vectors; the Pinecone index must use the same dimension and a compatible cosine metric. Ingest documents into the configured namespace:

```bash
python -m src.ingest --folders data docs
```

The existing `src/chains/rag.py` chain automatically loads the selected backend. Keep `VECTORSTORE_BACKEND=faiss` for local tests and CI; Pinecone credentials are never required for the deterministic fallback path. Pinecone network errors should be handled by the deployment's readiness and retry policy before exposing the RAG endpoint publicly.

## Research Metrics And Benchmarks

Latest recorded benchmark command:

```bash
python benchmarks/benchmark_hlm.py --iterations 100 --output benchmark-results.json
python -m json.tool benchmark-results.json
```

| Benchmark | Mean latency | Median latency | p95 latency | Evidence |
|---|---:|---:|---:|---|
| Tokenizer document encoding | 0.007402 ms | 0.006550 ms | 0.008000 ms | `benchmark-results.json` |
| Dataset materialization | 0.028561 ms | 0.025100 ms | 0.044400 ms | `benchmark-results.json` |
| Model forward pass | 2.207161 ms | 2.189750 ms | 2.790500 ms | `benchmark-results.json` |
| API safe fallback generation | 0.000217 ms | 0.000200 ms | 0.000200 ms | `benchmark-results.json` |

| Quality signal | Recorded value |
|---|---:|
| Tests | 13 passing |
| Runtime package coverage | 87% |
| Benchmark JSON validation | Passing |
| Live model downloads required for CI | 0 |
| Case-conflicting tracked log files | Resolved to `dailylog.md` |

See [metrics.md](metrics.md) for the full research metrics table and production target metrics.

## 9 Tier Deployment Hygiene

| Tier | Gate | Purpose |
|---:|---|---|
| 1 | Checkout source | Reproducible source snapshot |
| 2 | Python runtime setup | Standard Ubuntu latest runtime with pip cache |
| 3 | Dependency installation | Runtime and dev dependencies installed explicitly |
| 4 | Ruff static lint | Syntax, import, and maintainability checks |
| 5 | Ruff format verification | Consistent source formatting |
| 6 | Python import compilation | Import-time and syntax validation |
| 7 | Unit, API, and coverage tests | Regression coverage for core runtime behavior |
| 8 | Benchmark JSON validation | Machine-readable performance evidence |
| 9 | Docker, Bandit, and pip-audit | Deployment build, SAST, and dependency vulnerability hygiene |

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Run the API in safe fallback mode:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain hierarchical reasoning","max_tokens":64}'
```

Enable live model generation only when credentials, hardware, and model access are ready:

```bash
export HLM_ENABLE_LIVE_MODEL=true
export HLM_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export HLM_MODEL_REVISION=main
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Validation

```bash
ruff check api hierarchical_lm benchmarks tests
ruff format --check api hierarchical_lm benchmarks tests
pytest --cov=api --cov=hierarchical_lm --cov-report=term-missing
python benchmarks/benchmark_hlm.py --iterations 100 --output benchmark-results.json
python -m json.tool benchmark-results.json
python -m compileall -q api hierarchical_lm benchmarks tests src
```

## Deployment

Build the container:

```bash
docker build -t hierarchical-language-model:latest .
```

Run the local deployment blueprint:

```bash
docker compose -f deployment/docker-compose.yml up --build
```

## Known Gaps

- The deterministic benchmark uses compact synthetic inputs; it is not a large-corpus model-quality evaluation.
- Live Hugging Face generation requires explicit opt-in and valid model access.
- The RAG and dashboard modules remain prototype extensions and are not part of the core CI coverage gate.
- Production auth, rate limiting, TLS, model registry controls, and observability backends should be added before internet-facing deployment.

## Author

Corey Leath

AI / ML Engineer focused on LLM systems, MLOps, and distributed AI infrastructure.
