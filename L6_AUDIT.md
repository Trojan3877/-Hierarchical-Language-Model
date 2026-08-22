# L6 Engineering Audit

## Executive assessment

The repository has a meaningful research/software foundation: deterministic CI, a compact hierarchical PyTorch model, FastAPI boundaries, synthetic benchmark evidence, a from-scratch decoder-only research harness, retrieval prototypes, Docker packaging, and reproducibility-oriented documentation.

The main risk is narrative inflation across prototype and production-like surfaces. The strongest version of this project is an inspectable research system with disciplined evidence boundaries, not a claim of frontier LLM quality or internet-scale serving.

## Verified strengths

- Python 3.11 CI with Ruff, formatting, compile checks, pytest/coverage, benchmark generation, Docker build, Bandit, and pip-audit
- deterministic CI without gated model downloads
- explicit hierarchical tokenization/model path
- from-scratch decoder-only smoke/benchmark harness
- checked-in benchmark JSON
- FAISS default retrieval with Pinecone as an opt-in backend
- Dockerized service boundary
- research protocol documentation and citation metadata

## Critical gaps

### 1. Benchmark provenance

The checked-in benchmark records iteration count, harness, and CPU device, but not commit SHA, CPU model, OS, Python/PyTorch versions, warm-up count, memory, concurrency, or repeated-run uncertainty.

### 2. Service-boundary ambiguity

The Docker image serves `deployment/app.py`, while the README/API research narrative also describes `api/main.py`. These should either be unified or explicitly versioned as separate interfaces.

### 3. Prototype versus verified runtime

Retrieval, dashboard, hosted vector storage, and live Hugging Face paths should remain clearly separated from deterministic CI-supported behavior.

### 4. Model-quality evidence

Synthetic smoke loss is appropriate for plumbing validation but cannot support language-quality, reasoning, safety, or scaling-law claims. Real datasets, frozen evaluation protocols, multiple seeds, uncertainty, and data provenance are required.

### 5. Supply-chain promotion

The release path should add SBOM generation, container vulnerability scanning, and keyless image signing before making a stronger release-hardening claim.

## Promotion criteria

A future production-serving claim should require pinned model/runtime revisions, auth/rate limits, health/readiness semantics, load tests, p50/p95/p99 latency, TTFT/ITL for generation, error rates, resource utilization, observability, rollback evidence, signed images, and operational ownership.

A future research-quality improvement claim should require a frozen dataset/eval split, multiple seeds, uncertainty/confidence intervals, parameter/token/FLOP accounting, data-provenance records, and machine-readable experiment artifacts.
