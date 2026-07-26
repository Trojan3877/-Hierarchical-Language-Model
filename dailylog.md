# Engineering Operations & System Hygiene Daily Log

This log records notable architecture, deployment, and quality-system updates for the
Hierarchical Language Model repository.

## 2026-07-25 - Pinecone Retrieval Engineering Practice

- Added an opt-in Pinecone vector-store backend in [PR #7](https://github.com/CoreyLeath-code/-Hierarchical-Language-Model/pull/7) while retaining FAISS as the default for offline development and CI.
- Added explicit index and namespace configuration through deployment environment variables; Pinecone credentials remain outside source control.
- Preserved deterministic backend-selection tests and documented the embedding-dimension/index compatibility requirement.
- Opened [issue #8](https://github.com/CoreyLeath-code/-Hierarchical-Language-Model/issues/8) to track the operating practice: secret handling, namespace isolation, readiness/retry behavior, rollback to FAISS, and separate Pinecone quality/latency/error/cost benchmarks.
- Current boundary: Pinecone is an integration path, not a production authorization; deployment evidence and hosted retrieval benchmarks remain required.

## 2026-07-13

- Added deterministic benchmark capture for tokenizer, dataset, model forward pass, and API fallback paths.
- Upgraded CI into a 9-tier deployment hygiene workflow with linting, formatting, compilation, tests, benchmark validation, Docker build, Bandit, and dependency audit gates.
- Made live Hugging Face model loading opt-in so CI and local tests do not require gated model credentials or GPU hardware.
- Consolidated duplicate case-conflicting log files into this lowercase `dailylog.md` path for cross-platform checkout safety.

## 2026-07-03 - Production Modeling & CI/CD Validation Matrix

- Modular PyTorch core engineered into an isolated package layout: `config.py`, `tokenizer.py`, `dataset.py`, and `model.py`.
- Hierarchical ingestion pipeline implemented with token-level bidirectional GRUs mapped into document-level sequence processing.
- Automated regression verification added with native unit tests to assert tensor rank and shape contracts.
- Initial multi-stage GitHub Actions validation introduced for formatting, SAST, token hygiene, and test execution.

## 2025-12-03

- Wrote quality metrics documentation.
- Updated `CONTRIBUTING.md`.

## 2025-12-02

- Added MLflow experiment tracking.
- Dockerized the API.
- Created n8n workflow scaffolding.

## 2025-12-01

- Integrated Llama-3 inference scaffolding.
- Set up FastAPI endpoints.
