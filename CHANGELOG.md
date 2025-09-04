# 📑 Changelog
All notable changes to this project will be documented here.  
This project follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH).

---

## [1.0.0] - 2025-08-27
### Added
- Initial release of **Hierarchical Language Model (HLM)** 🚀
- Core encoders: Sentence → Paragraph → Document
- Training pipeline (`train.py`) and evaluation (`evaluate.py`)
- FastAPI inference API (`api/main.py`)
- Streamlit dashboard (`dashboard/app.py`)
- Config management (`config/config.yaml`)
- Unit tests (`tests/test_encoders.py`)
- Structured logging (`utils/logger.py`)
- MLflow experiment tracking (`utils/tracker.py`)
- Data ingestion pipeline (`data_ingest.py`)
- Exploratory notebook (`notebooks/exploration.ipynb`)

### Infrastructure
- Dockerfile for containerization
- Helm chart for Kubernetes deployment
- Makefile for developer convenience
- Pre-commit hooks (`.pre-commit-config.yaml`)
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`)
- `.gitignore`, `.env.example`, `requirements.txt`, `CONTRIBUTING.md`, `CITATION.cff`

---

## [Unreleased]
### Planned
- Pretrained weights integration
- Multi-task support (summarization, QA)
- Model optimization with quantization
- End-to-end cloud deployment demo (AWS/GCP/Render)
