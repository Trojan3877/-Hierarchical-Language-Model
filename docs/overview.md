# 🧠 Hierarchical Language Model (HLM) - Overview

## What is HLM?
The Hierarchical Language Model (HLM) captures text meaning at **three levels**:
- **Sentence-level:** Local context
- **Paragraph-level:** Relationships across sentences
- **Document-level:** Global meaning

## Why?
Traditional transformers handle sequences but don’t explicitly model hierarchy.  
HLM bridges that gap, improving **classification, summarization, and QA**.

## Key Features
- Modular encoders (Sentence → Paragraph → Document)
- Training & evaluation pipelines
- Inference API with FastAPI
- Streamlit dashboard for real-time metrics
- Config-driven (YAML) hyperparameter management
- Containerized (Docker + Helm)
- Experiment tracking with MLflow
- Professional repo structure with CI/CD, tests, and docs
