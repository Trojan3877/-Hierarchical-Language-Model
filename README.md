<div align="center">

<img src="assets/logo.png" alt="Hierarchical Language Model Logo" width="120"/>

# 🧠 Hierarchical Language Model (HLM)

A **capstone-ready**, **production-aware** AI project that models text hierarchically:  
Sentence → Paragraph → Document → Prediction  


### 📊 Project Health & Features

# Hierarchical Language Model (HLM)

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-purple.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Docker](https://img.shields.io/badge/Docker-ready-0db7ed.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![CI](https://github.com/Trojan3877/-Hierarchical-Language-Model/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/Trojan3877/-Hierarchical-Language-Model)
![Stars](https://img.shields.io/github/stars/Trojan3877/-Hierarchical-Language-Model?style=social)
![Forks](https://img.shields.io/github/forks/Trojan3877/-Hierarchical-Language-Model?style=social)

> A modular Hierarchical Language Model with Hugging Face + LangChain RAG integration.


### 🚀 Capstone & Deployment Readiness

![Capstone Ready](https://img.shields.io/badge/Status-Capstone%20Ready-brightgreen?logo=checkmarx&logoColor=white)
![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?logo=docker&logoColor=white)
![Docs](https://img.shields.io/badge/Docs-Available-brightgreen?logo=readthedocs&logoColor=white)
![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-brightgreen?logo=streamlit&logoColor=white)
![API](https://img.shields.io/badge/FastAPI-Available-brightgreen?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20Enabled-brightgreen?logo=mlflow&logoColor=white)

</div>

---


## 🚀 Tech Stack
![Build](https://img.shields.io/github/actions/workflow/status/Trojan3877/Hierarchical-Language-Model/ci.yml?branch=main&label=Build&logo=github&color=brightgreen)
![Tests](https://img.shields.io/github/actions/workflow/status/Trojan3877/Hierarchical-Language-Model/ci.yml?branch=main&label=Tests&logo=pytest&color=brightgreen)
![Python](https://img.shields.io/badge/Python-3.10-brightgreen?logo=python)
![License: MIT](https://img.shields.io/github/license/Trojan3877/Hierarchical-Language-Model?color=brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/Trojan3877/Hierarchical-Language-Model?logo=git&label=Last%20Commit&color=brightgreen)
![Repo Size](https://img.shields.io/github/repo-size/Trojan3877/Hierarchical-Language-Model?logo=github&label=Repo%20Size&color=brightgreen)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen?logo=github)
![Capstone Ready](https://img.shields.io/badge/Status-Capstone%20Ready-brightgreen?logo=checkmarx&logoColor=white)
![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?logo=docker&logoColor=white)
![Docs](https://img.shields.io/badge/Docs-Available-brightgreen?logo=readthedocs&logoColor=white)
![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-brightgreen?logo=streamlit&logoColor=white)
![API](https://img.shields.io/badge/FastAPI-Available-brightgreen?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20Enabled-brightgreen?logo=mlflow&logoColor=white)




⚡ Features
- Sentence-level encoder for local context  
- Paragraph-level encoder for cross-sentence context  
- Document-level encoder for global representation  
- API layer powered by **FastAPI**  
- Real-time metrics dashboard with **Streamlit**  
- Deployment-ready with **Docker** + **Kubernetes (Helm)**  

Architecture
## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Input
        Q[User Question]
    end

    subgraph Retrieval
        VS[FAISS Vector Store]
        EMB[Sentence-Transformer Embeddings]
        Q --> EMB --> VS
        VS -->|Top-k Docs| CTX[Context Chunks]
    end

    subgraph HLM
        SENC[Sentence Encoder]
        PENC[Paragraph Encoder]
        DENC[Document Encoder]
        CTX --> SENC --> PENC --> DENC
        DENC --> HCXT[Hierarchical Context]
    end

    subgraph Generation
        HF[Hugging Face Model]
        HCXT --> HF
        HF --> A[Answer]
    end

    subgraph API/UI
        API[FastAPI Endpoint]
        UI[Streamlit Dashboard]
        A --> API
        A --> UI
    end

```mermaid
flowchart TD
    A[Input Text] --> B[Sentence Encoder]
    B --> C[Paragraph Encoder]
    C --> D[Document Encoder]
    D --> E[Task Layer: QA / Summarization / Classification]

📊 Results (Sample Placeholder)
Task	Accuracy	F1 Score	Latency (ms)
Text Classification	91%	0.89	42 ms
Summarization	Rouge-L 0.47	N/A	58 ms

Hierarchical-Language-Model/
├── config/               # YAML configs
├── data/                 # sample text datasets
├── notebooks/            # Jupyter exploration
├── src/                  # core encoders + training
├── api/                  # FastAPI inference layer
├── dashboard/            # Streamlit metrics app
├── tests/                # unit tests
├── requirements.txt
├── Dockerfile
├── helm/                 # Kubernetes deployment
└── README.md


---

## 📚 Documentation

For full details, see the docs folder:  

- [📖 Project Overview](docs/overview.md)  
- [⚙️ Usage Guide](docs/usage.md)  
- [🏗️ System Design](docs/system_design.md)  

---

## 🔒 Policies & Standards
- [📑 Changelog](CHANGELOG.md)  
- [🛣️ Roadmap](ROADMAP.md)  
- [📊 Project Status](PROJECT_STATUS.md)  
- [📜 Code of Conduct](CODE_OF_CONDUCT.md)  
- [🔒 Security Policy](SECURITY.md)  
- [🤝 Contributing Guide](CONTRIBUTING.md)  
- [📖 Citation](CITATION.cff)  

---

## 🛠️ Quick Start

Clone the repo and check the [Usage Guide](docs/usage.md) for full instructions.  

```bash
git clone https://github.com/Trojan3877/Hierarchical-Language-Model.git
cd Hierarchical-Language-Model
pip install -r requirements.txt


## 🔒 Policies & Standards
- [📑 Changelog](CHANGELOG.md)  
- [🛣️ Roadmap](ROADMAP.md)  
- [📊 Project Status](PROJECT_STATUS.md)  
- [📜 Code of Conduct](CODE_OF_CONDUCT.md)  
- [🔒 Security Policy](SECURITY.md)  
- [🤝 Contributing Guide](CONTRIBUTING.md)  
- [📖 Citation](CITATION.cff)  

# -Hierarchical-Language-Model
This project implements a **Hierarchical Language Model (HLM)** that processes text at multiple levels of granularity.   The system uses sentence-level, paragraph-level, and document-level encoders to capture local and global context, producing richer representations for natural language tasks.  
