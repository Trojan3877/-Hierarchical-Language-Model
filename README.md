## 🚀 Tech Stack
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Helm-blue?logo=kubernetes)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

⚡ Features
- Sentence-level encoder for local context  
- Paragraph-level encoder for cross-sentence context  
- Document-level encoder for global representation  
- API layer powered by **FastAPI**  
- Real-time metrics dashboard with **Streamlit**  
- Deployment-ready with **Docker** + **Kubernetes (Helm)**  

Architecture

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


# -Hierarchical-Language-Model
This project implements a **Hierarchical Language Model (HLM)** that processes text at multiple levels of granularity.   The system uses sentence-level, paragraph-level, and document-level encoders to capture local and global context, producing richer representations for natural language tasks.  
