# Makefile for quick commands

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py

docker-build:
	docker build -t hlm:latest .

docker-run:
	docker run -p 8000:8000 hlm:latest

.PHONY: ingest api dash

# Build vector index from docs/ and data/
ingest:
\tpython -m src.ingest --folders data docs

# Run FastAPI API
api:
\tuvicorn api.main:app --reload --port 8080

# Run Streamlit dashboard
dash:
\tstreamlit run dashboard/rag_app.py
