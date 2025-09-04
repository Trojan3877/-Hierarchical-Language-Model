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
