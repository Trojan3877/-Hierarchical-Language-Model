# Contributing to Hierarchical-Language-Model

## How to Contribute
1. Fork the repository
2. Create a feature branch
3. Add tests
4. Submit a PR

## Code Style
- Python 3.10+
- PEP8 compliance
- Type hints required

## Workflows
Use MLflow UI to track experiments:
```bash
mlflow ui

---

# 8️⃣ — Tests

### 📁 `tests/test_api.py`

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200

def test_generate():
    res = client.post("/generate", json={"prompt": "Hello", "max_tokens": 10})
    assert res.status_code == 200
    assert "output" in res.json()
