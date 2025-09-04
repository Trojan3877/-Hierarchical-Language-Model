# api/main.py

"""
FastAPI Inference API for Hierarchical Language Model (HLM)
-----------------------------------------------------------
Provides endpoints to interact with the trained model.
Supports document classification via POST requests.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch

from src.train import HLM

# Initialize FastAPI app
app = FastAPI(
    title="Hierarchical Language Model API",
    description="FastAPI interface for document classification using HLM",
    version="1.0"
)

# Load model (for demo, using fresh model; in practice load trained weights)
model = HLM(num_classes=2)
model.eval()

# Define request schema
class DocumentInput(BaseModel):
    document: list[list[list[str]]]  # [batch, paragraphs, sentences]


@app.get("/")
def home():
    return {"message": "HLM API is running. Use /predict to classify documents."}


@app.post("/predict")
def predict(input_data: DocumentInput):
    """
    Accepts a nested list of documents and returns predictions.
    Example format:
    {
      "document": [
        [
          ["The movie was amazing!", "I loved the acting."],
          ["The cinematography was stunning.", "Highly recommend."]
        ]
      ]
    }
    """
    # Convert request into model input
    documents = input_data.document
    with torch.no_grad():
        outputs = model(documents)
        predictions = torch.argmax(outputs, dim=1).tolist()

    return {"predictions": predictions}
