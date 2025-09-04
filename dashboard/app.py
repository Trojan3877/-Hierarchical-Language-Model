# dashboard/app.py

"""
Streamlit Dashboard for Hierarchical Language Model (HLM)
---------------------------------------------------------
Provides a simple UI for text input, model predictions,
and visualization of evaluation metrics.
"""

import streamlit as st
import torch
from src.train import HLM
from src.evaluate import evaluate_model

# -----------------
# Setup
# -----------------
st.set_page_config(page_title="Hierarchical Language Model Dashboard", layout="wide")
st.title("🧠 Hierarchical Language Model (HLM) Dashboard")
st.write("This dashboard allows interactive testing and visualization of the HLM.")

# Load model (demo: untrained model — replace with trained weights later)
model = HLM(num_classes=2)
model.eval()


# -----------------
# Sidebar Controls
# -----------------
st.sidebar.header("⚙️ Settings")
task = st.sidebar.selectbox("Choose Task", ["Document Classification"])

# -----------------
# Document Input
# -----------------
st.subheader("📄 Input Document")
st.markdown("Enter text organized into paragraphs and sentences for classification.")

# Example input format
example = [
    [
        ["The food was delicious.", "The service was excellent."],
        ["I would definitely recommend this place.", "The atmosphere was great."]
    ]
]

user_input = st.text_area(
    "Enter a nested JSON-like structure (paragraphs -> sentences).",
    value=str(example),
    height=200
)

# -----------------
# Prediction
# -----------------
if st.button("🔮 Run Prediction"):
    try:
        # Convert string input into Python list
        documents = eval(user_input)

        with torch.no_grad():
            outputs = model(documents)
            predictions = torch.argmax(outputs, dim=1).tolist()

        st.success(f"Prediction Results: {predictions}")

    except Exception as e:
        st.error(f"Error processing input: {e}")

# -----------------
# Metrics Visualization (Placeholder)
# -----------------
st.subheader("📊 Evaluation Metrics")
st.write("Placeholder metrics until a trained model is evaluated.")

dummy_metrics = {"accuracy": 0.91, "precision": 0.90, "recall": 0.89, "f1": 0.90}
st.json(dummy_metrics)
