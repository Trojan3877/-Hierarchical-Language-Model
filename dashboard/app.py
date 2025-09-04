# dashboard/app.py

"""
Streamlit Demo for Hierarchical Language Model (HLM)
----------------------------------------------------
Provides a simple web interface to test document classification.
"""

import streamlit as st
import torch
from src.sentence_encoder import SentenceEncoder
from src.paragraph_encoder import ParagraphEncoder
from src.document_encoder import DocumentEncoder

# -----------------------
# Load Model Components
# -----------------------
@st.cache_resource
def load_model():
    sent_encoder = SentenceEncoder()
    para_encoder = ParagraphEncoder()
    doc_encoder = DocumentEncoder()
    return sent_encoder, para_encoder, doc_encoder

sent_encoder, para_encoder, doc_encoder = load_model()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="HLM Demo", layout="centered")
st.title("🧠 Hierarchical Language Model (HLM)")
st.write("Capstone-ready demo: Sentence → Paragraph → Document → Prediction")

# Text input
user_text = st.text_area("Enter a document for classification:", height=200)

if st.button("Run Prediction") and user_text.strip():
    with st.spinner("Encoding text..."):
        # Dummy pipeline (for demo only) – split into sentences and paragraphs
        sentences = [user_text.split(".")]
        paragraphs = [sentences]

        # Encode (simplified)
        with torch.no_grad():
            sent_embs = sent_encoder(sentences[0])
            para_emb = para_encoder(sent_embs.unsqueeze(0))
            doc_emb = doc_encoder(para_emb.unsqueeze(0))

        # Fake prediction head (random for now until model trained)
        prediction = torch.sigmoid(torch.randn(1)).item()
        label = "Positive" if prediction > 0.5 else "Negative"

    st.success(f"Prediction: **{label}** (score={prediction:.2f})")

    st.write("🔍 Embedding Shape:", doc_emb.shape)
    st.caption("Note: This is a demo. Replace with trained model weights for real predictions.")

