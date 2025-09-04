# src/train.py

"""
Training Script for Hierarchical Language Model (HLM)
-----------------------------------------------------
This script wires together the Sentence, Paragraph, and Document
Encoders into a full pipeline. Includes a sample classification
head for downstream tasks like sentiment analysis or topic classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from sentence_encoder import SentenceEncoder
from paragraph_encoder import ParagraphEncoder
from document_encoder import DocumentEncoder


class HLM(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(HLM, self).__init__()
        # Encoders
        self.sentence_encoder = SentenceEncoder()
        self.paragraph_encoder = ParagraphEncoder()
        self.document_encoder = DocumentEncoder()

        # Classification head
        doc_hidden_dim = self.document_encoder.hidden_dim * (2 if self.document_encoder.bidirectional else 1)
        self.classifier = nn.Linear(doc_hidden_dim, num_classes)

    def forward(self, documents: list[list[list[str]]]):
        """
        Full pipeline: sentences -> paragraphs -> document -> classification

        Args:
            documents (list): Nested list of strings:
                              [batch, paragraphs, sentences]

        Returns:
            torch.Tensor: Predictions [batch_size, num_classes]
        """
        batch_embeddings = []

        for doc in documents:
            paragraph_embeddings = []
            for paragraph in doc:
                # Encode sentences
                sent_embs = self.sentence_encoder(paragraph)  # [num_sents, hidden_dim]
                sent_embs = sent_embs.unsqueeze(0)  # Add batch dim
                # Encode paragraph
                para_emb = self.paragraph_encoder(sent_embs)  # [1, hidden_dim*2]
                paragraph_embeddings.append(para_emb)

            # Stack all paragraph embeddings for this doc
            paragraph_embeddings = torch.stack(paragraph_embeddings, dim=1)  # [1, num_paragraphs, hidden_dim*2]
            # Encode document
            doc_emb = self.document_encoder(paragraph_embeddings)  # [1, hidden_dim*2]
            batch_embeddings.append(doc_emb)

        # Concatenate all doc embeddings into batch
        doc_batch = torch.cat(batch_embeddings, dim=0)  # [batch_size, hidden_dim*2]

        # Classification head
        logits = self.classifier(doc_batch)
        return logits


# -----------------
# Training Function
# -----------------
def train_model(model, data, labels, epochs=2, lr=1e-4):
    """
    Simple training loop with dummy data.

    Args:
        model: HLM model
        data: Nested list of docs (batch, paragraphs, sentences)
        labels: Ground truth labels
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")


# -----------------
# Demo Run
# -----------------
if __name__ == "__main__":
    # Dummy dataset: 1 document, 2 paragraphs, 2 sentences each
    documents = [
        [
            ["The movie was great.", "I really enjoyed the acting."],
            ["The cinematography was stunning.", "It kept me engaged."]
        ]
    ]
    labels = torch.tensor([1])  # Example: positive sentiment

    model = HLM(num_classes=2)
    train_model(model, documents, labels, epochs=1)
