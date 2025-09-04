# src/document_encoder.py

"""
Document Encoder Module
-----------------------
Aggregates paragraph embeddings into a document-level representation.
Uses a bidirectional GRU to capture relationships across paragraphs
and generate a global embedding for the entire document.
"""

import torch
import torch.nn as nn

class DocumentEncoder(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, num_layers: int = 1, bidirectional: bool = True):
        """
        Initializes the document encoder.

        Args:
            input_dim (int): Dimension of paragraph embeddings (default 1024 if ParagraphEncoder is bidirectional with 512 units).
            hidden_dim (int): Hidden dimension for GRU/LSTM.
            num_layers (int): Number of recurrent layers.
            bidirectional (bool): Use bidirectional GRU for global context.
        """
        super(DocumentEncoder, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, paragraph_embeddings: torch.Tensor):
        """
        Encodes a sequence of paragraph embeddings into a document embedding.

        Args:
            paragraph_embeddings (torch.Tensor): [batch_size, num_paragraphs, input_dim]

        Returns:
            torch.Tensor: Document embedding [batch_size, hidden_dim * (2 if bidirectional else 1)]
        """
        # Run through GRU
        outputs, hidden = self.gru(paragraph_embeddings)

        # Concatenate forward & backward states if bidirectional
        if self.bidirectional:
            document_embedding = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            document_embedding = hidden[-1]

        return document_embedding
