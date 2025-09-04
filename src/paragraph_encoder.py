# src/paragraph_encoder.py

"""
Paragraph Encoder Module
------------------------
Aggregates sentence embeddings into a paragraph-level representation.
Uses a bidirectional GRU (or LSTM) to capture relationships between
sentences within a paragraph.
"""

import torch
import torch.nn as nn

class ParagraphEncoder(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, num_layers: int = 1, bidirectional: bool = True):
        """
        Initializes the paragraph encoder.

        Args:
            input_dim (int): Dimension of sentence embeddings (default 768 for BERT).
            hidden_dim (int): Hidden dimension for GRU/LSTM.
            num_layers (int): Number of recurrent layers.
            bidirectional (bool): Use bidirectional GRU for better context.
        """
        super(ParagraphEncoder, self).__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, sentence_embeddings: torch.Tensor):
        """
        Encodes a sequence of sentence embeddings into a paragraph embedding.

        Args:
            sentence_embeddings (torch.Tensor): [batch_size, num_sentences, input_dim]

        Returns:
            torch.Tensor: Paragraph embedding [batch_size, hidden_dim * (2 if bidirectional else 1)]
        """
        # Run through GRU
        outputs, hidden = self.gru(sentence_embeddings)

        # If bidirectional, concatenate forward & backward hidden states
        if self.bidirectional:
            paragraph_embedding = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            paragraph_embedding = hidden[-1]

        return paragraph_embedding
