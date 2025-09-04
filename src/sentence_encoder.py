# src/sentence_encoder.py

"""
Sentence Encoder Module
-----------------------
Encodes individual sentences into vector representations using
a pretrained Transformer (e.g., BERT) from Hugging Face.
This forms the first layer of the Hierarchical Language Model (HLM).
"""

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

class SentenceEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initializes the sentence encoder.

        Args:
            model_name (str): Hugging Face model name to load.
            device (str): "cuda" or "cpu" depending on availability.
        """
        super(SentenceEncoder, self).__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def forward(self, sentences):
        """
        Encodes a list of sentences into embeddings.

        Args:
            sentences (list of str): Input sentences.

        Returns:
            torch.Tensor: Sentence embeddings [batch_size, hidden_dim].
        """
        # Tokenize sentences
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Run through model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Take the [CLS] token embedding as sentence representation
        sentence_embeddings = outputs.last_hidden_state[:, 0, :]  

        return sentence_embeddings
