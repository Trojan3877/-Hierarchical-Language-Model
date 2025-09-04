# tests/test_encoders.py

"""
Unit Tests for HLM Encoders
---------------------------
Validates that Sentence, Paragraph, and Document encoders
work correctly and produce expected output shapes.
"""

import torch
import pytest

from src.sentence_encoder import SentenceEncoder
from src.paragraph_encoder import ParagraphEncoder
from src.document_encoder import DocumentEncoder


# -----------------
# Fixtures
# -----------------
@pytest.fixture
def dummy_sentences():
    return ["The cat sat on the mat.", "The dog barked loudly."]


@pytest.fixture
def dummy_paragraphs():
    return [
        ["The cat sat on the mat.", "The dog barked loudly."],
        ["The sun was shining.", "It was a beautiful day."]
    ]


# -----------------
# Tests
# -----------------
def test_sentence_encoder(dummy_sentences):
    encoder = SentenceEncoder()
    embeddings = encoder(dummy_sentences)
    assert embeddings.shape[0] == len(dummy_sentences)   # batch size = num sentences
    assert embeddings.shape[1] == 768                   # BERT hidden size


def test_paragraph_encoder(dummy_paragraphs):
    sent_encoder = SentenceEncoder()
    para_encoder = ParagraphEncoder()

    # Encode each paragraph into sentence embeddings
    paragraph_embeddings = []
    for paragraph in dummy_paragraphs:
        sent_embs = sent_encoder(paragraph)
        para_emb = para_encoder(sent_embs.unsqueeze(0))  # add batch dim
        paragraph_embeddings.append(para_emb)

    assert len(paragraph_embeddings) == len(dummy_paragraphs)
    assert paragraph_embeddings[0].shape[1] == 1024     # bidirectional GRU (512*2)


def test_document_encoder(dummy_paragraphs):
    sent_encoder = SentenceEncoder()
    para_encoder = ParagraphEncoder()
    doc_encoder = DocumentEncoder()

    # Encode paragraphs
    paragraph_embeddings = []
    for paragraph in dummy_paragraphs:
        sent_embs = sent_encoder(paragraph)
        para_emb = para_encoder(sent_embs.unsqueeze(0))
        paragraph_embeddings.append(para_emb)

    # Stack and encode document
    paragraph_embeddings = torch.stack(paragraph_embeddings, dim=1)
    doc_emb = doc_encoder(paragraph_embeddings)

    assert doc_emb.shape[1] == 1024  # bidirectional output


def test_pipeline(dummy_paragraphs):
    """Full pipeline: sentence -> paragraph -> document."""
    sent_encoder = SentenceEncoder()
    para_encoder = ParagraphEncoder()
    doc_encoder = DocumentEncoder()

    # Encode document
    paragraph_embeddings = []
    for paragraph in dummy_paragraphs:
        sent_embs = sent_encoder(paragraph)
        para_emb = para_encoder(sent_embs.unsqueeze(0))
        paragraph_embeddings.append(para_emb)

    paragraph_embeddings = torch.stack(paragraph_embeddings, dim=1)
    doc_emb = doc_encoder(paragraph_embeddings)

    assert isinstance(doc_emb, torch.Tensor)
    assert doc_emb.shape[1] == 1024
