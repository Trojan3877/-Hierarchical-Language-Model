# src/utils/generate_data.py

"""
Synthetic Dataset Generator
---------------------------
Generates synthetic labeled text data for quick testing of the HLM pipeline.
Useful when no dataset is available or when experimenting locally.
"""

import random
import pandas as pd
from pathlib import Path


POSITIVE_SENTENCES = [
    "The movie was amazing and the actors were fantastic.",
    "I loved the service and the atmosphere was great.",
    "The food tasted wonderful and the presentation was beautiful.",
    "This was an incredible experience, highly recommend!",
    "The product quality exceeded my expectations.",
    "The customer support was helpful and kind."
]

NEGATIVE_SENTENCES = [
    "The movie was boring and the acting was terrible.",
    "I hated the food, it was bland and overpriced.",
    "The service was slow and the staff was rude.",
    "What a waste of time and money.",
    "The product broke after one use.",
    "I had a horrible experience, never again."
]


def generate_dataset(n_samples: int = 1000, out_path: str = "data/synthetic_dataset.csv"):
    """
    Generate a synthetic text classification dataset.

    Args:
        n_samples (int): number of rows to generate
        out_path (str): where to save the dataset
    """
    texts, labels = [], []

    for _ in range(n_samples):
        if random.random() > 0.5:
            text = random.choice(POSITIVE_SENTENCES)
            label = 1
        else:
            text = random.choice(NEGATIVE_SENTENCES)
            label = 0
        texts.append(text)
        labels.append(label)

    df = pd.DataFrame({"text": texts, "label": labels})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Generated synthetic dataset with {n_samples} samples at {out_path}")
    return df


if __name__ == "__main__":
    # Example usage: generate 500 samples
    generate_dataset(500)
