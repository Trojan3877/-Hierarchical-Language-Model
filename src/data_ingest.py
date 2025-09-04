# src/data_ingest.py

"""
Data Ingestion Pipeline
-----------------------
Handles loading, cleaning, and batching text data
for the Hierarchical Language Model (HLM).
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logger import logger


class DataIngestor:
    def __init__(self, file_path: str, text_col: str = "text", label_col: str = "label"):
        """
        Initializes the data ingestor.

        Args:
            file_path (str): Path to CSV file containing dataset.
            text_col (str): Column name with text data.
            label_col (str): Column name with labels.
        """
        self.file_path = file_path
        self.text_col = text_col
        self.label_col = label_col
        self.df = None

    def load_data(self):
        """Load dataset from CSV"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        logger.info(f"Loaded dataset with {len(self.df)} rows from {self.file_path}")
        return self.df

    def clean_text(self, text: str) -> str:
        """Basic text cleaning (remove special chars, lowercase)"""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.strip()

    def preprocess(self):
        """Clean text column"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.df[self.text_col] = self.df[self.text_col].apply(self.clean_text)
        logger.info("Applied basic text cleaning.")
        return self.df

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split dataset into train, validation, and test sets"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        train_df, temp_df = train_test_split(
            self.df, test_size=(test_size + val_size), random_state=random_state, stratify=self.df[self.label_col]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=test_size / (test_size + val_size), random_state=random_state, stratify=temp_df[self.label_col]
        )

        logger.info(
            f"Data split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        return train_df, val_df, test_df
