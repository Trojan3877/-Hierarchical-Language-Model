# src/providers/hf_model.py
from __future__ import annotations
import os
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from dotenv import load_dotenv

load_dotenv()

class HFGenerator:
    """
    Thin wrapper over Hugging Face transformers pipeline.
    Supports both causal and seq2seq models.
    """
    def __init__(self, model_name: str, device: Optional[int] = None, max_new_tokens: int = 256, temperature: float = 0.2):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Choose task type based on model family
        if any(k in model_name.lower() for k in ["t5", "flan", "llama-guard", "ul2"]):
            task = "text2text-generation"
            ModelCls = AutoModelForSeq2SeqLM
        else:
            task = "text-generation"
            ModelCls = AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = ModelCls.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if device is None else None
        )

        self.pipe = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device if device is not None else None
        )

    def generate(self, prompt: str) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0.0,
        )
        # Pipeline returns list of dicts; unify to string
        if isinstance(out, list):
            text = out[0].get("generated_text") or out[0].get("summary_text") or ""
        else:
            text = str(out)
        return text.strip()
