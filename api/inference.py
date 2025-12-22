from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

class LLMEngine:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, prompt: str, max_new_tokens: int = 100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

llm_engine = LLMEngine()
