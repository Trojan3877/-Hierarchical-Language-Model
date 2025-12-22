from fastapi import FastAPI
from api.schemas import PromptRequest, PromptResponse
from api.inference import llm_engine

app = FastAPI(title="Hierarchical Language Model API")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/generate", response_model=PromptResponse)
def generate_text(request: PromptRequest):
    output = llm_engine.generate(request.prompt, max_new_tokens=request.max_tokens)
    return {"output": output}
