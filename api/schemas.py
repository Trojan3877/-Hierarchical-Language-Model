from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

class PromptResponse(BaseModel):
    output: str
