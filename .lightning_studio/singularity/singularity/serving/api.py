from __future__ import annotations

from typing import Any, Mapping

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = 256
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str


def create_app(config: Mapping[str, Any] | None = None) -> FastAPI:
    del config
    app = FastAPI(title="Singularity 32B", version="0.1.0")

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(text=request.prompt)

    return app


def run_api(config: Mapping[str, Any]) -> int:
    serving = config.get("serving", {})
    uvicorn.run(
        "singularity.serving.api:create_app",
        factory=True,
        host=str(serving.get("host", "0.0.0.0")),
        port=int(serving.get("port", 8000)),
    )
    return 0
