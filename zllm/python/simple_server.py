# simple_server.py - Simple FastAPI server for Mini-vLLM (mock responses)

import time
import uuid
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# Pydantic models for API
class GenerateRequest(BaseModel):
    """Request model for text generation"""

    prompt: str = Field(..., description="Input text prompt")
    max_tokens: Optional[int] = Field(50, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.8, description="Sampling temperature")
    top_p: Optional[float] = Field(0.95, description="Top-p (nucleus) sampling")
    top_k: Optional[int] = Field(50, description="Top-k sampling (-1 = disabled)")


class GenerateResponse(BaseModel):
    """Response model for text generation"""

    id: str = Field(..., description="Request ID")
    object: str = Field("text_completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field("mini_vllm", description="Model name")
    choices: List[Dict[str, Any]] = Field(..., description="Generated completions")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field("ok", description="Service status")
    version: str = Field("0.1.0", description="Service version")
    cuda_available: bool = Field(False, description="CUDA availability")


# Global state
is_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global is_initialized

    print("🚀 Starting Mini-vLLM server...")
    try:
        # Simple initialization
        is_initialized = True
        print("✓ Server initialized (mock mode)")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        is_initialized = False

    yield

    print("Shutting down Mini-vLLM server...")


# Create FastAPI app
app = FastAPI(
    title="Mini-vLLM API",
    description="Educational LLM Inference Engine API (Mock Mode)",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import torch

    return HealthResponse(
        status="ok" if is_initialized else "error",
        version="0.1.0",
        cuda_available=torch.cuda.is_available(),
    )


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Simple generation endpoint with mock response"""

    if not is_initialized:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        # Simulate processing time
        import asyncio

        await asyncio.sleep(0.1)

        # Create mock response based on input
        prompt_length = len(request.prompt.split())
        max_tokens = request.max_tokens or 50

        # Generate mock completion
        mock_completions = [
            f"{request.prompt} This is a mock response from Mini-vLLM running on Qwen/Qwen3-0.6B. The system is working correctly with CUDA support!",
            f"{request.prompt} Hello! I'm a simulated response. Mini-vLLM is successfully serving requests with advanced features like Flash Attention and RoPE embeddings.",
            f"{request.prompt} Great question! In this demo, we're showing how Mini-vLLM can handle text generation with proper tokenization and sampling strategies.",
        ]

        import random

        completion = random.choice(mock_completions)

        return {
            "generated_text": completion,
            "num_tokens": len(completion.split()),
            "model": "Qwen/Qwen3-0.6B",
            "parameters": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "max_tokens": max_tokens,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/v1/completions", response_model=GenerateResponse)
async def create_completion(request: GenerateRequest):
    """OpenAI-compatible completions endpoint with mock response"""

    if not is_initialized:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        # Create mock response
        request_id = str(uuid.uuid4())
        start_time = int(time.time())

        mock_text = f"{request.prompt} This is a demonstration of Mini-vLLM's OpenAI-compatible API. The system successfully loaded Qwen/Qwen3-0.6B and is ready for inference!"

        response = GenerateResponse(
            id=request_id,
            object="text_completion",
            created=start_time,
            model="mini_vllm-qwen3",
            choices=[
                {
                    "text": mock_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(mock_text.split()),
                "total_tokens": len(request.prompt.split()) + len(mock_text.split()),
            },
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def main():
    """Main entry point for running the server"""
    import argparse

    parser = argparse.ArgumentParser(description="Mini-vLLM Server (Mock Mode)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    print("🚀 Starting Mini-vLLM Server (Mock Mode)")
    print("📍 This demonstrates the API without full model inference")
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")

    uvicorn.run(
        "simple_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
