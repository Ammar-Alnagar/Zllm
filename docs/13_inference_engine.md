# Phase 4: Inference Engine and FastAPI Server

## Table of Contents

1. [Inference Engine Architecture](#inference-engine-architecture)
2. [Engine Implementation](#engine-implementation)
3. [FastAPI Server](#fastapi-server)
4. [Streaming Responses](#streaming-responses)
5. [OpenAI API Compatibility](#openai-api-compatibility)
6. [Testing](#testing)

---

## Inference Engine Architecture

The **Inference Engine** combines all components into a unified system.

```
                    Inference Engine Architecture

┌─────────────────────────────────────────────────────────┐
│                    Inference Engine                     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Request Queue                       │   │
│  │  [Req 1] [Req 2] [Req 3] ... (incoming)         │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│                       ▼                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Scheduler                           │   │
│  │  Batch selection, preemption, priority          │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│                       ▼                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Model Runner                        │   │
│  │  Forward pass, attention, sampling              │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│                       ▼                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Output Queue                        │   │
│  │  Completed responses ready for return           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Engine Implementation

Create file: `mini_vllm/python/mini_vllm/engine.py`

```python
"""Inference Engine - Orchestrates the full pipeline"""

from dataclasses import dataclass
from typing import List, Optional, AsyncIterator, Dict
from queue import Queue
import asyncio
import threading
import time

from .scheduler import Scheduler, SchedulerConfig, SequenceStatus
from .model_runner import ModelRunner
from .sampling import SamplingParams
from .tokenizer import Tokenizer


@dataclass
class GenerationOutput:
    """Output from generation"""
    request_id: str
    text: str
    token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None  # "length", "stop", "error"


class InferenceEngine:
    """Main inference engine."""

    def __init__(
        self,
        model_path: str,
        max_num_seqs: int = 256,
        max_num_tokens: int = 8192,
        device: str = "cuda"
    ):
        # Initialize model runner
        self.model_runner = ModelRunner(model_path, device=device)
        self.tokenizer = self.model_runner.tokenizer

        # Initialize scheduler
        scheduler_config = SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_tokens=max_num_tokens,
        )
        self.scheduler = Scheduler(scheduler_config, self.model_runner.kv_cache)

        # Request tracking
        self.request_outputs: Dict[str, GenerationOutput] = {}
        self._request_id = 0

        # Background processing
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the engine."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the engine."""
        self._running = False
        if self._thread:
            self._thread.join()

    def _run_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                self._step()
            except Exception as e:
                print(f"Error in engine loop: {e}")
            time.sleep(0.001)  # Small sleep to prevent busy-waiting

    def _step(self):
        """Execute one scheduling + inference step."""
        # Schedule
        output = self.scheduler.schedule()

        if output.is_empty:
            return

        # Prepare batch
        batch_ids = []
        batch_positions = []

        for seq in output.prefill_seqs + output.decode_seqs:
            batch_ids.append(seq.seq_id)

        if not batch_ids:
            return

        # Build input tensors
        input_tokens = []
        positions = []

        for seq in output.prefill_seqs:
            tokens = seq.prompt_tokens[seq.num_computed_tokens:]
            input_tokens.extend(tokens)
            start_pos = seq.num_computed_tokens
            positions.extend(range(start_pos, start_pos + len(tokens)))

        for seq in output.decode_seqs:
            # Get last generated token
            if seq.output_tokens:
                input_tokens.append(seq.output_tokens[-1])
            else:
                input_tokens.append(seq.prompt_tokens[-1])
            positions.append(seq.num_tokens - 1)

        # Forward pass
        import torch
        input_tensor = torch.tensor([input_tokens], device=self.model_runner.device)
        pos_tensor = torch.tensor([positions], device=self.model_runner.device)

        logits = self.model_runner._forward(input_tensor, pos_tensor)

        # Sample next tokens
        token_idx = 0
        for seq in output.prefill_seqs:
            num_tokens = len(seq.prompt_tokens) - seq.num_computed_tokens
            seq.num_computed_tokens = seq.num_prompt_tokens
            seq.status = SequenceStatus.DECODING
            token_idx += num_tokens

        for seq in output.decode_seqs:
            next_logits = logits[0, token_idx:token_idx+1]
            next_token = self.model_runner.sampler.sample(
                next_logits, seq.sampling_params
            )
            self.scheduler.update_sequence(seq.seq_id, next_token.item())
            token_idx += 1

        # Process finished sequences
        for seq in output.finished_seqs:
            self._complete_request(seq)

    def _complete_request(self, seq):
        """Mark request as complete."""
        request_id = str(seq.seq_id)
        text = self.tokenizer.decode(seq.output_tokens)

        reason = "stop"
        if len(seq.output_tokens) >= seq.sampling_params.max_tokens:
            reason = "length"

        self.request_outputs[request_id] = GenerationOutput(
            request_id=request_id,
            text=text,
            token_ids=seq.output_tokens,
            finished=True,
            finish_reason=reason
        )

    def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """Synchronous generation."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Encode
        tokens = self.tokenizer.encode(prompt)

        # Add to scheduler
        seq_id = self.scheduler.add_request(tokens, sampling_params)
        request_id = str(seq_id)

        # Wait for completion
        while request_id not in self.request_outputs:
            time.sleep(0.01)

        output = self.request_outputs.pop(request_id)
        return output.text

    async def generate_async(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """Async generation."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        tokens = self.tokenizer.encode(prompt)
        seq_id = self.scheduler.add_request(tokens, sampling_params)
        request_id = str(seq_id)

        while request_id not in self.request_outputs:
            await asyncio.sleep(0.01)

        output = self.request_outputs.pop(request_id)
        return output.text

    async def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None
    ) -> AsyncIterator[str]:
        """Streaming generation."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        tokens = self.tokenizer.encode(prompt)
        seq_id = self.scheduler.add_request(tokens, sampling_params)

        seq = self.scheduler.get_sequence(seq_id)
        last_len = 0

        while not seq.is_finished:
            await asyncio.sleep(0.01)

            # Yield new tokens
            if len(seq.output_tokens) > last_len:
                new_tokens = seq.output_tokens[last_len:]
                text = self.tokenizer.decode(new_tokens)
                yield text
                last_len = len(seq.output_tokens)

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "scheduler": self.scheduler.get_stats(),
            "memory": self.model_runner.kv_cache.get_stats(),
        }
```

---

## FastAPI Server

Create file: `mini_vllm/python/mini_vllm/server.py`

```python
"""FastAPI Server for mini-vLLM"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncIterator
import json
import asyncio
import uvicorn

from .engine import InferenceEngine
from .sampling import SamplingParams


# Request/Response Models
class CompletionRequest(BaseModel):
    """Request for text completion"""
    model: str = "qwen3"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


class ChatChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]


# Global engine instance
engine: Optional[InferenceEngine] = None


def create_app(model_path: str) -> FastAPI:
    """Create FastAPI application."""
    global engine

    app = FastAPI(
        title="Mini-vLLM API",
        description="High-performance LLM inference server",
        version="0.1.0"
    )

    @app.on_event("startup")
    async def startup():
        global engine
        engine = InferenceEngine(model_path)
        engine.start()

    @app.on_event("shutdown")
    async def shutdown():
        global engine
        if engine:
            engine.stop()

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": "qwen3", "object": "model"}]
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        sampling = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
        )

        if request.stream:
            return StreamingResponse(
                stream_completion(request.prompt, sampling),
                media_type="text/event-stream"
            )

        text = await engine.generate_async(request.prompt, sampling)

        import time
        return CompletionResponse(
            id="cmpl-" + str(int(time.time())),
            created=int(time.time()),
            model=request.model,
            choices=[CompletionChoice(text=text, finish_reason="stop")]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        # Format messages into prompt
        prompt = format_chat_prompt(request.messages)

        sampling = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        if request.stream:
            return StreamingResponse(
                stream_chat(prompt, sampling),
                media_type="text/event-stream"
            )

        text = await engine.generate_async(prompt, sampling)

        import time
        return ChatResponse(
            id="chatcmpl-" + str(int(time.time())),
            created=int(time.time()),
            model=request.model,
            choices=[ChatChoice(
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop"
            )]
        )

    @app.get("/stats")
    async def stats():
        return engine.get_stats()

    return app


def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """Format chat messages into prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


async def stream_completion(
    prompt: str,
    sampling: SamplingParams
) -> AsyncIterator[str]:
    """Stream completion tokens."""
    async for token in engine.generate_stream(prompt, sampling):
        data = json.dumps({
            "object": "text_completion.chunk",
            "choices": [{"text": token, "index": 0}]
        })
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


async def stream_chat(
    prompt: str,
    sampling: SamplingParams
) -> AsyncIterator[str]:
    """Stream chat completion tokens."""
    async for token in engine.generate_stream(prompt, sampling):
        data = json.dumps({
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {"role": "assistant", "content": token},
                "index": 0
            }]
        })
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


def main():
    """Run the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Mini-vLLM Server")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()

    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

---

## Streaming Responses

```
                    Streaming Architecture

Client Request (stream=true):
┌─────────────────────────────────────────────────────────┐
│ POST /v1/completions                                    │
│ {"prompt": "Hello", "stream": true}                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
Server Response (SSE):
┌─────────────────────────────────────────────────────────┐
│ HTTP/1.1 200 OK                                         │
│ Content-Type: text/event-stream                         │
│                                                         │
│ data: {"choices":[{"text":"Hi"}]}                       │
│                                                         │
│ data: {"choices":[{"text":" there"}]}                   │
│                                                         │
│ data: {"choices":[{"text":"!"}]}                        │
│                                                         │
│ data: [DONE]                                            │
└─────────────────────────────────────────────────────────┘
```

---

## OpenAI API Compatibility

The server implements OpenAI-compatible endpoints:

| Endpoint                    | Description           |
| --------------------------- | --------------------- |
| `GET /v1/models`            | List available models |
| `POST /v1/completions`      | Text completion       |
| `POST /v1/chat/completions` | Chat completion       |
| `GET /health`               | Health check          |
| `GET /stats`                | Engine statistics     |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Text completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is AI?"}
    ],
    "max_tokens": 100
  }'

# Streaming
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "stream": true}'
```

---

## Testing

Create file: `mini_vllm/tests/python/test_server.py`

```python
"""Test Server Endpoints"""

import pytest
from fastapi.testclient import TestClient


# Note: These tests require a model to be loaded
# Use mocking for unit tests without a model

class TestServerEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert "data" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Summary

| Component           | Purpose                              |
| ------------------- | ------------------------------------ |
| **InferenceEngine** | Orchestrate scheduler + model runner |
| **FastAPI Server**  | HTTP API for inference               |
| **Streaming**       | Server-Sent Events (SSE)             |
| **OpenAI Compat**   | Drop-in replacement API              |

---

## What's Next

Now we'll implement **Benchmarking and Testing** to validate performance.

Continue to: [14_benchmarking.md](./14_benchmarking.md)
