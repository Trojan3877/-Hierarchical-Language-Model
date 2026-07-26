"""Small decoder-only Transformer built from first principles.

This module is intentionally compact: it is a correctness and systems research
reference, not a claim of frontier-model capability. It provides a deterministic
smoke test and a hardware-aware scaling benchmark without downloading data or
weights.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelSpec:
    """Model dimensions are explicit so every experiment is reproducible."""

    vocab_size: int = 256
    context_length: int = 128
    layers: int = 2
    heads: int = 4
    width: int = 128
    intermediate: int = 512
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.width // self.heads


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.gate = nn.Linear(spec.width, spec.intermediate, bias=False)
        self.value = nn.Linear(spec.width, spec.intermediate, bias=False)
        self.down = nn.Linear(spec.intermediate, spec.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.value(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        if spec.width % spec.heads:
            raise ValueError("width must be divisible by heads")
        self.spec = spec
        self.qkv = nn.Linear(spec.width, 3 * spec.width, bias=False)
        self.output = nn.Linear(spec.width, spec.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        shape = (batch, tokens, self.spec.heads, self.spec.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.spec.dropout, is_causal=True
        )
        return self.output(attended.transpose(1, 2).reshape(batch, tokens, width))


class DecoderBlock(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(spec.width)
        self.attention = CausalSelfAttention(spec)
        self.feedforward_norm = RMSNorm(spec.width)
        self.feedforward = SwiGLU(spec)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class DecoderLM(nn.Module):
    """Decoder-only causal LM with tied token embeddings and a training loss."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.position_embedding = nn.Embedding(spec.context_length, spec.width)
        self.blocks = nn.ModuleList(DecoderBlock(spec) for _ in range(spec.layers))
        self.final_norm = RMSNorm(spec.width)
        self.lm_head = nn.Linear(spec.width, spec.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        _, tokens = input_ids.shape
        if tokens > self.spec.context_length:
            raise ValueError("input sequence exceeds context_length")
        positions = torch.arange(tokens, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def deterministic_batch(spec: ModelSpec, batch_size: int, device: str) -> tuple[Tensor, Tensor]:
    """Create a fixed synthetic next-token task; no external corpus is implied."""

    tokens = torch.randint(0, spec.vocab_size, (batch_size, spec.context_length + 1), device=device)
    return tokens[:, :-1], tokens[:, 1:]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def run_smoke(spec: ModelSpec, steps: int, batch_size: int, device: str) -> dict[str, Any]:
    torch.manual_seed(7)
    model = DecoderLM(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        input_ids, labels = deterministic_batch(spec, batch_size, device)
        result = model(input_ids, labels)
        loss = result["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "mode": "smoke",
        "model": asdict(spec),
        "parameters": model.parameter_count,
        "steps": steps,
        "batch_size": batch_size,
        "device": device,
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
        "loss_decreased": losses[-1] <= losses[0],
    }


@torch.no_grad()
def run_benchmark(spec: ModelSpec, iterations: int, batch_size: int, device: str) -> dict[str, Any]:
    torch.manual_seed(7)
    model = DecoderLM(spec).to(device).eval()
    input_ids, _ = deterministic_batch(spec, batch_size, device)
    for _ in range(5):
        model(input_ids)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        model(input_ids)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - started) * 1000)
    median_ms = statistics.median(durations)
    return {
        "mode": "benchmark",
        "model": asdict(spec),
        "parameters": model.parameter_count,
        "batch_size": batch_size,
        "tokens_per_batch": batch_size * spec.context_length,
        "device": device,
        "latency_ms": {
            "median": round(median_ms, 6),
            "p95": round(percentile(durations, 0.95), 6),
            "p99": round(percentile(durations, 0.99), 6),
        },
        "throughput_tokens_per_second": round(
            batch_size * spec.context_length * 1000 / median_ms, 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "benchmark"), default="smoke")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    spec = ModelSpec(
        layers=args.layers,
        width=args.width,
        heads=args.heads,
        context_length=args.context_length,
        intermediate=args.width * 4,
    )
    if args.mode == "smoke":
        result = run_smoke(spec, args.steps, args.batch_size, args.device)
    else:
        result = run_benchmark(spec, args.iterations, args.batch_size, args.device)
    result["environment"] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "seed": 7,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
