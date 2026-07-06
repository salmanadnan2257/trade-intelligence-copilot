"""Thin wrapper around Vertex AI Gemini generation with token-cost accounting.

Generation uses the google-genai SDK in Vertex mode. Every call records prompt
and output tokens so a query's total cost can be reported. A CostMeter is passed
through a single request so the API can return the real per-query cost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from . import config

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=config.GCP_PROJECT,
            location=config.VERTEX_GEN_LOCATION,
        )
    return _client


@dataclass
class Usage:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        p = config.PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        return (
            self.input_tokens / 1_000_000 * p["input"]
            + self.output_tokens / 1_000_000 * p["output"]
        )


@dataclass
class CostMeter:
    """Accumulates usage across the several model calls a single query makes."""

    calls: list[Usage] = field(default_factory=list)

    def add(self, usage: Usage) -> None:
        self.calls.append(usage)

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.calls)

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.calls)

    def summary(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "calls": [
                {
                    "model": u.model,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost_usd": round(u.cost_usd, 8),
                }
                for u in self.calls
            ],
        }


def generate(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.0,
    meter: CostMeter | None = None,
) -> str:
    """Run one generation call and return the text. Records usage on the meter."""
    model = model or config.MODEL_FAST
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system,
    )
    resp = client().models.generate_content(model=model, contents=prompt, config=cfg)
    um = resp.usage_metadata
    usage = Usage(
        model=model,
        input_tokens=getattr(um, "prompt_token_count", 0) or 0,
        output_tokens=getattr(um, "candidates_token_count", 0) or 0,
    )
    if meter is not None:
        meter.add(usage)
    return (resp.text or "").strip()


def generate_json(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    meter: CostMeter | None = None,
) -> Any:
    """Generate and parse JSON. Tolerates code-fenced output."""
    text = generate(prompt, model=model, system=system, meter=meter)
    return _parse_json(text)


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    # Trim to the outermost JSON object/array if the model added prose.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        if open_c in text:
            start = text.index(open_c)
            end = text.rfind(close_c)
            if end > start:
                return json.loads(text[start : end + 1])
    return json.loads(text)
