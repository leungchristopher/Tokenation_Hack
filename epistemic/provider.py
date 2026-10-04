"""OpenAI-compatible models (including hosted vLLM) and a deterministic offline mock."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Provider:
    name: str
    model: str
    base_url: str = ""
    temperature: float = 0.0
    calls: int = 0
    tokens: int = 0
    latency_s: float = 0.0
    deadline: float | None = None
    token_limit: int | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }

    def complete(self, prompt: str) -> str:
        raise NotImplementedError

    def output_token_limit(self) -> int:
        return 0

    def call_limit(self, prompt: str) -> str | None:
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            return "time_limit"
        estimated = len(prompt.encode("utf-8")) + 128 + self.output_token_limit()
        if self.token_limit is not None and self.tokens + estimated > self.token_limit:
            return "model_token_limit"
        return None


@dataclass
class MockProvider(Provider):
    """A deterministic fixture for tests."""

    name: str = "mock"
    model: str = "mock-deterministic"

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.tokens += len(prompt) // 4
        return "{}"


@dataclass
class OpenAICompatibleProvider(Provider):
    name: str = "openai_compatible"
    model: str = ""
    api_key_env: str = "EPISTEMIC_API_KEY"
    max_tokens: int = 700

    def metadata(self) -> dict[str, Any]:
        return super().metadata() | {"max_tokens": self.max_tokens}

    def output_token_limit(self) -> int:
        return self.max_tokens

    def complete(self, prompt: str) -> str:
        import httpx

        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set.")
        body: dict[str, Any] = {
            "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        self.calls += 1
        start = time.perf_counter()
        try:
            remaining = 180.0 if self.deadline is None else max(self.deadline - time.perf_counter(), 0.001)
            response = httpx.post(f"{self.base_url.rstrip('/')}/chat/completions", json=body,
                                  timeout=min(180.0, remaining),
                                  headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
            payload = response.json()
        finally:
            self.latency_s += time.perf_counter() - start
        self.tokens += int(payload.get("usage", {}).get("total_tokens", 0))
        return payload["choices"][0]["message"]["content"]


def get_provider(spec: str = "none", temperature: float = 0.0) -> Provider:
    if spec == "none":
        return Provider(name="none", model="none", temperature=temperature)
    if spec == "mock":
        return MockProvider(temperature=temperature)
    if spec.startswith("openai:"):
        model = spec.split(":", 1)[1] or os.environ.get("EPISTEMIC_MODEL", "")
        if not model:
            raise ValueError("Specify openai:<model> or set EPISTEMIC_MODEL.")
        return OpenAICompatibleProvider(
            model=model, base_url=os.environ.get("EPISTEMIC_BASE_URL", "https://api.openai.com/v1"),
            temperature=temperature,
        )
    raise KeyError(f"Unknown provider {spec!r}; use none, mock or openai:<model>.")
