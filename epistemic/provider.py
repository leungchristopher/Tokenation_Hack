"""OpenAI-compatible models (including hosted vLLM) and a deterministic offline mock."""

from __future__ import annotations

import json
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
    seed: int | None = None
    prompt_version: str = "v2"
    calls: int = 0
    tokens: int = 0
    latency_s: float = 0.0
    model_revision: str = "unspecified"
    inference_stack: str = "unspecified OpenAI-compatible server"
    seed_supported: bool = False
    deadline: float | None = None
    token_limit: int | None = None

    def metadata(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "seed": self.seed if self.seed_supported else None,
                "seed_supported": self.seed_supported, "prompt_version": self.prompt_version,
                "model_revision": self.model_revision, "inference_stack": self.inference_stack}

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
    """A deterministic no-claim fixture for evidence integration tests."""

    name: str = "mock"
    model: str = "mock-deterministic"
    seed_supported: bool = True
    model_revision: str = "sha256-v1"
    inference_stack: str = "local deterministic Python mock"

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.tokens += len(prompt) // 4
        return json.dumps({"claim_updates": []})


@dataclass
class OpenAICompatibleProvider(Provider):
    name: str = "openai_compatible"
    model: str = ""
    api_key_env: str = "EPISTEMIC_API_KEY"
    max_tokens: int = 700
    server_fingerprint: str | None = None

    def metadata(self) -> dict[str, Any]:
        return super().metadata() | {"max_tokens": self.max_tokens, "server_fingerprint": self.server_fingerprint}

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
        if self.seed is not None and self.seed_supported:
            body["seed"] = self.seed
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
        self.server_fingerprint = payload.get("system_fingerprint")
        return payload["choices"][0]["message"]["content"]


def get_provider(spec: str = "none", seed: int | None = 0, temperature: float = 0.0) -> Provider:
    if spec == "none":
        return Provider(name="none", model="none")
    if spec == "mock":
        return MockProvider(seed=seed, temperature=temperature)
    if spec.startswith("openai:"):
        model = spec.split(":", 1)[1] or os.environ.get("EPISTEMIC_MODEL", "")
        if not model:
            raise ValueError("Specify openai:<model> or set EPISTEMIC_MODEL.")
        return OpenAICompatibleProvider(
            model=model, base_url=os.environ.get("EPISTEMIC_BASE_URL", "https://api.openai.com/v1"),
            seed=seed, temperature=temperature,
            seed_supported=os.environ.get("EPISTEMIC_SEED_SUPPORTED", "false").lower() == "true",
            model_revision=os.environ.get("EPISTEMIC_MODEL_REVISION", "unspecified"),
            inference_stack=os.environ.get("EPISTEMIC_INFERENCE_STACK", "unspecified OpenAI-compatible server"),
        )
    raise KeyError(f"Unknown provider {spec!r}; use none, mock or openai:<model>.")
