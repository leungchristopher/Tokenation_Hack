"""Optional open-model chooser on Modal. Imported only when llm='modal'.

Default transport: an ephemeral, authenticated Modal app (`app.run`, no public URL) runs
Gemma 4 E4B with vLLM and stops when the session exits. If MODAL_LLM_URL is set, an
OpenAI-compatible Modal Endpoint is used instead, authenticated with MODAL_PROXY_TOKEN
(`<id>.<secret>`). Tokens are read from the environment and never logged.
"""
import asyncio
import os
import random
import time
from contextlib import asynccontextmanager

import modal

MODEL = os.getenv('MODAL_LLM_MODEL', 'google/gemma-4-E4B-it')
REVISION = 'ee0ef6023621cff504d758262d4e04895a5af4a2'
GPU = 'L40S'
MAX_TOKENS, CALL_TIMEOUT, STARTUP_TIMEOUT, ATTEMPTS = 1200, 120, 20*60, 3

image = (modal.Image.from_registry('nvidia/cuda:12.9.0-devel-ubuntu22.04', add_python='3.12')
         .entrypoint([]).uv_pip_install('vllm==0.21.0').env({'HF_XET_HIGH_PERFORMANCE': '1'}))
app = modal.App('tokenation-chooser')
hf_cache = modal.Volume.from_name('huggingface-cache', create_if_missing=True)


@app.cls(image=image, gpu=GPU, timeout=30*60, scaledown_window=60,
         volumes={'/root/.cache/huggingface': hf_cache})
class Chooser:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(MODEL, revision=REVISION, max_model_len=16384, gpu_memory_utilization=0.9,
                       limit_mm_per_prompt={'image': 0, 'audio': 0})

    @modal.method()
    def chat(self, messages, max_tokens=MAX_TOKENS):
        from vllm import SamplingParams
        start = time.monotonic()
        out = self.llm.chat(messages, SamplingParams(temperature=0, max_tokens=max_tokens),
                            use_tqdm=False)[0]
        return dict(text=out.outputs[0].text, prompt_tokens=len(out.prompt_token_ids),
                    completion_tokens=len(out.outputs[0].token_ids),
                    gpu_seconds=time.monotonic()-start, model=MODEL, revision=REVISION)


async def retry(call, attempts=ATTEMPTS):
    for attempt in range(attempts):
        try:
            return await call()
        except (ValueError, KeyError, TypeError):
            raise
        except Exception:
            if attempt == attempts-1:
                raise
            await asyncio.sleep(min(8, 0.5*2**attempt)*(1+random.random()))


class Client:
    """Callable text -> reply dict; keeps per-call usage for reporting."""
    def __init__(self, call):
        self.call, self.calls = call, []

    async def __call__(self, prompt):
        start = time.monotonic()
        reply = await retry(lambda: asyncio.wait_for(self.call([dict(role='user', content=prompt)]),
                                                     CALL_TIMEOUT))
        reply['latency_s'] = time.monotonic()-start
        self.calls.append({k:v for k,v in reply.items() if k != 'text'})
        return reply


def http_call(url, model):
    import httpx
    token = os.environ['MODAL_PROXY_TOKEN']

    async def call(messages):
        async with httpx.AsyncClient(timeout=CALL_TIMEOUT) as client:
            response = await client.post(url.rstrip('/')+'/v1/chat/completions',
                headers={'Authorization': 'Bearer '+token},
                json=dict(model=model, messages=messages, temperature=0, max_tokens=MAX_TOKENS))
        if response.status_code == 429 or response.status_code >= 500:
            raise ConnectionError(f'Transient endpoint status {response.status_code}')
        response.raise_for_status()
        body = response.json()
        usage = body.get('usage') or {}
        return dict(text=body['choices'][0]['message']['content'], model=body.get('model', model),
                    prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'))
    return call


@asynccontextmanager
async def session():
    """Yield a Client; an ephemeral Modal app is stopped on exit, even after errors."""
    if os.getenv('MODAL_LLM_URL'):
        yield Client(http_call(os.environ['MODAL_LLM_URL'], MODEL))
        return
    async with app.run.aio():
        chooser = Chooser()
        client = Client(chooser.chat.remote.aio)
        client.app_id = app.app_id
        # Cold start (image pull + weight load) is bounded separately from per-decision calls.
        await asyncio.wait_for(chooser.chat.remote.aio([dict(role='user', content='Reply OK.')], 4),
                               STARTUP_TIMEOUT)
        yield client
