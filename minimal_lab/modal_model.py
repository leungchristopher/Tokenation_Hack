"""Optional open-model chooser on Modal. Imported only when llm='modal'.

Self-hosted transports are ephemeral, authenticated Modal apps (`app.run`, no public URL)
that stop when the session exits. MODAL_LLM_BACKEND selects:
  gpu (default)  Gemma 4 E4B-it, vLLM, one L40S (needs a Modal payment method)
  cpu            Gemma 4 E2B-it Q4_0 GGUF, llama.cpp server on CPU (works on plan credits)
If MODAL_LLM_URL is set, an OpenAI-compatible Modal Endpoint is used instead, authenticated
with MODAL_PROXY_TOKEN (`<id>.<secret>`). Tokens come from the environment and are never logged.
"""
import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager

import modal

BACKEND = os.getenv('MODAL_LLM_BACKEND', 'gpu')
if BACKEND not in ('gpu', 'cpu'):
    raise ValueError('MODAL_LLM_BACKEND must be gpu or cpu.')
GPU_MODEL, GPU_REVISION = 'google/gemma-4-E4B-it', 'ee0ef6023621cff504d758262d4e04895a5af4a2'
CPU_MODEL, CPU_FILE = 'ggml-org/gemma-4-E2B-it-GGUF', 'gemma-4-E2B-it-Q4_0.gguf'
CPU_CORES, CPU_MEMORY_GIB, SLOTS = 8, 16, 4
MODEL, REVISION, HARDWARE, USD_PER_S = {  # https://modal.com/pricing
    'gpu': (GPU_MODEL, GPU_REVISION, 'L40S', 0.000542),
    'cpu': (f'{CPU_MODEL}/{CPU_FILE}', 'b4243c156154b6dca9324415f8c7ccc098b4aed1',
            f'{CPU_CORES} CPU cores', CPU_CORES*0.0000131+CPU_MEMORY_GIB*0.00000222)}[BACKEND]
MAX_TOKENS, ATTEMPTS, STARTUP_TIMEOUT = 1200, 3, 20*60
CALL_TIMEOUT = {'gpu': 120, 'cpu': 600}[BACKEND]
hf_cache = modal.Volume.from_name('huggingface-cache', create_if_missing=True)

gpu_app = modal.App('tokenation-chooser')


@gpu_app.cls(image=modal.Image.from_registry('nvidia/cuda:12.9.0-devel-ubuntu22.04', add_python='3.12')
             .entrypoint([]).uv_pip_install('vllm==0.21.0').env({'HF_XET_HIGH_PERFORMANCE': '1'}),
             gpu='L40S', timeout=30*60, scaledown_window=60, volumes={'/root/.cache/huggingface': hf_cache})
class Chooser:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(GPU_MODEL, revision=GPU_REVISION, max_model_len=16384, gpu_memory_utilization=0.9,
                       limit_mm_per_prompt={'image': 0, 'audio': 0})

    @modal.method()
    def chat(self, messages, max_tokens=MAX_TOKENS):
        from vllm import SamplingParams
        start = time.monotonic()
        out = self.llm.chat(messages, SamplingParams(temperature=0, max_tokens=max_tokens),
                            use_tqdm=False)[0]
        return dict(text=out.outputs[0].text, prompt_tokens=len(out.prompt_token_ids),
                    completion_tokens=len(out.outputs[0].token_ids), compute_s=time.monotonic()-start)


cpu_app = modal.App('tokenation-chooser-cpu')


@cpu_app.cls(image=modal.Image.from_registry('ghcr.io/ggml-org/llama.cpp:server', add_python='3.12')
             .entrypoint([]).env({'LLAMA_CACHE': '/root/.cache/huggingface/llama'}),
             cpu=CPU_CORES, memory=CPU_MEMORY_GIB*1024, timeout=45*60, scaledown_window=60,
             volumes={'/root/.cache/huggingface': hf_cache})
@modal.concurrent(max_inputs=SLOTS)
class CpuChooser:
    @modal.enter()
    def load(self):
        import subprocess
        import urllib.request
        # Bound to localhost inside the container; never exposed as a web endpoint.
        self.server = subprocess.Popen(['/app/llama-server', '-hf', f'{CPU_MODEL}:Q4_0', '--host', '127.0.0.1',
                                        '--port', '8080', '-c', str(8192*SLOTS), '-np', str(SLOTS),
                                        '-t', str(CPU_CORES), '--jinja'])
        for _ in range(STARTUP_TIMEOUT):
            try:
                urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)
                return
            except OSError:
                if self.server.poll() is not None:
                    raise RuntimeError('llama-server exited during startup.')
                time.sleep(1)
        raise TimeoutError('llama-server did not become healthy.')

    @modal.method()
    def chat(self, messages, max_tokens=MAX_TOKENS):
        import urllib.request
        start = time.monotonic()
        request = urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',
            data=json.dumps(dict(messages=messages, temperature=0, max_tokens=max_tokens)).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read())
        usage = body.get('usage') or {}
        return dict(text=body['choices'][0]['message']['content'], prompt_tokens=usage.get('prompt_tokens'),
                    completion_tokens=usage.get('completion_tokens'), compute_s=time.monotonic()-start)

    @modal.exit()
    def stop(self):
        self.server.terminate()


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
    """Callable prompt -> reply dict; keeps per-call usage for reporting."""
    def __init__(self, call):
        self.call, self.calls, self.app_id = call, [], None

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
        yield Client(http_call(os.environ['MODAL_LLM_URL'], os.getenv('MODAL_LLM_MODEL', 'auto')))
        return
    app, cls = (gpu_app, Chooser) if BACKEND == 'gpu' else (cpu_app, CpuChooser)
    async with app.run.aio():
        chooser = cls()
        client = Client(chooser.chat.remote.aio)
        client.app_id = app.app_id
        # Cold start (image pull + weight load) is bounded separately from per-decision calls.
        await asyncio.wait_for(chooser.chat.remote.aio([dict(role='user', content='Reply OK.')], 4),
                               STARTUP_TIMEOUT)
        yield client
