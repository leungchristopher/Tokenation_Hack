"""Local demo transport and rendering adapter. Run: python -m minimal_lab.live"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')

from minimal_lab.lab import Lab


class RenderedLab(Lab):
    """Observe this backend instance's physics steps; never edit upstream robotics code."""
    def __init__(self, *args, publish, **kwargs):
        import mujoco
        super().__init__(*args, on_action=lambda action, detail: publish(action=action, transfer=detail), **kwargs)
        self.publish = publish
        self.renderer = mujoco.Renderer(self.backend.model, height=480, width=640)
        self.original_step = self.backend.skills._step
        self.last_frame = 0
        self.backend.skills._step = self.step
        self.frame()

    def frame(self):
        import cv2
        self.renderer.update_scene(self.backend.data, camera='side')
        rgb = self.renderer.render()
        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError('Could not encode robot frame.')
        self.publish(frame=jpeg.tobytes())
        self.last_frame = time.monotonic()

    def step(self):
        self.original_step()
        if time.monotonic()-self.last_frame >= 1/12:
            self.frame()

    def close(self):
        self.backend.skills._step = self.original_step
        try:
            self.frame()
        finally:
            self.renderer.close()


class Demo:
    def __init__(self, args):
        self.args, self.lock, self.frame = args, threading.Lock(), b''
        self.state = dict(status='ready', phase='ready', graph=None, revision=0, frame_id=0,
                          budget=args.budget, mode='BO + Amass + LLM' if args.model else 'BO', cv=args.cv)

    def publish(self, **update):
        with self.lock:
            if 'frame' in update:
                self.frame = update.pop('frame')
                self.state['frame_id'] += 1
            else:
                self.state['revision'] += 1
            self.state.update(update)

    def event(self, stage, graph):
        # Snapshot before the simulation continues mutating its own graph.
        self.publish(phase=stage, graph=json.loads(json.dumps(graph)), action='')

    def start(self):
        with self.lock:
            if self.state['status'] == 'running':
                return False
            self.state.update(status='running', phase='starting', graph=None, error=None, scores=None,
                              download=None, action='', frame_id=0)
            self.state['revision'] += 1
            self.frame = b''
        threading.Thread(target=self.work, daemon=True).start()
        return True

    def work(self):
        from inspect_ai import Task, eval
        from inspect_ai.dataset import Sample
        from minimal_lab.task import demo_scorer, demo_solver
        a = self.args
        out = Path(a.out)/uuid4().hex[:10]
        try:
            if a.model and not os.getenv('AMASS_API_KEY'):
                raise ValueError('AMASS_API_KEY is required for literature mode.')
            task = Task(dataset=[Sample(id=f'seed-{a.seed}', input='Optimise the simulated assay.')],
                        solver=demo_solver(a.budget, a.seed, bool(a.model), bool(a.model), a.cv, str(out),
                            on_event=self.event, lab_factory=lambda *args, **kwargs:
                                RenderedLab(*args, publish=self.publish, **kwargs)), scorer=demo_scorer())
            log = eval(task, model=a.model or 'mockllm/model', display='none', ctl_server=False,
                       log_dir=str(out/'inspect'))[0]
            if log.status != 'success':
                raise RuntimeError(log.error.message if log.error else f'Inspect run {log.status}')
            scores = {k:v.value for k,v in log.samples[0].scores.items()}
            self.publish(status='complete', phase='complete', scores=scores,
                         download=str(out/f'seed-{a.seed}-epoch1'/'result.json'))
        except Exception as error:
            self.publish(status='error', phase='error', error=f'{type(error).__name__}: {error}')


def serve(demo, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, mime, status=200):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            with demo.lock:
                if path == '/state':
                    body, mime = json.dumps(demo.state).encode(), 'application/json'
                elif path == '/frame.jpg':
                    body, mime = demo.frame, 'image/jpeg'
                elif path == '/result.json' and demo.state.get('download'):
                    body, mime = Path(demo.state['download']).read_bytes(), 'application/json'
                elif path == '/':
                    body, mime = Path(__file__).with_name('live.html').read_bytes(), 'text/html; charset=utf-8'
                else:
                    return self.reply(b'Not found', 'text/plain', 404)
            self.reply(body, mime)

        def do_POST(self):
            # Local-only launch control; reject cross-origin browser requests.
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                return self.reply(b'Forbidden', 'text/plain', 403)
            if self.path != '/run':
                return self.reply(b'Not found', 'text/plain', 404)
            self.reply(b'{}', 'application/json', 202 if demo.start() else 409)

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--budget', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cv', type=float, default=0.15)
    parser.add_argument('--model', help='Inspect model ID; enables LLM selection and Amass research')
    parser.add_argument('--out', default='logs/live')
    args = parser.parse_args()
    if not 1 <= args.budget <= 24 or not 0 <= args.cv <= 1:
        parser.error('Use 1–24 experiments and a volume CV between 0 and 1.')
    server = serve(Demo(args), args.port)
    print(f'Open http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
