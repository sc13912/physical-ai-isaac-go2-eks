"""Isaac Lab --external_callback: mirror RSL-RL TensorBoard scalars to a Prometheus /metrics endpoint.

Usage:  isaaclab train --rl_library rsl_rl ... --external_callback promhook.start
        (the directory holding this file must be on PYTHONPATH)

Stdlib only: the Isaac Lab 3.0 kit-less image ships no prometheus_client.
Binds to $POD_IP (set via the downward API); falls back to 127.0.0.1 so it is never wider by default.
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_values: dict[str, float] = {}
_step = 0
_lock = threading.Lock()


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _fmt(v: float) -> str:
    if v != v:
        return "NaN"
    if v in (float("inf"), float("-inf")):
        return "+Inf" if v > 0 else "-Inf"
    return repr(v)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] != "/metrics":
            self.send_error(404)
            return
        with _lock:
            items = sorted(_values.items())
            step = _step
        lines = [
            "# HELP isaaclab_scalar Latest value of each RSL-RL TensorBoard scalar.",
            "# TYPE isaaclab_scalar gauge",
        ]
        lines += [f'isaaclab_scalar{{tag="{_escape(t)}"}} {_fmt(v)}' for t, v in items]
        lines += [
            "# HELP isaaclab_iteration Latest training iteration logged.",
            "# TYPE isaaclab_iteration gauge",
            f"isaaclab_iteration {step}",
        ]
        body = ("\n".join(lines) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _patch_viser():
    """Isaac Lab 3.0.0's NewtonViewerViser._log_plane_instances calls self._to_numpy, which the bundled
    newton 1.6.1 ViewerViser lacks -> 'no attribute _to_numpy' and the Viser server stops. Add it."""
    try:
        import numpy as np
        from isaaclab_visualizers.viser import viser_visualizer as vv
    except Exception:
        return
    cls = getattr(vv, "NewtonViewerViser", None)
    if cls is None or hasattr(cls, "_to_numpy"):
        return

    def _to_numpy(self, x):
        if x is None:
            return None
        if hasattr(x, "detach"):  # torch tensor (check first: CUDA tensors also have .numpy but it raises)
            return x.detach().cpu().numpy()
        if hasattr(x, "numpy") and not isinstance(x, np.ndarray):  # warp array
            return x.numpy()
        return np.asarray(x)

    cls._to_numpy = _to_numpy
    print("[promhook] patched NewtonViewerViser._to_numpy", flush=True)


def start():
    """Start the /metrics server, patch SummaryWriter.add_scalar and the Viser viewer. Returns None."""
    global _step
    _patch_viser()
    try:
        import demotasks

        demotasks.register()
    except Exception as e:  # noqa: BLE001 — never block a stock task because the demo task failed to import
        print(f"[promhook] demotasks not registered: {e}", flush=True)
    if os.environ.get("DEMO_CONTROLS") == "1" or os.environ.get("DEMO_CAMERA") == "1":
        import democtl

        democtl.install()
    host = os.environ.get("POD_IP", "127.0.0.1")
    port = int(os.environ.get("METRICS_PORT", "9500"))
    server = ThreadingHTTPServer((host, port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="promhook").start()
    print(f"[promhook] serving http://{host}:{port}/metrics", flush=True)

    from torch.utils.tensorboard import SummaryWriter

    original = SummaryWriter.add_scalar

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        global _step
        try:
            v = float(scalar_value.item() if hasattr(scalar_value, "item") else scalar_value)
            with _lock:
                _values[tag] = v
                if global_step is not None:
                    _step = max(_step, int(global_step))
        except Exception:
            pass
        return original(self, tag, scalar_value, global_step, *args, **kwargs)

    SummaryWriter.add_scalar = add_scalar
    return None
