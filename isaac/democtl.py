"""Demo controls in Viser. Camera panel (Overview / Follow robot N) whenever DEMO_CAMERA=1 or DEMO_CONTROLS=1;
joystick, push and the training-stage (checkpoint) picker only with DEMO_CONTROLS=1 (play pod, never training).

Loaded by promhook.start().
All writes to the simulation happen on the main sim thread, inside UniformVelocityCommand._update_command;
Viser GUI callbacks (Viser's thread) only set flags in STATE.
Targets Isaac Lab 3.0.0 / rsl-rl-lib 5.5.1 / viser 1.0.30 as shipped in nvcr.io/nvidia/isaac-lab:3.0.0-kitless.
"""

import faulthandler
import glob
import math
import os
import re
import sys
import threading
import time

STATE = {
    "manual": False, "vx": 0.0, "vy": 0.0, "wz": 0.0,
    "push": False,
    "pending_ckpt": None, "runner": None, "ckpt_dir": None, "ckpts": [],
    "dropdown": None, "status": None,
    "server": None, "visible_ids": None, "follow": False, "follow_idx": 0, "overview": False, "cam_tick": 0,
}
_lock = threading.Lock()


def _label(path: str, last: int, first: int = 0) -> str:
    it = int(re.search(r"model_(\d+)\.pt$", path).group(1))
    if it == first:
        return "Iteration 0 — untrained (random brain)" if first == 0 else f"Iteration {it} — start of this run (earlier brain)"
    if it >= last:
        return f"Iteration {it} — fully trained"
    return f"Iteration {it} — {round(100 * (it - first) / max(last - first, 1))}% through this run"


def _scan(ckpt_path: str) -> None:
    d = os.path.dirname(ckpt_path)
    files = sorted(glob.glob(os.path.join(d, "model_*.pt")), key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
    last = int(re.search(r"model_(\d+)", files[-1]).group(1)) if files else 0
    first = int(re.search(r"model_(\d+)", files[0]).group(1)) if files else 0
    STATE["ckpt_dir"] = d
    # keep ~6 stages (untrained … fully trained) instead of every save
    its = [int(re.search(r"model_(\d+)", p).group(1)) for p in files]
    picks = []
    for frac in (0.0, 0.03, 0.1, 0.25, 0.5, 1.0):
        if not files:
            break
        k = min(range(len(files)), key=lambda i: abs(its[i] - (first + frac * (last - first))))
        if k not in picks:
            picks.append(k)
    cur = [i for i, p in enumerate(files) if os.path.abspath(p) == os.path.abspath(ckpt_path)]
    picks = sorted(set(picks) | set(cur))
    STATE["ckpts"] = [(_label(files[i], last, first), files[i]) for i in picks]
    dd = STATE["dropdown"]
    if dd is not None and STATE["ckpts"]:
        dd.options = tuple(lbl for lbl, _ in STATE["ckpts"])
        sel = [lbl for lbl, p in STATE["ckpts"] if os.path.abspath(p) == os.path.abspath(ckpt_path)]
        if sel:
            dd.value = sel[0]


def _patch_runner() -> None:
    from rsl_rl.runners import on_policy_runner as opr

    orig = opr.OnPolicyRunner.load
    if getattr(orig, "_democtl", False):
        return

    def load(self, path, *a, **kw):
        out = orig(self, path, *a, **kw)
        STATE["runner"] = self
        STATE["loaded"] = os.path.abspath(path)
        try:
            _scan(path)
        except Exception as e:  # noqa: BLE001
            print(f"[democtl] checkpoint scan failed: {e}", flush=True)
        return out

    load._democtl = True
    opr.OnPolicyRunner.load = load


_HB = {"n": 0, "path": os.environ.get("DEMO_HEARTBEAT", "/tmp/heartbeat")}
_WD = {"s": float(os.environ.get("DEMO_WATCHDOG_S", "60")), "t": 0.0}


def _watchdog_beat() -> None:
    """Stall tracer, re-armed on every heartbeat (every 50 sim steps). If the sim loop stops stepping for
    DEMO_WATCHDOG_S seconds (default 60, 0 = off), faulthandler's C thread writes every thread's Python stack to
    stderr (`kubectl logs [--previous]`, grep 'Timeout ('), and repeats every DEMO_WATCHDOG_S until the loop resumes.
    Works even when the stuck thread holds the GIL. Armed only after the first sim step, so startup is not traced."""
    if _WD["s"] <= 0:
        return
    now = time.monotonic()
    if not _WD["t"]:
        print(f"[democtl-watchdog] armed: all-thread stack dump to stderr if no sim step for {_WD['s']:.0f}s", flush=True)
    elif now - _WD["t"] > _WD["s"]:
        print(f"[democtl-watchdog] sim loop resumed after {now - _WD['t']:.0f}s without a heartbeat", flush=True)
    _WD["t"] = now
    faulthandler.dump_traceback_later(_WD["s"], repeat=True, file=sys.stderr)
_NW = {"n": 0, "last": None, "reported": 0}


def _nan_watch(term) -> None:
    """Diagnostic (DEMO_NAN_WATCH=1 or a DEMO_VISIBLE_TILES training view): report robots whose state goes non-finite,
    with their last finite position and tile, and how many robots have fallen off the field (z < -2 m)."""
    import torch

    _NW["n"] += 1
    if _NW["n"] % 5:
        return
    d = term.robot.data
    pos = d.root_pos_w.torch[:, :3]
    parts = {"root_pos": pos}
    for name in ("root_lin_vel_w", "joint_pos", "joint_vel"):
        t = getattr(getattr(d, name, None), "torch", None)
        if t is not None:
            parts[name] = t
    bad = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
    which = []
    for name, t in parts.items():
        b = ~torch.isfinite(t.reshape(t.shape[0], -1)).all(1)
        if b.any():
            which.append(name)
        bad |= b
    if bad.any() and _NW["reported"] < 3:
        ter = STATE.get("terrain")
        ids = torch.nonzero(bad).flatten()[:6].tolist()
        info = []
        for i in ids:
            last = _NW["last"][i].tolist() if _NW["last"] is not None else None
            tile = (int(ter.terrain_levels[i]), int(ter.terrain_types[i])) if getattr(ter, "terrain_levels", None) is not None else None
            info.append((i, [round(x, 2) for x in last] if last else None, tile))
        print(f"[democtl-nan] call {_NW['n']}: {int(bad.sum())} robots non-finite in {which}; (env, last pos, (level,type)): {info}", flush=True)
        _NW["reported"] += 1
    if _NW["n"] % 1000 == 0:
        z = torch.nan_to_num(pos[:, 2], nan=0.0)
        print(f"[democtl-nan] call {_NW['n']}: robots below z=-2 m (off the field): {int((z < -2).sum())}, lowest z {float(z.min()):.1f}", flush=True)
    _NW["last"] = torch.where(torch.isfinite(pos), pos, _NW["last"] if _NW["last"] is not None else pos).clone()


def _patch_command() -> None:
    import torch
    from isaaclab.envs.mdp.commands import velocity_command as vc

    orig = vc.UniformVelocityCommand._update_command
    if getattr(orig, "_democtl", False):
        return

    def _update_command(self):
        orig(self)
        if not _NW.get("off") and (os.environ.get("DEMO_NAN_WATCH") == "1" or os.environ.get("DEMO_VISIBLE_TILES")):
            try:
                _nan_watch(self)
            except Exception as e:  # noqa: BLE001 — a diagnostic must never take the run down
                print(f"[democtl-nan] watch failed, disabled: {e}", flush=True)
                _NW["off"] = True
        _HB["n"] += 1
        if _HB["n"] % 50 == 1:  # liveness heartbeat: the sim loop is stepping (probe checks the file age)
            try:
                with open(_HB["path"], "a"):
                    pass
                os.utime(_HB["path"])
            except OSError:
                pass
            _watchdog_beat()
        if os.environ.get("DEMO_SELFTEST") == "1":
            _selftest_tick(self)
        elif os.environ.get("DEMO_SELFTEST") == "2":
            _abtest_tick(self)
        elif os.environ.get("DEMO_SELFTEST") == "3":
            _eval_tick(self)
        with _lock:
            pending, STATE["pending_ckpt"] = STATE["pending_ckpt"], None
            push, STATE["push"] = STATE["push"], False
            manual, vx, vy, wz = STATE["manual"], STATE["vx"], STATE["vy"], STATE["wz"]
        _camera_tick(self)
        if pending and STATE["runner"] is not None:
            try:
                STATE["runner"].load(pending)
                STATE["runner"].alg.eval_mode()
                if STATE["status"] is not None:
                    STATE["status"].content = f"**Brain loaded:** `{os.path.basename(pending)}`"
                print(f"[democtl] loaded {pending}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[democtl] load failed: {e}", flush=True)
        if manual:
            self.vel_command_b[:, 0] = vx
            self.vel_command_b[:, 1] = vy
            self.vel_command_b[:, 2] = wz
        if push:
            vel = self.robot.data.root_vel_w.torch.clone()
            n = vel.shape[0]
            m = float(os.environ.get("DEMO_PUSH", "1.0"))  # m/s; a bit above the training pushes (Go2 0.5)
            vel[:, 0] += torch.empty(n, device=vel.device).uniform_(-m, m)
            vel[:, 1] += torch.empty(n, device=vel.device).uniform_(-m, m)
            self.robot.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=None)  # None = all envs
            print("[democtl] push applied", flush=True)

    _update_command._democtl = True
    vc.UniformVelocityCommand._update_command = _update_command


def _patch_terrain() -> None:
    """Remember the terrain importer so the visible-env picker can choose robots by terrain tile (DEMO_VISIBLE_TILES),
    and keep those on-screen robots on their tiles."""
    import torch
    from isaaclab.terrains import terrain_importer as ti

    orig = ti.TerrainImporter.configure_env_origins
    if getattr(orig, "_democtl", False):
        return

    def configure_env_origins(self, *a, **kw):
        out = orig(self, *a, **kw)
        STATE["terrain"] = self
        return out

    configure_env_origins._democtl = True
    ti.TerrainImporter.configure_env_origins = configure_env_origins

    orig_update = ti.TerrainImporter.update_env_origins

    def update_env_origins(self, env_ids, move_up, move_down):
        # the robots on screen keep their tile (curriculum would scatter them over the 80 m field); <1% of envs
        pinned = STATE.get("pinned")
        if pinned is not None and torch.is_tensor(move_up) and torch.is_tensor(move_down):
            ids = env_ids
            if isinstance(ids, slice):  # reset passes slice(None) = every env
                ids = torch.arange(self.terrain_levels.shape[0], device=self.terrain_levels.device)[ids]
            ids = torch.as_tensor(ids, device=move_up.device)
            if ids.numel() and ids.numel() == move_up.shape[0] == move_down.shape[0]:
                mask = torch.isin(ids, pinned.to(ids.device))
                move_up = torch.where(mask, torch.zeros_like(move_up), move_up)
                move_down = torch.where(mask, torch.zeros_like(move_down), move_down)
        return orig_update(self, env_ids, move_up, move_down)

    ti.TerrainImporter.update_env_origins = update_env_origins


def _tile_ids(cap: int):
    """DEMO_VISIBLE_TILES="r0-r1,c0-c1" (curriculum rows = difficulty, cols = terrain type): pick up to `cap` robots
    spawned on that block of tiles, round-robin per tile. None if unset or the terrain has no curriculum grid."""
    spec = os.environ.get("DEMO_VISIBLE_TILES", "").strip()
    ter = STATE.get("terrain")
    if not spec or getattr(ter, "terrain_levels", None) is None:
        return None
    (r0, r1), (c0, c1) = [tuple(int(v) for v in part.split("-")) for part in spec.split(",")]
    per_tile = {}
    for i, (lv, ty) in enumerate(zip(ter.terrain_levels.tolist(), ter.terrain_types.tolist())):
        if r0 <= lv <= r1 and c0 <= ty <= c1:
            per_tile.setdefault((lv, ty), []).append(i)
    picked = []
    while len(picked) < cap and any(per_tile.values()):
        for key in sorted(per_tile):
            if per_tile[key] and len(picked) < cap:
                picked.append(per_tile[key].pop(0))
    print(f"[democtl] showing {len(picked)} robots on tiles rows {r0}-{r1} x cols {c0}-{c1} ({len(per_tile)} tiles), pinned there", flush=True)
    if picked:
        import torch

        STATE["pinned"] = torch.tensor(picked, dtype=torch.long)
    return sorted(picked) or None


def _patch_visible() -> None:
    """Show a compact k x k block of envs instead of a random sample spread over the whole 64x64 grid.

    Isaac Lab lays envs out with isaaclab.cloner.grid_transforms: row-major, num_cols = ceil(N/ceil(N/sqrt(N))),
    so env k sits at (k // num_cols, k % num_cols). We return an explicit id list from
    BaseVisualizer._compute_visualized_env_ids (explicit lists win in both resolve paths). Drawing only."""
    from isaaclab.visualizers import base_visualizer as bv

    orig = bv.BaseVisualizer._compute_visualized_env_ids
    if getattr(orig, "_democtl", False):
        return

    def _compute_visualized_env_ids(self):
        out = orig(self)
        cap = getattr(self.cfg, "max_visible_envs", None)
        prov = getattr(self, "_scene_data_provider", None)
        n = getattr(prov, "num_envs", 0) if prov is not None else 0
        # Default VisualizerCfg.randomly_sample_visible_envs=True picks `cap` random envs across the whole
        # 160 m field (why they looked scattered). Replace that (or the contiguous row) with a compact block,
        # unless the user gave explicit visible_env_indices.
        tiles = _tile_ids(int(cap)) if getattr(self.cfg, "visible_env_indices", None) is None and cap else None
        if tiles:
            out = tiles
        elif getattr(self.cfg, "visible_env_indices", None) is None and cap and n > int(cap):
            cap = int(cap)
            num_rows = int(math.ceil(n / math.sqrt(n)))
            num_cols = int(math.ceil(n / num_rows))
            k = int(math.ceil(math.sqrt(cap)))
            # centre of the field: the default Viser camera looks at the world origin
            r0, c0 = max((num_rows - k) // 2, 0), max((num_cols - k) // 2, 0)
            out = [(r0 + r) * num_cols + (c0 + c) for r in range(k) for c in range(k)
                   if (r0 + r) < num_rows and (c0 + c) < num_cols and (r0 + r) * num_cols + (c0 + c) < n][:cap]
            print(f"[democtl] showing a {k}x{k} block of envs at the field centre (grid {num_rows}x{num_cols}): {out[:8]}...", flush=True)
        STATE["visible_ids"] = list(out) if out is not None else None
        return out

    _compute_visualized_env_ids._democtl = True
    bv.BaseVisualizer._compute_visualized_env_ids = _compute_visualized_env_ids


def _set_camera(pos, look_at) -> None:
    srv = STATE["server"]
    if srv is None:
        return
    for client in list(srv.get_clients().values()):
        try:
            with client.atomic():
                client.camera.position = tuple(float(x) for x in pos)
                client.camera.look_at = tuple(float(x) for x in look_at)
        except Exception:  # noqa: BLE001
            pass


def _camera_tick(term) -> None:
    """Runs on the sim thread. Overview = one shot; Follow = re-aim at robot N every 5 steps."""
    if STATE["server"] is None:
        return
    if STATE["overview"]:
        STATE["overview"] = False
        ids = STATE["visible_ids"] or list(range(min(int(os.environ.get("DEMO_VISIBLE_ENVS", "16")), term.num_envs)))
        import torch

        p = term.robot.data.root_pos_w.torch[ids, :3].float()
        c = p.median(0).values.tolist()
        lo, hi = torch.quantile(p[:, :2], 0.1, dim=0), torch.quantile(p[:, :2], 0.9, dim=0)
        span = float((hi - lo).max())  # frame the middle 80% of the robots; one straggler shouldn't zoom the camera out
        d = max(7.0, 0.75 * span)
        _set_camera((c[0] - d, c[1] - d, max(5.0, 0.8 * d)), (c[0], c[1], 0.3))
    if STATE["follow"]:
        STATE["cam_tick"] += 1
        if STATE["cam_tick"] % 5 == 0:
            ids = STATE["visible_ids"] or list(range(term.num_envs))
            i = ids[min(int(STATE["follow_idx"]), len(ids) - 1)]
            p = term.robot.data.root_pos_w.torch[i, :3].tolist()
            _set_camera((p[0] - 1.6, p[1] - 1.6, p[2] + 0.7), (p[0], p[1], p[2]))


def _patch_terrain_mesh() -> None:
    """Keep a copy of the generated terrain mesh (generator terrains only; the flat task uses a ground plane)."""
    from isaaclab.terrains import terrain_importer as ti

    orig = ti.TerrainImporter.import_mesh
    if getattr(orig, "_democtl", False):
        return

    def import_mesh(self, name, mesh, *a, **kw):
        try:
            STATE["terrain_mesh"] = mesh.copy()
        except Exception:  # noqa: BLE001
            pass
        return orig(self, name, mesh, *a, **kw)

    import_mesh._democtl = True
    ti.TerrainImporter.import_mesh = import_mesh


def _add_terrain_tint(server) -> None:
    """Viewer-only terrain colouring so low relief (obstacle blocks/pits, waves) is visible. Matte single-colour meshes:
    every wall one subtle grey, tops in height bands (blue below ground, grey at ground, orange above). The original
    terrain is hidden in the viewer (_patch_hide_terrain), so nothing is drawn twice (no flicker). Physics and the
    robots' sensors are unchanged. DEMO_TERRAIN_TINT=0 turns it off."""
    if os.environ.get("DEMO_TERRAIN_TINT", "1") == "0" or STATE.get("terrain_mesh") is None:
        return
    import numpy as np

    m = STATE["terrain_mesh"]
    v = np.asarray(m.vertices, dtype=np.float32)
    f = np.asarray(m.faces)
    walls = np.abs(np.asarray(m.face_normals)[:, 2]) < 0.3
    zc = v[f][:, :, 2].mean(1)
    inf = float("inf")
    bands = [(-inf, -0.10, (52, 98, 170)), (-0.10, -0.05, (86, 136, 200)), (-0.05, -0.015, (150, 182, 222)),
             (-0.015, 0.015, (222, 222, 218)), (0.015, 0.05, (238, 196, 150)), (0.05, 0.10, (226, 150, 84)),
             (0.10, inf, (196, 112, 48))]

    def add(name, sel, color):
        if not sel.any():
            return 0
        used, inv = np.unique(f[sel], return_inverse=True)
        server.scene.add_mesh_simple(name, v[used], inv.reshape(-1, 3).astype(np.uint32), color=color,
                                     material="standard", side="double", cast_shadow=False, receive_shadow=True)
        return int(sel.sum())

    n = add("/demo_terrain/walls", walls, (150, 150, 146))
    for i, (lo, hi, c) in enumerate(bands):
        n += add(f"/demo_terrain/band{i}", (~walls) & (zc >= lo) & (zc < hi), c)
    print(f"[democtl] terrain colour layer added ({n} faces, walls grey, 7 height bands)", flush=True)


def _patch_hide_terrain() -> None:
    """Don't let Newton's Viser viewer draw the terrain mesh itself (only our colour layer), so no surface is drawn twice.
    Matches the mesh with the generated terrain's footprint; flat-ground tasks (no generated terrain) are untouched."""
    if os.environ.get("DEMO_TERRAIN_TINT", "1") == "0":
        return
    import numpy as np
    from newton._src.viewer import viewer_viser as nv

    cls = nv.ViewerViser
    if getattr(cls.log_instances, "_democtl", False):
        return
    big = set()
    orig_mesh, orig_inst = cls.log_mesh, cls.log_instances

    def log_mesh(self, name, points, indices, *a, **kw):
        try:
            ter = STATE.get("terrain_mesh")  # None on flat-ground tasks (ground plane): never hide anything there
            if ter is not None:
                p = points.numpy() if hasattr(points, "numpy") else np.asarray(points)
                span = (p.max(0) - p.min(0))[:2] if len(p) else None
                want = np.asarray(ter.bounds[1] - ter.bounds[0])[:2]
                if span is not None and np.all(np.abs(span - want) < 0.5):  # same footprint as the generated terrain
                    big.add(self._qualify(name))
                    kw["hidden"] = True
        except Exception:  # noqa: BLE001
            pass
        return orig_mesh(self, name, points, indices, *a, **kw)

    def log_instances(self, name, mesh, *a, **kw):
        if self._qualify(mesh) in big:
            kw["hidden"] = True
        return orig_inst(self, name, mesh, *a, **kw)

    log_instances._democtl = True
    cls.log_mesh, cls.log_instances = log_mesh, log_instances


def _patch_sidebar() -> None:
    from isaaclab_visualizers.viser import viser_visualizer as vv

    orig = vv.ViserVisualizer._setup_isaaclab_sidebar
    if getattr(orig, "_democtl", False):
        return

    def _setup_isaaclab_sidebar(self, server):
        orig(self, server)
        try:
            _add_controls(server)
        except Exception as e:  # noqa: BLE001
            print(f"[democtl] could not add controls: {e}", flush=True)
        try:
            _add_terrain_tint(server)
        except Exception as e:  # noqa: BLE001 — visual nicety; never block the viewer
            print(f"[democtl] could not add terrain tint: {e}", flush=True)

    _setup_isaaclab_sidebar._democtl = True
    vv.ViserVisualizer._setup_isaaclab_sidebar = _setup_isaaclab_sidebar


def _add_camera(server) -> None:
    gui = server.gui
    STATE["server"] = server

    @server.on_client_connect
    def _(_client):  # every new browser tab starts framed on the visible robots
        STATE["overview"] = True
    n = int(os.environ.get("DEMO_VISIBLE_ENVS", "16"))
    with gui.add_folder("Camera", expand_by_default=True):
        ov = gui.add_button("Overview (all robots)")

        @ov.on_click
        def _(_ev):
            STATE["follow"] = False
            STATE["overview"] = True

        fol = gui.add_checkbox("Follow one robot", initial_value=False)
        idx = gui.add_slider("Robot #", min=0, max=max(n - 1, 0), step=1, initial_value=0)

        def _sync_cam(_ev=None):
            STATE["follow"] = bool(fol.value)
            STATE["follow_idx"] = int(idx.value)

        fol.on_update(_sync_cam)
        idx.on_update(_sync_cam)
    print("[democtl] Viser camera controls added", flush=True)


def _add_controls(server) -> None:
    _add_camera(server)
    if os.environ.get("DEMO_CONTROLS") != "1":
        return
    gui = server.gui
    with gui.add_folder("Demo controls", expand_by_default=True):
        gui.add_markdown(
            "**Training stage** swaps the robots' brain to an earlier checkpoint. "
            "**Joystick** overrides the random commands. **Push** shoves every robot."
        )
        opts = tuple(lbl for lbl, _ in STATE["ckpts"]) or ("(loading checkpoints…)",)
        dd = gui.add_dropdown("Training stage", options=opts, initial_value=opts[-1])
        STATE["dropdown"] = dd

        @dd.on_update
        def _(_ev):
            if getattr(_ev, "client", None) is None:  # server-side options/value change, not a click
                return
            for lbl, p in STATE["ckpts"]:
                if lbl == dd.value and os.path.abspath(p) != STATE.get("loaded"):
                    with _lock:
                        STATE["pending_ckpt"] = p

        STATE["status"] = gui.add_markdown("**Brain loaded:** final checkpoint")

        manual = gui.add_checkbox("Joystick control", initial_value=False)
        # keep sliders inside the command ranges the policy was trained on (Go2 ±1.0)
        mx, my, mw = (float(os.environ.get(k, "1.0")) for k in ("DEMO_VX_MAX", "DEMO_VY_MAX", "DEMO_WZ_MAX"))
        vx = gui.add_slider("Forward / back (m/s)", min=-mx, max=mx, step=round(mx / 10, 3), initial_value=0.0)
        vy = gui.add_slider("Left / right (m/s)", min=-my, max=my, step=round(my / 10, 3), initial_value=0.0)
        wz = gui.add_slider("Turn (rad/s)", min=-mw, max=mw, step=round(mw / 10, 3), initial_value=0.0)

        def _sync(_ev=None):
            with _lock:
                STATE.update(manual=bool(manual.value), vx=float(vx.value), vy=float(vy.value), wz=float(wz.value))

        for h in (manual, vx, vy, wz):
            h.on_update(_sync)

        stop = gui.add_button("Stop (zero joystick)")

        @stop.on_click
        def _(_ev):
            vx.value = vy.value = wz.value = 0.0
            _sync()

        push = gui.add_button("Push robots!")

        @push.on_click
        def _(_ev):
            with _lock:
                STATE["push"] = True
    if STATE["ckpts"]:
        _scan(STATE["ckpts"][-1][1])
    print("[democtl] Viser demo controls added", flush=True)


_tick = {"n": 0}


def _selftest_tick(term) -> None:
    """DEMO_SELFTEST=1: drive every control once from the sim thread (no browser needed)."""
    _tick["n"] += 1
    n = _tick["n"]
    tvx = 0.75 * float(os.environ.get("DEMO_VX_MAX", "1.0"))
    twz = 0.5 * float(os.environ.get("DEMO_WZ_MAX", "1.0"))
    if n == 200:
        print(f"[democtl-selftest] step {n}: joystick vx={tvx:.2f} wz={twz:.2f} (trained brain)", flush=True)
        STATE.update(manual=True, vx=tvx, vy=0.0, wz=twz)
    elif n in (350, 390):
        v = term.robot.data.root_lin_vel_b.torch.mean(0).tolist(); w = term.robot.data.root_ang_vel_b.torch[:, 2].mean().item()
        print(f"[democtl-selftest] step {n}: cmd vx={tvx:.2f} wz={twz:.2f} -> actual vx={v[0]:.2f} vy={v[1]:.2f} wz={w:.2f}", flush=True)
    elif n == 400:
        print(f"[democtl-selftest] step {n}: push", flush=True)
        STATE["push"] = True
    elif n == 405:
        v = term.robot.data.root_lin_vel_b.torch.abs().mean(0).tolist()
        print(f"[democtl-selftest] step {n}: just after push |v| x={v[0]:.2f} y={v[1]:.2f}", flush=True)
    elif n == 500:
        v = term.robot.data.root_lin_vel_b.torch.mean(0).tolist()
        fell = int((term._env.episode_length_buf < 100).sum().item())
        print(f"[democtl-selftest] step {n}: 2 s after push, back on command? vx={v[0]:.2f} vy={v[1]:.2f}; robots reset since push: {fell}/{term.num_envs}", flush=True)
    elif n == 600 and STATE["ckpts"]:
        print(f"[democtl-selftest] step {n}: load {STATE['ckpts'][0][0]}", flush=True)
        STATE["pending_ckpt"] = STATE["ckpts"][0][1]
    elif n == 790:
        v = term.robot.data.root_lin_vel_b.torch.mean(0).tolist()
        print(f"[democtl-selftest] step {n}: untrained brain, cmd vx={tvx:.2f} -> actual vx={v[0]:.2f}", flush=True)
    elif n == 800 and STATE["ckpts"]:
        print(f"[democtl-selftest] step {n}: load {STATE['ckpts'][-1][0]}; joystick off", flush=True)
        STATE["pending_ckpt"] = STATE["ckpts"][-1][1]
        STATE["manual"] = False
    elif n == 1000:
        print(f"[democtl-selftest] step {n}: done", flush=True)


_ab = {"n": 0, "falls": {"final": 0, "first": 0}}


def _abtest_tick(term) -> None:
    """DEMO_SELFTEST=2: same joystick command (0.5 m/s forward) for 20 s each on the final brain, then the run's
    first checkpoint; count falls (terminations that are not time-outs)."""
    _ab["n"] += 1
    n = _ab["n"]
    tm = term._env.termination_manager
    if n == 50:
        STATE.update(manual=True, vx=0.5, vy=0.0, wz=0.0)
        print("[democtl-ab] phase 1: final brain, joystick 0.5 m/s forward, 20 s", flush=True)
    if 100 <= n < 1100:
        fell = int((tm.terminated & ~tm.time_outs).sum().item())
        _ab["falls"]["final" if n < 600 else "first"] += fell
    if n == 600 and STATE["ckpts"]:
        STATE["pending_ckpt"] = STATE["ckpts"][0][1]
        print(f"[democtl-ab] phase 2: {STATE['ckpts'][0][0]}, same command, 20 s", flush=True)
    if n == 1100:
        print(f"[democtl-ab] falls in 20 s with {term.num_envs} robots: final brain={_ab['falls']['final']}  first checkpoint={_ab['falls']['first']}", flush=True)
        STATE["manual"] = False
        if STATE["ckpts"]:
            STATE["pending_ckpt"] = STATE["ckpts"][-1][1]


_ev = {"n": 0}


def _eval_tick(term) -> None:
    """DEMO_SELFTEST=3: per-terrain pass test. Every robot is driven forward (DEMO_EVAL_VX, default 0.6 m/s) from its
    station centre in a random heading. Each episode is one attempt: PASS = gets >= DEMO_EVAL_DIST (2.0 m) from its station
    centre (past the obstacles), FALL = terminated (not a time-out) before that, STUCK = timed out before that. Prints a table every 1000 steps.
    Runs on the sim thread; resets for this step have already happened when _update_command runs."""
    import torch

    env = term._env
    n = term.num_envs
    _ev["n"] += 1
    k = _ev["n"]
    pos = term.robot.data.root_pos_w.torch[:, :2]
    tm = env.termination_manager
    if k == 1:
        STATE.update(manual=True, vx=float(os.environ.get("DEMO_EVAL_VX", "0.6")), vy=0.0, wz=0.0)
        names = list(env.cfg.scene.terrain.terrain_generator.sub_terrains.keys())
        cols = env.scene.terrain.terrain_types.tolist()
        _ev["name"] = [names[int(c)] if int(c) < len(names) else str(c) for c in cols]
        _ev["start"] = env.scene.env_origins[:, :2].clone()  # station centre; never changes
        _ev["maxd"] = torch.zeros(n, device=pos.device)
        _ev["open"] = torch.ones(n, dtype=torch.bool, device=pos.device)
        _ev["res"] = {nm: [0, 0, 0] for nm in names}  # pass, fall, stuck
        _ev["stuckd"] = {nm: [] for nm in names}  # how far stuck robots got [m]
        _ev["dist"] = float(os.environ.get("DEMO_EVAL_DIST", "2.0"))
        print(f"[democtl-eval] {n} robots on stations {sorted(set(_ev['name']))}, vx={STATE['vx']}, pass at {_ev['dist']} m", flush=True)
        return
    ended = tm.terminated | tm.time_outs  # these envs were just reset; their attempt ended last step
    for i in torch.nonzero(ended & _ev["open"]).flatten().tolist():
        r = _ev["res"][_ev["name"][i]]
        if bool(tm.time_outs[i]) and not bool(tm.terminated[i]):
            r[2] += 1
            _ev["stuckd"][_ev["name"][i]].append(float(_ev["maxd"][i]))
        else:
            r[1] += 1
    # new attempt for reset envs (distance is always measured from the station centre)
    _ev["maxd"][ended] = 0.0
    _ev["open"][ended] = True
    d = (pos - _ev["start"]).norm(dim=1)
    _ev["maxd"] = torch.maximum(_ev["maxd"], d)
    newly = _ev["open"] & (_ev["maxd"] >= _ev["dist"])
    for i in torch.nonzero(newly).flatten().tolist():
        _ev["res"][_ev["name"][i]][0] += 1
    _ev["open"][newly] = False
    if k % 1000 == 0:
        rows = []
        for nm, (p, f, s_) in _ev["res"].items():
            tot = p + f + s_
            sd = _ev["stuckd"][nm]
            extra = f" (stuck got {sum(sd) / len(sd):.1f} m)" if sd else ""
            rows.append(f"{nm}: {p}/{tot} pass ({100 * p / tot:.0f}%), fall {f}, stuck {s_}{extra}" if tot else f"{nm}: no attempts yet")
        print(f"[democtl-eval] step {k}: " + " | ".join(rows), flush=True)


def install() -> None:
    fns = (_patch_runner, _patch_command, _patch_sidebar, _patch_terrain, _patch_terrain_mesh, _patch_hide_terrain, _patch_visible) if os.environ.get("DEMO_CONTROLS") == "1" else (_patch_command, _patch_sidebar, _patch_terrain, _patch_terrain_mesh, _patch_hide_terrain, _patch_visible)
    for fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"[democtl] {fn.__name__} failed: {e}", flush=True)
    print("[democtl] installed", flush=True)
