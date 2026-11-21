"""Top-down labelled maps of the demo terrains + a fingerprint of the obstacle page's course.
usage (inside an Isaac Lab pod): python terrain_map.py <dir-with-demotasks.py> <out-dir> [--map]
Prints PLAYHASH <sha> so two demotasks versions can be compared; --map also writes obstacle-page-map.png and
compact-training-map.png (needs the version of demotasks that has the compact course)."""

import hashlib
import os
import sys

src, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
os.environ.setdefault("DEMO_PLAY_DIFFICULTY", "0.55,0.75")  # what the obstacle page (8083) runs

import numpy as np  # noqa: E402

import demotasks  # noqa: E402
from isaaclab.terrains.terrain_generator import TerrainGenerator  # noqa: E402

calls = []
_orig = TerrainGenerator._get_terrain_mesh


def _spy(self, difficulty, cfg):
    calls.append(float(difficulty))
    return _orig(self, difficulty, cfg)


TerrainGenerator._get_terrain_mesh = _spy


def build(cfg):
    calls.clear()
    np.random.seed(0)  # height-field terrains draw from the global numpy RNG (unseeded in Isaac Lab) -> pin it to compare
    return cfg.class_type(cfg, device="cpu"), list(calls)


NAMES = {"stairs_up": "Stairs up", "stairs_down": "Stairs down", "obstacle_field": "Obstacle field",
         "stepping_stones": "Stepping stones", "gaps": "Gaps", "waves": "Waves"}


def describe(key, d):
    if key in ("stairs_up", "stairs_down"):
        return f"pyramid\n{100 * (0.03 + 0.12 * d):.0f} cm steps"
    if key == "obstacle_field":
        return f"12 blocks + pits\nup to ±{100 * (0.05 + 0.10 * d):.0f} cm"
    if key == "stepping_stones":
        return f"{100 * (0.9 - 0.4 * d):.0f} cm stones\n{100 * (0.2 + 0.1 * d):.0f} cm slots, 15 cm deep"
    if key == "gaps":
        return f"{100 * (0.15 + 0.15 * d):.0f} cm trench ring\n15 cm deep"
    if key == "waves":
        return f"gentle waves\n±{100 * (0.02 + 0.10 * d):.0f} cm"
    return key


play_cfg = demotasks._course(train=False, hole=0.15, visible=True)
gen, diffs = build(play_cfg)
m = gen.terrain_mesh
print("PLAYHASH", hashlib.sha256(np.round(m.vertices, 4).tobytes() + m.faces.tobytes()).hexdigest()[:16],
      "difficulties", [round(x, 3) for x in diffs], flush=True)

if "--map" in sys.argv:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(gen, title, labels, path, figsize):
        m = gen.terrain_mesh
        v, f = np.asarray(m.vertices), np.asarray(m.faces)
        z = v[f][:, :, 2].mean(1)
        a, b, c = v[f][:, 0, :2], v[f][:, 1, :2], v[f][:, 2, :2]
        keep = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) > 1e-9  # drop walls
        f, z = f[keep], z[keep]
        order = np.argsort(z)
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        fig.patch.set_facecolor("#161d2c")
        ax.set_facecolor("#161d2c")
        tpc = ax.tripcolor(v[:, 1], v[:, 0], f[order], facecolors=z[order], cmap="terrain", vmin=-0.4, vmax=0.4,
                           edgecolors="none", rasterized=True)
        for (x, y), text in labels:
            ax.text(y, x, text, ha="center", va="center", fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#161d2ccc", ec="#76b900", lw=0.8))
        ax.set_aspect("equal")
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="#aab4c8", labelsize=7)
        ax.set_xlabel("metres", color="#aab4c8", fontsize=8)
        cb = fig.colorbar(tpc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("height (m)", color="#aab4c8", fontsize=8)
        cb.ax.tick_params(colors="#aab4c8", labelsize=7)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print("wrote", path, flush=True)

    keys = list(play_cfg.sub_terrains.keys())
    o = gen.terrain_origins  # (1, 6, 3): station k = keys[k]
    labels = [((o[0, k, 0] + (3.4 if o[0, k, 0] > 0 else -3.4), o[0, k, 1]), f"{NAMES[key]}\n{describe(key, diffs[k])}")
              for k, key in enumerate(keys)]  # label outside its tile, away from the middle
    draw(gen, "Obstacle page (/obstacle): 6 stations, 5 m each, difficulty 0.55-0.75", labels,
         os.path.join(out, "obstacle-page-map.png"), (9, 7))

    cc = demotasks._course(train=True, hole=0.15, visible=True, compact=True)
    cgen, cd = build(cc)
    ck = list(cc.sub_terrains.keys())
    co = cgen.terrain_origins  # (rows=difficulty, cols=type, 3)
    labels = [((co[0, c, 0] - 3.6 - 1.4 * (c % 2), co[0, c, 1]), NAMES[k]) for c, k in enumerate(ck)]  # staggered
    labels += [((co[r, 0, 0], co[r, 0, 1] - 3.8), f"level {r}") for r in range(co.shape[0])]
    draw(cgen, "Compact training field (/train-obstacle): 5 difficulty levels x 6 terrains, 5 m tiles", labels,
         os.path.join(out, "compact-training-map.png"), (10, 8))
