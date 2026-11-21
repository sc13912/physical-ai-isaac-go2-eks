"""Demo task: Go2 on a harder obstacle course. Registered from promhook.start() (Isaac Lab --external_callback).

Demo-Velocity-ObstacleCourse-UnitreeGo2
  train: 10 difficulty rows x 20 columns of 8 m tiles (curriculum), 9 terrain types
  play:  6 stations, 5 m each, laid out 2 x 3 (GridTerrainGenerator) (stairs up, stairs down, obstacle field,
         stepping stones, gaps, waves), medium-hard difficulty, slower random commands
Builds on the stock Isaac-Velocity-Rough-UnitreeGo2 config (same robot, sensors, rewards, observations),
so a policy trained on the stock rough task can be fine-tuned here (--checkpoint).
"""

import os

import gymnasium as gym
import numpy as np
import torch
import trimesh

import isaaclab.envs.mdp as base_mdp
import isaaclab.terrains as tg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshGapTerrainCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass
from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

TASK_ID = "Demo-Velocity-ObstacleCourse-UnitreeGo2"


def shallow_gap_terrain(difficulty: float, cfg: "ShallowGapTerrainCfg"):
    """Isaac Lab gap_terrain (platform + outer ring, 1 m tall, no floor = 1 m-deep moat) plus a floor `gap_depth` below
    the surface inside the moat, so a missed step is recoverable instead of a 1 m drop. Geometry otherwise identical."""
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    h = 1.0
    c = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -h / 2)
    inner = (cfg.platform_width + 2 * gap_width, cfg.platform_width + 2 * gap_width)
    meshes = list(make_border(cfg.size, inner, h, c))
    meshes.append(trimesh.creation.box((cfg.platform_width, cfg.platform_width, h), trimesh.transformations.translation_matrix(c)))
    floor_h = h - cfg.gap_depth
    meshes.append(trimesh.creation.box((inner[0], inner[1], floor_h),
                                       trimesh.transformations.translation_matrix((c[0], c[1], -cfg.gap_depth - floor_h / 2))))
    return meshes, np.array([c[0], c[1], 0.0])


@configclass
class ShallowGapTerrainCfg(MeshGapTerrainCfg):
    function: str = "demotasks:shallow_gap_terrain"
    gap_depth: float = 0.25


class GridTerrainGenerator(TerrainGenerator):
    """Stock curriculum generator with 1 row x N stations (same tiles, same seed, same 2-robots-per-station spawn),
    but the stations are placed physically as grid_shape (default 2 x 3) so the whole course fits one camera view.
    Station k goes to grid cell divmod(k, grid_cols): row 0 = stairs up/down, obstacle field; row 1 = stones, gaps, waves."""

    def __init__(self, cfg: "GridTerrainGeneratorCfg", device: str = "cpu"):
        gr, gc = cfg.grid_shape
        if gr * gc < cfg.num_rows * cfg.num_cols:
            raise ValueError(f"grid {gr}x{gc} cannot hold {cfg.num_rows}x{cfg.num_cols} stations")
        super().__init__(cfg, device)
        # the base class centred the course for num_rows x num_cols; re-centre it for the physical grid
        d = np.array([cfg.size[0] * (cfg.num_rows - gr) / 2, cfg.size[1] * (cfg.num_cols - gc) / 2, 0.0])
        self.terrain_mesh.apply_translation(d)
        self.terrain_origins += d
        for name, v in self.flat_patches.items():
            self.flat_patches[name] = v + torch.tensor(d, dtype=torch.float, device=self.device)
        xy = np.round(self.terrain_origins[..., :2], 2).tolist()
        print(f"[demotasks] {gr}x{gc} course, station centres (x, y): {xy}", flush=True)

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        prow, pcol = divmod(row * self.cfg.num_cols + col, self.cfg.grid_shape[1])
        d = np.array([(prow - row) * self.cfg.size[0], (pcol - col) * self.cfg.size[1], 0.0])
        mesh.apply_translation(d)  # the base class then adds its own (row, col) offset -> lands on (prow, pcol)
        super()._add_sub_terrain(mesh, np.asarray(origin, dtype=float) + d, row, col, sub_terrain_cfg)

    def _add_terrain_border(self):
        gr, gc = self.cfg.grid_shape
        sx, sy, b = self.cfg.size[0], self.cfg.size[1], self.cfg.border_width
        meshes = make_border((gr * sx + 2 * b, gc * sy + 2 * b), (gr * sx, gc * sy), height=abs(self.cfg.border_height),
                             position=(gr * sx / 2, gc * sy / 2, -self.cfg.border_height / 2))
        border = trimesh.util.concatenate(meshes)
        border.update_faces(~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1))
        self.terrain_meshes.append(border)


@configclass
class GridTerrainGeneratorCfg(tg.TerrainGeneratorCfg):
    class_type: type = GridTerrainGenerator
    grid_shape: tuple = (2, 3)  # physical (rows along x, cols along y); DEMO_PLAY_GRID="1,6" = the old straight line


def _course(train: bool, hole: float = 0.25, visible: bool = False, compact: bool = False) -> tg.TerrainGeneratorCfg:
    """train: 10 x 20 tiles of 8 m, 9 terrain types (rows = difficulty).  play: the 6 stations, 5 m each, laid out 2 x 3.
    compact (training): the same 6 stations as play, 5 m tiles, 5 difficulty rows x 6 columns (~25 x 30 m)."""
    big = train and not compact     # the full-size training field
    edge = 1.0 if big else 0.25  # inner border of mesh tiles
    stairs = dict(step_height_range=(0.03, 0.15), step_width=0.3, platform_width=3.0 if big else 2.0,
                  border_width=edge, holes=False)
    subs = {
        "stairs_up": tg.MeshPyramidStairsTerrainCfg(**stairs),
        "stairs_down": tg.MeshInvertedPyramidStairsTerrainCfg(**stairs),
        "obstacle_field": tg.HfDiscreteObstaclesTerrainCfg(
            obstacle_height_mode="choice", obstacle_width_range=(0.25, 0.6), obstacle_height_range=(0.05, 0.15),
            num_obstacles=25 if big else 12, platform_width=1.5, border_width=0.25),
        # visible=True: every hole/gap >= 2 height-scan cells (scan resolution 0.1 m). 0.1 m slots alias in the scan,
        # the policy can't see them and feet wedge (measured: stuck robots at 1.2-1.6 m from the station centre).
        "stepping_stones": tg.HfSteppingStonesTerrainCfg(
            stone_height_max=0.0, stone_width_range=(0.5, 0.9) if visible else (0.45, 1.0),
            stone_distance_range=(0.2, 0.3) if visible else (0.05, 0.18),
            holes_depth=-hole, platform_width=1.5, border_width=0.25),  # shallow holes: a slip is recoverable, not a trap
        "gaps": ShallowGapTerrainCfg(gap_width_range=(0.15, 0.30) if visible else (0.05, 0.22), gap_depth=hole,
                                     platform_width=2.5 if big else 2.0),
        "waves": tg.HfWaveTerrainCfg(amplitude_range=(0.02, 0.12), num_waves=3 if big else 2, border_width=0.25),
    }
    if big:
        subs["boxes"] = tg.MeshRandomGridTerrainCfg(grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0)
        subs["slopes"] = tg.HfPyramidSlopedTerrainCfg(slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25)
        subs["rough"] = tg.HfRandomUniformTerrainCfg(noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25)
        share = {"stairs_up": 0.15, "stairs_down": 0.15, "obstacle_field": 0.15, "stepping_stones": 0.10,
                 "gaps": 0.10, "waves": 0.10, "boxes": 0.10, "slopes": 0.10, "rough": 0.05}
        # DEMO_TRAIN_FOCUS="stepping_stones=0.25,gaps=0.2": set those shares, scale the rest to fill 1.0
        focus = {k: float(v) for k, v in (kv.split("=") for kv in os.environ.get("DEMO_TRAIN_FOCUS", "").split(",") if "=" in kv)}
        if focus:
            rest = sum(v for k, v in share.items() if k not in focus)
            left = max(1.0 - sum(focus.values()), 0.0)
            share = {k: focus.get(k, v * left / rest if rest else 0.0) for k, v in share.items()}
    else:
        share = {k: 1.0 / len(subs) for k in subs}
    for k, v in subs.items():
        v.proportion = share[k]
    grid = {} if train else {"grid_shape": tuple(int(x) for x in os.environ.get("DEMO_PLAY_GRID", "2,3").split(","))}
    return (tg.TerrainGeneratorCfg if train else GridTerrainGeneratorCfg)(
        **grid,
        size=(8.0, 8.0) if big else (5.0, 5.0),
        border_width=20.0 if big else (10.0 if compact else 3.0),
        num_rows=10 if big else (5 if compact else 1),
        num_cols=20 if big else len(subs),
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, use_cache=False,
        curriculum=True,  # rows = difficulty, columns = terrain types (in order) -> one station per column in play
        # play ~ what the policy has mastered; DEMO_PLAY_DIFFICULTY="lo,hi" overrides (used by the evaluation runs)
        difficulty_range=(0.0, 1.0) if train else tuple(float(x) for x in os.environ.get("DEMO_PLAY_DIFFICULTY", "0.35,0.55").split(",")),
        sub_terrains=subs,
        # play: fixed seed = same station order + difficulties every run; DEMO_TERRAIN_SEED overrides. Not byte-identical:
        # Isaac Lab's height-field terrains (obstacle-field placement etc.) draw from numpy's global RNG, which is unseeded.
        seed=None if train else int(os.environ.get("DEMO_TERRAIN_SEED", "42")),
    )


@configclass
class Go2ObstacleCourseEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _course(train=True)

    def play_mode(self):
        super().play_mode()  # stock: random spawn, no curriculum, no pushes
        self.scene.terrain.terrain_generator = _course(train=False)
        r = self.commands.base_velocity.ranges  # slower wandering so robots stay near their station
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.5, 0.5), (-0.3, 0.3), (-0.5, 0.5)


@configclass
class Go2ObstacleCoursePPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "unitree_go2_obstacles"


SHALLOW_TASK_ID = "Demo-Velocity-ObstacleCourse-Shallow-UnitreeGo2"
VISIBLE_TASK_ID = "Demo-Velocity-ObstacleCourse-Visible-UnitreeGo2"
COMPACT_TASK_ID = "Demo-Velocity-ObstacleCourse-Compact-UnitreeGo2"


@configclass
class Go2ObstacleCourseShallowEnvCfg(UnitreeGo2RoughEnvCfg):
    """Same course, but stepping-stone holes and the gap moat are only 0.10 m deep: a foot that slips in steps out
    (0.25 m slots ~10 cm wide wedged Go2 feet)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _course(train=True, hole=0.10)

    def play_mode(self):
        super().play_mode()
        self.scene.terrain.terrain_generator = _course(train=False, hole=0.10)
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.5, 0.5), (-0.3, 0.3), (-0.5, 0.5)


@configclass
class Go2ObstacleCourseShallowPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "unitree_go2_obstacles_shallow"


@configclass
class Go2ObstacleCourseVisibleEnvCfg(UnitreeGo2RoughEnvCfg):
    """Holes/gaps 0.2-0.3 m wide (visible to the 0.1 m height scan) and 0.15 m deep (recoverable)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _course(train=True, hole=0.15, visible=True)

    def play_mode(self):
        super().play_mode()
        self.scene.terrain.terrain_generator = _course(train=False, hole=0.15, visible=True)
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.5, 0.5), (-0.3, 0.3), (-0.5, 0.5)


@configclass
class Go2ObstacleCourseVisiblePPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "unitree_go2_obstacles_visible"


@configclass
class Go2ObstacleCourseCompactEnvCfg(UnitreeGo2RoughEnvCfg):
    """Training on the obstacle page's 6 stations, 5 difficulty rows x 6 columns of 5 m tiles (~25 x 30 m): every
    terrain type sits in one camera view. Play = the Visible 2 x 3 course."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _course(train=True, hole=0.15, visible=True, compact=True)
        # On a 25 x 30 m field robots walk off the edge and fall into the void (seen: z -120 m), which ends in NaN
        # observations. Lowest terrain is about -0.75 m, so 2 m below the field = off it: end that attempt.
        self.terminations.off_field = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": -2.0})

    def play_mode(self):
        super().play_mode()
        self.scene.terrain.terrain_generator = _course(train=False, hole=0.15, visible=True)
        r = self.commands.base_velocity.ranges
        r.lin_vel_x, r.lin_vel_y, r.ang_vel_z = (-0.5, 0.5), (-0.3, 0.3), (-0.5, 0.5)


@configclass
class Go2ObstacleCourseCompactPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    experiment_name = "unitree_go2_obstacles_compact"


def register() -> None:
    for tid, env_cls, agent_cls in ((TASK_ID, "Go2ObstacleCourseEnvCfg", "Go2ObstacleCoursePPORunnerCfg"),
                                    (SHALLOW_TASK_ID, "Go2ObstacleCourseShallowEnvCfg", "Go2ObstacleCourseShallowPPORunnerCfg"),
                                    (VISIBLE_TASK_ID, "Go2ObstacleCourseVisibleEnvCfg", "Go2ObstacleCourseVisiblePPORunnerCfg"),
                                    (COMPACT_TASK_ID, "Go2ObstacleCourseCompactEnvCfg", "Go2ObstacleCourseCompactPPORunnerCfg")):
        if tid in gym.registry:
            continue
        gym.register(
            id=tid,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{env_cls}",
                "rsl_rl_cfg_entry_point": f"{__name__}:{agent_cls}",
                "default_agent": "rsl_rl",
            },
        )
        print(f"[demotasks] registered {tid}", flush=True)
