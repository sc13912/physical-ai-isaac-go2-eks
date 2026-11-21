# References — Isaac Lab on EKS demo (all links checked 2026-11-05)

## Robot learning stack (NVIDIA / open source)
- Isaac Lab (BSD-3) — https://github.com/isaac-sim/IsaacLab · v3.0.0 release: https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0
- Isaac Lab docs — install / kit-less container: https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html · Viser visualizer: https://isaac-sim.github.io/IsaacLab/main/source/concepts/visualization.html
- Isaac Lab container (`nvcr.io/nvidia/isaac-lab:3.0.0-kitless`) — https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab
- Isaac Lab Web Demos (in-browser physics) — https://isaac-sim.github.io/IsaacLab-Web-Demos/
- Isaac Sim 6.2 GPU requirements (why L40S/RTX, not H100) — https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html
- Newton physics engine (Apache-2.0, Linux Foundation) — https://github.com/newton-physics/newton · docs: https://newton-physics.github.io/newton/latest/guide/installation.html
- NVIDIA Warp — https://github.com/NVIDIA/warp
- MuJoCo Warp (solver used by Newton, `newton_mjwarp`) — https://github.com/google-deepmind/mujoco_warp
- RSL-RL (PPO trainer) — https://github.com/leggedrobotics/rsl_rl
- Viser (browser 3D viewer) — https://github.com/nerfstudio-project/viser

## Robot
- Unitree Go2 — https://www.unitree.com/go2
- Go2 licence in Isaac Lab (BSD-3, © Unitree Robotics) — https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/licenses/assets/unitree-license.txt
- Go2 task config (rough terrain) — https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/go2/rough_env_cfg.py
- Terrain generator (stock rough course) — https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/terrains/config/rough.py
- Isaac Lab task list (has `Isaac-Velocity-Flat-UnitreeGo2`, `Isaac-Velocity-Rough-UnitreeGo2`) — https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/environments.html
- Go2 asset config in Isaac Lab (`UNITREE_GO2_CFG` → `go2.usd`) — https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
- Unitree's own models and training repos: URDFs https://github.com/unitreerobotics/unitree_ros (BSD-3) · MuJoCo https://github.com/unitreerobotics/unitree_mujoco · Isaac Lab envs https://github.com/unitreerobotics/unitree_rl_lab (Apache-2.0; built on Isaac Lab 2.3)
- AWS Architecture Icons (diagram) — https://aws.amazon.com/architecture/icons/
- Go2 3D model (NVIDIA asset server) — https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd

## Kubernetes / EKS
- Karpenter — getting started: https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/ · capacity reservations (NodeClass): https://karpenter.sh/docs/concepts/nodeclasses/
- Kueue (job queueing) — https://kueue.sigs.k8s.io/docs/overview/
- NVIDIA device plugin — https://github.com/NVIDIA/k8s-device-plugin
- NVIDIA DCGM exporter (GPU metrics) — https://github.com/NVIDIA/dcgm-exporter · Grafana dashboard 12239: https://grafana.com/grafana/dashboards/12239
- EKS AL2023 NVIDIA AMI (v20261031, driver 580) — https://github.com/awslabs/amazon-eks-ami/releases
- Amazon EFS CSI driver — https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html

## AWS
- EC2 G6e (NVIDIA L40S) — https://aws.amazon.com/ec2/instance-types/g6e/ · L40S specs: https://www.nvidia.com/en-us/data-center/l40s/
- On-Demand Capacity Reservations (billing, cancel anytime) — https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-pricing-billing.html
- Amazon Managed Service for Prometheus — managed collector: https://docs.aws.amazon.com/prometheus/latest/userguide/AMP-collector-how-to.html
- Amazon Managed Grafana + IAM Identity Center — https://docs.aws.amazon.com/grafana/latest/userguide/authentication-in-AMG-SSO.html

## This demo
- Runbook — `README.md` in this repo
