#!/bin/bash
# usage: scripts/eval.sh CHECKPOINT DIFFICULTY [NAME] [TASK] [TERRAIN_SEED]   -> per-station pass/fall/stuck for 120 robots (20 per station), 6000 steps
set -e
export IMAGE=${IMAGE:-$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_DEFAULT_REGION:?set AWS_DEFAULT_REGION (README step 0)}.amazonaws.com/isaac-lab:3.0.0-kitless} CHECKPOINT=$1 PLAY_DIFFICULTY=$2
kubectl -n isaac delete pod play-${3:-eval} --ignore-not-found --wait=true >/dev/null
ROBOT=${3:-eval} TASK=${4:-Demo-Velocity-ObstacleCourse-Visible-UnitreeGo2} SELFTEST=3 VX_MAX=1.0 VY_MAX=1.0 WZ_MAX=1.0 PUSH=1.0 PRESETS=presets=newton_mjwarp \
  envsubst "\$IMAGE \$ROBOT \$TASK \$CHECKPOINT \$SELFTEST \$VX_MAX \$VY_MAX \$WZ_MAX \$PUSH \$PRESETS \$PLAY_DIFFICULTY" < "$(dirname "$0")/../k8s/play-pod.yaml" \
  | sed "s/--num_envs 16/--num_envs 120/; s/ --real-time//" | sed "s/value: \"${PLAY_DIFFICULTY}\"/value: \"${PLAY_DIFFICULTY}\"\n        - name: DEMO_TERRAIN_SEED\n          value: \"${5:-42}\"/" | kubectl apply -f - >/dev/null
for i in $(seq 1 200); do kubectl -n isaac logs play-${3:-eval} 2>/dev/null | grep -qE "democtl-eval\] step 6000|Traceback" && break
  case $(kubectl -n isaac get pod play-${3:-eval} -o jsonpath='{.status.phase}') in Failed|Succeeded) break;; esac; sleep 5; done
kubectl -n isaac logs play-${3:-eval} 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "democtl-eval\] (step 6000|[0-9]+ robots)|Traceback|[Ee]rror" | tail -3
kubectl -n isaac delete pod play-${3:-eval} --wait=false >/dev/null
