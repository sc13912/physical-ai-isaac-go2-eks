# Physical AI on Amazon EKS — Isaac Lab robot-learning demo

For details, please refer to my blog: [Physical AI on EKS: NVIDIA Isaac Lab training for the Unitree Go2](https://route179.dev/2026/11/20/physical-ai-on-eks-isaac-lab-unitree-go2/).

Run every command from the repo root. For my setup, the cluster `isaaclab-cluster` is deployed in ap-northeast-2 (Seoul) using an existing VPC with private subnets in two AZs, 2a and 2b (nodes), and public subnets in 2a/2c (ALB), plus an on-demand capacity reservation for 4× g6e.4xlarge (NVIDIA L40S) in ap-northeast-2b. Adjust the values based on your own environment.

## 0. Your values
```bash
export AWS_DEFAULT_REGION=ap-northeast-2 CLUSTER_NAME=isaaclab-cluster
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export VPC_ID=vpc-xxxxxxxx
export PRIVATE_SUBNET_1=subnet-xxxxxxxx PRIVATE_SUBNET_2=subnet-xxxxxxxx              # private subnets in two AZs (nodes)
export PUBLIC_SUBNETS="subnet-xxxxxxxx, subnet-xxxxxxxx"                               # public subnets in two AZs, at least one AZ shared with the private subnets (ALB, step 17)
export CAPACITY_RESERVATION_ID=cr-xxxxxxxx                                             # g6e.4xlarge reservation in one of the private subnets' AZs (none: delete capacityReservationSelectorTerms in k8s/karpenter-gpu.yaml)
export ADMIN_CIDR=203.0.113.10/32                                                      # your public IP (EKS API endpoint access)
```

## 1. Tools
```bash
eksctl version; kubectl version --client; helm version --short; docker --version
aws sts get-caller-identity
```
Want: eksctl 0.231.0, kubectl v1.36.6, helm v4.3.1, and your account ID.

## 2. Karpenter IAM
```bash
export KARPENTER_VERSION=1.15.1
curl -fsSL https://raw.githubusercontent.com/aws/karpenter-provider-aws/v${KARPENTER_VERSION}/website/content/en/preview/getting-started/getting-started-with-karpenter/cloudformation.yaml > infra/karpenter-cfn.yaml
aws cloudformation deploy --stack-name Karpenter-${CLUSTER_NAME} --template-file infra/karpenter-cfn.yaml \
  --capabilities CAPABILITY_NAMED_IAM --parameter-overrides ClusterName=${CLUSTER_NAME} \
  --tags project=physical-ai-eks
```

## 3. Cluster
```bash
export PRIVATE_AZ_1=$(aws ec2 describe-subnets --subnet-ids ${PRIVATE_SUBNET_1} --query 'Subnets[0].AvailabilityZone' --output text)
export PRIVATE_AZ_2=$(aws ec2 describe-subnets --subnet-ids ${PRIVATE_SUBNET_2} --query 'Subnets[0].AvailabilityZone' --output text)
envsubst '$AWS_ACCOUNT_ID $AWS_DEFAULT_REGION $CLUSTER_NAME $VPC_ID $PRIVATE_AZ_1 $PRIVATE_SUBNET_1 $PRIVATE_AZ_2 $PRIVATE_SUBNET_2 $ADMIN_CIDR' < infra/cluster.yaml | eksctl create cluster -f -
kubectl get nodes
```
Want: 2 system nodes Ready (~20 min).

## 4. Karpenter
```bash
export KARPENTER_VERSION=1.15.1
helm registry logout public.ecr.aws || true
helm upgrade --install karpenter oci://public.ecr.aws/karpenter/karpenter --version ${KARPENTER_VERSION} \
  --namespace kube-system --set settings.clusterName=${CLUSTER_NAME} --set settings.interruptionQueue=${CLUSTER_NAME} \
  --set settings.enableZonalShift=true --set controller.resources.requests.cpu=1 --set controller.resources.requests.memory=1Gi \
  --set controller.resources.limits.cpu=1 --set controller.resources.limits.memory=1Gi --wait
```

## 5. Mirror the Isaac Lab image to ECR
```bash
export ECR=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/isaac-lab:3.0.0-kitless
aws ecr create-repository --repository-name isaac-lab --image-scanning-configuration scanOnPush=true --tags Key=project,Value=physical-ai-eks
aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR}
docker pull nvcr.io/nvidia/isaac-lab:3.0.0-kitless && docker tag nvcr.io/nvidia/isaac-lab:3.0.0-kitless ${IMAGE} && docker push ${IMAGE}
```

## 6. GPU NodePool + NVIDIA device plugin
```bash
export CLUSTER_SG=$(aws eks describe-cluster --name ${CLUSTER_NAME} --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)
envsubst '$CLUSTER_NAME $CLUSTER_SG $PRIVATE_SUBNET_1 $PRIVATE_SUBNET_2 $CAPACITY_RESERVATION_ID' < k8s/karpenter-gpu.yaml | kubectl apply -f -
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin && helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin --version 0.20.1 -n nvidia-device-plugin --create-namespace
```

## 7. GPU smoke test
```bash
export IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/isaac-lab:3.0.0-kitless
kubectl create ns isaac --dry-run=client -o yaml | kubectl apply -f -
envsubst '$IMAGE' < k8s/smoke-test.yaml | kubectl apply -f -
kubectl -n isaac wait --for=jsonpath='{.status.phase}'=Running pod/smoke-gpu --timeout=30m   # GPU node launch + 6 GB image pull
kubectl -n isaac logs -f smoke-gpu
```
Want: `L40S`, `Physics newton_mjwarp`, `Steps per second` per iteration. Then `kubectl -n isaac delete pod smoke-gpu`.

## 8. EFS for logs + checkpoints
```bash
export CLUSTER_SG=$(aws eks describe-cluster --name ${CLUSTER_NAME} --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)
export EFS_SG=$(aws ec2 create-security-group --group-name isaaclab-efs --description "EFS for ${CLUSTER_NAME}" --vpc-id ${VPC_ID} \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=Name,Value=isaaclab-efs},{Key=project,Value=physical-ai-eks}]' --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id ${EFS_SG} --protocol tcp --port 2049 --source-group ${CLUSTER_SG}
export EFS_ID=$(aws efs create-file-system --encrypted --throughput-mode elastic \
  --tags Key=Name,Value=isaaclab-logs Key=project,Value=physical-ai-eks --query FileSystemId --output text)
for s in ${PRIVATE_SUBNET_1} ${PRIVATE_SUBNET_2}; do
  aws efs create-mount-target --file-system-id ${EFS_ID} --subnet-id $s --security-groups ${EFS_SG}; done
envsubst '$EFS_ID' < k8s/storage.yaml | kubectl apply -f -
```
Want: `kubectl -n isaac get pvc isaac-logs` → Bound.

## 9. Kueue
```bash
helm upgrade -i kueue oci://registry.k8s.io/kueue/charts/kueue --version 0.20.4 -n kueue-system --create-namespace --wait
kubectl apply -f k8s/kueue.yaml
```

## 10. Training jobs
```bash
export IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/isaac-lab:3.0.0-kitless
export NUM_ENVS=4096 RUNS=1 VIZ="" TRAIN_FOCUS="" VISIBLE_TILES=""
kubectl -n isaac create configmap promhook --from-file=isaac/promhook.py --from-file=isaac/democtl.py --dry-run=client -o yaml | kubectl apply -f -
ROBOT=go2long TASK=Isaac-Velocity-Flat-UnitreeGo2 MAX_ITER=1500 envsubst '$IMAGE $ROBOT $TASK $NUM_ENVS $MAX_ITER $VIZ $RUNS $TRAIN_FOCUS $VISIBLE_TILES' < k8s/train-job.yaml | kubectl apply -f -
ROBOT=h1     TASK=Isaac-Velocity-Flat-H1         MAX_ITER=500  envsubst '$IMAGE $ROBOT $TASK $NUM_ENVS $MAX_ITER $VIZ $RUNS $TRAIN_FOCUS $VISIBLE_TILES' < k8s/train-job.yaml | kubectl apply -f -
ROBOT=anymal TASK=Isaac-Velocity-Flat-AnymalD    MAX_ITER=300  envsubst '$IMAGE $ROBOT $TASK $NUM_ENVS $MAX_ITER $VIZ $RUNS $TRAIN_FOCUS $VISIBLE_TILES' < k8s/train-job.yaml | kubectl apply -f -
ROBOT=g1     TASK=Isaac-Velocity-Flat-G1         MAX_ITER=1500 envsubst '$IMAGE $ROBOT $TASK $NUM_ENVS $MAX_ITER $VIZ $RUNS $TRAIN_FOCUS $VISIBLE_TILES' < k8s/train-job.yaml | kubectl apply -f -
kubectl -n isaac get workloads,jobs,pods -o wide; kubectl get nodeclaims
```

## 11. Watch the robots
```bash
kubectl -n isaac wait --for=condition=complete job/train-go2long --timeout=60m   # the flat-walking brain from step 10
# trained Go2 + "Demo controls" (stage picker, joystick, push), always-on Deployment (self-heals), not queued
export IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/isaac-lab:3.0.0-kitless VX_MAX=1.0 VY_MAX=1.0 WZ_MAX=1.0 PUSH=1.0 PRESETS=presets=newton_mjwarp CONFIGMAP=promhook
export CHECKPOINT='$(ls /workspace/isaaclab/logs/rsl_rl/unitree_go2_flat/*_go2long/model_1499.pt | tail -1)'   # resolved inside the pod
ROBOT=go2 TASK=Isaac-Velocity-Flat-UnitreeGo2 NUM_ENVS=16 PLAY_DIFFICULTY=0.35,0.55 \
  envsubst '$IMAGE $ROBOT $TASK $CHECKPOINT $NUM_ENVS $VX_MAX $VY_MAX $WZ_MAX $PUSH $PRESETS $PLAY_DIFFICULTY $CONFIGMAP' < k8s/play-deploy.yaml | kubectl apply -f -
setsid nohup ./scripts/viewer-forward.sh 8080 app=isaac-play,robot=go2  >/tmp/fwd-8080.log 2>&1 < /dev/null &   # play view
# live training view: relearns flat walking from scratch every cycle (600 it, ~7 min), forever
ROBOT=go2 TASK=Isaac-Velocity-Flat-UnitreeGo2 NUM_ENVS=4096 MAX_ITER=600 RUNS=100000 VIZ="--viz viser --max_visible_envs 32" TRAIN_FOCUS="" VISIBLE_TILES="" \
  envsubst '$IMAGE $ROBOT $TASK $NUM_ENVS $MAX_ITER $VIZ $RUNS $TRAIN_FOCUS $VISIBLE_TILES' < k8s/train-job.yaml | kubectl apply -f -
setsid nohup ./scripts/viewer-forward.sh 8081 app=isaac-train,robot=go2 >/tmp/fwd-8081.log 2>&1 < /dev/null &   # live training view
```
Open http://localhost:8080 and http://localhost:8081. `CHECKPOINT=pretrained` = NVIDIA's policy. One-shot self-test (`SELFTEST=1`, drives every control once): `k8s/play-pod.yaml`.

## 12. Metrics: DCGM exporter + AMP managed scraper
```bash
helm repo add dcgm https://nvidia.github.io/dcgm-exporter/helm-charts && helm repo update
helm upgrade -i dcgm dcgm/dcgm-exporter --version 4.8.4 -n monitoring --create-namespace -f observability/dcgm-values.yaml
export AMP_ARN=$(aws amp create-workspace --alias isaaclab --tags project=physical-ai-eks --query arn --output text)
export CLUSTER_ARN=$(aws eks describe-cluster --name ${CLUSTER_NAME} --query cluster.arn --output text)
export CLUSTER_SG=$(aws eks describe-cluster --name ${CLUSTER_NAME} --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)
aws amp create-scraper --alias isaaclab --tags project=physical-ai-eks \
  --source eksConfiguration="{clusterArn='${CLUSTER_ARN}',securityGroupIds=['${CLUSTER_SG}'],subnetIds=['${PRIVATE_SUBNET_1}','${PRIVATE_SUBNET_2}']}" \
  --scrape-configuration configurationBlob=$(envsubst '$CLUSTER_NAME' < observability/scrape-config.yaml | base64 -w0) \
  --destination ampConfiguration="{workspaceArn='${AMP_ARN}'}"
```
Want: `aws amp list-scrapers --query 'scrapers[].status'` → ACTIVE (~15 min).

## 13. Amazon Managed Grafana
```bash
export ACC=${AWS_ACCOUNT_ID} SSO_USER_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx   # your IAM Identity Center user ID
cat > /tmp/amg-trust.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"grafana.amazonaws.com"},"Action":"sts:AssumeRole",
 "Condition":{"StringEquals":{"aws:SourceAccount":"$ACC"},"StringLike":{"aws:SourceArn":"arn:aws:grafana:${AWS_DEFAULT_REGION}:$ACC:/workspaces/*"}}}]}
EOF
export ROLE=$(aws iam create-role --role-name isaaclab-grafana-role --assume-role-policy-document file:///tmp/amg-trust.json --tags Key=project,Value=physical-ai-eks --query Role.Arn --output text)
aws iam attach-role-policy --role-name isaaclab-grafana-role --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess
export WS=$(aws grafana create-workspace --workspace-name isaaclab --account-access-type CURRENT_ACCOUNT --workspace-role-arn ${ROLE} \
  --authentication-providers AWS_SSO --permission-type CUSTOMER_MANAGED --workspace-data-sources PROMETHEUS \
  --tags project=physical-ai-eks --query workspace.id --output text)
aws grafana describe-workspace --workspace-id ${WS} --query workspace.status --output text   # wait for ACTIVE
aws grafana update-permissions --workspace-id ${WS} --update-instruction-batch \
  '[{"action":"ADD","role":"ADMIN","users":[{"id":"'"${SSO_USER_ID}"'","type":"SSO_USER"}]}]'
export SA=$(aws grafana create-workspace-service-account --workspace-id ${WS} --name provisioner --grafana-role ADMIN --query id --output text)
export TOKEN=$(aws grafana create-workspace-service-account-token --workspace-id ${WS} --service-account-id ${SA} --name t1 --seconds-to-live 3600 --query serviceAccountToken.key --output text)
export GURL=https://$(aws grafana describe-workspace --workspace-id ${WS} --query workspace.endpoint --output text)
export AMP_URL=https://aps-workspaces.${AWS_DEFAULT_REGION}.amazonaws.com/workspaces/$(aws amp list-workspaces --alias isaaclab --query 'workspaces[0].workspaceId' --output text)/
curl -s -X POST -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' ${GURL}/api/datasources -d "{\"name\":\"AMP isaaclab\",\"type\":\"prometheus\",\"access\":\"proxy\",\"url\":\"${AMP_URL}\",\"isDefault\":true,\"jsonData\":{\"httpMethod\":\"POST\",\"sigV4Auth\":true,\"sigV4AuthType\":\"ec2_iam_role\",\"sigV4Region\":\"${AWS_DEFAULT_REGION}\"}}"
jq '{dashboard: ., overwrite: true}' observability/dashboard.json | curl -s -X POST -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' ${GURL}/api/dashboards/db -d @-
echo ${GURL}/d/isaaclab-eks
```
Want: sign in with your IAM Identity Center user; dashboard "Isaac Lab on EKS — robots learning live".

## 14. Go2 obstacle course
```bash
export IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/isaac-lab:3.0.0-kitless NUM_ENVS=4096 RUNS=1 VIZ="" VISIBLE_TILES=""
kubectl -n isaac create configmap promhook --from-file=isaac/promhook.py --from-file=isaac/democtl.py --from-file=terrain/demotasks.py --dry-run=client -o yaml | kubectl apply -f -
# the brain: stock rough terrain 1,500 it (~28 min), then the obstacle course 1,000 it on top (~18 min)
ROBOT=go2rough TASK=Isaac-Velocity-Rough-UnitreeGo2 MAX_ITER=1500 TRAIN_FOCUS="" \
  envsubst '$IMAGE $ROBOT $TASK $NUM_ENVS $MAX_ITER $VIZ $RUNS $TRAIN_FOCUS $VISIBLE_TILES' < k8s/train-job.yaml | kubectl apply -f -
kubectl -n isaac wait --for=condition=complete job/train-go2rough --timeout=90m
ROBOT=go2visible TASK=Demo-Velocity-ObstacleCourse-Visible-UnitreeGo2 MAX_ITER=1000 TRAIN_FOCUS="stepping_stones=0.25,gaps=0.25" \
  VIZ='--checkpoint $(ls /workspace/isaaclab/logs/rsl_rl/unitree_go2_rough/*_go2rough/model_1499.pt | tail -1)' \
  envsubst '$IMAGE $ROBOT $TASK $NUM_ENVS $MAX_ITER $VIZ $RUNS $TRAIN_FOCUS $VISIBLE_TILES' < k8s/train-job.yaml | kubectl apply -f -
kubectl -n isaac wait --for=condition=complete job/train-go2visible --timeout=60m
# play it: 12 robots on the 6 stations (2 x 3)
export CHECKPOINT='$(ls /workspace/isaaclab/logs/rsl_rl/unitree_go2_obstacles_visible/*_go2visible/model_2498.pt | tail -1)'   # rough model_1499 + 1,000 iterations
export VX_MAX=1.0 VY_MAX=1.0 WZ_MAX=1.0 PUSH=1.0 PRESETS=presets=newton_mjwarp CONFIGMAP=promhook   # 6 stations laid out 2x3 (DEMO_PLAY_GRID)
ROBOT=go2rough TASK=Demo-Velocity-ObstacleCourse-Visible-UnitreeGo2 NUM_ENVS=12 PLAY_DIFFICULTY=0.55,0.75 \
  envsubst '$IMAGE $ROBOT $TASK $CHECKPOINT $NUM_ENVS $VX_MAX $VY_MAX $WZ_MAX $PUSH $PRESETS $PLAY_DIFFICULTY $CONFIGMAP' < k8s/play-deploy.yaml | kubectl apply -f -
setsid nohup ./scripts/viewer-forward.sh 8083 app=isaac-play,robot=go2rough >/tmp/fwd-8083.log 2>&1 < /dev/null &
```

## 15. Terrain pass test
```bash
./scripts/eval.sh "$CHECKPOINT" 0.55,0.75 e1 Demo-Velocity-ObstacleCourse-Visible-UnitreeGo2 42   # 120 robots, 20 per station, ~4 min
```
Want: every station 100% pass (our runs: `results/eval-results.log`).

## 16. Health check
```bash
kubectl -n isaac get deploy play-go2 play-go2rough; kubectl -n isaac get pods -l app=isaac-train,robot=go2   # RESTARTS should stay low
kubectl -n isaac get events --field-selector reason=Unhealthy | grep -i liveness                           # want: nothing
kubectl get nodepool gpu -o jsonpath='{.spec.disruption.budgets}'                                           # [{"nodes":"0"}]
```
Change code: update the ConfigMap (step 14), then `kubectl -n isaac rollout restart deploy/play-go2 deploy/play-go2rough` (each viewer restarts in place: about 2 min blank). Training picks the new code up at its next run.

## 17. Public viewers: ALB + Cognito sign-in
```bash
# AWS Load Balancer Controller (Pod Identity). Skip create-policy if AWSLoadBalancerControllerIAMPolicy already exists in the account.
curl -fsSL https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v3.6.2/docs/install/iam_policy.json -o /tmp/lbc-iam-policy.json
aws iam create-policy --policy-name AWSLoadBalancerControllerIAMPolicy --policy-document file:///tmp/lbc-iam-policy.json
eksctl create podidentityassociation --cluster ${CLUSTER_NAME} --namespace kube-system --service-account-name aws-load-balancer-controller \
  --role-name ${CLUSTER_NAME}-aws-lbc --permission-policy-arns arn:aws:iam::${AWS_ACCOUNT_ID}:policy/AWSLoadBalancerControllerIAMPolicy
helm repo add eks https://aws.github.io/eks-charts && helm repo update eks
helm upgrade -i aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --version 3.6.2 \
  --set clusterName=${CLUSTER_NAME} --set region=${AWS_DEFAULT_REGION} --set vpcId=${VPC_ID} --set enableServiceMutatorWebhook=false --wait

# Self-signed certificate -> ACM (the browser warns once; use an ACM certificate for your own domain instead if you have one)
openssl req -x509 -newkey rsa:2048 -nodes -days 30 -keyout /tmp/viewers.key -out /tmp/viewers.crt -subj "/CN=isaac-viewers" \
  -addext "subjectAltName=DNS:*.${AWS_DEFAULT_REGION}.elb.amazonaws.com" -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature,keyEncipherment" -addext "extendedKeyUsage=serverAuth"
export CERT_ARN=$(aws acm import-certificate --certificate fileb:///tmp/viewers.crt --private-key fileb:///tmp/viewers.key --query CertificateArn --output text)
shred -u /tmp/viewers.key

# Cognito: admin-only user pool, hosted sign-in domain, app client for the ALB, one user
export COGNITO_DOMAIN=isaac-viewers-$(openssl rand -hex 3)
export POOL_ID=$(aws cognito-idp create-user-pool --pool-name isaac-viewers --admin-create-user-config AllowAdminCreateUserOnly=true --query UserPool.Id --output text)
export COGNITO_POOL_ARN=$(aws cognito-idp describe-user-pool --user-pool-id ${POOL_ID} --query UserPool.Arn --output text)
aws cognito-idp create-user-pool-domain --user-pool-id ${POOL_ID} --domain ${COGNITO_DOMAIN} --managed-login-version 2
CLIENT_ARGS="--client-name alb --allowed-o-auth-flows code --allowed-o-auth-scopes openid --allowed-o-auth-flows-user-pool-client
  --supported-identity-providers COGNITO --explicit-auth-flows ALLOW_USER_SRP_AUTH ALLOW_REFRESH_TOKEN_AUTH --prevent-user-existence-errors ENABLED"
export COGNITO_CLIENT_ID=$(aws cognito-idp create-user-pool-client --user-pool-id ${POOL_ID} ${CLIENT_ARGS} --generate-secret \
  --callback-urls https://localhost/oauth2/idpresponse --query UserPoolClient.ClientId --output text)
aws cognito-idp create-managed-login-branding --user-pool-id ${POOL_ID} --client-id ${COGNITO_CLIENT_ID} --use-cognito-provided-values
aws cognito-idp admin-create-user --user-pool-id ${POOL_ID} --username viewer --message-action SUPPRESS
read -rs PW && aws cognito-idp admin-set-user-password --user-pool-id ${POOL_ID} --username viewer --password "$PW" --permanent; unset PW

# ALB + viewer proxy (pinned to the ALB's AZs), then point the app client's callback at the ALB
export ALB_AZS=$(aws ec2 describe-subnets --subnet-ids $(echo ${PUBLIC_SUBNETS} | tr -d ,) --query 'Subnets[].AvailabilityZone' --output json | jq -c .)
envsubst '$CERT_ARN $PUBLIC_SUBNETS $ALB_AZS $COGNITO_POOL_ARN $COGNITO_CLIENT_ID $COGNITO_DOMAIN' < k8s/alb-ingress.yaml | kubectl apply -f -
until ALB_DNS=$(kubectl -n isaac get ingress isaac-viewers -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'); [ -n "$ALB_DNS" ]; do sleep 10; done
export ALB_DNS=$(echo ${ALB_DNS} | tr A-Z a-z)
aws cognito-idp update-user-pool-client --user-pool-id ${POOL_ID} --client-id ${COGNITO_CLIENT_ID} ${CLIENT_ARGS} \
  --callback-urls "https://${ALB_DNS}/oauth2/idpresponse"     # update resets omitted fields, hence CLIENT_ARGS again
```
Want: `curl -sk -o /dev/null -w '%{http_code}\n' https://$ALB_DNS/play/` → `302` (anonymous → Cognito sign-in). Browser: `https://$ALB_DNS/` → accept the certificate warning once → sign in as `viewer` → `/play/` `/train/` `/obstacle/` `/train-obstacle/`.
Teardown: `kubectl -n isaac delete ingress isaac-viewers` while the Load Balancer Controller is still installed (otherwise the ALB is left behind), then delete the Cognito user pool and the ACM certificate.

## 18. Obstacle training loop
```bash
export IMAGE=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/isaac-lab:3.0.0-kitless ROBOT=go2obstacle TASK=Demo-Velocity-ObstacleCourse-Compact-UnitreeGo2 \
  NUM_ENVS=4096 MAX_ITER=600 RUNS=100000 TRAIN_FOCUS="" VISIBLE_TILES=0-4,0-5      # viewer: 30 robots, one pinned to each of the 30 tiles
export VIZ='--viz viser --max_visible_envs 30 --checkpoint $(ls /workspace/isaaclab/logs/rsl_rl/unitree_go2_rough/*_go2rough/model_1499.pt | tail -1)'
envsubst '$IMAGE $ROBOT $TASK $NUM_ENVS $MAX_ITER $VIZ $RUNS $TRAIN_FOCUS $VISIBLE_TILES' < k8s/train-job.yaml | kubectl apply -f -
setsid nohup ./scripts/viewer-forward.sh 8084 app=isaac-train,robot=go2obstacle >/tmp/fwd-8084.log 2>&1 < /dev/null &
```
Want: `kubectl -n isaac logs job/train-go2obstacle | grep pinned` → `showing 30 robots on tiles rows 0-4 x cols 0-5 (30 tiles), pinned there`. Maps: `terrain/compact-training-map.png`, `terrain/obstacle-page-map.png` (`terrain/terrain_map.py`). Route: `k8s/alb-ingress.yaml` (Service `viser-train-obstacle` + nginx location), then `kubectl -n isaac exec deploy/viewer-proxy -- nginx -s reload`.
