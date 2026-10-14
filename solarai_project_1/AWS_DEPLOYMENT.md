# ☀️ SolarAI — AWS Deployment Guide

## Architecture

```
Internet → ALB → ECS Fargate (SolarAI container)
                      ↓
               S3 (model artifacts)
               ECR (Docker image)
```

---

## Step 1: Build & Push Docker Image to ECR

```bash
# Set variables
REGION=ap-south-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO=solarai-app

# Create ECR repo
aws ecr create-repository --repository-name $REPO --region $REGION

# Login
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build & push
docker build -t $REPO .
docker tag $REPO:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
```

---

## Step 2: Upload Model to S3

```bash
aws s3 mb s3://solarai-models-$(date +%s) --region $REGION
aws s3 sync saved_model/ s3://your-bucket/saved_model/
aws s3 cp dataset.csv s3://your-bucket/dataset.csv
```

---

## Step 3: Deploy on ECS Fargate

```bash
# Create cluster
aws ecs create-cluster --cluster-name solarai-cluster

# Register task definition
cat > task-definition.json << EOF
{
  "family": "solarai-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [{
    "name": "solarai",
    "image": "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/solarai-app:latest",
    "portMappings": [{"containerPort": 8501, "protocol": "tcp"}],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/solarai",
        "awslogs-region": "${REGION}",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
EOF

aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster solarai-cluster \
  --service-name solarai-service \
  --task-definition solarai-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

---

## Step 4: Quick Deploy (Docker Compose — Dev/EC2)

```bash
# On EC2 (t2.large or better)
sudo yum install -y docker
sudo systemctl start docker
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
     -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

git clone <your-repo> solarai && cd solarai
docker-compose up -d

# Access at http://<EC2-Public-IP>:8501
```

---

## Recommended EC2 Instance

| Use Case | Instance | RAM | vCPU |
|---|---|---|---|
| Dev/Demo | t3.medium | 4 GB | 2 |
| Production | m5.large | 8 GB | 2 |
| Training | g4dn.xlarge | 16 GB | 4+GPU |

---

## Cost Estimate (ap-south-1 Mumbai)

| Resource | Cost/month |
|---|---|
| ECS Fargate (1 task, t-equiv) | ~$25 |
| ECR storage | ~$2 |
| S3 (models + data) | ~$1 |
| Data Transfer | ~$5 |
| **Total** | **~$33/month** |