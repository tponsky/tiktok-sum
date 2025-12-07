#!/bin/bash

# TikTok RAG Video Search Deployment Script
# This script deploys to your production server

set -e

echo "=========================================="
echo "TikTok Video RAG Deployment"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="tiktok-rag"
IMAGE_NAME="tiktok-rag:latest"
PORT="8080"

echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check if we're in the right directory
if [ ! -f "rag_api.py" ]; then
    echo -e "${RED}Error: Not in the correct directory. Please run from project root.${NC}"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

echo -e "${YELLOW}Step 2: Loading environment variables...${NC}"

# Load .env file if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | grep -v '^$' | xargs)
    echo -e "${GREEN}✓ Loaded .env file${NC}"
else
    echo -e "${YELLOW}Warning: No .env file found${NC}"
fi

# Check required variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY is not set${NC}"
    exit 1
fi

if [ -z "$PINECONE_API_KEY" ]; then
    echo -e "${RED}Error: PINECONE_API_KEY is not set${NC}"
    exit 1
fi

if [ -z "$JWT_SECRET_KEY" ]; then
    echo -e "${YELLOW}Warning: JWT_SECRET_KEY is not set${NC}"
    echo "Using default secret key (not recommended for production)"
fi

echo -e "${GREEN}✓ Environment variables OK${NC}"
echo ""

echo -e "${YELLOW}Step 3: Stopping existing container (if any)...${NC}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping $CONTAINER_NAME..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    echo "Removing $CONTAINER_NAME..."
    docker rm $CONTAINER_NAME 2>/dev/null || true
    echo -e "${GREEN}✓ Old container removed${NC}"
else
    echo "No existing container found"
fi

echo ""

echo -e "${YELLOW}Step 4: Building Docker image...${NC}"
docker build --no-cache -t $IMAGE_NAME .

echo -e "${GREEN}✓ Image built successfully${NC}"
echo ""

echo -e "${YELLOW}Step 5: Starting new container...${NC}"

# Create data directory if it doesn't exist
mkdir -p $(pwd)/data

# Build docker run command with all environment variables
docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e PINECONE_API_KEY="$PINECONE_API_KEY" \
  -e JWT_SECRET_KEY="$JWT_SECRET_KEY" \
  -e STRIPE_SECRET_KEY="$STRIPE_SECRET_KEY" \
  -e STRIPE_WEBHOOK_SECRET="$STRIPE_WEBHOOK_SECRET" \
  -e APP_URL="$APP_URL" \
  -e RESEND_API_KEY="$RESEND_API_KEY" \
  -e DATA_DIR="/app/data" \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  $IMAGE_NAME

echo -e "${GREEN}✓ Container started${NC}"
echo ""

echo -e "${YELLOW}Step 6: Waiting for service to start...${NC}"
sleep 3

# Check if container is running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${GREEN}✓ Container is running${NC}"
else
    echo -e "${RED}✗ Container failed to start${NC}"
    echo ""
    echo "Logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo ""

echo -e "${YELLOW}Step 7: Testing API...${NC}"

sleep 2

# Test the API
if curl -s -f http://localhost:$PORT/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API is responding${NC}"
else
    echo -e "${YELLOW}⚠ API test skipped (no /health endpoint or not ready yet)${NC}"
    echo "Check logs with: docker logs $CONTAINER_NAME"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=========================================="
echo ""
echo "Container name: $CONTAINER_NAME"
echo "Image: $IMAGE_NAME"
echo "Port: $PORT"
echo ""
echo "Useful commands:"
echo "  View logs:        docker logs $CONTAINER_NAME"
echo "  Follow logs:      docker logs -f $CONTAINER_NAME"
echo "  Stop container:   docker stop $CONTAINER_NAME"
echo "  Restart:          docker restart $CONTAINER_NAME"
echo "  Shell access:     docker exec -it $CONTAINER_NAME /bin/bash"
echo ""

# Show recent logs
echo "Recent logs:"
echo "---"
docker logs --tail 20 $CONTAINER_NAME
echo ""

echo -e "${GREEN}✓ App is now running at http://localhost:$PORT${NC}"
echo ""
echo "Access via your domain:"
echo "  https://tiktoksum.staycurrentapp.com"
echo ""
