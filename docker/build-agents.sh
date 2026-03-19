#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-evoskill-agents}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================================"
echo "  Building EvoSkill Agents Image"
echo "============================================================"
echo ""
echo "Image name: $FULL_IMAGE_NAME"
echo "Build context: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"
docker build \
    -f docker/Dockerfile.agents \
    -t "$FULL_IMAGE_NAME" \
    .

echo ""
echo "============================================================"
echo "  Build completed successfully!"
echo "============================================================"
echo ""
echo "Image: $FULL_IMAGE_NAME"
echo ""
echo "To verify:"
echo "  docker run --rm $FULL_IMAGE_NAME node --version"
echo "  docker run --rm $FULL_IMAGE_NAME claude --version"
echo "  docker run --rm $FULL_IMAGE_NAME codex --version"
