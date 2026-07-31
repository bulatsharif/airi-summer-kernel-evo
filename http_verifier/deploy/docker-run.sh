#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image="${CUTE_HARNESS_IMAGE:-cute-harness:latest}"
container="${CUTE_HARNESS_CONTAINER:-cute-harness}"
network="${CUTE_HARNESS_NETWORK:-cute-harness-network}"
host_port="${CUTE_HARNESS_HOST_PORT:-18080}"
env_file="${CUTE_HARNESS_ENV_FILE:-${script_dir}/service.env}"

if ! docker network inspect "$network" >/dev/null 2>&1; then
  docker network create "$network" >/dev/null
fi

docker volume create cute-harness-profiles >/dev/null
docker volume create cute-harness-cache >/dev/null

# Named volumes are initially root-owned. Initialize them without exposing any
# host directory to the service container.
docker run --rm --user 0 \
  --mount source=cute-harness-profiles,target=/profiles \
  --mount source=cute-harness-cache,target=/cache \
  "$image" \
  sh -c 'chown -R 10001:10001 /profiles /cache'

if docker container inspect "$container" >/dev/null 2>&1; then
  docker rm -f "$container" >/dev/null
fi
docker run -d \
  --name "$container" \
  --restart unless-stopped \
  --init \
  --read-only \
  --user 10001:10001 \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 512 \
  --memory 64g \
  --shm-size 8g \
  --tmpfs /tmp:rw,nosuid,nodev,size=4g,mode=1777 \
  --network "$network" \
  --publish "0.0.0.0:${host_port}:18080" \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia0 \
  --mount type=bind,source=/usr/lib/x86_64-linux-gnu,target=/host-driver,readonly \
  --mount source=cute-harness-profiles,target=/profiles \
  --mount source=cute-harness-cache,target=/cache \
  --env-file "$env_file" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env LD_LIBRARY_PATH=/host-driver \
  --env CUTE_HARNESS_ARTIFACT_DIR=/profiles \
  --env CUTE_HARNESS_PYTHON=/usr/bin/python3 \
  --env CUTE_HARNESS_NSYS=/usr/local/bin/nsys \
  --env CUTE_DSL_CACHE_DIR=/cache/cute \
  --env TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
  --env TRITON_CACHE_DIR=/cache/triton \
  --log-opt max-size=20m \
  --log-opt max-file=3 \
  "$image" >/dev/null

echo "Started $container from $image on port $host_port"
