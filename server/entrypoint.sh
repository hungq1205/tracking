#!/bin/bash
# Keeps /app a live checkout of ${REPO_REF} — updated fresh on every
# container start rather than baked into the image at build time, so
# `docker run` always executes the newest code pushed to the branch
# without needing an image rebuild. See CLAUDE.md's Deployment section.
#
# Model weights (DA3METRIC-LARGE.onnx) live outside /app (DA3_ONNX_PATH,
# baked in at build time) specifically so they're never touched by the
# clone/reset below.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/hungq1205/tracking}"
REPO_REF="${REPO_REF:-vi-slam}"

if [ -d /app/.git ]; then
    echo "[entrypoint] /app is already a checkout — updating to latest ${REPO_REF}..."
    git -C /app fetch --depth 1 origin "${REPO_REF}"
    git -C /app reset --hard "origin/${REPO_REF}"
else
    echo "[entrypoint] Cloning ${REPO_REF} into /app..."
    git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" /app
fi

pip install --no-deps -e /app/scan_server/Depth-Anything-3

exec "$@"
