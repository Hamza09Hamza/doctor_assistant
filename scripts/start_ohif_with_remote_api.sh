#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 https://YOUR-NGROK-DEV-DOMAIN" >&2
  exit 2
fi

REMOTE_API="${1%/}"
if [[ ! "$REMOTE_API" =~ ^https:// ]]; then
  echo "Remote API must be an https:// ngrok URL" >&2
  exit 2
fi

curl --fail --silent --show-error \
  -H 'ngrok-skip-browser-warning: doctor-assistant' \
  "$REMOTE_API/health"
echo

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/viewer/ohif/platform/app"

# OHIF's current Rspack toolchain requires Node 24+. Prefer the side-by-side
# Homebrew runtime so the system's older `node`/`pnpm` cannot select a stale binding.
NODE24_BIN="/opt/homebrew/opt/node@24/bin"
if [[ -x "$NODE24_BIN/node" ]]; then
  export PATH="$NODE24_BIN:$PATH"
fi
RSPACK_BIN="$REPO_ROOT/viewer/ohif/node_modules/.bin/rspack"
if [[ ! -x "$RSPACK_BIN" ]]; then
  echo "OHIF dependencies are missing. From viewer/ohif, run:" >&2
  echo "  PATH=/opt/homebrew/opt/node@24/bin:\$PATH NODE_OPTIONS=--max-old-space-size=2048 corepack pnpm install --frozen-lockfile" >&2
  exit 1
fi

# Keep the viewer itself bounded on the 16 GB Mac. This launcher owns the UI
# compiler process, so an inherited large Node heap must not defeat the guard;
# all Torch/MONAI/MedSAM2 work stays in Colab.
export NODE_OPTIONS="--max-old-space-size=2048"
export NODE_ENV=development
export OHIF_OPEN=false
export PROXY_TARGET=http://localhost:3000/pacs/dicom-web
export PROXY_DOMAIN=http://localhost:8042
export PROXY_PATH_REWRITE_FROM=/pacs/dicom-web
export PROXY_PATH_REWRITE_TO=/dicom-web
export DOCTOR_ASSISTANT_API_TARGET="$REMOTE_API"
export APP_CONFIG=config/doctor_assistant.js

echo "OHIF API proxy -> $DOCTOR_ASSISTANT_API_TARGET"
echo "OHIF will remain at http://localhost:3000 and use local Orthanc for DICOM."
exec "$RSPACK_BIN" serve --config .webpack/webpack.pwa.js
