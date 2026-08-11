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

# Keep the viewer itself bounded on the 16 GB Mac. This is only the UI compiler;
# all Torch/MedSAM2 work stays in Colab.
export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=4096}"
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
exec pnpm exec rspack serve --config .webpack/webpack.pwa.js
