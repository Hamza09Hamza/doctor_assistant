#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="${REPO_ROOT}/data/orthanc-macos"
CONFIG_PATH="${REPO_ROOT}/deployments/orthanc-macos.json"
STORAGE_PATH="${REPO_ROOT}/data/orthanc-storage"

if curl --fail --silent --user doctor_assistant:doctor_assistant \
  http://localhost:8042/system >/dev/null 2>&1; then
  echo "Orthanc is already running at http://localhost:8042"
  exit 0
fi

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [PATH_TO_EXTRACTED_ORTHANC_PACKAGE]" >&2
  exit 2
fi

SEARCH_ROOT="${1:-$INSTALL_ROOT}"
if [[ ! -d "$SEARCH_ROOT" ]]; then
  echo "Orthanc is not installed. Run: bash scripts/install_orthanc_macos.sh" >&2
  exit 1
fi

ORTHANC_BIN="$(find "$SEARCH_ROOT" -type f \( -name 'Orthanc.exec' -o -name 'Orthanc' \) -print -quit)"
DICOMWEB_PLUGIN="$(find "$SEARCH_ROOT" -type f -name 'OrthancDicomWeb*.dylib' -print -quit)"
if [[ -z "$ORTHANC_BIN" ]]; then
  echo "Could not find Orthanc.exec under: $SEARCH_ROOT" >&2
  exit 1
fi
if [[ -z "$DICOMWEB_PLUGIN" ]]; then
  echo "Could not find the DICOMweb plugin under: $SEARCH_ROOT" >&2
  exit 1
fi

mkdir -p "$STORAGE_PATH"
export DOCTOR_ASSISTANT_ORTHANC_STORAGE="$STORAGE_PATH"
export DOCTOR_ASSISTANT_ORTHANC_PLUGINS="$(dirname "$DICOMWEB_PLUGIN")"

echo "Starting native Orthanc (no Docker, no local AI inference)..."
echo "  REST/DICOMweb: http://localhost:8042"
echo "  Storage: $DOCTOR_ASSISTANT_ORTHANC_STORAGE"
echo "Leave this terminal open; press Ctrl-C to stop Orthanc."
exec "$ORTHANC_BIN" "$CONFIG_PATH"
