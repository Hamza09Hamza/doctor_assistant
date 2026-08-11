#!/usr/bin/env bash
set -euo pipefail

# Official universal package: native on both Apple Silicon and Intel Macs.
ORTHANC_MACOS_VERSION="${ORTHANC_MACOS_VERSION:-26.4.2}"
ORTHANC_ARCHIVE="Orthanc-macOS-${ORTHANC_MACOS_VERSION}.zip"
ORTHANC_URL="https://orthanc.uclouvain.be/downloads/macos/packages/universal/${ORTHANC_ARCHIVE}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="${REPO_ROOT}/data/orthanc-macos"
VERSION_ROOT="${INSTALL_ROOT}/${ORTHANC_MACOS_VERSION}"
ARCHIVE_PATH="${INSTALL_ROOT}/${ORTHANC_ARCHIVE}"
DOWNLOADED_ARCHIVE="${HOME}/Downloads/${ORTHANC_ARCHIVE}"

find_orthanc() {
  find "$VERSION_ROOT" -type f \( -name 'Orthanc.exec' -o -name 'Orthanc' \) -print -quit
}

if [[ -n "$(find_orthanc 2>/dev/null)" ]]; then
  echo "Orthanc macOS ${ORTHANC_MACOS_VERSION} is already installed in:"
  echo "  $VERSION_ROOT"
  exit 0
fi

mkdir -p "$VERSION_ROOT"

if [[ -f "$DOWNLOADED_ARCHIVE" ]]; then
  ARCHIVE_PATH="$DOWNLOADED_ARCHIVE"
  echo "Using the package already downloaded by the browser:"
  echo "  $ARCHIVE_PATH"
elif [[ ! -f "$ARCHIVE_PATH" ]]; then
  echo "Downloading the official Orthanc macOS package (~320 MB)..."
  CURL_TLS_ARGS=()
  if [[ -x "${REPO_ROOT}/.venv-mlx/bin/python" ]]; then
    CERT_BUNDLE="$("${REPO_ROOT}/.venv-mlx/bin/python" -c \
      'import certifi; print(certifi.where())' 2>/dev/null || true)"
    if [[ -f "$CERT_BUNDLE" ]]; then
      CURL_TLS_ARGS=(--cacert "$CERT_BUNDLE")
    fi
  fi
  PARTIAL_PATH="${ARCHIVE_PATH}.partial"
  if ! curl --fail --location --progress-bar "${CURL_TLS_ARGS[@]}" \
    "$ORTHANC_URL" --output "$PARTIAL_PATH"; then
    if [[ "${ORTHANC_ALLOW_INSECURE_DOWNLOAD:-0}" != "1" ]]; then
      rm -f "$PARTIAL_PATH"
      echo "TLS verification failed. Download the package in Safari, or explicitly" >&2
      echo "retry with ORTHANC_ALLOW_INSECURE_DOWNLOAD=1." >&2
      exit 1
    fi
    echo "Retrying the same official URL with curl TLS verification disabled..." >&2
    curl --fail --location --progress-bar --insecure \
      "$ORTHANC_URL" --output "$PARTIAL_PATH"
  fi
  mv "$PARTIAL_PATH" "$ARCHIVE_PATH"
fi

echo "Extracting Orthanc..."
ditto -x -k "$ARCHIVE_PATH" "$VERSION_ROOT"

ORTHANC_BIN="$(find_orthanc)"
DICOMWEB_PLUGIN="$(find "$VERSION_ROOT" -type f -name 'OrthancDicomWeb*.dylib' -print -quit)"
if [[ -z "$ORTHANC_BIN" || -z "$DICOMWEB_PLUGIN" ]]; then
  echo "The package did not contain the expected Orthanc executable and DICOMweb plugin." >&2
  exit 1
fi

chmod +x "$ORTHANC_BIN"
if command -v codesign >/dev/null 2>&1; then
  codesign --verify --deep --strict "$ORTHANC_BIN"
fi
echo "Installed Orthanc macOS ${ORTHANC_MACOS_VERSION} in:"
echo "  $VERSION_ROOT"
echo "Start it with: bash scripts/start_orthanc_macos.sh"
