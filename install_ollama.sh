#!/bin/sh
# This script installs Ollama on Linux for local user installation.

set -eu

# Set the user directory where Ollama will be installed
OLLAMA_INSTALL_DIR="$HOME/ollama"

# The rest of the setup remains as before:
red="$( (/usr/bin/tput bold || :; /usr/bin/tput setaf 1 || :) 2>&-)"
plain="$( (/usr/bin/tput sgr0 || :) 2>&-)"

status() { echo ">>> $*" >&2; }
error() { echo "${red}ERROR:${plain} $*"; exit 1; }
warning() { echo "${red}WARNING:${plain} $*"; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf $TEMP_DIR; }
trap cleanup EXIT

available() { command -v $1 >/dev/null; }
require() {
    local MISSING=''
    for TOOL in $*; do
        if ! available $TOOL; then
            MISSING="$MISSING $TOOL"
        fi
    done

    echo $MISSING
}

[ "$(uname -s)" = "Linux" ] || error 'This script is intended to run on Linux only.'

ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) error "Unsupported architecture: $ARCH" ;;
esac

# Skip WSL2 checks and other system-related setup for now
VER_PARAM="${OLLAMA_VERSION:+?version=$OLLAMA_VERSION}"

NEEDS=$(require curl awk grep sed tee xargs)
if [ -n "$NEEDS" ]; then
    status "ERROR: The following tools are required but missing:"
    for NEED in $NEEDS; do
        echo "  - $NEED"
    done
    exit 1
fi

# Ensure the directory for Ollama is created in the user's home directory
mkdir -p "$OLLAMA_INSTALL_DIR"
status "Installing ollama to $OLLAMA_INSTALL_DIR"

# Download the correct Linux version based on architecture
status "Downloading Linux ${ARCH} bundle"
curl --fail --show-error --location --progress-bar \
    "https://ollama.com/download/ollama-linux-${ARCH}.tgz${VER_PARAM}" | \
     tar -xzf - -C "$OLLAMA_INSTALL_DIR"

# Make sure the binary is linked to the user's local bin directory
mkdir -p "$HOME/bin"
ln -sf "$OLLAMA_INSTALL_DIR/ollama" "$HOME/bin/ollama"

status "Installation completed. Ollama is installed in $OLLAMA_INSTALL_DIR."

# No system-level configurations here (no systemd or users created)
