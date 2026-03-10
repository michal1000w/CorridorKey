#!/bin/bash

# Ensure script stops on error
set -e

# Path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure uv is installed and in PATH
if ! command -v uv &> /dev/null; then
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "uv not found. Installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        if [ -f "$HOME/.cargo/bin/uv" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        elif [ -f "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
fi

# Detect Apple Silicon and install MLX version automatically
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo "Apple Silicon detected. Ensuring MLX backend is installed..."
    uv pip install "corridorkey-mlx@git+https://github.com/nikopueringer/corridorkey-mlx.git"
    export CORRIDORKEY_BACKEND=mlx
    export CORRIDORKEY_DEVICE=mps
fi

# Enable OpenEXR Support
export OPENCV_IO_ENABLE_OPENEXR=1

echo "Starting CorridorKey Inference..."
echo "Scanning ClipsForInference for Ready Clips (Input + Alpha)..."

# Run Manager (uv handles the virtual environment automatically)
uv run python "${SCRIPT_DIR}/corridorkey_cli.py" --action run_inference

echo "Inference Complete."
