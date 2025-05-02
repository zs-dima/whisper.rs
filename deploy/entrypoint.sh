#!/bin/bash
set -e

MODEL_DIR="/app/models"
MODEL_NAME="${WHISPER_MODEL_NAME:-ggml-base.en-q5_1.bin}"
MODEL_URL="${WHISPER_MODEL_URL:-https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL_NAME?download=true}"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH. Downloading from $MODEL_URL..."
    wget -O "$MODEL_PATH" "$MODEL_URL"
else
    echo "Model found at $MODEL_PATH."
fi

# Pass model path as env var or argument if needed by your app
exec whisper-server