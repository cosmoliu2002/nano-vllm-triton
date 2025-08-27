#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config_bench.yaml"
PYTHON_SCRIPT="${SCRIPT_DIR}/bench.py"
PARSE_SCRIPT="${SCRIPT_DIR}/parse_config.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script $PYTHON_SCRIPT not found!"
    exit 1
fi

if [ ! -f "$PARSE_SCRIPT" ]; then
    echo "Error: Parse script $PARSE_SCRIPT not found!"
    exit 1
fi

MODEL_PATH=$(python "$PARSE_SCRIPT" "$CONFIG_FILE" "model_path")
ENGINE=$(python "$PARSE_SCRIPT" "$CONFIG_FILE" "engine")


PYTHON_ARGS=(
    "--model-path" "$MODEL_PATH"
    "--engine" "$ENGINE"
)

python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"