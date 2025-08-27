#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config_example.yaml"
PYTHON_SCRIPT="${SCRIPT_DIR}/example.py"
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
TENSOR_PARALLEL_SIZE=$(python "$PARSE_SCRIPT" "$CONFIG_FILE" "tensor_parallel_size")
ENFORCE_EAGER=$(python "$PARSE_SCRIPT" "$CONFIG_FILE" "enforce_eager")
TEMPERATURE=$(python "$PARSE_SCRIPT" "$CONFIG_FILE" "temperature")
MAX_TOKENS=$(python "$PARSE_SCRIPT" "$CONFIG_FILE" "max_tokens")


PYTHON_ARGS=(
    "--model-path" "$MODEL_PATH"
    "--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE"
    "--enforce-eager" "$ENFORCE_EAGER"
    "--temperature" "$TEMPERATURE"
    "--max-tokens" "$MAX_TOKENS"
)

python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"