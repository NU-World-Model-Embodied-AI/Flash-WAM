#!/usr/bin/env bash

# Assemble a complete serveable Flash-WAM root without copying model weights.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 ONLINE_TRANSFORMER_DIR EVAL_MODEL_DIR" >&2
  exit 2
fi

TRANSFORMER_DIR=$(cd -- "$1" && pwd)
EVAL_MODEL_DIR=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_MODEL_DIR=${BASE_MODEL_DIR:-$(cd -- "${SCRIPT_DIR}/../hf_assets/FlashWAM-RoboTwin" && pwd)}

[[ -f "${TRANSFORMER_DIR}/config.json" ]] || { echo "Missing transformer config: ${TRANSFORMER_DIR}" >&2; exit 1; }
for component in vae tokenizer text_encoder; do
  [[ -d "${BASE_MODEL_DIR}/${component}" ]] || { echo "Missing base component: ${BASE_MODEL_DIR}/${component}" >&2; exit 1; }
done

mkdir -p -- "${EVAL_MODEL_DIR}"
link_component() {
  local name=$1
  local source=$2
  local target="${EVAL_MODEL_DIR}/${name}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    [[ "$(readlink -f -- "${target}")" == "$(readlink -f -- "${source}")" ]] || {
      echo "Refusing to replace existing ${target}; it points to a different source." >&2
      exit 1
    }
    return
  fi
  ln -s -- "${source}" "${target}"
}

link_component transformer "${TRANSFORMER_DIR}"
for component in vae tokenizer text_encoder; do
  link_component "${component}" "${BASE_MODEL_DIR}/${component}"
done
printf 'Prepared Flash-WAM evaluation model: %s\n' "$(cd -- "${EVAL_MODEL_DIR}" && pwd)"
