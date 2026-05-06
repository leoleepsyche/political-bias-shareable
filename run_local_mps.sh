#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_python_with_torch() {
  local candidate
  local -a candidates=()

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=(
    "${REPO_ROOT}/.venv/bin/python"
    "${HOME}/anaconda3/envs/torch_m1/bin/python3"
    "python3"
  )

  for candidate in "${candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  ./run_local_mps.sh <script.py> [args...]

Examples:
  ./run_local_mps.sh run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --input-csv data/ideoinst_clean/ideology_840_opinion.csv --device mps --output-dir outputs/test
  ./run_local_mps.sh run_experiment.py --model mistralai/Mistral-7B-Instruct-v0.3 --input-csv data/ideoinst_clean/ideology_840_opinion.csv --device mps --languages en it --output-dir outputs/mistral
EOF
  exit 1
fi

PYTHON_PATH="$(find_python_with_torch || true)"
if [[ -z "${PYTHON_PATH}" ]]; then
  echo "Could not find a Python interpreter with torch installed." >&2
  echo "Set PYTHON_BIN=/path/to/python or create .venv with torch." >&2
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "${PYTHON_PATH}" "$@"
