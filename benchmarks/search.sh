#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
PARENT="${ROOT}/../.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing ${PYTHON}. Create it with:"
  echo "  cd benchmarks && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
for arg in "$@"; do
  if [[ "$arg" == "rank-pool-ce" || "$arg" == "finetune-ce" || "$arg" == "pool-first-stage" || "$arg" == "rerank-retrieved" || "$arg" == "finetune-dense" || "$arg" == "local-dense" || "$arg" == "mine-retrieved-lists" || "$arg" == "finetune-4class" || "$arg" == "colbert-local" || "$arg" == "rank-corpus" || "$arg" == "ltr-head" || "$arg" == "backfill-entity-text" ]]; then
    if [[ -x "$PARENT" ]]; then
      PYTHON="$PARENT"
    fi
    if [[ "$arg" == "backfill-entity-text" && -f "${HOME}/.brainapi/source/.env" ]]; then
      set -a
      # shellcheck disable=SC1091
      source "${HOME}/.brainapi/source/.env"
      set +a
      export SEARCH_ENABLED="${SEARCH_ENABLED:-true}"
      export DATA_DB="${DATA_DB:-postgresql}"
      export VECTOR_DB="${VECTOR_DB:-postgresql}"
      export GRAPH_DB="${GRAPH_DB:-networkx}"
      export ENV="${ENV:-development}"
    fi
    break
  fi
done
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m search "$@"
