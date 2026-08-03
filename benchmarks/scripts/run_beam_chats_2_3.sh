#!/usr/bin/env bash
set +e
cd /Users/christiannonis/Documents/Projects/brainapi2/benchmarks
unset BRAINPAT_TOKEN BRAINAPI_TOKEN
set -a; source ./.env; set +a
export TERM=dumb PYTHONUNBUFFERED=1

ensure_api() {
  for _ in 1 2 3 4 5 6; do
    if curl -s -o /dev/null --max-time 2 http://127.0.0.1:8000/docs >/dev/null; then
      return 0
    fi
    if ! lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
      (
        cd /Users/christiannonis/.brainapi/source
        ./.venv/bin/uvicorn src.services.api.app:app --host 127.0.0.1 --port 8000 >> /tmp/uvicorn-stable.log 2>&1 &
      )
    fi
    if ! pgrep -f 'celery -A src.workers.app worker' >/dev/null; then
      (
        cd /Users/christiannonis/.brainapi/source
        ./.venv/bin/celery -A src.workers.app worker --loglevel=info --pool=threads \
          -Q ingest_data,finalize_ingestion,process_architect_relationships,consolidate_graph,chatbot_memory,ingest_file,ingest_structured_data \
          >> /tmp/brainapi-beam-worker.log 2>&1 &
      )
    fi
    sleep 5
  done
}

completed_count() {
  local sample=$1
  .venv/bin/python - "$sample" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
sample = sys.argv[1]
p = Path(f"runs/beam-100k-{sample}-clean/ingest.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
last = {}
for r in rows:
    last[r.get("unit_id")] = r.get("status")
print(Counter(last.values()).get("completed", 0))
PY
}

for sample in 2 3; do
  echo "==== CHAT $sample ===="
  for try in $(seq 1 30); do
    ensure_api
    done_n=$(completed_count "$sample")
    echo "try=$try completed=$done_n $(date -u +%H:%M:%S)"
    if [[ "$done_n" -ge 70 ]]; then
      break
    fi
    ./beam.sh ingest \
      --size 100K --sample "$sample" \
      --run "beam-100k-${sample}-clean" \
      --brain "beam100k${sample}clean" \
      --concurrency "${BEAM_INGEST_CONCURRENCY:-2}" \
      --timeout 1200
  done
  done_n=$(completed_count "$sample")
  echo "INGEST${sample}_DONE=$done_n"
  if [[ "$done_n" -lt 50 ]]; then
    echo "skip eval for $sample (ingest incomplete)"
    continue
  fi
  abilities=(
    abstention contradiction_resolution event_ordering information_extraction
    instruction_following knowledge_update multi_session_reasoning
    preference_following summarization temporal_reasoning
  )
  for ab in "${abilities[@]}"; do
    ensure_api
    ./beam.sh evaluate \
      --size 100K --sample "$sample" \
      --run "beam-100k-${sample}-clean-product" \
      --brain "beam100k${sample}clean" \
      --profile product \
      --abilities "$ab" \
      --concurrency 1 \
      --historical-limit 12 --max-passages 12 --max-facts 40 \
      --no-ppr
    ensure_api
    ./beam.sh evaluate \
      --size 100K --sample "$sample" \
      --run "beam-100k-${sample}-clean-product" \
      --brain "beam100k${sample}clean" \
      --profile product \
      --abilities "$ab" \
      --concurrency 1 \
      --historical-limit 12 --max-passages 12 --max-facts 40 \
      --no-ppr
  done
  .venv/bin/python - <<PY
from pathlib import Path
from beam.report import write_report
r = write_report(Path("runs/beam-100k-${sample}-clean-product"))
print(
    "CHAT${sample}_HEADLINE",
    (r.get("metrics") or {}).get("headline_score"),
    "n",
    (r.get("metrics") or {}).get("n_questions"),
)
PY
done
echo MULTI_DONE
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/beam-100k-23-done.DONE
