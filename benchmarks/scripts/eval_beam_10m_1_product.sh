#!/usr/bin/env bash
# Evaluate-only BEAM 10M/1 product on existing brain (no ingest / no wipe).
# Never touches beam1m1clean.
set +e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
unset BRAINPAT_TOKEN BRAINAPI_TOKEN
set -a; source ./.env; set +a
export TERM=dumb PYTHONUNBUFFERED=1

SRC="${BRAINAPI_SRC:-$HOME/.brainapi/source}"
UV_LOG=/tmp/uvicorn-beam10m.log
CEL_LOG=/tmp/brainapi-beam-worker.log
RUN_ID="${BEAM_10M_RUN:-beam-10m-1-clean}"
BRAIN_ID="${BEAM_10M_BRAIN:-beam10m1clean}"
EVAL_RUN_ID="${BEAM_10M_EVAL_RUN:-beam-10m-1-clean-product}"
SAMPLE_ID="${BEAM_10M_SAMPLE:-1}"

ensure_api() {
  for _ in 1 2 3 4 5 6 7 8; do
    if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
      code=$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' http://127.0.0.1:8000/docs || echo 000)
      if [[ "$code" == "200" || "$code" == "401" ]]; then
        if ! pgrep -f '/.venv/bin/celery -A src.workers.app worker' >/dev/null 2>&1; then
          nohup bash -c "cd '$SRC' && exec ./.venv/bin/celery -A src.workers.app worker --loglevel=info --pool=threads -Q ingest_data,finalize_ingestion,process_architect_relationships,consolidate_graph,chatbot_memory,ingest_file,ingest_structured_data" \
            >> "$CEL_LOG" 2>&1 &
          sleep 4
        fi
        return 0
      fi
    fi
    if ! lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
      nohup bash -c "cd '$SRC' && exec ./.venv/bin/uvicorn src.services.api.app:app --host 127.0.0.1 --port 8000" \
        >> "$UV_LOG" 2>&1 &
      sleep 5
    fi
  done
  return 1
}

echo "START_10M_EVAL $(date -u +%Y-%m-%dT%H:%M:%SZ) brain=$BRAIN_ID eval_run=$EVAL_RUN_ID"
ensure_api || { echo FATAL_API_DOWN; exit 1; }

for ab in \
  abstention \
  contradiction_resolution \
  event_ordering \
  information_extraction \
  instruction_following \
  knowledge_update \
  multi_session_reasoning \
  preference_following \
  summarization \
  temporal_reasoning
do
  echo "==== EVAL $ab $(date -u +%H:%M:%S) ===="
  ensure_api
  ./beam.sh evaluate \
    --size 10M --sample "$SAMPLE_ID" \
    --run "$EVAL_RUN_ID" \
    --brain "$BRAIN_ID" \
    --profile product \
    --abilities "$ab" \
    --concurrency 1 \
    --historical-limit 12 --max-passages 12 --max-facts 40 \
    --no-ppr
done

./beam.sh report --run "$EVAL_RUN_ID"
.venv/bin/python - <<PY
import json, statistics
from pathlib import Path
ingest = Path('runs/${RUN_ID}/ingest.jsonl')
rows=[json.loads(l) for l in ingest.read_text().splitlines() if l.strip()] if ingest.exists() else []
last={}
for r in rows: last[r.get('unit_id')]=r
comp=[r for r in last.values() if r.get('status')=='completed' and (r.get('cost') or {}).get('llm_source_multiplier') is not None]
mult=[r['cost']['llm_source_multiplier'] for r in comp]
print('INGEST completed', sum(1 for r in last.values() if r.get('status')=='completed'), '/', len(last))
print('PERMANENT_FAILS', sorted(
  uid for uid,r in last.items()
  if r.get('status')=='permanent_failed'
  or (r.get('status')=='failed' and '8192' in (r.get('error') or ''))
))
print('MULTIPLIER n', len(mult),
      'median', round(statistics.median(mult),1) if mult else None,
      'mean', round(statistics.mean(mult),1) if mult else None)
rep_path=Path('runs/${EVAL_RUN_ID}/report.json')
if rep_path.exists():
  rep=json.loads(rep_path.read_text())
  m=rep.get('metrics') or {}
  hs=m.get('headline_score')
  print('HEADLINE', hs, f"({(hs or 0)*100:.1f}%)" if isinstance(hs,(int,float)) else '')
  print('PER', m.get('per_ability'))
PY
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/beam-10m-1.DONE
echo EVAL_DONE
