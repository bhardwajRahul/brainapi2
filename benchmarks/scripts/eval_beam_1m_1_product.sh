#!/usr/bin/env bash
# Evaluate-only BEAM 1M/1 product on existing brain (no ingest / no wipe).
set +e
cd /Users/christiannonis/Documents/Projects/brainapi2/benchmarks
unset BRAINPAT_TOKEN BRAINAPI_TOKEN
set -a; source ./.env; set +a
export TERM=dumb PYTHONUNBUFFERED=1

SRC=/Users/christiannonis/.brainapi/source
UV_LOG=/tmp/uvicorn-beam1m.log
CEL_LOG=/tmp/brainapi-beam-worker.log

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

echo "START_1M_EVAL $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
    --size 1M --sample 1 \
    --run beam-1m-1-clean-product \
    --brain beam1m1clean \
    --profile product \
    --abilities "$ab" \
    --concurrency 1 \
    --historical-limit 12 --max-passages 12 --max-facts 40 \
    --no-ppr
done

./beam.sh report --run beam-1m-1-clean-product
.venv/bin/python - <<'PY'
import json, statistics
from pathlib import Path
rows=[json.loads(l) for l in Path('runs/beam-1m-1-clean/ingest.jsonl').read_text().splitlines() if l.strip()]
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
rep_path=Path('runs/beam-1m-1-clean-product/report.json')
if rep_path.exists():
  rep=json.loads(rep_path.read_text())
  m=rep.get('metrics') or {}
  hs=m.get('headline_score')
  print('HEADLINE', hs, f"({(hs or 0)*100:.1f}%)" if isinstance(hs,(int,float)) else '')
  print('PER', m.get('per_ability'))
  print('GOAL_B_65', 'MET' if (hs or 0) >= 0.65 else 'MISS')
PY
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/beam-1m-1.DONE
echo EVAL_DONE
