#!/usr/bin/env bash
# Resilient BEAM 1M/1 ingest + product evaluate.
# Resume-safe. LISTEN-based ensure_api (never pkill healthy uvicorn).
# Uses flock so watchdog/ensure never double-bind :8000.
#
# Parallel ingest (next re-ingest — do NOT wipe live eval brains casually):
#   BEAM_INGEST_CONCURRENCY=2|3  (default 2; ≤4; need CELERY_WORKER_CONCURRENCY ≥ N)
# Observed wall ~40h @ concurrency 1 for ~625 turns → ~20–25h @2, ~14–18h @3.
set +e
cd /Users/christiannonis/Documents/Projects/brainapi2/benchmarks
unset BRAINPAT_TOKEN BRAINAPI_TOKEN
set -a; source ./.env; set +a
export TERM=dumb PYTHONUNBUFFERED=1

SRC=/Users/christiannonis/.brainapi/source
UV_LOG=/tmp/uvicorn-beam1m.log
CEL_LOG=/tmp/brainapi-beam-worker.log
LOCK_DIR=/tmp/beam1m-locks
mkdir -p "$LOCK_DIR"

# Harness parallel units. Keep ≤ CELERY_WORKER_CONCURRENCY (product .env default 4).
BEAM_INGEST_CONCURRENCY="${BEAM_INGEST_CONCURRENCY:-2}"
BEAM_INGEST_TIMEOUT="${BEAM_INGEST_TIMEOUT:-2400}"
CELERY_WORKER_CONCURRENCY="${CELERY_WORKER_CONCURRENCY:-4}"

api_ok() {
  lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 || return 1
  # Longer timeout under concurrent ingest poll load.
  code=$(curl -s -o /dev/null --max-time 5 -w '%{http_code}' http://127.0.0.1:8000/docs || echo 000)
  [[ "$code" == "200" || "$code" == "401" ]]
}

start_uvicorn() {
  (
    flock -n 9 || return 0
    if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    nohup bash -c "cd '$SRC' && exec ./.venv/bin/uvicorn src.services.api.app:app --host 127.0.0.1 --port 8000" \
      >> "$UV_LOG" 2>&1 &
    echo $! > /tmp/beam1m-uvicorn.pid
    sleep 5
  ) 9>"$LOCK_DIR/uvicorn.lock"
}

start_celery() {
  (
    flock -n 9 || return 0
    # Match the real worker binary, not bash wrappers that mention celery in -c strings.
    if pgrep -f '/.venv/bin/celery -A src.workers.app worker' >/dev/null 2>&1; then
      return 0
    fi
    nohup bash -c "cd '$SRC' && exec ./.venv/bin/celery -A src.workers.app worker --loglevel=info --pool=threads \
      --concurrency=${CELERY_WORKER_CONCURRENCY} \
      -Q ingest_data,finalize_ingestion,process_architect_relationships,consolidate_graph,chatbot_memory,ingest_file,ingest_structured_data" \
      >> "$CEL_LOG" 2>&1 &
    echo $! > /tmp/beam1m-celery.pid
    sleep 5
  ) 9>"$LOCK_DIR/celery.lock"
}

ensure_api() {
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if api_ok; then
      start_celery
      return 0
    fi
    start_uvicorn
    start_celery
    sleep 3
  done
  return 1
}

# Terminal units for stall/target: completed + permanent_failed (+ legacy embed-8192 failed).
terminal_count() {
  .venv/bin/python - <<'PY'
import json
import re
from pathlib import Path
from collections import Counter
p = Path("runs/beam-1m-1-clean/ingest.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
last = {}
for r in rows:
    last[r.get("unit_id")] = r
pat = re.compile(r"maximum context length is 8192", re.I)
n = 0
for r in last.values():
    st = r.get("status")
    err = r.get("error") or ""
    if st in {"completed", "partial_failed", "permanent_failed"}:
        n += 1
    elif st == "failed" and pat.search(err):
        n += 1
print(n)
PY
}

completed_count() {
  terminal_count
}

watchdog() {
  while [[ ! -f /tmp/beam-1m-1.STOP ]]; do
    if ! lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
      echo "WATCHDOG_API_RESTART $(date -u +%H:%M:%S)"
      start_uvicorn
    fi
    if ! pgrep -f '/.venv/bin/celery -A src.workers.app worker' >/dev/null 2>&1; then
      echo "WATCHDOG_CELERY_RESTART $(date -u +%H:%M:%S)"
      start_celery
    fi
    sleep 30
  done
}

# Accept near-complete ingest: a few units permanently fail (embed 8192 context).
TARGET=620
STALL_LIMIT=3
prev_n=-1
stalls=0
rm -f /tmp/beam-1m-1.STOP /tmp/beam-1m-1.DONE
echo "START_1M_INGEST $(date -u +%Y-%m-%dT%H:%M:%SZ) concurrency=$BEAM_INGEST_CONCURRENCY celery_c=$CELERY_WORKER_CONCURRENCY"
ensure_api || { echo FATAL_API_DOWN; exit 1; }
watchdog &
WD_PID=$!
echo "watchdog_pid=$WD_PID"

for try in $(seq 1 300); do
  ensure_api || { echo API_DOWN; sleep 20; continue; }
  done_n=$(completed_count)
  echo "try=$try terminal=$done_n $(date -u +%H:%M:%S)"
  if [[ "$done_n" -ge "$TARGET" ]]; then
    break
  fi
  if [[ "$done_n" -eq "$prev_n" ]]; then
    stalls=$((stalls + 1))
    if [[ "$stalls" -ge "$STALL_LIMIT" ]]; then
      echo "INGEST_STALLED terminal=$done_n (no progress ${stalls} tries) — proceed to eval"
      break
    fi
  else
    stalls=0
  fi
  prev_n=$done_n
  ./beam.sh ingest \
    --size 1M --sample 1 \
    --run beam-1m-1-clean \
    --brain beam1m1clean \
    --concurrency "$BEAM_INGEST_CONCURRENCY" \
    --timeout "$BEAM_INGEST_TIMEOUT"
  sleep 2
done

done_n=$(completed_count)
echo "INGEST_DONE=$done_n"
# Do not STOP keepalive globally; only stop this script's watchdog.
kill "$WD_PID" 2>/dev/null
wait "$WD_PID" 2>/dev/null

if [[ "$done_n" -lt 500 ]]; then
  echo "INGEST_INCOMPLETE abort eval ($done_n/625)"
  date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/beam-1m-1.DONE
  exit 2
fi

# Portable loop (bash array with multiline words crashed as commands in prior run).
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
import json, statistics, re
from pathlib import Path
rows=[json.loads(l) for l in Path('runs/beam-1m-1-clean/ingest.jsonl').read_text().splitlines() if l.strip()]
last={}
for r in rows: last[r.get('unit_id')]=r
pat=re.compile(r'maximum context length is 8192', re.I)
comp=[r for r in last.values() if r.get('status')=='completed' and (r.get('cost') or {}).get('llm_source_multiplier') is not None]
mult=[r['cost']['llm_source_multiplier'] for r in comp]
perm=sorted(
  uid for uid,r in last.items()
  if r.get('status')=='permanent_failed'
  or (r.get('status')=='failed' and pat.search(r.get('error') or ''))
)
print('MULTIPLIER n', len(mult),
      'median', round(statistics.median(mult),1) if mult else None,
      'mean', round(statistics.mean(mult),1) if mult else None)
print('PERMANENT_FAILS', perm)
rep_path=Path('runs/beam-1m-1-clean-product/report.json')
if rep_path.exists():
  rep=json.loads(rep_path.read_text())
  m=rep.get('metrics') or {}
  print('HEADLINE', m.get('headline_score'))
  print('PER', m.get('per_ability'))
PY
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/beam-1m-1.DONE
echo MULTI_DONE
