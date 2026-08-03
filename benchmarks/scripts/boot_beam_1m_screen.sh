#!/usr/bin/env bash
# Boot BEAM 1M ingest inside screen. Keep this file free of long uvicorn/celery
# command strings in the screen argv — use this script path only:
#   screen -dmS beam1m /path/to/boot_beam_1m_screen.sh
set +e
SRC=/Users/christiannonis/.brainapi/source
UV_LOG=/tmp/uvicorn-beam1m.log
CEL_LOG=/tmp/brainapi-beam-worker.log
UV_PID=/tmp/beam1m-uvicorn.pid
CEL_PID=/tmp/beam1m-celery.pid

start_api() {
  if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    return 0
  fi
  nohup bash -c "cd '$SRC' && exec ./.venv/bin/uvicorn src.services.api.app:app --host 127.0.0.1 --port 8000" \
    >> "$UV_LOG" 2>&1 &
  echo $! > "$UV_PID"
  sleep 5
}

start_worker() {
  if pgrep -f 'celery -A src.workers.app worker' >/dev/null 2>&1; then
    return 0
  fi
  CELERY_C="${CELERY_WORKER_CONCURRENCY:-4}"
  nohup bash -c "cd '$SRC' && exec ./.venv/bin/celery -A src.workers.app worker --loglevel=info --pool=threads --concurrency=${CELERY_C} -Q ingest_data,finalize_ingestion,process_architect_relationships,consolidate_graph,chatbot_memory,ingest_file,ingest_structured_data" \
    >> "$CEL_LOG" 2>&1 &
  echo $! > "$CEL_PID"
  sleep 5
}

start_api
start_worker
exec /tmp/run_beam_1m_1_ingest_eval.sh >> /tmp/beam-1m-1-run.log 2>&1
