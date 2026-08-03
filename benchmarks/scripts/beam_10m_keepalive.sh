#!/usr/bin/env bash
# Tiny keep-alive for BEAM 10M: only ensures API LISTEN + celery worker. Never pkill.
# screen -dmS beam10m-keep bash benchmarks/scripts/beam_10m_keepalive.sh
#
# Safe under concurrent ingest: flock around starts; longer /docs probe.
# Does not restart a healthy LISTEN uvicorn (avoids killing live eval).
set +e
SRC="${BRAINAPI_SRC:-$HOME/.brainapi/source}"
UV_LOG=/tmp/uvicorn-beam10m.log
CEL_LOG=/tmp/brainapi-beam-worker.log
LOCK_DIR=/tmp/beam10m-locks
mkdir -p "$LOCK_DIR"
CELERY_WORKER_CONCURRENCY="${CELERY_WORKER_CONCURRENCY:-4}"

echo "KEEPALIVE_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/beam-10m-keepalive.log
while [[ ! -f /tmp/beam-10m-1.STOP ]]; do
  if ! lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "API_RESTART $(date -u +%H:%M:%S)" >> /tmp/beam-10m-keepalive.log
    (
      flock -n 9 || exit 0
      if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
        exit 0
      fi
      nohup bash -c "cd '$SRC' && exec ./.venv/bin/uvicorn src.services.api.app:app --host 127.0.0.1 --port 8000" \
        >> "$UV_LOG" 2>&1 &
      echo $! > /tmp/beam10m-uvicorn.pid
      sleep 5
    ) 9>"$LOCK_DIR/uvicorn.lock"
  else
    code=$(curl -s -o /dev/null --max-time 5 -w '%{http_code}' http://127.0.0.1:8000/docs || echo 000)
    if [[ "$code" != "200" && "$code" != "401" ]]; then
      echo "API_BUSY_OR_SLOW code=$code $(date -u +%H:%M:%S)" >> /tmp/beam-10m-keepalive.log
    fi
  fi
  if ! pgrep -f '/.venv/bin/celery -A src.workers.app worker' >/dev/null 2>&1; then
    echo "CELERY_RESTART $(date -u +%H:%M:%S)" >> /tmp/beam-10m-keepalive.log
    (
      flock -n 9 || exit 0
      if pgrep -f '/.venv/bin/celery -A src.workers.app worker' >/dev/null 2>&1; then
        exit 0
      fi
      nohup bash -c "cd '$SRC' && exec ./.venv/bin/celery -A src.workers.app worker --loglevel=info --pool=threads \
        --concurrency=${CELERY_WORKER_CONCURRENCY} \
        -Q ingest_data,finalize_ingestion,process_architect_relationships,consolidate_graph,chatbot_memory,ingest_file,ingest_structured_data" \
        >> "$CEL_LOG" 2>&1 &
      echo $! > /tmp/beam10m-celery.pid
      sleep 5
    ) 9>"$LOCK_DIR/celery.lock"
  fi
  sleep 20
done
echo "KEEPALIVE_STOP $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> /tmp/beam-10m-keepalive.log
