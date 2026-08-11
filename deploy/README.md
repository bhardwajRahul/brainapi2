# BrainAPI deploy profiles

Choose **heavy** or **light** when bringing up a new deployment.

| | Heavy | Light |
| --- | --- | --- |
| Compose | [`docker-compose.heavy.yaml`](docker-compose.heavy.yaml) | [`docker-compose.light.yaml`](docker-compose.light.yaml) |
| Env template | [`env.heavy.example`](env.heavy.example) | [`env.light.example`](env.light.example) |
| Graph | Neo4j | NetworkX on Postgres |
| Data | MongoDB | PostgreSQL |
| Vectors | Milvus | PostgreSQL + pgvector |
| LLMs | GCP Vertex (small) + Azure (large) | DeepSeek (small + large) |
| Embeddings | Azure | OpenAI (`text-embedding-3-large`) |
| Container limits | Full (current production) | Smaller footprint (`mem_limit` / `cpus`) |
| Suggested VM | ~8–16 GB RAM / 4+ vCPU | ~2 GB RAM / 1 vCPU |

## Quick start

```bash
# 1. Pick a profile and install env
cp deploy/env.light.example /root/.env   # or env.heavy.example
# edit secrets: BRAINPAT_TOKEN, DEEPSEEK_API_KEY, OPENAI_API_KEY, …

# 2. Bring up the matching compose
docker compose -f deploy/docker-compose.light.yaml up -d
# or
docker compose -f deploy/docker-compose.heavy.yaml up -d
```

## Light sizing

| Service | Heavy | Light |
| --- | --- | --- |
| nginx | 128m / 0.2 CPU | 16m / 0.025 CPU |
| redis | 2048m / 0.5 CPU (`maxmemory 1500mb`) | 256m / 0.0625 CPU (`maxmemory 192mb`) |
| postgres | — (not in heavy compose) | 512m / 0.25 CPU |
| neo4j / milvus / mongo / etcd / minio | present | omitted |
| api / worker / mcp | 4g / 2 CPU each | 512m / 0.25 CPU each |
| celery `--concurrency` | 2 | 1 |
| `CELERY_WORKER_CONCURRENCY` | 4 | 1 |

## Notes

- Both profiles expect `/root/.env` and nginx TLS paths under `/srv/nginx` (same as the legacy root `example-docker-compose.yaml`).
- Heavy still mounts `/root/gcp_credentials.json` for Vertex; light does not need GCP credentials.
- Do not mix profiles on the same host without wiping volumes — DB backends differ.
