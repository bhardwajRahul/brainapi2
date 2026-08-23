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
# 1. Copy a profile template outside the repository and replace every secret.
install -m 600 deploy/env.light.example /etc/brainapi/light.env
# Set BRAINAPI_ENV_FILE to the absolute file path in /etc/brainapi/light.env.

# 2. Validate first, then start and wait for health.
docker compose --env-file /etc/brainapi/light.env \
  -f deploy/docker-compose.light.yaml config -q
docker compose --env-file /etc/brainapi/light.env \
  -f deploy/docker-compose.light.yaml up -d --wait
# or
docker compose --env-file /etc/brainapi/heavy.env \
  -f deploy/docker-compose.heavy.yaml up -d --wait
```

Only nginx publishes host ports (`80` and `443`). The API, embedded Console,
and MCP are reached through `/`, `/console/`, and `/mcp`. The checked-in TLS
server rejects handshakes until an operator mounts a certificate-specific
nginx server file; do not expose port 80 beyond a trusted proxy without HTTPS.

## Light sizing

| Service | Heavy | Light |
| --- | --- | --- |
| nginx | 128m / 0.2 CPU | 64m / 0.1 CPU |
| redis | 2048m / 0.5 CPU (`maxmemory 1500mb`) | 256m / 0.1 CPU (`maxmemory 192mb`) |
| postgres | — (not in heavy compose) | 768m / 0.35 CPU |
| neo4j / milvus / mongo / etcd / minio | present | omitted |
| api / worker / mcp | 2–4g / 1–2 CPU | 768m / 0.5 CPU each |
| celery `--concurrency` | 2 | 1 |
| `CELERY_WORKER_CONCURRENCY` | 4 | 1 |

## Notes

- `BRAINAPI_IMAGE`, `BRAINAPI_ENV_FILE`, Redis credentials, and profile backend
  credentials are required and fail during Compose interpolation when absent.
- Mount cloud credentials with a private operator-owned Compose override; no
  host-specific credential path is present in the production profiles.
- Do not mix profiles under the same Compose project name. Their volume layouts
  and restore order differ.

## Backup and restore

Backups default to `/srv/brainapi/backups` and are created with mode `0700`.
They contain checksummed dumps and stopped-volume archives, but never secrets.

```bash
deploy/brainapi-backup backup --profile light \
  --env-file /etc/brainapi/light.env --project-name brainapi
deploy/brainapi-backup verify --profile light \
  --env-file /etc/brainapi/light.env --archive /srv/brainapi/backups/light-UTCSTAMP
deploy/brainapi-backup restore --profile light \
  --env-file /etc/brainapi/light.env --archive /srv/brainapi/backups/light-UTCSTAMP
```

Backup stops nginx, API, MCP, and workers before capture. Restore refuses a
running stack, a profile/image mismatch, a checksum mismatch, or any non-empty
target volume. Use a documented maintenance window and retain the generated
manifest with the release artifacts.

## Public documentation sandbox

The anonymous sandbox is an opt-in extension of the light profile. Merge
[`env.public-demo.example`](env.public-demo.example) into the deployment's
private environment, deploy the current API and worker image, and seed only the
dedicated `agentdemo` brain:

```bash
docker compose --env-file /etc/brainapi/light.env -f deploy/docker-compose.light.yaml up -d
docker compose --env-file /etc/brainapi/light.env -f deploy/docker-compose.light.yaml exec brainapi \
  python scripts/seed_public_demo.py --api-url http://localhost:8000
```

The seed command downloads the published V2 documentation, records its SHA-256
marker, skips an already-current corpus, and polls the asynchronous ingestion
task. It requires the private system `BRAINPAT_TOKEN` inside the deployment; do
not expose that token through the gateway. After the first successful seed,
`BRAIN_CREATION_ALLOWED` may be returned to the operator's normal setting.

Expose the API origin as `api.brain-api.dev`, with TLS, and put the Cloudflare
gateway in front of root-domain `/api/*`. The application itself deliberately
does not implement client rate limiting; the checked-in gateway owns the
30-request/60-second demo limit.
