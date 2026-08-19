# brainapi

Interactive installer and runtime CLI for [BrainAPI](https://github.com/Lumen-Labs/brainapi2).

## Install

```bash
npm install -g brainapi-tui
```

This puts a `brainapi` binary on your `$PATH`.

## Quick start

```bash
brainapi init    # clone the project, set up a Python venv, configure services
brainapi start   # start docker compose services + run the API
brainapi doctor  # check Python, Docker, Ollama, and configured services
```

### Non-interactive / hybrid init

Any value you pass as a flag skips that wizard prompt; everything else is still asked.

```bash
brainapi init \
  --vector-db=postgresql \
  --data-db=postgresql \
  --graph-db=networkx \
  --models-mode=remote \
  --llm-small-provider=openai \
  --llm-large-provider=openai \
  --embeddings-provider=openai \
  --openai-api-key=sk-... \
  --llm-small=gpt-4o-mini \
  --llm-large=gpt-4o \
  --brainpat-token=brainpat_... \
  --no-plugins \
  --no-start-services
```

`brainapi --init ...` is an alias for `brainapi init ...`.

| Flag group | Examples |
| --- | --- |
| Defaults / DBs | `--defaults`, `--vector-db`, `--data-db`, `--graph-db` |
| Pipeline / runtime | `--ocr-mode`, `--services-runtime`, `--start-services`, `--no-start-services` |
| Models | `--models-mode`, `--llm-small-provider`, `--llm-large-provider`, `--embeddings-provider`, `--llm-small`, `--llm-large`, `--embedding-model`, `--embedding-dimensions` |
| Ollama | `--ollama-host`, `--ollama-port`, `--ollama-small`, `--ollama-large`, `--ollama-embeddings` |
| GCP | `--gcp-credentials`, `--gcp-project`, `--gcp-small`, `--gcp-large`, `--gcp-embedding` |
| Azure | `--azure-endpoint`, `--azure-api-version`, `--azure-key`, `--azure-small`, `--azure-large`, `--azure-embedding-endpoint`, `--azure-embedding-key`, `--azure-embedding` |
| OpenAI / Anthropic / DeepSeek | `--openai-api-key`, `--openai-base-url`, `--openai-small`, `--openai-large`, `--openai-embedding`, `--anthropic-api-key`, `--anthropic-small`, `--anthropic-large`, `--deepseek-api-key`, `--deepseek-small`, `--deepseek-large` |
| Bedrock | `--aws-region`, `--aws-access-key-id`, `--aws-secret-access-key`, `--aws-session-token`, `--bedrock-small`, `--bedrock-large`, `--bedrock-embedding` |
| Connections | `--redis-host`, `--redis-port`, `--postgres-*`, `--neo4j-*`, `--milvus-*`, `--mongo-*` |
| Auth / plugins | `--brainpat-token`, `--plugin <name[@version]>` (repeatable), `--no-plugins` |

If any config flag is present (or `--defaults`), the “Use default settings?” question is skipped.

## Commands

| Command           | Description                                                                                        |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| `brainapi init`   | Full bootstrap: clone the repo, create a venv, install deps, and run setup (flags skip prompts).   |
| `brainapi start`  | Start docker compose containers for the chosen services and launch the API.                        |
| `brainapi config` | Re-run the interactive flow and rewrite `.env`.                                                    |
| `brainapi doctor` | Check that Python, Docker, Ollama, GCP credentials, and configured services are reachable.         |
| `brainapi update` | `git pull` the project and `pip install -e .` again.                                               |
| `brainapi plugins install <name>` | Install a plugin from the registry (e.g. `brainapi plugins install chatbot`).          |
| `brainapi plugins list`           | List installed plugins (`--remote` to browse the registry).                            |
| `brainapi plugins uninstall <name>` | Remove an installed plugin.                                                          |
| `brainapi reset` | Purge Celery queues and/or wipe Redis + Postgres brain state (interactive). |
| `brainapi reset --brain <id> -y` | Wipe one brain's Redis keys and DROP its `brain_*` Postgres database. |
| `brainapi reset --all -y` | Redis `FLUSHDB` + DROP all `brain_*` databases + clear `data_brains`. |
| `brainapi reset --queues-only -y` | Purge Celery queue keys only. |

### `brainapi start` options

| Option                             | Description                                                                                                               |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--pipeline accurate\|lightweight` | Set `PIPELINE_MODE` in `~/.brainapi/source/.env` before start (accurate = full pipeline; lightweight = faster ingestion). |

## Where things live

- Source: `~/.brainapi/source/`
- Python venv: `~/.brainapi/source/.venv/`
- Env file: `~/.brainapi/source/.env`
- Install state: `~/.brainapi/state.json`

## Configuration

The interactive flow asks (any matching CLI flag skips that prompt):

1. Use default settings? (NetworkX + Postgres + pgvector + remote GCP Vertex)
2. Otherwise: Vector DB → Data DB → Graph DB → Models mode
3. For `remote` models mode: pick provider, then provide GCP credentials, project, and model names
4. For `local` models mode: probe Ollama, wait for it to start, then verify pulled models
5. Connection details for only the services you actually selected
6. `BRAINPAT_TOKEN` (generate or paste)
7. Optionally start the docker compose containers now

If Python (>=3.11) or Docker is missing, the TUI walks you through installing them with platform-aware commands and retries detection automatically.

## Development

```bash
cd tui
npm install
npm run dev    # builds in watch mode
node dist/index.js init   # try it
```

### Publishing

```bash
npm version <patch|minor|major>
npm publish
```

`prepublishOnly` runs the build automatically so the published tarball always contains `dist/`. Only the `dist/`, `README.md`, and `LICENSE` files are included (see the `files` field in `package.json`).

If someone runs any command before `brainapi init` (e.g. `brainapi doctor` or `brainapi start`), the CLI prints a notice and runs `init` first.

## Environment overrides

| Variable            | Default                                       | Description                       |
| ------------------- | --------------------------------------------- | --------------------------------- |
| `BRAINAPI_REPO_URL` | `https://github.com/Lumen-Labs/brainapi2.git` | The git repo cloned on `init`.    |
| `BRAINAPI_HOME`     | `$HOME/.brainapi`                             | Where the source and state live.  |
| `BRAINAPI_BRANCH`   | `main`                                        | Branch to checkout after cloning. |
