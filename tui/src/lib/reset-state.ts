import { createHash } from "node:crypto";
import { ENV_KEYS, SERVICE_DEFAULTS } from "../constants.js";
import type { EnvFile } from "./env.js";
import { getEnvValue } from "./env.js";
import { runQuiet } from "./exec.js";
import { which } from "./exec.js";

const CELERY_QUEUE_KEYS = [
  "ingest_data",
  "process_architect_relationships",
  "ingest_structured_data",
  "consolidate_graph",
  "finalize_ingestion",
  "ingest_file",
  "unacked",
  "unacked_index",
  "unacked_mutex",
] as const;

const BRAIN_DB_PREFIX = "brain_";
const MAX_DBNAME_LEN = 63;
const HASH_SUFFIX_LEN = 12;

export type ResetScope =
  | { kind: "queues" }
  | { kind: "brain"; brainId: string }
  | { kind: "all" };

export interface ResetOptions {
  yes?: boolean;
  redisOnly?: boolean;
  dbOnly?: boolean;
}

export interface ResetPlan {
  scope: ResetScope;
  redisHost: string;
  redisPort: number;
  postgres?: {
    host: string;
    port: number;
    username: string;
    password: string;
    systemDatabase: string;
    maintenanceDatabase: string;
  };
  brainDatabases: string[];
  redisActions: string[];
  dbActions: string[];
}

export interface ResetResult {
  redisDeletedKeys: number;
  databasesDropped: string[];
  registryRowsDeleted: number;
  warnings: string[];
}

function requireEnv(env: EnvFile, key: string, fallback?: string): string {
  return getEnvValue(env, key) ?? fallback ?? "";
}

export function brainDbName(brainId: string): string {
  if (!brainId.trim()) {
    throw new Error("brain_id must be a non-empty string");
  }
  const sanitized =
    brainId
      .toLowerCase()
      .replace(/[^a-z0-9_]+/g, "_")
      .replace(/^_+|_+$/g, "") || "x";
  const candidate = `${BRAIN_DB_PREFIX}${sanitized}`;
  if (candidate.length <= MAX_DBNAME_LEN) return candidate;

  const digest = createHash("sha256")
    .update(brainId)
    .digest("hex")
    .slice(0, HASH_SUFFIX_LEN);
  const headBudget = MAX_DBNAME_LEN - BRAIN_DB_PREFIX.length - digest.length - 1;
  const head = sanitized.slice(0, headBudget).replace(/_+$/g, "");
  return `${BRAIN_DB_PREFIX}${head}_${digest}`;
}

export function brainIdFromDbName(dbName: string): string | null {
  if (!dbName.startsWith(BRAIN_DB_PREFIX)) return null;
  const suffix = dbName.slice(BRAIN_DB_PREFIX.length);
  return suffix || null;
}

export function usesPostgres(env: EnvFile): boolean {
  const graph = getEnvValue(env, ENV_KEYS.graphDb);
  const data = getEnvValue(env, ENV_KEYS.dataDb);
  const vector = getEnvValue(env, ENV_KEYS.vectorDb);
  return (
    graph === "networkx" || data === "postgresql" || vector === "postgresql"
  );
}

export function buildResetPlan(env: EnvFile, scope: ResetScope): ResetPlan {
  const redisHost =
    requireEnv(env, ENV_KEYS.redisHost, SERVICE_DEFAULTS.redis.host) ||
    SERVICE_DEFAULTS.redis.host;
  const redisPort = Number(
    requireEnv(env, ENV_KEYS.redisPort, String(SERVICE_DEFAULTS.redis.port)) ||
      SERVICE_DEFAULTS.redis.port,
  );

  const plan: ResetPlan = {
    scope,
    redisHost,
    redisPort,
    brainDatabases: [],
    redisActions: [],
    dbActions: [],
  };

  if (scope.kind === "queues") {
    plan.redisActions.push("Purge Celery queue keys and unacked buffers");
  } else if (scope.kind === "brain") {
    plan.redisActions.push(`Delete Redis keys for brain "${scope.brainId}"`);
    plan.redisActions.push("Purge Celery queue keys and unacked buffers");
    plan.brainDatabases = [brainDbName(scope.brainId)];
    plan.dbActions.push(`DROP DATABASE ${plan.brainDatabases[0]}`);
    plan.dbActions.push(
      `DELETE FROM data_brains WHERE name_key/id = "${scope.brainId}"`,
    );
  } else {
    plan.redisActions.push("FLUSHDB (clear entire Redis database)");
    plan.dbActions.push("DROP all brain_* databases");
    plan.dbActions.push("DELETE all rows from data_brains");
  }

  if (usesPostgres(env)) {
    plan.postgres = {
      host:
        requireEnv(env, ENV_KEYS.postgresHost, SERVICE_DEFAULTS.postgresql.host) ||
        SERVICE_DEFAULTS.postgresql.host,
      port: Number(
        requireEnv(
          env,
          ENV_KEYS.postgresPort,
          String(SERVICE_DEFAULTS.postgresql.port),
        ) || SERVICE_DEFAULTS.postgresql.port,
      ),
      username:
        requireEnv(
          env,
          ENV_KEYS.postgresUsername,
          SERVICE_DEFAULTS.postgresql.username,
        ) || SERVICE_DEFAULTS.postgresql.username,
      password:
        requireEnv(
          env,
          ENV_KEYS.postgresPassword,
          SERVICE_DEFAULTS.postgresql.password,
        ) || SERVICE_DEFAULTS.postgresql.password,
      systemDatabase:
        requireEnv(env, ENV_KEYS.postgresSystemDatabase, "brainapi") ||
        "brainapi",
      maintenanceDatabase:
        requireEnv(env, ENV_KEYS.postgresMaintenanceDatabase, "postgres") ||
        "postgres",
    };
  } else {
    plan.dbActions = [];
    plan.brainDatabases = [];
  }

  return plan;
}

async function ensureBin(bin: string): Promise<void> {
  if (!(await which(bin))) {
    throw new Error(
      `Required binary "${bin}" was not found on PATH. Install it or add it to PATH.`,
    );
  }
}

async function redisCli(
  host: string,
  port: number,
  args: string[],
): Promise<{ ok: boolean; stdout: string; stderr: string }> {
  return runQuiet("redis-cli", ["-h", host, "-p", String(port), ...args]);
}

async function purgeCeleryQueues(
  host: string,
  port: number,
): Promise<number> {
  const result = await redisCli(host, port, ["DEL", ...CELERY_QUEUE_KEYS]);
  if (!result.ok) {
    throw new Error(
      `Failed to purge Celery queues: ${result.stderr || result.stdout}`,
    );
  }
  const n = Number(result.stdout.trim() || "0");
  return Number.isFinite(n) ? n : 0;
}

async function deleteKeysByPattern(
  host: string,
  port: number,
  pattern: string,
): Promise<number> {
  const scanned = await redisCli(host, port, ["--scan", "--pattern", pattern]);
  if (!scanned.ok) {
    throw new Error(
      `Failed to scan Redis keys (${pattern}): ${scanned.stderr || scanned.stdout}`,
    );
  }
  const keys = scanned.stdout
    .split(/\r?\n/)
    .map((k) => k.trim())
    .filter(Boolean);
  if (keys.length === 0) return 0;

  let deleted = 0;
  const chunkSize = 100;
  for (let i = 0; i < keys.length; i += chunkSize) {
    const chunk = keys.slice(i, i + chunkSize);
    const result = await redisCli(host, port, ["DEL", ...chunk]);
    if (!result.ok) {
      throw new Error(
        `Failed to delete Redis keys: ${result.stderr || result.stdout}`,
      );
    }
    deleted += Number(result.stdout.trim() || "0") || chunk.length;
  }
  return deleted;
}

async function psql(
  pg: NonNullable<ResetPlan["postgres"]>,
  database: string,
  sql: string,
): Promise<{ ok: boolean; stdout: string; stderr: string }> {
  return runQuiet(
    "psql",
    [
      "-h",
      pg.host,
      "-p",
      String(pg.port),
      "-U",
      pg.username,
      "-d",
      database,
      "-v",
      "ON_ERROR_STOP=1",
      "-t",
      "-A",
      "-c",
      sql,
    ],
    {
      env: {
        ...process.env,
        PGPASSWORD: pg.password,
      },
    },
  );
}

export async function listBrainDatabases(
  env: EnvFile,
): Promise<Array<{ brainId: string; database: string }>> {
  if (!usesPostgres(env)) return [];
  await ensureBin("psql");
  const plan = buildResetPlan(env, { kind: "all" });
  if (!plan.postgres) return [];

  const result = await psql(
    plan.postgres,
    plan.postgres.maintenanceDatabase,
    `SELECT datname FROM pg_database WHERE datname LIKE 'brain\\_%' ESCAPE '\\' AND datname <> '${plan.postgres.systemDatabase.replace(/'/g, "''")}' ORDER BY 1;`,
  );
  if (!result.ok) {
    throw new Error(
      `Failed to list brain databases: ${result.stderr || result.stdout}`,
    );
  }
  return result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((database) => ({
      database,
      brainId: brainIdFromDbName(database) ?? database,
    }));
}

async function dropBrainDatabase(
  pg: NonNullable<ResetPlan["postgres"]>,
  database: string,
): Promise<void> {
  if (!/^[a-z0-9_]+$/.test(database)) {
    throw new Error(`Refusing to drop unsafe database name: ${database}`);
  }
  if (database === pg.systemDatabase || database === pg.maintenanceDatabase) {
    throw new Error(
      `Refusing to drop protected database "${database}" (system/maintenance).`,
    );
  }
  if (!database.startsWith(BRAIN_DB_PREFIX) || database === "brainapi") {
    throw new Error(
      `Refusing to drop non-brain database "${database}". Expected prefix ${BRAIN_DB_PREFIX}.`,
    );
  }
  const terminate = await psql(
    pg,
    pg.maintenanceDatabase,
    `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${database}' AND pid <> pg_backend_pid();`,
  );
  if (!terminate.ok) {
    throw new Error(
      `Failed to terminate connections to ${database}: ${terminate.stderr || terminate.stdout}`,
    );
  }
  const drop = await psql(
    pg,
    pg.maintenanceDatabase,
    `DROP DATABASE IF EXISTS ${database};`,
  );
  if (!drop.ok) {
    throw new Error(
      `Failed to drop ${database}: ${drop.stderr || drop.stdout}`,
    );
  }
}

async function ensureSystemDatabase(
  pg: NonNullable<ResetPlan["postgres"]>,
): Promise<void> {
  if (!/^[a-z0-9_]+$/.test(pg.systemDatabase)) {
    throw new Error(`Unsafe system database name: ${pg.systemDatabase}`);
  }
  const result = await psql(
    pg,
    pg.maintenanceDatabase,
    `SELECT 1 FROM pg_database WHERE datname = '${pg.systemDatabase}';`,
  );
  if (!result.ok) {
    throw new Error(
      `Failed to check system database: ${result.stderr || result.stdout}`,
    );
  }
  if (result.stdout.trim()) return;
  const created = await psql(
    pg,
    pg.maintenanceDatabase,
    `CREATE DATABASE ${pg.systemDatabase};`,
  );
  if (!created.ok) {
    throw new Error(
      `Failed to recreate system database ${pg.systemDatabase}: ${created.stderr || created.stdout}`,
    );
  }
}

async function deleteBrainRegistryRows(
  pg: NonNullable<ResetPlan["postgres"]>,
  brainId: string | null,
): Promise<number> {
  const sql =
    brainId === null
      ? "DELETE FROM data_brains RETURNING id;"
      : `DELETE FROM data_brains WHERE name_key = '${brainId.replace(/'/g, "''")}' OR id = '${brainId.replace(/'/g, "''")}' RETURNING id;`;
  const result = await psql(pg, pg.systemDatabase, sql);
  if (!result.ok) {
    // Table may not exist yet on fresh installs.
    if (/data_brains/i.test(result.stderr) && /does not exist/i.test(result.stderr)) {
      return 0;
    }
    throw new Error(
      `Failed to clean data_brains registry: ${result.stderr || result.stdout}`,
    );
  }
  return result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean).length;
}

export async function executeReset(
  env: EnvFile,
  scope: ResetScope,
  options: ResetOptions = {},
): Promise<ResetResult> {
  const celeryBackend = getEnvValue(env, ENV_KEYS.celeryBackend) ?? "redis";
  if (celeryBackend !== "redis" && !options.dbOnly) {
    throw new Error(
      `reset currently supports CELERY_BACKEND=redis only (got "${celeryBackend}").`,
    );
  }

  const plan = buildResetPlan(env, scope);
  const warnings: string[] = [];
  let redisDeletedKeys = 0;
  const databasesDropped: string[] = [];
  let registryRowsDeleted = 0;

  const doRedis = !options.dbOnly;
  const doDb = !options.redisOnly && Boolean(plan.postgres) && scope.kind !== "queues";

  if (doRedis) {
    await ensureBin("redis-cli");
    if (scope.kind === "all") {
      const flush = await redisCli(plan.redisHost, plan.redisPort, ["FLUSHDB"]);
      if (!flush.ok) {
        throw new Error(`Redis FLUSHDB failed: ${flush.stderr || flush.stdout}`);
      }
      redisDeletedKeys = -1; // unknown count
    } else {
      if (scope.kind === "brain") {
        redisDeletedKeys += await deleteKeysByPattern(
          plan.redisHost,
          plan.redisPort,
          `${scope.brainId}:*`,
        );
        const systemKey = await redisCli(plan.redisHost, plan.redisPort, [
          "DEL",
          `system:brain:${scope.brainId}`,
        ]);
        if (systemKey.ok) {
          redisDeletedKeys += Number(systemKey.stdout.trim() || "0") || 0;
        }
      }
      redisDeletedKeys += await purgeCeleryQueues(plan.redisHost, plan.redisPort);
    }
  }

  if (doDb && plan.postgres) {
    await ensureBin("psql");
    await ensureSystemDatabase(plan.postgres);
    let targets = plan.brainDatabases;
    if (scope.kind === "all") {
      const listed = await listBrainDatabases(env);
      targets = listed.map((row) => row.database);
    }
    for (const database of targets) {
      if (
        database === plan.postgres.systemDatabase ||
        database === plan.postgres.maintenanceDatabase ||
        database === "brainapi"
      ) {
        warnings.push(`Skipped protected database "${database}".`);
        continue;
      }
      await dropBrainDatabase(plan.postgres, database);
      databasesDropped.push(database);
    }
    if (scope.kind === "brain") {
      registryRowsDeleted = await deleteBrainRegistryRows(
        plan.postgres,
        scope.brainId,
      );
    } else if (scope.kind === "all") {
      registryRowsDeleted = await deleteBrainRegistryRows(plan.postgres, null);
    }
  } else if (!options.redisOnly && !plan.postgres && scope.kind !== "queues") {
    warnings.push(
      "Postgres is not configured for this install (GRAPH_DB/DATA_DB/VECTOR_DB); skipped database wipe.",
    );
  }

  if (options.redisOnly) {
    warnings.push("Skipped Postgres (--redis-only).");
  }
  if (options.dbOnly) {
    warnings.push("Skipped Redis (--db-only).");
  }

  return {
    redisDeletedKeys,
    databasesDropped,
    registryRowsDeleted,
    warnings,
  };
}
