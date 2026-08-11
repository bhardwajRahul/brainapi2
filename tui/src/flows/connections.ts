import * as p from "@clack/prompts";
import { SERVICE_DEFAULTS } from "../constants.js";
import type { InitFlagOverrides } from "../lib/init-flags.js";
import { askPassword, askText, pickOne } from "../lib/prompts.js";
import type {
  Connections,
  DbChoices,
  MilvusConnection,
  MongoConnection,
  Neo4jConnection,
  PostgresConnection,
  RedisConnection,
} from "../types.js";

function portValidator(value: string): string | undefined {
  const n = Number(value);
  if (!Number.isInteger(n) || n <= 0 || n > 65535)
    return "Port must be 1-65535";
  return undefined;
}

async function askRedis(flags?: InitFlagOverrides): Promise<RedisConnection> {
  p.log.step("Redis (required for cache + Celery)");
  let host = flags?.redisHost;
  if (host === undefined) {
    const raw = await askText({
      message: "Redis host",
      placeholder: SERVICE_DEFAULTS.redis.host,
      defaultValue: SERVICE_DEFAULTS.redis.host,
    });
    host = raw.trim() || SERVICE_DEFAULTS.redis.host;
  }
  let port = flags?.redisPort;
  if (port === undefined) {
    const raw = await askText({
      message: "Redis port",
      placeholder: String(SERVICE_DEFAULTS.redis.port),
      defaultValue: String(SERVICE_DEFAULTS.redis.port),
      validate: portValidator,
    });
    port = Number(raw) || SERVICE_DEFAULTS.redis.port;
  }
  return { host, port };
}

async function askPostgres(flags?: InitFlagOverrides): Promise<PostgresConnection> {
  p.log.step("PostgreSQL");
  p.note(
    [
      "BrainAPI uses a single Postgres database to host all brains.",
      "Each brain is isolated logically via a `brain_id` column, not via separate databases.",
      "Press Enter to accept the default name.",
    ].join("\n"),
    "Heads up"
  );
  let host = flags?.postgresHost;
  if (host === undefined) {
    const raw = await askText({
      message: "Postgres host",
      placeholder: SERVICE_DEFAULTS.postgresql.host,
      defaultValue: SERVICE_DEFAULTS.postgresql.host,
    });
    host = raw.trim() || SERVICE_DEFAULTS.postgresql.host;
  }
  let port = flags?.postgresPort;
  if (port === undefined) {
    const raw = await askText({
      message: "Postgres port",
      placeholder: String(SERVICE_DEFAULTS.postgresql.port),
      defaultValue: String(SERVICE_DEFAULTS.postgresql.port),
      validate: portValidator,
    });
    port = Number(raw) || SERVICE_DEFAULTS.postgresql.port;
  }
  let username = flags?.postgresUsername;
  if (username === undefined) {
    const raw = await askText({
      message: "Postgres username",
      placeholder: SERVICE_DEFAULTS.postgresql.username,
      defaultValue: SERVICE_DEFAULTS.postgresql.username,
    });
    username = raw.trim() || SERVICE_DEFAULTS.postgresql.username;
  }
  let password = flags?.postgresPassword;
  if (password === undefined) {
    password = await askPassword({
      message: "Postgres password (default is 'password')",
      validate: (value) =>
        value.length === 0 ? "Password is required" : undefined,
    });
  }
  let systemDatabase = flags?.postgresSystemDb;
  if (systemDatabase === undefined) {
    const raw = await askText({
      message: `System database (brains registry, default ${SERVICE_DEFAULTS.postgresql.systemDatabase})`,
      placeholder: SERVICE_DEFAULTS.postgresql.systemDatabase,
      defaultValue: SERVICE_DEFAULTS.postgresql.systemDatabase,
    });
    systemDatabase =
      raw.trim() || SERVICE_DEFAULTS.postgresql.systemDatabase;
  }
  let maintenanceDatabase = flags?.postgresMaintenanceDb;
  if (maintenanceDatabase === undefined) {
    const raw = await askText({
      message: `Maintenance database (for CREATE DATABASE, default ${SERVICE_DEFAULTS.postgresql.maintenanceDatabase})`,
      placeholder: SERVICE_DEFAULTS.postgresql.maintenanceDatabase,
      defaultValue: SERVICE_DEFAULTS.postgresql.maintenanceDatabase,
    });
    maintenanceDatabase =
      raw.trim() || SERVICE_DEFAULTS.postgresql.maintenanceDatabase;
  }
  return {
    host,
    port,
    username,
    password,
    systemDatabase,
    maintenanceDatabase,
  };
}

async function askNeo4j(flags?: InitFlagOverrides): Promise<Neo4jConnection> {
  p.log.step("Neo4j");
  let host = flags?.neo4jHost;
  if (host === undefined) {
    const raw = await askText({
      message: "Neo4j host",
      placeholder: SERVICE_DEFAULTS.neo4j.host,
      defaultValue: SERVICE_DEFAULTS.neo4j.host,
    });
    host = raw.trim() || SERVICE_DEFAULTS.neo4j.host;
  }
  let port = flags?.neo4jPort;
  if (port === undefined) {
    const raw = await askText({
      message: "Neo4j Bolt port",
      placeholder: String(SERVICE_DEFAULTS.neo4j.port),
      defaultValue: String(SERVICE_DEFAULTS.neo4j.port),
      validate: portValidator,
    });
    port = Number(raw) || SERVICE_DEFAULTS.neo4j.port;
  }
  let username = flags?.neo4jUsername;
  if (username === undefined) {
    const raw = await askText({
      message: "Neo4j username",
      placeholder: SERVICE_DEFAULTS.neo4j.username,
      defaultValue: SERVICE_DEFAULTS.neo4j.username,
    });
    username = raw.trim() || SERVICE_DEFAULTS.neo4j.username;
  }
  let password = flags?.neo4jPassword;
  if (password === undefined) {
    password = await askPassword({
      message: "Neo4j password (default is 'your_password')",
      validate: (value) =>
        value.length === 0 ? "Password is required" : undefined,
    });
  }
  return { host, port, username, password };
}

async function askMilvus(flags?: InitFlagOverrides): Promise<MilvusConnection> {
  p.log.step("Milvus");

  let target = flags?.milvusDeployment;
  if (flags?.milvusUri || flags?.milvusToken) {
    target = "managed";
  }
  if (target === undefined) {
    target = await pickOne<"local" | "managed">({
      message: "Milvus deployment",
      options: [
        { value: "local", label: "Local docker compose" },
        { value: "managed", label: "Managed (Zilliz Cloud) — uri + token" },
      ],
      initialValue: "local",
    });
  }

  if (target === "managed") {
    let uri = flags?.milvusUri;
    if (uri === undefined) {
      uri = (
        await askText({
          message: "Milvus URI",
          placeholder: "https://your-cluster.api.gcp-us-west1.zillizcloud.com",
          validate: (value) =>
            value.trim().length === 0 ? "URI is required" : undefined,
        })
      ).trim();
    }
    let token = flags?.milvusToken;
    if (token === undefined) {
      token = await askPassword({
        message: "Milvus token",
        validate: (value) =>
          value.length === 0 ? "Token is required" : undefined,
      });
    }
    return {
      host: flags?.milvusHost ?? SERVICE_DEFAULTS.milvus.host,
      port: flags?.milvusPort ?? SERVICE_DEFAULTS.milvus.port,
      uri,
      token,
    };
  }

  let host = flags?.milvusHost;
  if (host === undefined) {
    const raw = await askText({
      message: "Milvus host",
      placeholder: SERVICE_DEFAULTS.milvus.host,
      defaultValue: SERVICE_DEFAULTS.milvus.host,
    });
    host = raw.trim() || SERVICE_DEFAULTS.milvus.host;
  }
  let port = flags?.milvusPort;
  if (port === undefined) {
    const raw = await askText({
      message: "Milvus port",
      placeholder: String(SERVICE_DEFAULTS.milvus.port),
      defaultValue: String(SERVICE_DEFAULTS.milvus.port),
      validate: portValidator,
    });
    port = Number(raw) || SERVICE_DEFAULTS.milvus.port;
  }
  return { host, port };
}

async function askMongo(flags?: InitFlagOverrides): Promise<MongoConnection> {
  p.log.step("MongoDB");
  let host = flags?.mongoHost;
  if (host === undefined) {
    const raw = await askText({
      message: "Mongo host",
      placeholder: SERVICE_DEFAULTS.mongo.host,
      defaultValue: SERVICE_DEFAULTS.mongo.host,
    });
    host = raw.trim() || SERVICE_DEFAULTS.mongo.host;
  }
  let port = flags?.mongoPort;
  if (port === undefined) {
    const raw = await askText({
      message: "Mongo port",
      placeholder: String(SERVICE_DEFAULTS.mongo.port),
      defaultValue: String(SERVICE_DEFAULTS.mongo.port),
      validate: portValidator,
    });
    port = Number(raw) || SERVICE_DEFAULTS.mongo.port;
  }
  let username = flags?.mongoUsername;
  if (username === undefined) {
    const raw = await askText({
      message: "Mongo username",
      placeholder: SERVICE_DEFAULTS.mongo.username,
      defaultValue: SERVICE_DEFAULTS.mongo.username,
    });
    username = raw.trim() || SERVICE_DEFAULTS.mongo.username;
  }
  let password = flags?.mongoPassword;
  if (password === undefined) {
    password = await askPassword({
      message: "Mongo password (default is 'password')",
      validate: (value) =>
        value.length === 0 ? "Password is required" : undefined,
    });
  }
  return { host, port, username, password };
}

export async function askConnections(
  dbs: DbChoices,
  flags?: InitFlagOverrides,
): Promise<Connections> {
  const redis = await askRedis(flags);
  const result: Connections = { redis };

  const needsPostgres =
    dbs.vectorDb === "postgresql" ||
    dbs.dataDb === "postgresql" ||
    dbs.graphDb === "networkx";
  if (needsPostgres) {
    result.postgresql = await askPostgres(flags);
  }
  if (dbs.graphDb === "neo4j") {
    result.neo4j = await askNeo4j(flags);
  }
  if (dbs.vectorDb === "milvus") {
    result.milvus = await askMilvus(flags);
  }
  if (dbs.dataDb === "mongo") {
    result.mongo = await askMongo(flags);
  }

  return result;
}
