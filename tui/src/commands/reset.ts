import * as p from "@clack/prompts";
import pc from "picocolors";
import { readEnvFile, getEnvValue } from "../lib/env.js";
import { envFilePath } from "../lib/paths.js";
import { readState } from "../lib/state.js";
import { ENV_KEYS } from "../constants.js";
import {
  buildResetPlan,
  brainDbName,
  executeReset,
  listBrainDatabases,
  type ResetScope,
} from "../lib/reset-state.js";

export interface ResetCommandOptions {
  brain?: string;
  all?: boolean;
  queuesOnly?: boolean;
  redisOnly?: boolean;
  dbOnly?: boolean;
  yes?: boolean;
}

async function ensureInstalled(): Promise<void> {
  const state = await readState();
  if (!state || !state.envWritten) {
    p.cancel(
      "No brainapi install detected. Run " + pc.cyan("brainapi init") + " first.",
    );
    process.exit(1);
  }
}

function summarizePlan(
  scope: ResetScope,
  redisOnly: boolean,
  dbOnly: boolean,
): string[] {
  const lines: string[] = [];
  if (scope.kind === "queues") {
    lines.push("Purge Celery queues in Redis (leave brain data intact)");
  } else if (scope.kind === "brain") {
    lines.push(`Reset brain ${pc.cyan(scope.brainId)}`);
    if (!dbOnly) {
      lines.push(`- Redis: delete ${scope.brainId}:* keys + purge Celery queues`);
    }
    if (!redisOnly) {
      lines.push(
        `- Postgres: DROP ${brainDbName(scope.brainId)} + registry row`,
      );
    }
  } else {
    lines.push("Reset ALL brains");
    if (!dbOnly) lines.push("- Redis: FLUSHDB");
    if (!redisOnly) {
      lines.push("- Postgres: DROP every brain_* database + clear data_brains");
    }
  }
  lines.push("");
  lines.push(
    pc.yellow("Stop `brainapi start` first so workers are not holding DB connections."),
  );
  return lines;
}

async function resolveScope(
  opts: ResetCommandOptions,
): Promise<ResetScope> {
  if (opts.queuesOnly) return { kind: "queues" };
  if (opts.all) return { kind: "all" };
  if (opts.brain) {
    const brainId = opts.brain.trim();
    if (!brainId) {
      throw new Error("Brain id cannot be empty.");
    }
    return { kind: "brain", brainId };
  }

  const mode = await p.select({
    message: "What should be cleared?",
    options: [
      {
        value: "brain",
        label: "One brain",
        hint: "Redis keys + DROP that brain's Postgres DB",
      },
      {
        value: "all",
        label: "All brains",
        hint: "Redis FLUSHDB + DROP all brain_* databases",
      },
      {
        value: "queues",
        label: "Celery queues only",
        hint: "Purge pending/running queue keys in Redis",
      },
    ],
  });
  if (p.isCancel(mode)) {
    p.cancel("Reset cancelled.");
    process.exit(0);
  }

  if (mode === "queues") return { kind: "queues" };
  if (mode === "all") return { kind: "all" };

  const env = await readEnvFile(envFilePath());
  let hint = "e.g. locomoconv26";
  try {
    const brains = await listBrainDatabases(env);
    if (brains.length > 0) {
      hint = brains.map((b) => b.brainId).slice(0, 8).join(", ");
      if (brains.length > 8) hint += ", …";
    }
  } catch {
    // listing is optional for the prompt
  }

  const brain = await p.text({
    message: "Brain id to reset",
    placeholder: hint,
    validate: (value) => {
      if (!value.trim()) return "Brain id is required";
      if (!/^[A-Za-z0-9_-]+$/.test(value.trim())) {
        return "Use alphanumeric brain ids (hyphen/underscore ok)";
      }
    },
  });
  if (p.isCancel(brain)) {
    p.cancel("Reset cancelled.");
    process.exit(0);
  }
  return { kind: "brain", brainId: String(brain).trim() };
}

export async function runReset(opts: ResetCommandOptions = {}): Promise<void> {
  await ensureInstalled();
  p.intro(pc.bgCyan(pc.black(" brainapi ")) + " " + pc.dim("reset"));

  if (opts.redisOnly && opts.dbOnly) {
    p.log.error("Use only one of --redis-only / --db-only.");
    process.exit(1);
  }
  const exclusive = [opts.brain, opts.all, opts.queuesOnly].filter(Boolean).length;
  if (exclusive > 1) {
    p.log.error("Use only one of --brain / --all / --queues-only.");
    process.exit(1);
  }

  const env = await readEnvFile(envFilePath());
  const scope = await resolveScope(opts);
  const plan = buildResetPlan(env, scope);

  p.note(summarizePlan(scope, Boolean(opts.redisOnly), Boolean(opts.dbOnly)).join("\n"), "Plan");

  if (plan.postgres) {
    p.log.info(
      `Postgres ${plan.postgres.host}:${plan.postgres.port} (system db ${plan.postgres.systemDatabase})`,
    );
  }
  p.log.info(`Redis ${plan.redisHost}:${plan.redisPort}`);

  const celeryBackend = getEnvValue(env, ENV_KEYS.celeryBackend) ?? "redis";
  if (celeryBackend !== "redis" && !opts.dbOnly) {
    p.log.error(
      `reset supports CELERY_BACKEND=redis only (current: ${celeryBackend}). Use --db-only to wipe Postgres.`,
    );
    process.exit(1);
  }

  if (!opts.yes) {
    const ok = await p.confirm({
      message: "This is destructive. Continue?",
      initialValue: false,
    });
    if (p.isCancel(ok) || !ok) {
      p.cancel("Reset cancelled.");
      process.exit(0);
    }
  }

  const spinner = p.spinner();
  spinner.start("Clearing state…");
  try {
    const result = await executeReset(env, scope, {
      yes: opts.yes,
      redisOnly: opts.redisOnly,
      dbOnly: opts.dbOnly,
    });
    spinner.stop("Done.");

    const lines: string[] = [];
    if (result.redisDeletedKeys === -1) {
      lines.push("Redis: FLUSHDB completed");
    } else if (!opts.dbOnly) {
      lines.push(`Redis: deleted ~${result.redisDeletedKeys} key(s)`);
    }
    if (result.databasesDropped.length > 0) {
      lines.push(`Postgres: dropped ${result.databasesDropped.join(", ")}`);
    }
    if (result.registryRowsDeleted > 0) {
      lines.push(`Registry: removed ${result.registryRowsDeleted} data_brains row(s)`);
    }
    for (const warning of result.warnings) {
      lines.push(pc.yellow(warning));
    }
    if (lines.length > 0) {
      p.note(lines.join("\n"), "Result");
    }
    p.outro(
      pc.green("State cleared.") +
        " Restart with " +
        pc.cyan("brainapi start") +
        " when ready.",
    );
  } catch (err) {
    spinner.stop("Failed.");
    throw err;
  }
}
