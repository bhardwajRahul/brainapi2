import { access, cp } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as p from "@clack/prompts";
import pc from "picocolors";
import { sourcePath } from "./paths.js";

const SYNC_PATHS = ["src", "console"] as const;

const SKIP_DIR_NAMES = new Set([
  "node_modules",
  ".venv",
  "__pycache__",
  ".git",
  ".turbo",
  ".next",
]);

function shouldSkipSyncEntry(srcPath: string): boolean {
  return SKIP_DIR_NAMES.has(path.basename(srcPath));
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await access(target);
    return true;
  } catch {
    return false;
  }
}

export async function devWorkspaceRoot(): Promise<string | null> {
  const fromEnv = process.env.BRAINAPI_DEV_SOURCE?.trim();
  if (fromEnv && (await pathExists(path.join(fromEnv, "src", "config.py")))) {
    return fromEnv;
  }

  let dir = path.dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 6; i++) {
    if (await pathExists(path.join(dir, "src", "config.py"))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

export async function syncDevSourceIfAvailable(): Promise<boolean> {
  const root = await devWorkspaceRoot();
  if (!root) return false;

  const destRoot = sourcePath();
  let synced = false;
  for (const rel of SYNC_PATHS) {
    const src = path.join(root, rel);
    const dest = path.join(destRoot, rel);
    if (!(await pathExists(src))) continue;
    await cp(src, dest, {
      recursive: true,
      force: true,
      filter: (source) => !shouldSkipSyncEntry(source),
    });
    synced = true;
  }
  if (synced) {
    p.log.info(
      `Synced Python sources from ${pc.cyan(root)} → ${pc.cyan(destRoot)}`,
    );
  }
  return synced;
}
