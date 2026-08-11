import * as p from "@clack/prompts";
import { askUseDefaults, DEFAULT_DBS } from "./defaults.js";
import { askDatabases } from "./services.js";
import { askModels } from "./models.js";
import { askPipeline } from "./pipeline.js";
import { askConnections } from "./connections.js";
import { askAuth } from "./auth.js";
import { askServicesRuntime } from "./services-runtime.js";
import { askPlugins } from "./plugins.js";
import { isPromptBack } from "../lib/prompts.js";
import {
  hasConfigOverrides,
  type InitFlagOverrides,
} from "../lib/init-flags.js";
import type {
  AuthChoices,
  Connections,
  DbChoices,
  InitChoices,
  ModelsChoices,
  PluginChoice,
  PipelineChoices,
  ServicesRuntime,
} from "../types.js";

export interface SetupDraft {
  dbs?: DbChoices;
  servicesRuntime?: ServicesRuntime;
  models?: ModelsChoices;
  pipeline?: PipelineChoices;
  connections?: Connections;
  auth?: AuthChoices;
  plugins?: PluginChoice[];
  usedDefaults?: boolean;
}

export interface RunSetupWizardOptions {
  firstStepBack?: "return" | "cancel";
  stepBackHint?: string;
  firstStepBackHint?: string;
  flags?: InitFlagOverrides;
}

export function toInitChoices(draft: SetupDraft): InitChoices {
  return {
    dbs: draft.dbs!,
    servicesRuntime: draft.servicesRuntime!,
    models: draft.models!,
    pipeline: draft.pipeline!,
    connections: draft.connections!,
    auth: draft.auth!,
    plugins: draft.plugins ?? [],
    usedDefaults: draft.usedDefaults ?? false,
  };
}

function seedDraftFromFlags(
  draft: SetupDraft,
  flags: InitFlagOverrides | undefined,
): void {
  if (!flags) return;

  if (flags.defaults) {
    draft.usedDefaults = true;
    draft.dbs = { ...DEFAULT_DBS };
    draft.pipeline = { ocrMode: flags.ocrMode ?? "docparser" };
  }

  if (
    flags.vectorDb !== undefined ||
    flags.dataDb !== undefined ||
    flags.graphDb !== undefined
  ) {
    draft.dbs = {
      vectorDb: flags.vectorDb ?? draft.dbs?.vectorDb ?? DEFAULT_DBS.vectorDb,
      dataDb: flags.dataDb ?? draft.dbs?.dataDb ?? DEFAULT_DBS.dataDb,
      graphDb: flags.graphDb ?? draft.dbs?.graphDb ?? DEFAULT_DBS.graphDb,
    };
  }

  if (flags.ocrMode) {
    draft.pipeline = { ocrMode: flags.ocrMode };
  }

  if (flags.servicesRuntime) {
    draft.servicesRuntime = flags.servicesRuntime;
  }

  if (flags.brainpatToken) {
    draft.auth = { brainpatToken: flags.brainpatToken };
  }

  if (flags.noPlugins) {
    draft.plugins = [];
  } else if (flags.plugins) {
    draft.plugins = flags.plugins;
  }
}

export async function runSetupWizard(
  draft: SetupDraft,
  opts: RunSetupWizardOptions = {},
): Promise<boolean> {
  const flags = opts.flags;
  seedDraftFromFlags(draft, flags);

  const stepBackHint = opts.stepBackHint ?? "Previous step";
  const firstStepBackHint = opts.firstStepBackHint ?? stepBackHint;
  let step = 0;

  const skipUseDefaults =
    Boolean(flags?.defaults) || (flags !== undefined && hasConfigOverrides(flags));

  while (step >= 0) {
    if (step === 0) {
      if (skipUseDefaults) {
        if (flags?.defaults) {
          draft.usedDefaults = true;
          draft.dbs = draft.dbs ?? { ...DEFAULT_DBS };
          draft.pipeline = draft.pipeline ?? { ocrMode: "docparser" };
          step = 2;
          continue;
        }
        draft.usedDefaults = false;
        step = 1;
        continue;
      }

      const useDefaults = await askUseDefaults({
        allowBack: true,
        backHint: firstStepBackHint,
      });
      if (isPromptBack(useDefaults)) {
        if (opts.firstStepBack === "cancel") {
          p.cancel("Setup cancelled.");
          process.exit(0);
        }
        return false;
      }
      draft.usedDefaults = useDefaults;
      if (useDefaults) {
        draft.dbs = DEFAULT_DBS;
        draft.pipeline = { ocrMode: "docparser" };
        step = 2;
        continue;
      }
      step = 1;
      continue;
    }

    if (step === 1) {
      const allDbsPrechosen =
        flags?.vectorDb !== undefined &&
        flags?.dataDb !== undefined &&
        flags?.graphDb !== undefined;
      if (allDbsPrechosen && draft.dbs) {
        step = 2;
        continue;
      }
      const dbs = await askDatabases({
        allowBack: true,
        backHint: stepBackHint,
        initial: draft.dbs,
        prechosen: flags
          ? {
              vectorDb: flags.vectorDb,
              dataDb: flags.dataDb,
              graphDb: flags.graphDb,
            }
          : undefined,
      });
      if (isPromptBack(dbs)) {
        step = 0;
        continue;
      }
      draft.dbs = dbs;
      step = 2;
      continue;
    }

    if (step === 2) {
      if (!draft.dbs) {
        step = 1;
        continue;
      }
      if (flags?.servicesRuntime) {
        draft.servicesRuntime = flags.servicesRuntime;
        step = 3;
        continue;
      }
      const runtime = await askServicesRuntime(draft.dbs, {
        allowBack: true,
        backHint: stepBackHint,
        initialValue: draft.servicesRuntime,
        prechosen: flags?.servicesRuntime,
      });
      if (isPromptBack(runtime)) {
        step = draft.usedDefaults ? 0 : 1;
        continue;
      }
      draft.servicesRuntime = runtime;
      step = 3;
      continue;
    }

    if (step === 3) {
      const models = draft.usedDefaults
        ? await askModels({
            prechosenMode: flags?.modelsMode ?? "remote",
            initialMode: draft.models?.mode,
            flags,
          })
        : await askModels({
            allowBack: true,
            backHint: stepBackHint,
            initialMode: draft.models?.mode,
            prechosenMode: flags?.modelsMode,
            flags,
          });
      if (isPromptBack(models)) {
        step = 2;
        continue;
      }
      draft.models = models;
      step = draft.usedDefaults ? 5 : 4;
      continue;
    }

    if (step === 4) {
      const pipeline = await askPipeline({
        allowBack: true,
        backHint: stepBackHint,
        initialOcrMode: draft.pipeline?.ocrMode,
        prechosenOcrMode: flags?.ocrMode,
      });
      if (isPromptBack(pipeline)) {
        step = 3;
        continue;
      }
      draft.pipeline = pipeline;
      step = 5;
      continue;
    }

    if (step === 5) {
      if (!draft.dbs) {
        step = draft.usedDefaults ? 0 : 1;
        continue;
      }
      draft.connections = await askConnections(draft.dbs, flags);
      step = 6;
      continue;
    }

    if (step === 6) {
      if (flags?.brainpatToken) {
        draft.auth = { brainpatToken: flags.brainpatToken };
        step = 7;
        continue;
      }
      const auth = await askAuth({
        allowBack: true,
        backHint: stepBackHint,
        prechosenToken: flags?.brainpatToken,
      });
      if (isPromptBack(auth)) {
        step = 5;
        continue;
      }
      draft.auth = auth;
      step = 7;
      continue;
    }

    if (step === 7) {
      if (flags?.noPlugins || flags?.plugins !== undefined) {
        draft.plugins = flags.noPlugins ? [] : (flags.plugins ?? []);
        return true;
      }
      const plugins = await askPlugins({
        allowBack: true,
        backHint: stepBackHint,
        initial: draft.plugins,
        prechosen: flags?.plugins,
        skip: flags?.noPlugins,
      });
      if (isPromptBack(plugins)) {
        step = 6;
        continue;
      }
      draft.plugins = plugins;
      return true;
    }
  }

  return false;
}
