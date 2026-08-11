import type {
  DataDb,
  GraphDb,
  ModelProvider,
  ModelsMode,
  OcrMode,
  PluginChoice,
  ServicesRuntime,
  VectorDb,
} from "../types.js";

export interface InitFlagOverrides {
  defaults?: boolean;
  vectorDb?: VectorDb;
  dataDb?: DataDb;
  graphDb?: GraphDb;
  ocrMode?: OcrMode;
  servicesRuntime?: ServicesRuntime;
  startServices?: boolean;

  modelsMode?: ModelsMode;
  llmSmallProvider?: ModelProvider;
  llmLargeProvider?: ModelProvider;
  embeddingsProvider?: Exclude<ModelProvider, "anthropic" | "deepseek">;
  llmSmall?: string;
  llmLarge?: string;
  embeddingModel?: string;
  embeddingDimensions?: number;

  ollamaHost?: string;
  ollamaPort?: number;
  ollamaSmall?: string;
  ollamaLarge?: string;
  ollamaEmbeddings?: string;

  gcpCredentials?: string;
  gcpProject?: string;
  gcpSmall?: string;
  gcpLarge?: string;
  gcpEmbedding?: string;

  azureEndpoint?: string;
  azureApiVersion?: string;
  azureKey?: string;
  azureSmall?: string;
  azureLarge?: string;
  azureEmbeddingEndpoint?: string;
  azureEmbeddingKey?: string;
  azureEmbedding?: string;

  openaiApiKey?: string;
  openaiBaseUrl?: string;
  openaiSmall?: string;
  openaiLarge?: string;
  openaiEmbedding?: string;

  anthropicApiKey?: string;
  anthropicSmall?: string;
  anthropicLarge?: string;

  deepseekApiKey?: string;
  deepseekSmall?: string;
  deepseekLarge?: string;

  awsRegion?: string;
  awsAccessKeyId?: string;
  awsSecretAccessKey?: string;
  awsSessionToken?: string;
  bedrockSmall?: string;
  bedrockLarge?: string;
  bedrockEmbedding?: string;

  redisHost?: string;
  redisPort?: number;
  postgresHost?: string;
  postgresPort?: number;
  postgresUsername?: string;
  postgresPassword?: string;
  postgresSystemDb?: string;
  postgresMaintenanceDb?: string;
  neo4jHost?: string;
  neo4jPort?: number;
  neo4jUsername?: string;
  neo4jPassword?: string;
  milvusDeployment?: "local" | "managed";
  milvusHost?: string;
  milvusPort?: number;
  milvusUri?: string;
  milvusToken?: string;
  mongoHost?: string;
  mongoPort?: number;
  mongoUsername?: string;
  mongoPassword?: string;

  brainpatToken?: string;
  plugins?: PluginChoice[];
  noPlugins?: boolean;
}

const VECTOR_DBS = new Set<VectorDb>(["postgresql", "milvus"]);
const DATA_DBS = new Set<DataDb>(["postgresql", "mongo"]);
const GRAPH_DBS = new Set<GraphDb>(["networkx", "neo4j"]);
const OCR_MODES = new Set<OcrMode>(["docparser", "docling"]);
const RUNTIMES = new Set<ServicesRuntime>(["docker", "manual"]);
const MODELS_MODES = new Set<ModelsMode>(["local", "remote"]);
const LLM_PROVIDERS = new Set<ModelProvider>([
  "ollama",
  "openai",
  "anthropic",
  "deepseek",
  "azure",
  "gcp_vertex",
  "amazon_bedrock",
]);
const EMBEDDING_PROVIDERS = new Set<Exclude<ModelProvider, "anthropic" | "deepseek">>([
  "ollama",
  "openai",
  "azure",
  "gcp_vertex",
  "amazon_bedrock",
]);

function splitFlag(arg: string): { name: string; inline?: string } {
  const eq = arg.indexOf("=");
  if (eq === -1) return { name: arg };
  return { name: arg.slice(0, eq), inline: arg.slice(eq + 1) };
}

function takeValue(
  inline: string | undefined,
  args: string[],
  flag: string,
): { ok: true; value: string } | { ok: false; error: string } {
  if (inline !== undefined) {
    if (inline.length === 0) {
      return { ok: false, error: `${flag} requires a value` };
    }
    return { ok: true, value: inline };
  }
  const next = args.shift();
  if (next === undefined || next.startsWith("-")) {
    if (next !== undefined) args.unshift(next);
    return { ok: false, error: `${flag} requires a value` };
  }
  return { ok: true, value: next };
}

function parsePort(
  raw: string,
  flag: string,
): { ok: true; value: number } | { ok: false; error: string } {
  const n = Number(raw);
  if (!Number.isInteger(n) || n <= 0 || n > 65535) {
    return { ok: false, error: `${flag} must be an integer port 1-65535` };
  }
  return { ok: true, value: n };
}

function parsePositiveInt(
  raw: string,
  flag: string,
): { ok: true; value: number } | { ok: false; error: string } {
  const n = Number(raw);
  if (!Number.isInteger(n) || n <= 0) {
    return { ok: false, error: `${flag} must be a positive integer` };
  }
  return { ok: true, value: n };
}

function parsePluginSpec(raw: string): PluginChoice {
  const at = raw.lastIndexOf("@");
  if (at > 0) {
    return {
      name: raw.slice(0, at),
      source: "registry",
      version: raw.slice(at + 1),
    };
  }
  return { name: raw, source: "registry" };
}

/** True when any wizard-config flag is present (skips “use defaults?”). */
export function hasConfigOverrides(flags: InitFlagOverrides): boolean {
  const {
    defaults: _defaults,
    startServices: _start,
    ...rest
  } = flags;
  return Object.values(rest).some((v) => v !== undefined);
}

export type ApplyInitFlagResult =
  | { kind: "applied" }
  | { kind: "unknown" }
  | { kind: "error"; error: string };

/** Apply a single init flag. Mutates `flags` and may shift `args` for the value. */
export function tryApplyInitFlag(
  raw: string,
  args: string[],
  flags: InitFlagOverrides,
): ApplyInitFlagResult {
  if (!raw.startsWith("-")) return { kind: "unknown" };

  const { name, inline } = splitFlag(raw);

  const setString = (key: keyof InitFlagOverrides, flagName: string): ApplyInitFlagResult => {
    const taken = takeValue(inline, args, flagName);
    if (!taken.ok) return { kind: "error", error: taken.error };
    (flags as Record<string, unknown>)[key] = taken.value;
    return { kind: "applied" };
  };

  const setPort = (key: keyof InitFlagOverrides, flagName: string): ApplyInitFlagResult => {
    const taken = takeValue(inline, args, flagName);
    if (!taken.ok) return { kind: "error", error: taken.error };
    const port = parsePort(taken.value, flagName);
    if (!port.ok) return { kind: "error", error: port.error };
    (flags as Record<string, unknown>)[key] = port.value;
    return { kind: "applied" };
  };

  switch (name) {
    case "--defaults":
      flags.defaults = true;
      return { kind: "applied" };
    case "--vector-db": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as VectorDb;
      if (!VECTOR_DBS.has(v)) {
        return {
          kind: "error",
          error: `Invalid --vector-db "${taken.value}". Use postgresql or milvus.`,
        };
      }
      flags.vectorDb = v;
      return { kind: "applied" };
    }
    case "--data-db": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as DataDb;
      if (!DATA_DBS.has(v)) {
        return {
          kind: "error",
          error: `Invalid --data-db "${taken.value}". Use postgresql or mongo.`,
        };
      }
      flags.dataDb = v;
      return { kind: "applied" };
    }
    case "--graph-db": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as GraphDb;
      if (!GRAPH_DBS.has(v)) {
        return {
          kind: "error",
          error: `Invalid --graph-db "${taken.value}". Use networkx or neo4j.`,
        };
      }
      flags.graphDb = v;
      return { kind: "applied" };
    }
    case "--ocr-mode": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as OcrMode;
      if (!OCR_MODES.has(v)) {
        return {
          kind: "error",
          error: `Invalid --ocr-mode "${taken.value}". Use docparser or docling.`,
        };
      }
      flags.ocrMode = v;
      return { kind: "applied" };
    }
    case "--services-runtime": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as ServicesRuntime;
      if (!RUNTIMES.has(v)) {
        return {
          kind: "error",
          error: `Invalid --services-runtime "${taken.value}". Use docker or manual.`,
        };
      }
      flags.servicesRuntime = v;
      return { kind: "applied" };
    }
    case "--start-services":
      flags.startServices = true;
      return { kind: "applied" };
    case "--no-start-services":
      flags.startServices = false;
      return { kind: "applied" };
    case "--models-mode": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as ModelsMode;
      if (!MODELS_MODES.has(v)) {
        return {
          kind: "error",
          error: `Invalid --models-mode "${taken.value}". Use local or remote.`,
        };
      }
      flags.modelsMode = v;
      return { kind: "applied" };
    }
    case "--llm-small-provider": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as ModelProvider;
      if (!LLM_PROVIDERS.has(v)) {
        return {
          kind: "error",
          error: `Invalid --llm-small-provider "${taken.value}".`,
        };
      }
      flags.llmSmallProvider = v;
      return { kind: "applied" };
    }
    case "--llm-large-provider": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as ModelProvider;
      if (!LLM_PROVIDERS.has(v)) {
        return {
          kind: "error",
          error: `Invalid --llm-large-provider "${taken.value}".`,
        };
      }
      flags.llmLargeProvider = v;
      return { kind: "applied" };
    }
    case "--embeddings-provider": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase() as Exclude<
        ModelProvider,
        "anthropic" | "deepseek"
      >;
      if (!EMBEDDING_PROVIDERS.has(v)) {
        return {
          kind: "error",
          error: `Invalid --embeddings-provider "${taken.value}".`,
        };
      }
      flags.embeddingsProvider = v;
      return { kind: "applied" };
    }
    case "--llm-small":
      return setString("llmSmall", name);
    case "--llm-large":
      return setString("llmLarge", name);
    case "--embedding-model":
      return setString("embeddingModel", name);
    case "--embedding-dimensions": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const n = parsePositiveInt(taken.value, name);
      if (!n.ok) return { kind: "error", error: n.error };
      flags.embeddingDimensions = n.value;
      return { kind: "applied" };
    }
    case "--ollama-host":
      return setString("ollamaHost", name);
    case "--ollama-port":
      return setPort("ollamaPort", name);
    case "--ollama-small":
      return setString("ollamaSmall", name);
    case "--ollama-large":
      return setString("ollamaLarge", name);
    case "--ollama-embeddings":
      return setString("ollamaEmbeddings", name);
    case "--gcp-credentials":
      return setString("gcpCredentials", name);
    case "--gcp-project":
      return setString("gcpProject", name);
    case "--gcp-small":
      return setString("gcpSmall", name);
    case "--gcp-large":
      return setString("gcpLarge", name);
    case "--gcp-embedding":
      return setString("gcpEmbedding", name);
    case "--azure-endpoint":
      return setString("azureEndpoint", name);
    case "--azure-api-version":
      return setString("azureApiVersion", name);
    case "--azure-key":
      return setString("azureKey", name);
    case "--azure-small":
      return setString("azureSmall", name);
    case "--azure-large":
      return setString("azureLarge", name);
    case "--azure-embedding-endpoint":
      return setString("azureEmbeddingEndpoint", name);
    case "--azure-embedding-key":
      return setString("azureEmbeddingKey", name);
    case "--azure-embedding":
      return setString("azureEmbedding", name);
    case "--openai-api-key":
      return setString("openaiApiKey", name);
    case "--openai-base-url":
      return setString("openaiBaseUrl", name);
    case "--openai-small":
      return setString("openaiSmall", name);
    case "--openai-large":
      return setString("openaiLarge", name);
    case "--openai-embedding":
      return setString("openaiEmbedding", name);
    case "--anthropic-api-key":
      return setString("anthropicApiKey", name);
    case "--anthropic-small":
      return setString("anthropicSmall", name);
    case "--anthropic-large":
      return setString("anthropicLarge", name);
    case "--deepseek-api-key":
      return setString("deepseekApiKey", name);
    case "--deepseek-small":
      return setString("deepseekSmall", name);
    case "--deepseek-large":
      return setString("deepseekLarge", name);
    case "--aws-region":
      return setString("awsRegion", name);
    case "--aws-access-key-id":
      return setString("awsAccessKeyId", name);
    case "--aws-secret-access-key":
      return setString("awsSecretAccessKey", name);
    case "--aws-session-token":
      return setString("awsSessionToken", name);
    case "--bedrock-small":
      return setString("bedrockSmall", name);
    case "--bedrock-large":
      return setString("bedrockLarge", name);
    case "--bedrock-embedding":
      return setString("bedrockEmbedding", name);
    case "--redis-host":
      return setString("redisHost", name);
    case "--redis-port":
      return setPort("redisPort", name);
    case "--postgres-host":
      return setString("postgresHost", name);
    case "--postgres-port":
      return setPort("postgresPort", name);
    case "--postgres-username":
      return setString("postgresUsername", name);
    case "--postgres-password":
      return setString("postgresPassword", name);
    case "--postgres-system-db":
      return setString("postgresSystemDb", name);
    case "--postgres-maintenance-db":
      return setString("postgresMaintenanceDb", name);
    case "--neo4j-host":
      return setString("neo4jHost", name);
    case "--neo4j-port":
      return setPort("neo4jPort", name);
    case "--neo4j-username":
      return setString("neo4jUsername", name);
    case "--neo4j-password":
      return setString("neo4jPassword", name);
    case "--milvus-deployment": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      const v = taken.value.toLowerCase();
      if (v !== "local" && v !== "managed") {
        return {
          kind: "error",
          error: `Invalid --milvus-deployment "${taken.value}". Use local or managed.`,
        };
      }
      flags.milvusDeployment = v;
      return { kind: "applied" };
    }
    case "--milvus-host":
      return setString("milvusHost", name);
    case "--milvus-port":
      return setPort("milvusPort", name);
    case "--milvus-uri":
      return setString("milvusUri", name);
    case "--milvus-token":
      return setString("milvusToken", name);
    case "--mongo-host":
      return setString("mongoHost", name);
    case "--mongo-port":
      return setPort("mongoPort", name);
    case "--mongo-username":
      return setString("mongoUsername", name);
    case "--mongo-password":
      return setString("mongoPassword", name);
    case "--brainpat-token":
      return setString("brainpatToken", name);
    case "--plugin": {
      const taken = takeValue(inline, args, name);
      if (!taken.ok) return { kind: "error", error: taken.error };
      if (!flags.plugins) flags.plugins = [];
      flags.plugins.push(parsePluginSpec(taken.value));
      return { kind: "applied" };
    }
    case "--no-plugins":
      flags.noPlugins = true;
      return { kind: "applied" };
    default:
      return { kind: "unknown" };
  }
}
