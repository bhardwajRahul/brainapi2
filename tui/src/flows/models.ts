import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { homedir } from "node:os";
import * as p from "@clack/prompts";
import pc from "picocolors";
import {
  ANTHROPIC_DEFAULT_LARGE_MODEL,
  ANTHROPIC_DEFAULT_SMALL_MODEL,
  AZURE_DEFAULT_EMBEDDING_MODEL,
  AZURE_DEFAULT_SMALL_MODEL,
  AZURE_DEFAULT_LARGE_API_VERSION,
  AZURE_DEFAULT_LARGE_MODEL,
  BEDROCK_DEFAULT_EMBEDDING_MODEL,
  BEDROCK_DEFAULT_LARGE_MODEL,
  BEDROCK_DEFAULT_REGION,
  BEDROCK_DEFAULT_SMALL_MODEL,
  DEEPSEEK_DEFAULT_LARGE_MODEL,
  DEEPSEEK_DEFAULT_SMALL_MODEL,
  GCP_DEFAULT_EMBEDDING_MODEL,
  GCP_DEFAULT_LARGE_MODEL,
  GCP_DEFAULT_SMALL_MODEL,
  OPENAI_DEFAULT_EMBEDDING_MODEL,
  OPENAI_DEFAULT_LARGE_MODEL,
  OPENAI_DEFAULT_SMALL_MODEL,
} from "../constants.js";
import {
  askPassword,
  askText,
  isPromptBack,
  pickOne,
  type PromptBack,
} from "../lib/prompts.js";
import type {
  AzureChoices,
  BedrockChoices,
  DeepSeekChoices,
  GcpChoices,
  ModelProvider,
  ModelsChoices,
  ModelsMode,
  AnthropicChoices,
  OpenAIChoices,
} from "../types.js";
import { askOllama } from "./ollama.js";
import { attachEmbeddingDimensions } from "./embedding-dimensions.js";

function expandHome(filePath: string): string {
  if (filePath.startsWith("~")) {
    return path.join(homedir(), filePath.slice(1));
  }
  return filePath;
}

interface GcpCredentialsCheck {
  error?: string;
  projectId?: string;
}

async function inspectGcpCredentials(filePath: string): Promise<GcpCredentialsCheck> {
  const expanded = expandHome(filePath.trim());
  if (!expanded) return { error: "Path is required" };
  try {
    const info = await stat(expanded);
    if (!info.isFile()) return { error: "Path is not a file" };
    const raw = await readFile(expanded, "utf8");
    const parsed = JSON.parse(raw) as { project_id?: unknown };
    const projectId =
      typeof parsed.project_id === "string" && parsed.project_id.trim().length > 0
        ? parsed.project_id.trim()
        : undefined;
    return { projectId };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { error: `Could not read credentials JSON: ${message}` };
  }
}

async function askGcp(opts: {
  needLlm: boolean;
  needEmbeddings: boolean;
}): Promise<GcpChoices> {
  p.log.step(
    opts.needLlm && opts.needEmbeddings
      ? "Configure GCP Vertex"
      : opts.needEmbeddings
        ? "Configure GCP Vertex embeddings"
        : "Configure GCP Vertex LLMs",
  );

  let credentialsPath = "";
  let projectIdFromFile: string | undefined;
  while (true) {
    const raw = await askText({
      message: "Path to GCP service-account credentials JSON",
      placeholder: "~/.config/gcloud/brainapi.json",
    });
    const result = await inspectGcpCredentials(raw);
    if (!result.error) {
      credentialsPath = expandHome(raw.trim());
      projectIdFromFile = result.projectId;
      break;
    }
    p.log.error(result.error);
  }

  let projectId: string;
  if (projectIdFromFile) {
    projectId = projectIdFromFile;
    p.log.info(`Using GCP project ${pc.cyan(projectId)} (from credentials file)`);
  } else {
    p.log.warn(
      "No project_id field found in the credentials JSON — please enter it manually.",
    );
    const entered = await askText({
      message: "GCP project id",
      placeholder: "my-gcp-project",
      validate: (value) => (value.trim().length === 0 ? "Project id is required" : undefined),
    });
    projectId = entered.trim();
  }

  let smallLlmModel = GCP_DEFAULT_SMALL_MODEL;
  let largeLlmModel = GCP_DEFAULT_LARGE_MODEL;
  if (opts.needLlm) {
    const smallModel = await askText({
      message: "GCP small LLM model",
      placeholder: GCP_DEFAULT_SMALL_MODEL,
      defaultValue: GCP_DEFAULT_SMALL_MODEL,
    });
    const largeModel = await askText({
      message: "GCP large LLM model",
      placeholder: GCP_DEFAULT_LARGE_MODEL,
      defaultValue: GCP_DEFAULT_LARGE_MODEL,
    });
    smallLlmModel = smallModel.trim() || GCP_DEFAULT_SMALL_MODEL;
    largeLlmModel = largeModel.trim() || GCP_DEFAULT_LARGE_MODEL;
  }

  let embeddingModel = GCP_DEFAULT_EMBEDDING_MODEL;
  if (opts.needEmbeddings) {
    const embedding = await askText({
      message: "GCP embedding model",
      placeholder: GCP_DEFAULT_EMBEDDING_MODEL,
      defaultValue: GCP_DEFAULT_EMBEDDING_MODEL,
    });
    embeddingModel = embedding.trim() || GCP_DEFAULT_EMBEDDING_MODEL;
  }

  return {
    credentialsPath,
    projectId,
    smallLlmModel,
    largeLlmModel,
    embeddingModel,
  };
}

async function askAzure(opts: {
  needLlm: boolean;
  needEmbeddings: boolean;
}): Promise<AzureChoices> {
  p.log.step(
    opts.needLlm && opts.needEmbeddings
      ? "Configure Azure OpenAI"
      : opts.needEmbeddings
        ? "Configure Azure OpenAI embeddings"
        : "Configure Azure OpenAI LLMs",
  );

  let llmEndpoint = "";
  let llmApiVersion = AZURE_DEFAULT_LARGE_API_VERSION;
  let llmSubscriptionKey = "";
  let smallLlmModel = AZURE_DEFAULT_SMALL_MODEL;
  let largeLlmModel = AZURE_DEFAULT_LARGE_MODEL;
  if (opts.needLlm) {
    llmEndpoint = (
      await askText({
        message: "Azure LLM endpoint",
        placeholder: "https://yourproject.openai.azure.com",
        validate: (value) =>
          value.trim().length === 0 ? "Endpoint is required" : undefined,
      })
    ).trim();
    llmApiVersion = (
      await askText({
        message: "Azure LLM API version",
        placeholder: AZURE_DEFAULT_LARGE_API_VERSION,
        defaultValue: AZURE_DEFAULT_LARGE_API_VERSION,
      })
    ).trim() || AZURE_DEFAULT_LARGE_API_VERSION;
    llmSubscriptionKey = (
      await askPassword({
        message: "Azure LLM subscription key",
        validate: (value) =>
          value.trim().length === 0 ? "API key is required" : undefined,
      })
    ).trim();
    smallLlmModel =
      (
        await askText({
          message: "Azure small LLM deployment/model",
          placeholder: AZURE_DEFAULT_SMALL_MODEL,
          defaultValue: AZURE_DEFAULT_SMALL_MODEL,
        })
      ).trim() || AZURE_DEFAULT_SMALL_MODEL;
    largeLlmModel =
      (
        await askText({
          message: "Azure large LLM deployment/model",
          placeholder: AZURE_DEFAULT_LARGE_MODEL,
          defaultValue: AZURE_DEFAULT_LARGE_MODEL,
        })
      ).trim() || AZURE_DEFAULT_LARGE_MODEL;
  }

  let embeddingEndpoint = "";
  let embeddingKey = "";
  let embeddingModel = AZURE_DEFAULT_EMBEDDING_MODEL;
  if (opts.needEmbeddings) {
    embeddingEndpoint = (
      await askText({
        message: "Azure embeddings endpoint URL",
        placeholder:
          "https://yourproject.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15",
        validate: (value) =>
          value.trim().length === 0 ? "Endpoint URL is required" : undefined,
      })
    ).trim();
    embeddingKey = (
      await askPassword({
        message: "Azure embeddings API key",
        validate: (value) =>
          value.trim().length === 0 ? "API key is required" : undefined,
      })
    ).trim();
    embeddingModel =
      (
        await askText({
          message: "Azure embedding model/deployment",
          placeholder: AZURE_DEFAULT_EMBEDDING_MODEL,
          defaultValue: AZURE_DEFAULT_EMBEDDING_MODEL,
        })
      ).trim() || AZURE_DEFAULT_EMBEDDING_MODEL;
  }

  return {
    smallLlmModel,
    largeLlmModel,
    llmApiVersion,
    llmEndpoint,
    llmSubscriptionKey,
    embeddingEndpoint,
    embeddingKey,
    embeddingModel,
  };
}

async function askBedrock(opts: {
  needLlm: boolean;
  needEmbeddings: boolean;
}): Promise<BedrockChoices> {
  p.log.step(
    opts.needLlm && opts.needEmbeddings
      ? "Configure Amazon Bedrock"
      : opts.needEmbeddings
        ? "Configure Amazon Bedrock embeddings"
        : "Configure Amazon Bedrock LLMs",
  );
  const region = await askText({
    message: "AWS region",
    placeholder: BEDROCK_DEFAULT_REGION,
    defaultValue: BEDROCK_DEFAULT_REGION,
  });
  const accessKeyId = await askText({
    message: "AWS access key id",
    validate: (value) =>
      value.trim().length === 0 ? "Access key id is required" : undefined,
  });
  const secretAccessKey = await askPassword({
    message: "AWS secret access key",
    validate: (value) =>
      value.trim().length === 0 ? "Secret access key is required" : undefined,
  });
  const sessionToken = await askPassword({
    message: "AWS session token (optional)",
    validate: () => undefined,
  });

  let smallLlmModel = BEDROCK_DEFAULT_SMALL_MODEL;
  let largeLlmModel = BEDROCK_DEFAULT_LARGE_MODEL;
  if (opts.needLlm) {
    smallLlmModel =
      (
        await askText({
          message: "Bedrock small LLM model id",
          placeholder: BEDROCK_DEFAULT_SMALL_MODEL,
          defaultValue: BEDROCK_DEFAULT_SMALL_MODEL,
        })
      ).trim() || BEDROCK_DEFAULT_SMALL_MODEL;
    largeLlmModel =
      (
        await askText({
          message: "Bedrock large LLM model id",
          placeholder: BEDROCK_DEFAULT_LARGE_MODEL,
          defaultValue: BEDROCK_DEFAULT_LARGE_MODEL,
        })
      ).trim() || BEDROCK_DEFAULT_LARGE_MODEL;
  }

  let embeddingModel = BEDROCK_DEFAULT_EMBEDDING_MODEL;
  if (opts.needEmbeddings) {
    embeddingModel =
      (
        await askText({
          message: "Bedrock embedding model id",
          placeholder: BEDROCK_DEFAULT_EMBEDDING_MODEL,
          defaultValue: BEDROCK_DEFAULT_EMBEDDING_MODEL,
        })
      ).trim() || BEDROCK_DEFAULT_EMBEDDING_MODEL;
  }

  return {
    region: region.trim() || BEDROCK_DEFAULT_REGION,
    accessKeyId: accessKeyId.trim(),
    secretAccessKey: secretAccessKey.trim(),
    sessionToken: sessionToken.trim() || undefined,
    smallLlmModel,
    largeLlmModel,
    embeddingModel,
  };
}

async function askOpenAI(opts: {
  needLlm: boolean;
  needEmbeddings: boolean;
}): Promise<OpenAIChoices> {
  p.log.step(
    opts.needLlm && opts.needEmbeddings
      ? "Configure OpenAI"
      : opts.needEmbeddings
        ? "Configure OpenAI embeddings"
        : "Configure OpenAI LLMs",
  );

  const apiKey = await askPassword({
    message: "OpenAI API key",
    validate: (value) =>
      value.trim().length === 0 ? "API key is required" : undefined,
  });
  const baseUrl = await askText({
    message: "OpenAI base URL (optional)",
    placeholder: "https://api.openai.com/v1",
    defaultValue: "",
  });

  let smallLlmModel = OPENAI_DEFAULT_SMALL_MODEL;
  let largeLlmModel = OPENAI_DEFAULT_LARGE_MODEL;
  if (opts.needLlm) {
    const small = await askText({
      message: "OpenAI small LLM model",
      placeholder: OPENAI_DEFAULT_SMALL_MODEL,
      defaultValue: OPENAI_DEFAULT_SMALL_MODEL,
    });
    const large = await askText({
      message: "OpenAI large LLM model",
      placeholder: OPENAI_DEFAULT_LARGE_MODEL,
      defaultValue: OPENAI_DEFAULT_LARGE_MODEL,
    });
    smallLlmModel = small.trim() || OPENAI_DEFAULT_SMALL_MODEL;
    largeLlmModel = large.trim() || OPENAI_DEFAULT_LARGE_MODEL;
  }

  let embeddingModel = OPENAI_DEFAULT_EMBEDDING_MODEL;
  if (opts.needEmbeddings) {
    const embedding = await askText({
      message: "OpenAI embedding model",
      placeholder: OPENAI_DEFAULT_EMBEDDING_MODEL,
      defaultValue: OPENAI_DEFAULT_EMBEDDING_MODEL,
    });
    embeddingModel = embedding.trim() || OPENAI_DEFAULT_EMBEDDING_MODEL;
  }

  return {
    apiKey: apiKey.trim(),
    baseUrl: baseUrl.trim() || undefined,
    smallLlmModel,
    largeLlmModel,
    embeddingModel,
  };
}

async function askAnthropic(): Promise<AnthropicChoices> {
  p.log.step("Configure Anthropic");
  const apiKey = await askPassword({
    message: "Anthropic API key",
    validate: (value) =>
      value.trim().length === 0 ? "API key is required" : undefined,
  });
  const smallLlmModel = await askText({
    message: "Anthropic small LLM model",
    placeholder: ANTHROPIC_DEFAULT_SMALL_MODEL,
    defaultValue: ANTHROPIC_DEFAULT_SMALL_MODEL,
  });
  const largeLlmModel = await askText({
    message: "Anthropic large LLM model",
    placeholder: ANTHROPIC_DEFAULT_LARGE_MODEL,
    defaultValue: ANTHROPIC_DEFAULT_LARGE_MODEL,
  });
  return {
    apiKey: apiKey.trim(),
    smallLlmModel: smallLlmModel.trim() || ANTHROPIC_DEFAULT_SMALL_MODEL,
    largeLlmModel: largeLlmModel.trim() || ANTHROPIC_DEFAULT_LARGE_MODEL,
  };
}

async function askDeepSeek(): Promise<DeepSeekChoices> {
  p.log.step("Configure DeepSeek");
  const apiKey = await askPassword({
    message: "DeepSeek API key",
    validate: (value) =>
      value.trim().length === 0 ? "API key is required" : undefined,
  });
  const smallLlmModel = await askText({
    message: "DeepSeek small LLM model",
    placeholder: DEEPSEEK_DEFAULT_SMALL_MODEL,
    defaultValue: DEEPSEEK_DEFAULT_SMALL_MODEL,
  });
  const largeLlmModel = await askText({
    message: "DeepSeek large LLM model",
    placeholder: DEEPSEEK_DEFAULT_LARGE_MODEL,
    defaultValue: DEEPSEEK_DEFAULT_LARGE_MODEL,
  });
  return {
    apiKey: apiKey.trim(),
    smallLlmModel: smallLlmModel.trim() || DEEPSEEK_DEFAULT_SMALL_MODEL,
    largeLlmModel: largeLlmModel.trim() || DEEPSEEK_DEFAULT_LARGE_MODEL,
  };
}

async function askLlmProvider(message: string, initialValue: ModelProvider): Promise<ModelProvider> {
  return pickOne<ModelProvider>({
    message,
    options: [
      { value: "ollama", label: "Ollama" },
      { value: "openai", label: "OpenAI" },
      { value: "anthropic", label: "Anthropic" },
      { value: "deepseek", label: "DeepSeek" },
      { value: "azure", label: "Azure OpenAI" },
      { value: "gcp_vertex", label: "Google Cloud — Vertex AI" },
      { value: "amazon_bedrock", label: "Amazon Bedrock" },
    ],
    initialValue,
  });
}

async function askEmbeddingsProvider(
  message: string,
  initialValue: Exclude<ModelProvider, "anthropic" | "deepseek">,
): Promise<Exclude<ModelProvider, "anthropic" | "deepseek">> {
  return pickOne<Exclude<ModelProvider, "anthropic" | "deepseek">>({
    message,
    options: [
      { value: "ollama", label: "Ollama" },
      { value: "openai", label: "OpenAI" },
      { value: "azure", label: "Azure OpenAI" },
      { value: "gcp_vertex", label: "Google Cloud — Vertex AI" },
      { value: "amazon_bedrock", label: "Amazon Bedrock" },
    ],
    initialValue,
  });
}

export async function askModels(options?: {
  prechosenMode?: ModelsMode;
  allowBack?: false;
  initialMode?: ModelsMode;
}): Promise<ModelsChoices>;
export async function askModels(options: {
  prechosenMode?: ModelsMode;
  allowBack: true;
  backHint?: string;
  initialMode?: ModelsMode;
}): Promise<ModelsChoices | PromptBack>;
export async function askModels(options?: {
  prechosenMode?: ModelsMode;
  allowBack?: boolean;
  backHint?: string;
  initialMode?: ModelsMode;
}): Promise<ModelsChoices | PromptBack> {
  let mode = options?.prechosenMode;
  if (!mode) {
    const picked = options?.allowBack
      ? await pickOne<ModelsMode>({
          message: "Models mode",
          options: [
            { value: "remote", label: "Remote (cloud provider)" },
            { value: "local", label: "Local (Ollama)" },
          ],
          initialValue: options.initialMode ?? "remote",
          allowBack: true,
          backHint: options.backHint,
        })
      : await pickOne<ModelsMode>({
          message: "Models mode",
          options: [
            { value: "remote", label: "Remote (cloud provider)" },
            { value: "local", label: "Local (Ollama)" },
          ],
          initialValue: options?.initialMode ?? "remote",
        });
    if (isPromptBack(picked)) {
      return picked;
    }
    mode = picked;
  }

  if (mode === "local") {
    const ollama = await askOllama();
    return attachEmbeddingDimensions(
      {
        mode,
        llmSmallProvider: "ollama",
        llmLargeProvider: "ollama",
        embeddingsProvider: "ollama",
        ollama,
      },
      {
        allowBack: options?.allowBack,
        backHint: options?.backHint,
      },
    );
  }

  const llmSmallProvider = await askLlmProvider("Small LLM provider", "gcp_vertex");
  const llmLargeProvider = await askLlmProvider("Large LLM provider", "azure");
  const embeddingsProvider = await askEmbeddingsProvider("Embeddings provider", "azure");

  const providers = new Set([llmSmallProvider, llmLargeProvider, embeddingsProvider]);

  let ollama = undefined;
  let gcp = undefined;
  let azure = undefined;
  let openai = undefined;
  let anthropic = undefined;
  let deepseek = undefined;
  let bedrock = undefined;

  if (providers.has("ollama")) {
    ollama = await askOllama();
  }
  if (providers.has("gcp_vertex")) {
    gcp = await askGcp({
      needLlm:
        llmSmallProvider === "gcp_vertex" || llmLargeProvider === "gcp_vertex",
      needEmbeddings: embeddingsProvider === "gcp_vertex",
    });
  }
  if (providers.has("azure")) {
    azure = await askAzure({
      needLlm: llmSmallProvider === "azure" || llmLargeProvider === "azure",
      needEmbeddings: embeddingsProvider === "azure",
    });
  }
  if (providers.has("openai")) {
    openai = await askOpenAI({
      needLlm: llmSmallProvider === "openai" || llmLargeProvider === "openai",
      needEmbeddings: embeddingsProvider === "openai",
    });
  }
  if (providers.has("anthropic")) {
    anthropic = await askAnthropic();
  }
  if (providers.has("deepseek")) {
    deepseek = await askDeepSeek();
  }
  if (providers.has("amazon_bedrock")) {
    bedrock = await askBedrock({
      needLlm:
        llmSmallProvider === "amazon_bedrock" ||
        llmLargeProvider === "amazon_bedrock",
      needEmbeddings: embeddingsProvider === "amazon_bedrock",
    });
  }

  return attachEmbeddingDimensions(
    {
      mode,
      llmSmallProvider,
      llmLargeProvider,
      embeddingsProvider,
      ollama,
      gcp,
      azure,
      openai,
      anthropic,
      deepseek,
      bedrock,
    },
    {
      allowBack: options?.allowBack,
      backHint: options?.backHint,
    },
  );
}
