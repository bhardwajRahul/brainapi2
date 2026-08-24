import * as p from "@clack/prompts";
import pc from "picocolors";
import { isPromptBack, pickOne, type PromptBack } from "../lib/prompts.js";
import type { SearchChoices, SearchRetrieval } from "../types.js";

const DEFAULT_SEARCH: SearchChoices = {
  enabled: false,
  retrieval: "hybrid",
};

export function defaultSearchChoices(): SearchChoices {
  return { ...DEFAULT_SEARCH };
}

export async function askSearch(options?: {
  allowBack?: false;
  initial?: SearchChoices;
  dataDb?: string;
}): Promise<SearchChoices>;
export async function askSearch(options: {
  allowBack: true;
  backHint?: string;
  initial?: SearchChoices;
  dataDb?: string;
}): Promise<SearchChoices | PromptBack>;
export async function askSearch(options?: {
  allowBack?: boolean;
  backHint?: string;
  initial?: SearchChoices;
  dataDb?: string;
}): Promise<SearchChoices | PromptBack> {
  p.log.step("Search");
  if (options?.dataDb && options.dataDb !== "postgresql") {
    p.log.info(
      pc.dim(
        "Hybrid search needs DATA_DB=postgresql. Leaving SEARCH_ENABLED=false.",
      ),
    );
    return { enabled: false, retrieval: "hybrid" };
  }

  const enabledChoice = options?.allowBack
    ? await pickOne<"yes" | "no">({
        message: "Enable hybrid search API (/retrieve/search)?",
        options: [
          {
            value: "no",
            label: "No (default)",
            hint: "memory path unchanged; search routes return 404",
          },
          {
            value: "yes",
            label: "Yes",
            hint: "BM25 + dense ANN; fused by default; p50 < 200 ms ex-embed",
          },
        ],
        initialValue: options.initial?.enabled ? "yes" : "no",
        allowBack: true,
        backHint: options.backHint,
      })
    : await pickOne<"yes" | "no">({
        message: "Enable hybrid search API (/retrieve/search)?",
        options: [
          {
            value: "no",
            label: "No (default)",
            hint: "memory path unchanged; search routes return 404",
          },
          {
            value: "yes",
            label: "Yes",
            hint: "BM25 + dense ANN; fused by default; p50 < 200 ms ex-embed",
          },
        ],
        initialValue: options?.initial?.enabled ? "yes" : "no",
      });
  if (isPromptBack(enabledChoice)) {
    return enabledChoice;
  }
  if (enabledChoice === "no") {
    return { enabled: false, retrieval: "hybrid" };
  }

  const retrieval = options?.allowBack
    ? await pickOne<SearchRetrieval>({
        message: "Search retrieval",
        options: [
          {
            value: "hybrid",
            label: "Both fused (default)",
            hint: "parallel BM25 ∪ dense, RRF",
          },
          { value: "dense", label: "Dense only" },
          { value: "bm25", label: "BM25 only" },
        ],
        initialValue: options.initial?.retrieval ?? "hybrid",
        allowBack: true,
        backHint: options.backHint,
      })
    : await pickOne<SearchRetrieval>({
        message: "Search retrieval",
        options: [
          {
            value: "hybrid",
            label: "Both fused (default)",
            hint: "parallel BM25 ∪ dense, RRF",
          },
          { value: "dense", label: "Dense only" },
          { value: "bm25", label: "BM25 only" },
        ],
        initialValue: options?.initial?.retrieval ?? "hybrid",
      });
  if (isPromptBack(retrieval)) {
    return retrieval;
  }
  return { enabled: true, retrieval };
}
