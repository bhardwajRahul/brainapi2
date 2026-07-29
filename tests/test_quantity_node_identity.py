import json
import os
import re
import unittest


ENV_DEFAULTS = {
    "BRAINPAT_TOKEN": "test-token",
    "MODELS_MODE": "local",
    "EMBEDDINGS_LOCAL_MODEL": "local-model",
    "EMBEDDINGS_SMALL_MODEL": "small-model",
    "EMBEDDING_NODES_DIMENSION": "3",
    "EMBEDDING_TRIPLETS_DIMENSION": "3",
    "EMBEDDING_OBSERVATIONS_DIMENSION": "3",
    "EMBEDDING_DATA_DIMENSION": "3",
    "EMBEDDING_RELATIONSHIPS_DIMENSION": "3",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "MILVUS_HOST": "localhost",
    "MILVUS_PORT": "19530",
    "MONGO_CONNECTION_STRING": "mongodb://localhost:27017",
    "CELERY_WORKER_CONCURRENCY": "1",
    "OLLAMA_HOST": "localhost",
    "OLLAMA_PORT": "11434",
    "OLLAMA_LLM_SMALL_MODEL": "small",
    "OLLAMA_LLM_LARGE_MODEL": "large",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)

from src.constants.prompts import architect_agent, janitor_agent, scout_agent
from src.core.saving.identity import stable_node_id


_PLACEHOLDER_TYPES = {
    "MONEY",
    "UNIT",
    "UNITS",
    "AMOUNT",
    "QUANTITY",
    "COUNT",
    "CURRENCY",
    "NUMBER",
}
_ENTITY_PATTERN = re.compile(r"\{\{[^{}]*\"type\"\s*:[^{}]*\}\}")


def _prompt_constants(module) -> dict[str, str]:
    return {
        name: value
        for name, value in vars(module).items()
        if name.isupper() and isinstance(value, str)
    }


def _demonstrated_entities(prompt: str) -> list[dict]:
    entities: list[dict] = []
    for match in _ENTITY_PATTERN.finditer(prompt):
        body = match.group(0).replace("{{", "{").replace("}}", "}")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("type") and parsed.get("name"):
            entities.append(parsed)
    return entities


def _shared_hub_entities(prompt: str) -> list[dict]:
    shared: list[dict] = []
    for entity in _demonstrated_entities(prompt):
        entity_type = str(entity["type"]).strip()
        name = str(entity["name"]).strip()
        if entity_type.upper() in _PLACEHOLDER_TYPES:
            shared.append(entity)
        elif name.lower() == entity_type.lower():
            shared.append(entity)
    return shared


def _fact_node_ids(prompt: str, event_name: str, happened_at: str) -> set[str]:
    ids = {stable_node_id(event_name, "EVENT", happened_at)}
    for entity in _shared_hub_entities(prompt):
        ids.add(stable_node_id(entity["name"], entity["type"]))
    return ids


class QuantityNodeIdentityTests(unittest.TestCase):
    def test_two_monetary_amounts_do_not_share_a_node_id(self):
        for module in (scout_agent, architect_agent):
            for name, prompt in _prompt_constants(module).items():
                acme = _fact_node_ids(prompt, "Raised", "19/01/2026")
                globex = _fact_node_ids(prompt, "Raised", "03/04/2026")
                self.assertEqual(
                    acme & globex,
                    set(),
                    f"{module.__name__}.{name} demonstrates a node shared by two "
                    f"unrelated monetary facts",
                )

    def test_prompts_demonstrate_no_type_named_placeholder_nodes(self):
        offenders: list[str] = []
        for module in (scout_agent, architect_agent):
            for name, prompt in _prompt_constants(module).items():
                for entity in _shared_hub_entities(prompt):
                    offenders.append(
                        f"{module.__name__}.{name}: "
                        f"{entity['type']}/{entity['name']}"
                    )
        self.assertEqual(offenders, [])

    def test_prompts_do_not_instruct_unit_entities(self):
        offenders: list[str] = []
        for module in (scout_agent, architect_agent):
            for name, prompt in _prompt_constants(module).items():
                if re.search(r"\"Unit\".*as an entity", prompt):
                    offenders.append(f"{module.__name__}.{name}")
                if "quantitative units" in prompt:
                    offenders.append(f"{module.__name__}.{name}")
        self.assertEqual(offenders, [])

    def test_janitor_still_prescribes_amount_migration(self):
        prompts = "\n".join(_prompt_constants(janitor_agent).values())
        self.assertIn("Amount Migration", prompts)
        self.assertIn("amount-nodes", prompts)

    def test_type_named_identity_is_constant_across_facts(self):
        self.assertEqual(
            stable_node_id("Money", "MONEY"),
            stable_node_id("Money", "MONEY"),
        )
        self.assertNotEqual(
            stable_node_id("Raised", "EVENT", "19/01/2026"),
            stable_node_id("Raised", "EVENT", "03/04/2026"),
        )


if __name__ == "__main__":
    unittest.main()
