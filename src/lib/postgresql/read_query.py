from __future__ import annotations

import re

MAX_READ_QUERY_ROWS = 100
READ_QUERY_TIMEOUT_MS = 10000

_READ_QUERY_START = re.compile(
    r"^\s*(SELECT|WITH|EXPLAIN|TABLE)\b",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_SQL = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"COPY|MERGE|CALL|DO|VACUUM|REINDEX|CLUSTER|"
    r"LISTEN|NOTIFY|LOAD|LOCK|DISCARD|RESET"
    r")\b",
    re.IGNORECASE,
)
_CYPHER_MARKERS = re.compile(
    r"\b(MATCH|RETURN|OPTIONAL\s+MATCH|CREATE\s*\(|DETACH\s+DELETE|SET\s+n\.|REMOVE\s+n\.)\b",
    re.IGNORECASE,
)
_LEGACY_TABLE_NAMES = re.compile(
    r"\b(FROM|JOIN|TABLE|INTO|UPDATE)\s+(nodes|relationships)\b",
    re.IGNORECASE,
)
_WITH_RECURSIVE = re.compile(r"\bWITH\s+RECURSIVE\b", re.IGNORECASE)


class ReadQueryValidationError(Exception):
    pass


def _paren_balance(sql: str) -> int:
    """Return open-paren count minus close-paren count, ignoring quoted strings."""
    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(sql):
        ch = sql[i]
        if in_single:
            if ch == "'" and i + 1 < len(sql) and sql[i + 1] == "'":
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    return depth


def validate_read_only_sql(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ReadQueryValidationError("Empty query")
    parts = [part.strip() for part in normalized.split(";") if part.strip()]
    if len(parts) != 1:
        raise ReadQueryValidationError("Multiple statements are not allowed")
    normalized = parts[0]
    if _CYPHER_MARKERS.search(normalized):
        raise ReadQueryValidationError(
            "Cypher is not supported on PostgreSQL. Use SQL against kg_nodes "
            "and kg_relationships (for example: SELECT uuid, data->>'name' AS name "
            "FROM kg_nodes WHERE data->>'name' ILIKE '%alice%' LIMIT 50)."
        )
    if _LEGACY_TABLE_NAMES.search(normalized):
        raise ReadQueryValidationError(
            "Unknown table. Use kg_nodes and kg_relationships "
            "(not nodes/relationships)."
        )
    if _FORBIDDEN_SQL.search(normalized):
        raise ReadQueryValidationError("Only read-only SELECT queries are allowed")
    if not _READ_QUERY_START.match(normalized):
        raise ReadQueryValidationError(
            "Query must start with SELECT, WITH, EXPLAIN, or TABLE"
        )
    # EXPLAIN of Cypher still starts with EXPLAIN; catch residual Cypher body.
    if normalized.upper().startswith("EXPLAIN") and _CYPHER_MARKERS.search(
        normalized[7:]
    ):
        raise ReadQueryValidationError(
            "Cypher is not supported on PostgreSQL. Use SQL EXPLAIN SELECT ..."
        )

    balance = _paren_balance(normalized)
    if balance != 0:
        raise ReadQueryValidationError(
            "Incomplete SQL (unbalanced parentheses). Send one complete statement "
            "in a single tool call — do not split WITH RECURSIVE CTEs across calls. "
            "Example: WITH RECURSIVE walk AS ("
            "SELECT source_uuid, target_uuid, rel_type, 1 AS depth "
            "FROM kg_relationships WHERE source_uuid = '<uuid>' "
            "UNION ALL "
            "SELECT r.source_uuid, r.target_uuid, r.rel_type, w.depth + 1 "
            "FROM walk w JOIN kg_relationships r ON r.source_uuid = w.target_uuid "
            "WHERE w.depth < 3"
            ") SELECT * FROM walk LIMIT 50"
        )
    if _WITH_RECURSIVE.search(normalized):
        # A recursive CTE must both define the CTE and select from it.
        if not re.search(r"\)\s*SELECT\b", normalized, re.IGNORECASE):
            raise ReadQueryValidationError(
                "Incomplete WITH RECURSIVE query. Include the outer SELECT after "
                "the CTE, e.g. ') SELECT * FROM walk LIMIT 50'. Do not send only "
                "the CTE body or a fragment of the UNION ALL branch."
            )

    return normalized
