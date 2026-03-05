# Agent Task: Registry Query `--since` Filter

## Objective
Add a `--since <YYYY-MM-DD>` date filter to the registry query CLI and evaluate
whether materialized views are needed at current scale.

## Files to modify
- `src/fisheye/registry/query.py` — the query CLI

## Implementation: `--since` filter

### 1. Add argument (in `_parse_args`, around line 37 after the dpf args)
Add a new argument following the existing pattern:
```python
parser.add_argument("--since", type=str, help="Only datasets created on or after this date (YYYY-MM-DD).")
```

### 2. Add filter clause (in `_build_query`, around line 170 after the dpf clauses)
Follow the existing `add_clause` pattern:
```python
if args.since:
    add_clause("AND d.created_utc >= ?", args.since)
```
The `datasets.created_utc` column stores ISO-8601 strings, so plain string
comparison with a `YYYY-MM-DD` value works correctly in SQLite.

### 3. Tests
Add a test in the appropriate test file under `tests/` that:
- Inserts a couple of datasets with different `created_utc` values
- Calls the query with `--since` and confirms only the expected rows return
- Follow existing test patterns in the registry test files

## Item 2: Materialized views

With only ~114 rows in `datasets`, materialized views provide no measurable
benefit. Mark this item as **deferred** — add a note to the todo doc that it
should be revisited if the registry grows past ~10k rows.

## After implementation
Update `docs/registry_query_todo.md`:
- Check off `--since` item and add a status note
- Update the materialized views item to note deferral with rationale

## Reference: existing patterns
- Arguments: see lines 14-69 in `query.py`
- Filter clauses: see `_build_query` lines 141-170 — uses `add_clause(sql_fragment, value)`
- DB schema: `datasets.created_utc TEXT` (line 1968 in `db.py`)
- The `--dpf-min` / `--dpf-max` filters are the closest analog to `--since`
