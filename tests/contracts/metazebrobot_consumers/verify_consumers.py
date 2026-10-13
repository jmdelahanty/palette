#!/usr/bin/env python3
"""Check MetaZebrobot consumer expectations against the producer's schema.

Stdlib only. Reads consumers.json next to this file and verifies, for each
consumer, that every operation, field, schema_version, and error response it
relies on is present in MetaZebrobot's pinned consumer OpenAPI slice.

    # Pinned file, fetched from GitHub at the pinned commit (default)
    python3 verify_consumers.py

    # A local copy of docs/api/consumer_openapi.json instead of fetching
    python3 verify_consumers.py --openapi-file ~/gitrepos/metazebrobot/docs/api/consumer_openapi.json

    # Also check the running service: GET /version must report the pinned
    # digest, and its live /openapi.json must satisfy every expectation
    python3 verify_consumers.py --live http://127.0.0.1:8000

    # Only one consumer's expectations
    python3 verify_consumers.py --consumer orange

Exit status is 0 when everything checks out, 1 otherwise.
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
ERROR_SCHEMA = "ApiErrorResponse"


class Report:
    def __init__(self):
        self.failures = 0

    def ok(self, message):
        print(f"  ok    {message}")

    def fail(self, message):
        self.failures += 1
        print(f"  FAIL  {message}")


def fetch(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def schema_name(ref):
    return ref.rsplit("/", 1)[-1]


def deref(node, doc):
    """Follow $ref and Optional (anyOf [X, null]) wrappers to a concrete schema."""
    while True:
        if "$ref" in node:
            node = doc["components"]["schemas"][schema_name(node["$ref"])]
        elif "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            if len(options) != 1:
                return node
            node = options[0]
        else:
            return node


def response_schema(operation, status, doc):
    response = operation.get("responses", {}).get(status)
    if response is None:
        return None
    schema = response.get("content", {}).get("application/json", {}).get("schema")
    return schema


def resolve_field(root, field, doc):
    """Resolve a dotted field path; ``name[]`` steps into an array's items."""
    node = deref(root, doc)
    for part in field.split("."):
        is_array = part.endswith("[]")
        name = part[:-2] if is_array else part
        properties = node.get("properties", {})
        if name not in properties:
            return None, f"no property {name!r}"
        prop = properties[name]
        node = deref(prop, doc)
        if is_array:
            if node.get("type") != "array":
                return None, f"{name!r} is not an array"
            node = deref(node["items"], doc)
    return prop, None


def check_expectation(expectation, doc, report):
    method, path = expectation["operation"].split(" ", 1)
    operation = doc.get("paths", {}).get(path, {}).get(method.lower())
    label = expectation["operation"]
    if operation is None:
        report.fail(f"{label}: operation not served")
        return

    root = response_schema(operation, "200", doc)
    if root is None:
        report.fail(f"{label}: no JSON 200 response schema")
        return

    for field in expectation.get("fields", []):
        prop, problem = resolve_field(root, field, doc)
        if problem:
            report.fail(f"{label}: {field}: {problem}")
        else:
            report.ok(f"{label}: {field}")

    expected_version = expectation.get("schema_version")
    if expected_version is not None:
        prop, problem = resolve_field(root, "schema_version", doc)
        served = (prop or {}).get("default")
        if problem or served != expected_version:
            report.fail(f"{label}: schema_version is {served!r}, expected {expected_version}")
        else:
            report.ok(f"{label}: schema_version == {expected_version}")

    for status in expectation.get("errors", []):
        schema = response_schema(operation, status, doc)
        if not schema or schema_name(schema.get("$ref", "")) != ERROR_SCHEMA:
            report.fail(f"{label}: {status} does not return {ERROR_SCHEMA}")
            continue
        detail, problem = resolve_field(schema, "detail", doc)
        detail = deref(detail, doc) if detail else {}
        if problem or "error" not in detail.get("required", []):
            report.fail(f"{label}: {status} {ERROR_SCHEMA}.detail.error not required")
        else:
            report.ok(f"{label}: {status} -> {ERROR_SCHEMA} with detail.error")


def check_consumers(spec, doc, only, report):
    for name, consumer in spec["consumers"].items():
        if only and name != only:
            continue
        print(f"[{name}] ({consumer.get('status', 'unknown')})")
        expectations = consumer.get("relies_on", [])
        if not expectations:
            print("  -     nothing declared yet")
        for expectation in expectations:
            check_expectation(expectation, doc, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", default=str(HERE / "consumers.json"))
    parser.add_argument("--openapi-file", help="Local consumer_openapi.json instead of fetching")
    parser.add_argument("--live", metavar="URL", help="Also check a running MetaZebrobot")
    parser.add_argument("--consumer", help="Only check this consumer")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    producer = spec["producer"]
    pinned_sha = producer["consumer_openapi_sha256"]
    report = Report()

    print(f"MetaZebrobot pin: {producer['commit'][:12]}  sha256 {pinned_sha[:12]}")

    # 1. The pinned file itself.
    try:
        if args.openapi_file:
            source = args.openapi_file
            text = Path(args.openapi_file).read_text()
        else:
            source = RAW_URL.format(repo=producer["repo"], commit=producer["commit"],
                                    path=producer["consumer_openapi"])
            text = fetch(source)
    except (OSError, urllib.error.URLError) as exc:
        print(f"[pinned schema] {source}")
        report.fail(f"could not read pinned schema: {exc}")
        return 1

    print(f"[pinned schema] {source}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest == pinned_sha:
        report.ok("sha256 matches pin")
    else:
        report.fail(f"sha256 {digest} != pinned {pinned_sha}")
    check_consumers(spec, json.loads(text), args.consumer, report)

    # 2. Optionally, the running service.
    if args.live:
        base = args.live.rstrip("/")
        print(f"[live] {base}")
        try:
            version = json.loads(fetch(f"{base}/version"))
        except urllib.error.HTTPError as exc:
            version = None
            report.fail(f"GET /version returned {exc.code} (service older than /version?)")
        except (OSError, urllib.error.URLError, ValueError) as exc:
            version = None
            report.fail(f"GET /version failed: {exc}")
        if version:
            commit = version.get("service_commit") or "unknown"
            dirty = " (dirty)" if version.get("service_commit_dirty") else ""
            print(f"  -     service_commit {commit}{dirty}, started {version.get('started_at_utc')}")
            if version.get("consumer_schema_sha256") == pinned_sha:
                report.ok("live consumer_schema_sha256 matches pin")
            else:
                report.fail(
                    f"live consumer_schema_sha256 {version.get('consumer_schema_sha256')} "
                    f"!= pinned {pinned_sha}"
                )
        try:
            live_doc = json.loads(fetch(f"{base}/openapi.json"))
        except (OSError, urllib.error.URLError, ValueError) as exc:
            report.fail(f"GET /openapi.json failed: {exc}")
        else:
            check_consumers(spec, live_doc, args.consumer, report)

    print("PASS" if report.failures == 0 else f"FAILED ({report.failures})")
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
