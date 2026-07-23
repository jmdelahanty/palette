from __future__ import annotations

import re
from pathlib import Path


_FIXED_UNICODE_PATTERNS = (
    re.compile(r"dtype\s*=\s*f?['\"]<U\d*"),
    re.compile(r"np\.dtype\(\s*f?['\"]<U\d*"),
    re.compile(r"astype\(\s*f?['\"]<U\d*"),
)

# Keep this list empty by default. If a compatibility exception is required,
# add the repo-relative path here with a short code comment in the file.
_ALLOWLIST: tuple[str, ...] = ()


def test_runtime_source_has_no_fixed_unicode_string_writes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_root = repo_root / "src"

    violations: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        rel = path.relative_to(repo_root).as_posix()
        if rel in _ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(pat.search(line) for pat in _FIXED_UNICODE_PATTERNS):
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert not violations, (
        "Found fixed-width unicode (<U...) write patterns in runtime source.\n"
        "Use VariableLengthUTF8() or reason_bytes-compatible encodings instead.\n"
        + "\n".join(violations)
    )
