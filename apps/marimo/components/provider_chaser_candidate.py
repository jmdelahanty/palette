"""Read-only adapter for unpromoted provider-aware chaser candidates."""

from __future__ import annotations

from pathlib import Path

from fisheye.analysis.provider_chaser_distance_candidates import (
    MANIFEST_DIGEST_ATTR,
    PARENT_PATH,
    validate_provider_chaser_distance_candidate,
)

from .common import normalize_path
from .registry import (
    PROVIDER_CHASER_CANDIDATE_RENDERER,
    InteractiveSpecOption,
)


def _exact_candidate_run_path(option: InteractiveSpecOption) -> str:
    run_path = normalize_path(option.run_path)
    prefix = f"{PARENT_PATH}/"
    if not run_path.startswith(prefix):
        raise ValueError(
            "Provider chaser candidate must be selected by its exact candidate run path."
        )
    run_name = run_path.removeprefix(prefix)
    if not run_name or "/" in run_name or run_name in {".", ".."}:
        raise ValueError("Provider chaser candidate run name is invalid.")
    return run_path


def available_provider_chaser_candidate_analysis_ids(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
) -> tuple[str, ...]:
    """Return read-only views only after exact consolidated manifest validation."""

    if option.renderer != PROVIDER_CHASER_CANDIDATE_RENDERER:
        raise ValueError("Selected option is not a provider chaser candidate.")
    run_path = _exact_candidate_run_path(option)
    manifest_sha256 = str(option.attrs.get(MANIFEST_DIGEST_ATTR) or "").strip()
    if not manifest_sha256:
        raise ValueError("Provider chaser candidate has no manifest digest authority.")
    archive = Path(zarr_path)
    validation = validate_provider_chaser_distance_candidate(
        archive / run_path,
        use_consolidated=True,
        archive_path=archive,
        archive_run_path=run_path,
        expected_manifest_sha256=manifest_sha256,
    )
    if not validation.get("valid"):
        raise ValueError(
            "Provider chaser candidate failed exact manifest validation: "
            f"{validation.get('errors', [])!r}."
        )
    if option.spec.get("candidate_status") != "unpromoted_selector_ineligible":
        raise ValueError("Provider chaser candidate status is not explicitly unpromoted.")
    return ("static_artifacts", "provenance")
