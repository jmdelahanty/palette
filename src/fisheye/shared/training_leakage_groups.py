"""Conservative biological/cohort grouping for training-set splits."""

from __future__ import annotations

from typing import Sequence


def resolve_training_leakage_group(
    *,
    recording_id: str,
    subject_ids: Sequence[str],
    started_utc: str | None,
) -> tuple[str, str]:
    """Return one stable group that must remain inside a single data split.

    Registered biological subjects are the strongest available identity.  For
    historical multi-camera acquisitions without subject registration, the
    acquisition start timestamp keeps sibling camera recordings together.
    The recording identity is the final fail-closed fallback.
    """

    recording = str(recording_id).strip()
    if not recording:
        raise ValueError("recording_id is required for leakage grouping.")
    subjects = tuple(
        sorted({str(value).strip() for value in subject_ids if str(value).strip()})
    )
    if subjects:
        prefix = "subject" if len(subjects) == 1 else "subjects"
        return f"{prefix}:{','.join(subjects)}", "registered_subject"
    started = str(started_utc).strip() if started_utc is not None else ""
    if started:
        return f"acquisition_cohort:{started}", "acquisition_start_fallback"
    return f"recording:{recording}", "recording_fallback"


__all__ = ["resolve_training_leakage_group"]
