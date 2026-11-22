"""Orange's constant-QP all-intra crop profile, and the rule that admits it.

Orange crop recordings are either lossless (``tuning`` "lossless", QP 0) or,
since Orange edc5450 (first recording 2026_11_10_14_14_34), the constant-QP
profile ``external_crop_hevc_cqp_gop1``: HEVC, tuning "hq", one constant QP
from app config ``recording.crop.external_ipc.qp`` (1..51), every frame an
IDR. Lossless crops keep their existing checks unchanged; this module decides
only whether a non-lossless crop is exactly that constant-QP profile.

The rule (``palette.orange_crop_cqp_gop1.v1``) reads the recorder summary:

- top-level ``tuning`` "hq" and ``video_metadata.encoder``: codec hevc, profile_name
  ``external_crop_hevc_cqp_gop1``, tuning "hq", rate_control_strategy cqp,
  resolved_gop_length 1, and ``qp`` equal to the rate-control QP;
- ``rate_control``: resolved_strategy cqp, const_qp_active true and an
  integer const_qp from 1 to 51. It is the authoritative rate control; the
  encoder block's bitrate and VBV fields are not read.

``requested_gop_length`` is Orange's shard routing period (25), not the
encoder GOP, and is not read. Any other combination is refused.

The QP the recorder actually received is sealed in the start snapshot at
``session.effective_configuration.environment.ORANGE_CROP_EXTERNAL_QP``
(``{"value": "29", "source": ...}``; the value is a decimal string).
``require_start_snapshot_qp`` refuses a recording whose snapshot does not
carry it or disagrees with the summary.
"""

from __future__ import annotations

from typing import Any, Mapping

CQP_GOP1_RULE_ID = "palette.orange_crop_cqp_gop1.v1"
CQP_GOP1_PROFILE = "external_crop_hevc_cqp_gop1"
CQP_TUNING = "hq"
QP_RANGE = (1, 51)
START_SNAPSHOT_QP_ENV = "ORANGE_CROP_EXTERNAL_QP"


class CropCompressionRefusal(ValueError):
    """The crop summary is not exactly an admitted crop compression profile."""


def _strict_int(value: Any) -> int | None:
    return value if type(value) is int else None


def cqp_gop1_compression(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The admitted constant-QP declaration of one crop summary, or a refusal.

    Returns ``{"rule", "profile_name", "tuning", "qp"}``. Raises
    ``CropCompressionRefusal`` naming every field that does not match.
    """

    metadata = summary.get("video_metadata")
    encoder = metadata.get("encoder") if isinstance(metadata, Mapping) else None
    rate = summary.get("rate_control")
    if not isinstance(encoder, Mapping) or not isinstance(rate, Mapping):
        raise CropCompressionRefusal(
            "constant-QP crop summary lacks video_metadata.encoder or rate_control"
        )
    expected_encoder = {
        "codec": "hevc",
        "profile_name": CQP_GOP1_PROFILE,
        "tuning": CQP_TUNING,
        "rate_control_strategy": "cqp",
        "resolved_gop_length": 1,
    }
    mismatches = {
        f"encoder.{key}": encoder.get(key)
        for key, expected in expected_encoder.items()
        if encoder.get(key) != expected or type(encoder.get(key)) is not type(expected)
    }
    if summary.get("tuning") != CQP_TUNING:
        mismatches["tuning"] = summary.get("tuning")
    if rate.get("resolved_strategy") != "cqp":
        mismatches["rate_control.resolved_strategy"] = rate.get("resolved_strategy")
    if rate.get("const_qp_active") is not True:
        mismatches["rate_control.const_qp_active"] = rate.get("const_qp_active")
    qp = _strict_int(rate.get("const_qp"))
    if qp is None or not QP_RANGE[0] <= qp <= QP_RANGE[1]:
        mismatches["rate_control.const_qp"] = rate.get("const_qp")
    elif _strict_int(encoder.get("qp")) != qp:
        mismatches["encoder.qp"] = encoder.get("qp")
    if mismatches:
        raise CropCompressionRefusal(
            f"crop is neither lossless nor {CQP_GOP1_RULE_ID}: {mismatches}"
        )
    return {
        "rule": CQP_GOP1_RULE_ID,
        "profile_name": CQP_GOP1_PROFILE,
        "tuning": CQP_TUNING,
        "qp": qp,
    }


LOSSLESS_RULE_ID = "palette.orange_crop_lossless.v1"


def classify_crop_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Lossless or exactly constant-QP; anything else is refused.

    For training sources that previously read crops without any compression
    check. Lossless means the summary's top-level ``tuning`` is "lossless"
    and its encoder block, when present, does not say otherwise. Every other
    crop must satisfy ``palette.orange_crop_cqp_gop1.v1``.
    """

    metadata = summary.get("video_metadata")
    encoder = metadata.get("encoder") if isinstance(metadata, Mapping) else None
    encoder_tuning = encoder.get("tuning") if isinstance(encoder, Mapping) else None
    if summary.get("tuning") == "lossless":
        if encoder_tuning not in (None, "lossless"):
            raise CropCompressionRefusal(
                f"crop summary tuning 'lossless' contradicts encoder tuning {encoder_tuning!r}"
            )
        return {"rule": LOSSLESS_RULE_ID, "tuning": "lossless", "lossless": True}
    return {**cqp_gop1_compression(summary), "lossless": False}


def start_snapshot_crop_qp(snapshot: Mapping[str, Any]) -> int:
    """The crop QP Orange's recorder received, from the sealed start snapshot."""

    session = snapshot.get("session")
    effective = session.get("effective_configuration") if isinstance(session, Mapping) else None
    environment = effective.get("environment") if isinstance(effective, Mapping) else None
    entry = environment.get(START_SNAPSHOT_QP_ENV) if isinstance(environment, Mapping) else None
    value = entry.get("value") if isinstance(entry, Mapping) else None
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise CropCompressionRefusal(
            "start snapshot session.effective_configuration.environment."
            f"{START_SNAPSHOT_QP_ENV}.value is not a decimal string: {value!r}"
        )
    return int(value)


def require_start_snapshot_qp(snapshot: Mapping[str, Any], qp: int) -> None:
    sealed = start_snapshot_crop_qp(snapshot)
    if sealed != qp:
        raise CropCompressionRefusal(
            f"crop summary QP {qp} disagrees with the start snapshot's "
            f"{START_SNAPSHOT_QP_ENV} {sealed}"
        )


__all__ = [
    "CQP_GOP1_PROFILE",
    "CQP_GOP1_RULE_ID",
    "CropCompressionRefusal",
    "LOSSLESS_RULE_ID",
    "classify_crop_summary",
    "cqp_gop1_compression",
    "require_start_snapshot_qp",
    "start_snapshot_crop_qp",
]
