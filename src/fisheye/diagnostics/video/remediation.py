from __future__ import annotations

from pathlib import Path

from .models import Remediation, VideoDiagnosticsReport


def build_remediation(report: VideoDiagnosticsReport) -> list[Remediation]:
    actions: list[Remediation] = []
    if report.overall_status == "pass":
        return actions

    video_path = Path(report.file_info.path)
    stem = video_path.stem
    codec = (report.stream_info.codec or report.container.codec or "").lower()

    if codec == "hevc" and bool(report.container.needs_fix):
        actions.append(
            Remediation(
                issue="Missing HEVC sync samples",
                description="Re-encode the HEVC file with an explicit GOP so MP4 sync-sample entries are rebuilt.",
                command=(
                    f"ffmpeg -i {video_path} -c:v hevc_nvenc -preset p4 -rc vbr -cq 23 "
                    f"-g 60 -bf 0 {stem}_hevc_fixed.mp4"
                ),
            )
        )

    if codec == "hevc" and (
        (report.gop.max_keyframe_interval_s is not None and report.gop.max_keyframe_interval_s > 10.0)
        or bool(report.gop.b_frames_present)
    ):
        actions.append(
            Remediation(
                issue="Sparse keyframes in HEVC",
                description="Re-encode to H.264 with regular keyframes for broad compatibility.",
                command=(
                    f"ffmpeg -i {video_path} -c:v libx264 -preset medium -crf 23 "
                    f"-pix_fmt yuv420p -g 60 {stem}_compatible.mp4"
                ),
            )
        )
        actions.append(
            Remediation(
                issue="Sparse keyframes in HEVC",
                description="If you want to stay on HEVC, re-encode with a fixed GOP and no B-frames.",
                command=(
                    f"ffmpeg -i {video_path} -c:v hevc_nvenc -preset p4 -rc vbr -cq 23 "
                    f"-g 60 -bf 0 {stem}_hevc_fixed.mp4"
                ),
            )
        )

    if report.timing.status == "fail" or any(item.status == "fail" for item in report.decode):
        actions.append(
            Remediation(
                issue="General compatibility",
                description="Re-encode to a conservative H.264 output to salvage a problematic file.",
                command=(
                    f"ffmpeg -i {video_path} -c:v libx264 -preset medium -crf 23 "
                    f"-pix_fmt yuv420p {stem}_fixed.mp4"
                ),
            )
        )

    if report.timing.gap_count > 0:
        actions.append(
            Remediation(
                issue="Recording pipeline investigation",
                description=(
                    "Timestamp gaps suggest dropped or malformed frames. Investigate the writer, "
                    "encoder queue depth, and disk I/O in the acquisition pipeline."
                ),
            )
        )

    return actions
