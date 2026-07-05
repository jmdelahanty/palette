"""Post-completion labeler queue metadata helpers."""

from __future__ import annotations

from typing import Mapping

from .assignment_store import LabelingStore
from .admin_dashboard import _store_consistency_report
from .web_auth import (
    DASHBOARD_PATH,
    DATASET_QUEUE_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _dashboard_url_for_expected_user,
)
from .web_policy import LABELING_HOME_PATH, PERSONAL_WORK_PATH
from .work_queue import (
    _add_direct_start_contracts_to_work_tasks,
    _add_work_summary_fields,
    _labeler_work_completion_fields,
)


def _post_completion_queue_metadata(
    store: LabelingStore,
    *,
    user: str,
    expected_user: str | None = None,
) -> dict[str, object]:
    resolved_user = str(user or "").strip()
    expected = str(expected_user or resolved_user).strip()
    guarded_user = expected or resolved_user
    check_report = _store_consistency_report(store)
    reassignment_session_safety = check_report.get("reassignment_session_safety", {})
    work = store.task_summary_for_user(resolved_user, include_completed=True)
    work["include_completed"] = True
    work["expected_user"] = guarded_user
    work["labeler_landing_page_path"] = "/"
    work["labeling_home_page_path"] = LABELING_HOME_PATH
    work["dashboard_path"] = DASHBOARD_PATH
    work["dataset_queue_page_path"] = DATASET_QUEUE_PATH
    work["personal_work_page_path"] = PERSONAL_WORK_PATH
    work["personal_dataset_queue_page_path"] = PERSONAL_DATASET_QUEUE_PATH
    work["expected_user_labeler_landing_url"] = _dashboard_url_for_expected_user("/", guarded_user)
    work["expected_user_labeling_home_url"] = _dashboard_url_for_expected_user(
        LABELING_HOME_PATH,
        guarded_user,
    )
    work["expected_user_dashboard_url"] = _dashboard_url_for_expected_user(DASHBOARD_PATH, guarded_user)
    work["expected_user_dataset_queue_url"] = _dashboard_url_for_expected_user(DATASET_QUEUE_PATH, guarded_user)
    work["expected_user_personal_work_url"] = _dashboard_url_for_expected_user(PERSONAL_WORK_PATH, guarded_user)
    work["expected_user_personal_dataset_queue_url"] = _dashboard_url_for_expected_user(
        PERSONAL_DATASET_QUEUE_PATH,
        guarded_user,
    )
    work["preferred_labeler_entrypoint"] = "personal_datasets_waiting_queue"
    work["preferred_labeler_entry_url"] = work["expected_user_personal_dataset_queue_url"]
    work["personalized_labeler_entrypoint"] = "personal_datasets_waiting_queue"
    work["personalized_labeler_entry_url"] = work["expected_user_personal_dataset_queue_url"]
    _add_work_summary_fields(
        work,
        reassignment_session_safety=(
            reassignment_session_safety if isinstance(reassignment_session_safety, Mapping) else {}
        ),
    )
    _add_direct_start_contracts_to_work_tasks(
        work,
        expected_user=guarded_user,
        reassignment_session_safety=(
            reassignment_session_safety if isinstance(reassignment_session_safety, Mapping) else {}
        ),
    )
    completion = (
        work.get("labeler_work_completion")
        if isinstance(work.get("labeler_work_completion"), Mapping)
        else {}
    )
    dataset_queue_state = (
        work.get("dataset_queue_state")
        if isinstance(work.get("dataset_queue_state"), Mapping)
        else {}
    )
    direct_start_summary = (
        work.get("direct_browser_start_contract_summary")
        if isinstance(work.get("direct_browser_start_contract_summary"), Mapping)
        else {}
    )
    next_labeler_action = str(
        completion.get("labeler_action")
        or work.get("labeler_action")
        or "open_dataset_queue"
    )
    next_labeler_url = str(
        work.get("expected_user_personal_dataset_queue_url")
        or work.get("expected_user_dataset_queue_url")
        or PERSONAL_DATASET_QUEUE_PATH
    )
    return_expected_user = guarded_user
    return_personal_dataset_queue_url = str(
        work.get("expected_user_personal_dataset_queue_url")
        or next_labeler_url
        or PERSONAL_DATASET_QUEUE_PATH
    )
    return_personal_work_url = str(
        work.get("expected_user_personal_work_url")
        or work.get("expected_user_dashboard_url")
        or PERSONAL_WORK_PATH
    )
    return_labeling_home_url = str(
        work.get("expected_user_labeling_home_url")
        or _dashboard_url_for_expected_user(LABELING_HOME_PATH, guarded_user)
        or LABELING_HOME_PATH
    )
    post_completion_queue = {
        "schema": "palette.web_labeling_post_completion_queue.v1",
        "resolved_user": resolved_user,
        "expected_user": guarded_user,
        "expected_user_guard_checked_server_side": True,
        "expected_user_guard_present": bool(expected),
        "expected_user_matches_resolved_user": not expected or expected == resolved_user,
        "include_completed": True,
        "next_labeler_action": next_labeler_action,
        "next_labeler_url": next_labeler_url,
        "next_labeler_url_role": "preferred_queue",
        "return_expected_user": return_expected_user,
        "return_labeling_home_url": return_labeling_home_url,
        "return_labeling_home_expected_user_guarded": bool(
            return_labeling_home_url and "expected_user=" in return_labeling_home_url
        ),
        "return_personal_dataset_queue_url": return_personal_dataset_queue_url,
        "return_personal_dataset_queue_expected_user_guarded": bool(
            return_personal_dataset_queue_url and "expected_user=" in return_personal_dataset_queue_url
        ),
        "return_personal_work_url": return_personal_work_url,
        "return_personal_work_expected_user_guarded": bool(
            return_personal_work_url and "expected_user=" in return_personal_work_url
        ),
        "preferred_labeler_entrypoint": str(work.get("preferred_labeler_entrypoint") or ""),
        "preferred_labeler_entry_url": str(work.get("preferred_labeler_entry_url") or ""),
        "personalized_labeler_entrypoint": str(work.get("personalized_labeler_entrypoint") or ""),
        "personalized_labeler_entry_url": str(work.get("personalized_labeler_entry_url") or ""),
        "expected_user_labeling_home_url": str(work.get("expected_user_labeling_home_url") or ""),
        "expected_user_personal_dataset_queue_url": str(
            work.get("expected_user_personal_dataset_queue_url") or ""
        ),
        "expected_user_dataset_queue_url": str(work.get("expected_user_dataset_queue_url") or ""),
        "expected_user_personal_work_url": str(work.get("expected_user_personal_work_url") or ""),
        "expected_user_dashboard_url": str(work.get("expected_user_dashboard_url") or ""),
        "dataset_queue_state": dict(dataset_queue_state),
        "labeler_work_completion": dict(completion),
        **_labeler_work_completion_fields(completion),
        "labeler_start_ready": bool(work.get("labeler_start_ready")),
        "labeler_start_status": str(work.get("labeler_start_status") or ""),
        "labeler_action": str(work.get("labeler_action") or ""),
        "labeler_start_message": str(work.get("labeler_start_message") or ""),
        "labeler_start_operator_action": str(work.get("labeler_start_operator_action") or ""),
        "progress_summary": (
            dict(work.get("progress_summary"))
            if isinstance(work.get("progress_summary"), Mapping)
            else {}
        ),
        "dataset_queue_summary": (
            dict(work.get("dataset_queue_summary"))
            if isinstance(work.get("dataset_queue_summary"), Mapping)
            else {}
        ),
        "direct_browser_start_contract_summary": dict(direct_start_summary),
        "browser_label_write_target": str(
            direct_start_summary.get("browser_label_write_target") or "training_zarr"
        ),
        "browser_writes_csv_or_handoff_files": bool(
            direct_start_summary.get("browser_writes_csv_or_handoff_files")
        ),
        "browser_writes_handoff_csv": bool(direct_start_summary.get("browser_writes_handoff_csv")),
        "browser_writes_intermediate_csv": bool(
            direct_start_summary.get("browser_writes_intermediate_csv")
        ),
        "browser_has_direct_zarr_write_authority": bool(
            direct_start_summary.get("browser_has_direct_zarr_write_authority")
        ),
    }
    return {
        "post_completion_queue": post_completion_queue,
        "post_completion_next_labeler_action": next_labeler_action,
        "post_completion_next_labeler_url": next_labeler_url,
        "post_completion_next_labeler_url_role": "preferred_queue",
        "post_completion_return_expected_user": return_expected_user,
        "post_completion_return_labeling_home_url": return_labeling_home_url,
        "post_completion_return_labeling_home_expected_user_guarded": bool(
            return_labeling_home_url and "expected_user=" in return_labeling_home_url
        ),
        "post_completion_return_personal_dataset_queue_url": return_personal_dataset_queue_url,
        "post_completion_return_personal_dataset_queue_expected_user_guarded": bool(
            return_personal_dataset_queue_url and "expected_user=" in return_personal_dataset_queue_url
        ),
        "post_completion_return_personal_work_url": return_personal_work_url,
        "post_completion_return_personal_work_expected_user_guarded": bool(
            return_personal_work_url and "expected_user=" in return_personal_work_url
        ),
        "labeler_work_completion": dict(completion),
        **_labeler_work_completion_fields(completion),
    }
