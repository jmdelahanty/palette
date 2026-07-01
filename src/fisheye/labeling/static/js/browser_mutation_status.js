
    let latestMutationSupportReference = "";
    function renderMutationSupportReference(text) {
      if (!text) return;
      const target = document.getElementById("operator-support");
      if (!target) return;
      const esc = (typeof escapeSupportText === "function") ? escapeSupportText : (value) => String(value == null ? "" : value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;");
      const personalQueueHref = esc((typeof sessionReturnHref === "function") ? sessionReturnHref("dataset-queue", "/my-datasets") : "/my-datasets");
      const personalWorkHref = esc((typeof sessionReturnHref === "function") ? sessionReturnHref("work-dashboard", "/my-work") : "/my-work");
      target.className = "operator-support active";
      target.innerHTML =
        "<details>" +
        "<summary>Operator support reference</summary>" +
        "<p>This operator support reference is only needed if the operator asks for audit details; give audit event id and server target from the block below. Return to <a href=\"" + personalQueueHref + "\">your personalized dataset queue</a> or <a href=\"" + personalWorkHref + "\">your personalized work dashboard</a> before reopening stale or superseded work.</p>" +
        "<pre>" + esc(text) + "</pre>" +
        "<button type=\"button\" onclick=\"copySessionSupport(this)\">Copy support details</button>" +
        "</details>";
    }
    function renderMutationSupportReferenceSoon(text) {
      if (!text) return;
      window.setTimeout(() => renderMutationSupportReference(text), 0);
    }
    function setMutationSupportReference(result, mutation, eventId, eventType, target) {
      const queue = (result && result.post_completion_queue) || {};
      const context = (mutation && mutation.authorization_context) || (result && result.authorization_context) || {};
      const returnExpectedUser = String(
        mutation.return_expected_user ||
        context.return_expected_user ||
        queue.return_expected_user ||
        queue.expected_user ||
        ""
      );
      const returnPersonalDatasetQueueUrl = String(
        mutation.return_personal_dataset_queue_url ||
        context.return_personal_dataset_queue_url ||
        queue.return_personal_dataset_queue_url ||
        queue.expected_user_personal_dataset_queue_url ||
        queue.personalized_labeler_entry_url ||
        queue.next_labeler_url ||
        ""
      );
      const returnPersonalWorkUrl = String(
        mutation.return_personal_work_url ||
        context.return_personal_work_url ||
        queue.return_personal_work_url ||
        queue.expected_user_personal_work_url ||
        ""
      );
      latestMutationSupportReference = [
        "audit_event_id=" + eventId,
        eventType ? "audit_event_type=" + eventType : "",
        mutation.task_id ? "task_id=" + mutation.task_id : "",
        mutation.recording_id ? "recording_id=" + mutation.recording_id : "",
        "server_target=" + target,
        mutation.browser_label_write_target ? "browser_label_write_target=" + mutation.browser_label_write_target : "",
        mutation.label_mutation_target_kind ? "label_mutation_target_kind=" + mutation.label_mutation_target_kind : "",
        mutation.browser_writes_csv_or_handoff_files !== undefined ? "browser_writes_csv_or_handoff_files=" + mutation.browser_writes_csv_or_handoff_files : "",
        mutation.browser_has_direct_zarr_write_authority !== undefined ? "browser_has_direct_zarr_write_authority=" + mutation.browser_has_direct_zarr_write_authority : "",
        mutation.handoff_artifacts_are_metadata_only !== undefined ? "handoff_artifacts_are_metadata_only=" + mutation.handoff_artifacts_are_metadata_only : "",
        returnExpectedUser ? "return_expected_user=" + returnExpectedUser : "",
        returnPersonalDatasetQueueUrl ? "return_personal_dataset_queue_url=" + returnPersonalDatasetQueueUrl : "",
        returnPersonalDatasetQueueUrl ? "return_personal_dataset_queue_expected_user_guarded=" + String(returnPersonalDatasetQueueUrl.includes("expected_user=")) : "",
        returnPersonalWorkUrl ? "return_personal_work_url=" + returnPersonalWorkUrl : "",
        returnPersonalWorkUrl ? "return_personal_work_expected_user_guarded=" + String(returnPersonalWorkUrl.includes("expected_user=")) : ""
      ].filter(Boolean).join("\n");
      const button = document.getElementById("copy-mutation-support-reference");
      if (button) {
        button.hidden = false;
        button.textContent = "Copy support reference";
      }
    }
    function copyMutationSupportReference(button) {
      const text = latestMutationSupportReference;
      if (!text) return;
      const markCopied = () => {
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = "Copy support reference"; }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(markCopied).catch(() => {
          const textarea = document.createElement("textarea");
          textarea.value = text;
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          textarea.remove();
          markCopied();
        });
        return;
      }
      const textarea = document.createElement("textarea");
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
      markCopied();
    }
    function mutationStatusSuffix(result) {
      const mutation = (result && result.mutation) || {};
      const eventId = String(mutation.audit_event_id || "");
      const eventType = String(mutation.audit_event_type || "");
      const target = String(mutation.data_plane_write_target || "server-owned assigned task Zarr scope");
      if (!eventId) return "";
      setMutationSupportReference(result, mutation, eventId, eventType, target);
      renderMutationSupportReferenceSoon(latestMutationSupportReference);
      return " Operator support reference available below.";
    }
    function postCompletionQueueUrl(result) {
      const queue = (result && result.post_completion_queue) || {};
      return String(
        queue.next_labeler_url ||
        queue.expected_user_personal_dataset_queue_url ||
        queue.personalized_labeler_entry_url ||
        queue.preferred_labeler_entry_url ||
        queue.expected_user_dataset_queue_url ||
        "/my-datasets"
      );
    }
    function postCompletionStatusText(result) {
      const queue = (result && result.post_completion_queue) || {};
      const completion = queue.labeler_work_completion || result.labeler_work_completion || {};
      const status = String(completion.status || "");
      if (status === "complete") return "Task marked complete. All assigned labeling work is complete.";
      if (status === "waiting") return "Task marked complete. Returning to your datasets waiting queue.";
      if (status === "blocked") return "Task marked complete. Returning to your queue; operator action is required before more labeling.";
      if (status === "unassigned") return "Task marked complete. No active assignments remain.";
      return "Task marked complete. Returning to your datasets waiting queue.";
    }
    function handleTaskCompletionSuccess(result) {
      setStatus(postCompletionStatusText(result));
      const nextUrl = postCompletionQueueUrl(result);
      window.setTimeout(() => { window.location.href = nextUrl; }, 350);
    }
