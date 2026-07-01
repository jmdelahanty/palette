
    async function readApiPayload(response) {
      try {
        return await response.json();
      } catch (error) {
        return {
          ok: false,
          error: "invalid_json_response",
          details: "The server returned a non-JSON response with status " + (response.status || "unknown") + "."
        };
      }
    }

    function apiFailure(response, data, fallbackError) {
      const errorCode = String((data && data.error) || fallbackError);
      const details = String((data && (data.details || data.message)) || response.statusText || errorCode);
      const failure = new Error(details);
      failure.operatorSupport = {
        error: errorCode,
        status: response.status || "unknown",
        details: details,
        session_closure_event: data && data.session_closure_event ? data.session_closure_event : null
      };
      return failure;
    }

    function normalizedFailure(error, fallbackError) {
      if (error && error.operatorSupport) return error;
      const message = String((error && error.message) || error || fallbackError);
      const failure = new Error(message);
      failure.operatorSupport = {
        error: fallbackError,
        status: "client",
        details: message
      };
      return failure;
    }

    function escapeSupportText(value) {
      return String(value == null ? "" : value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;");
    }

    function clearOperatorSupport() {
      const target = document.getElementById("operator-support");
      if (!target) return;
      target.className = "operator-support";
      target.innerHTML = "";
    }

    function operatorSupportText(error, fallbackError) {
      const failure = normalizedFailure(error, fallbackError);
      const support = failure.operatorSupport || {};
      return [
        "error=" + (support.error || fallbackError),
        "status=" + (support.status || "client"),
        "details=" + (support.details || failure.message || ""),
        support.session_closure_event ? "session_closure_event=" + JSON.stringify(support.session_closure_event) : ""
      ].filter(Boolean).join("\n");
    }

    function copySessionSupport(button) {
      const block = button.closest(".operator-support");
      const text = block && block.querySelector("pre") ? block.querySelector("pre").textContent : "";
      const markCopied = () => {
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = "Copy support details"; }, 1800);
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

    function sessionReturnHref(kind, fallback) {
      const link = document.querySelector(`[data-session-return="${kind}"]`);
      return link ? (link.getAttribute("href") || fallback) : fallback;
    }

    function showOperatorSupport(error, fallbackError) {
      const failure = normalizedFailure(error, fallbackError);
      setStatus(failure.message || fallbackError, true);
      const target = document.getElementById("operator-support");
      if (!target) return;
      const supportText = operatorSupportText(failure, fallbackError);
      const personalQueueHref = escapeSupportText(sessionReturnHref("dataset-queue", "/my-datasets"));
      const personalWorkHref = escapeSupportText(sessionReturnHref("work-dashboard", "/my-work"));
      target.className = "operator-support active";
      target.innerHTML =
        "<details>" +
        "<summary>What to send the operator</summary>" +
        "<p>Stop and send these details to the operator. Return to <a href=\"" + personalQueueHref + "\">your personalized dataset queue</a> or <a href=\"" + personalWorkHref + "\">your personalized work dashboard</a> before reopening stale or superseded work.</p>" +
        "<pre>" + escapeSupportText(supportText) + "</pre>" +
        "<button type=\"button\" onclick=\"copySessionSupport(this)\">Copy support details</button>" +
        "</details>";
    }
