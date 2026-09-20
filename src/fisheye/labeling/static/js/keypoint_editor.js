    const sessionId = window.PALETTE_KEYPOINT_SESSION_ID || "";
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let payload = null;
    let points = [];
    let activePoint = -1;
    let dragging = false;
    let showText = true;
    let foregroundBusy = false;
    let applyBusy = false;
    let localGeneration = 0;
    let checkpointStateGeneration = 0;
    let foregroundOperationGeneration = 0;
    let uncertainApplyAttempt = null;
    const foregroundControlIds = [
      "nav-prev-button",
      "nav-next-button",
      "save-button",
      "save-next-button",
      "no-keypoints-button",
      "detection-issue-button"
    ];
    const checkpointStateFields = [
      "checkpoint_save_supported",
      "save_mode",
      "unapplied_session_edit_count",
      "active_session_edit_count",
      "applying_session_edit_count",
      "pending_apply_effect_count",
      "selected_session_edit_count",
      "checkpoint_snapshot_sha256",
      "apply_available",
      "resumable_apply_id",
      "resumable_checkpoint_snapshot_sha256"
    ];
    const keypointPalette = [
      "#e4572e",
      "#1479ff",
      "#00a86b",
      "#f0b429",
      "#b83280",
      "#7c3aed",
      "#00a6a6",
      "#d97706",
      "#ef4444",
      "#4b5563",
      "#84cc16",
      "#06b6d4"
    ];

    function keypointColor(index) {
      return keypointPalette[Math.abs(Number(index) || 0) % keypointPalette.length];
    }

    function decodeKeypoints(values) {
      // JSON null is a missing landmark; Number(null) would invent (0, 0).
      return values.map((point) => point.map((v) =>
        typeof v === "number" && Number.isFinite(v) ? v : NaN));
    }

    function setStatus(text, isError=false) {
      const node = document.getElementById("status");
      node.textContent = text;
      node.className = isError ? "status error" : "status";
      if (!isError) clearOperatorSupport();
    }

    function stateCount(state, name) {
      const value = Number(state?.[name]);
      return Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
    }

    function supportsCheckpointSave(state=payload?.state) {
      return state?.checkpoint_save_supported === true;
    }

    function isDirectDeltaSave(state=payload?.state) {
      return state?.save_mode === "immutable_delta_direct_v1";
    }

    function hasPendingCheckpointWork(state=payload?.state) {
      if (!supportsCheckpointSave(state) || isDirectDeltaSave(state)) return false;
      return stateCount(state, "unapplied_session_edit_count") > 0
        || stateCount(state, "active_session_edit_count") > 0
        || stateCount(state, "applying_session_edit_count") > 0
        || Boolean(state?.checkpoint_snapshot_sha256)
        || state?.apply_available === true
        || Boolean(state?.resumable_apply_id)
        || Boolean(uncertainApplyAttempt);
    }

    function recoveredApplyAttempt(state=payload?.state) {
      const applyId = state?.resumable_apply_id;
      const snapshotDigest = state?.resumable_checkpoint_snapshot_sha256;
      return applyId && snapshotDigest
        ? {applyId: String(applyId), snapshotDigest: String(snapshotDigest)}
        : null;
    }

    function canStartApply(state=payload?.state) {
      if (!supportsCheckpointSave(state) || state?.save_mode !== "checkpoint_v1") return false;
      if (uncertainApplyAttempt || recoveredApplyAttempt(state)) return true;
      return state?.apply_available === true && Boolean(state?.checkpoint_snapshot_sha256);
    }

    function refreshControls() {
      foregroundControlIds.forEach((id) => {
        const node = document.getElementById(id);
        if (node) node.disabled = foregroundBusy;
      });
      const pendingGuard = hasPendingCheckpointWork();
      const reviewButton = document.getElementById("set-review-button");
      const reviewSelect = document.getElementById("review-state");
      const completeButton = document.getElementById("complete-task-button");
      if (reviewButton) reviewButton.disabled = foregroundBusy || pendingGuard;
      if (reviewSelect) reviewSelect.disabled = foregroundBusy || pendingGuard;
      if (completeButton) completeButton.disabled = foregroundBusy || pendingGuard;
      const applyButton = document.getElementById("apply-button");
      if (applyButton) applyButton.disabled = foregroundBusy || applyBusy || !canStartApply();
    }

    function beginForeground() {
      if (foregroundBusy) return false;
      foregroundBusy = true;
      foregroundOperationGeneration += 1;
      localGeneration += 1;
      dragging = false;
      refreshControls();
      return true;
    }

    function endForeground() {
      foregroundBusy = false;
      refreshControls();
    }

    function checkpointGuardMessage() {
      return "Apply all saved keypoint checkpoints before changing review status or completing this task.";
    }

    const viewport = createImageCanvasViewport(canvas, draw);

    function decodeRawImage(image) {
      const raw = atob(image.pixels);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      const h = image.shape[0];
      const w = image.shape[1];
      const c = image.shape.length >= 3 ? image.shape[2] : 1;
      const out = new ImageData(w, h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const src = (y * w + x) * c;
          const dst = (y * w + x) * 4;
          if (c === 1) {
            const v = bytes[y * w + x];
            out.data[dst] = v;
            out.data[dst + 1] = v;
            out.data[dst + 2] = v;
          } else {
            out.data[dst] = bytes[src];
            out.data[dst + 1] = bytes[src + 1] ?? bytes[src];
            out.data[dst + 2] = bytes[src + 2] ?? bytes[src];
          }
          out.data[dst + 3] = 255;
        }
      }
      return out;
    }

    function fitView(redraw=true) {
      viewport.fit(redraw);
    }

    function constrainView() {
      viewport.constrain();
    }

    function prepareImageSurface() {
      const image = decodeRawImage(payload.roi_image);
      const sizeChanged = viewport.imageWidth !== image.width || viewport.imageHeight !== image.height;
      viewport.setImageData(image, {resetView: sizeChanged});
    }

    function draw() {
      if (!payload || !viewport.hasImage()) return;
      viewport.drawImage();
      const pointRadius = Math.max(2, Math.min(5, canvas.width / 120));
      const labelFontPx = Math.max(8, Math.min(12, canvas.width / 38));
      const labelOffset = pointRadius + 3;
      ctx.lineWidth = Math.max(1, Math.min(2, canvas.width / 320));
      ctx.font = `${labelFontPx}px Trebuchet MS`;
      points.forEach((point, index) => {
        const x = Number(point[0]);
        const y = Number(point[1]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        const [canvasX, canvasY] = imageToCanvas(x, y);
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, pointRadius, 0, Math.PI * 2);
        ctx.fillStyle = keypointColor(index);
        ctx.fill();
        ctx.lineWidth = index === activePoint ? Math.max(2, Math.min(3, canvas.width / 180)) : Math.max(1, Math.min(2, canvas.width / 320));
        ctx.strokeStyle = index === activePoint ? "#101410" : "white";
        ctx.stroke();
        if (index === activePoint) {
          ctx.beginPath();
          ctx.arc(canvasX, canvasY, pointRadius + 3, 0, Math.PI * 2);
          ctx.strokeStyle = "white";
          ctx.stroke();
        }
        if (showText) {
          ctx.fillStyle = "white";
          ctx.fillText(payload.labels[index] || String(index + 1), canvasX + labelOffset, canvasY - labelOffset);
        }
      });
    }

    function renderPoints() {
      const rows = points.map((point, index) => {
        const x = Number(point[0]);
        const y = Number(point[1]);
        const label = payload.labels[index] || String(index + 1);
        const marker = index === activePoint ? "▶ " : "";
        const color = keypointColor(index);
        const coordinates = Number.isFinite(x) && Number.isFinite(y) ? `${x.toFixed(1)}, ${y.toFixed(1)}` : "missing";
        return `<div class="point-row" role="button" tabindex="0" data-point-index="${index}"><b><span style="display:inline-block;width:0.75em;height:0.75em;border-radius:999px;background:${color};margin-right:0.4em;border:1px solid rgba(0,0,0,.24);"></span>${marker}${label}</b><span>${coordinates}</span></div>`;
      }).join("");
      document.getElementById("points").innerHTML = rows;
    }

    function renderSummary() {
      const state = payload.state || {};
      const immutableDeltaReview = isDirectDeltaSave(state)
        || (Boolean(state.immutable_base) && state.edit_storage === "delta_generation");
      const recoveredReview = Boolean(state.recovered_roi_only);
      const mutableReviewControls = document.getElementById("mutable-review-controls");
      const immutableReviewNote = document.getElementById("immutable-delta-review-note");
      if (mutableReviewControls) mutableReviewControls.hidden = immutableDeltaReview || recoveredReview;
      if (immutableReviewNote) {
        immutableReviewNote.hidden = !immutableDeltaReview && !recoveredReview;
        if (recoveredReview) immutableReviewNote.textContent = "Recovered training labels are saved per row. Registry approval and export are separate steps.";
      }
      const editStorage = immutableDeltaReview
        ? `delta ${state.delta_run || ""}/${state.delta_generation || ""}`
        : "mutable run";
      const checkpointSupported = supportsCheckpointSave(state);
      const directDeltaSave = isDirectDeltaSave(state);
      const knownSaveMode = checkpointSupported || directDeltaSave;
      const unappliedCount = stateCount(state, "unapplied_session_edit_count");
      const activeCount = stateCount(state, "active_session_edit_count");
      const applyingCount = stateCount(state, "applying_session_edit_count");
      const pendingApplyEffectCount = stateCount(state, "pending_apply_effect_count");
      const finishApply = pendingApplyEffectCount > 0
        && Boolean(state.resumable_apply_id)
        && Boolean(state.resumable_checkpoint_snapshot_sha256);
      const checkpointStatus = document.getElementById("checkpoint-status");
      if (checkpointStatus) {
        checkpointStatus.hidden = !knownSaveMode;
        checkpointStatus.textContent = directDeltaSave
          ? "Save mode: direct immutable delta; each Save is applied immediately to the task delta."
          : finishApply
            ? "Labels are applied; finish recording this Apply."
            : `${unappliedCount} saved checkpoint${unappliedCount === 1 ? "" : "s"} pending Apply (${activeCount} ready, ${applyingCount} applying).`;
      }
      const applyControls = document.getElementById("apply-controls");
      if (applyControls) {
        applyControls.hidden = !checkpointSupported || state.save_mode !== "checkpoint_v1";
      }
      const directSaveNote = document.getElementById("immutable-direct-save-note");
      if (directSaveNote) directSaveNote.hidden = !directDeltaSave;
      const applyButton = document.getElementById("apply-button");
      if (applyButton) applyButton.textContent = finishApply
        ? "Finish Apply"
        : "Apply saved checkpoints";
      const applyHelp = document.getElementById("apply-help");
      if (applyHelp) applyHelp.textContent = finishApply
        ? "The labels are already applied. Finish Apply before review approval or task completion."
        : "Apply writes the saved snapshot under the exclusive canonical writer. You can keep reviewing other rows while it runs.";
      const saveButton = document.getElementById("save-button");
      const saveNextButton = document.getElementById("save-next-button");
      if (saveButton) saveButton.textContent = !knownSaveMode
        ? "Save"
        : directDeltaSave ? "Save direct delta" : "Save checkpoint";
      if (saveNextButton) saveNextButton.textContent = !knownSaveMode
        ? "Save + next"
        : directDeltaSave ? "Save direct delta + next" : "Save checkpoint + next";
      document.getElementById("summary").innerHTML = `
        <p><b>ROI</b> ${payload.roi_idx} / <b>${payload.frame_index_domain === "legacy_training_sample_row" ? "source training row" : "frame"}</b> ${payload.frame_idx}</p>
        <p><b>Position</b> ${state.position + 1} of ${state.total}</p>
        <p><b>Run</b> ${state.refined_run || ""}</p>
        <p><b>Edit storage</b> ${editStorage}</p>
        <p><b>Reason</b> ${payload.reason || ""}</p>
      `;
      refreshControls();
    }

    function adoptRoi(roi, responseState=null) {
      const previousActivePoint = activePoint;
      const nextState = responseState || roi?.state || {};
      payload = {...roi, state: nextState};
      points = decodeKeypoints(Array.isArray(payload.points) ? payload.points : []);
      const firstMissing = points.findIndex((point) =>
        point.some((value) => !Number.isFinite(value)));
      activePoint = firstMissing >= 0
        ? firstMissing
        : previousActivePoint >= 0 && previousActivePoint < points.length
          ? previousActivePoint
          : -1;
      prepareImageSurface();
      renderSummary();
      renderPoints();
      draw();
      localGeneration += 1;
      checkpointStateGeneration += 1;
    }

    function mergeCheckpointState(state) {
      if (!payload || !state) return;
      const merged = {...(payload.state || {})};
      checkpointStateFields.forEach((name) => {
        if (Object.prototype.hasOwnProperty.call(state, name)) merged[name] = state[name];
      });
      payload.state = merged;
      checkpointStateGeneration += 1;
      renderSummary();
    }

    function captureLocalState() {
      return {
        generation: localGeneration,
        checkpointStateGeneration,
        foregroundOperationGeneration,
        targetToken: payload?.state?.target_token
      };
    }

    function canAdoptResponseRoi(captured, response) {
      return !response?.current_target_changed
        && localGeneration === captured.generation
        && payload?.state?.target_token === captured.targetToken;
    }

    function canMergeApplyState(captured) {
      return checkpointStateGeneration === captured.checkpointStateGeneration
        && foregroundOperationGeneration === captured.foregroundOperationGeneration;
    }

    async function api(path, options={}) {
      const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/keypoints${path}`, options);
      const data = await readApiPayload(response);
      if (!response.ok || !data.ok) {
        const failure = apiFailure(response, data, "session_request_failed");
        failure.apiData = data;
        throw failure;
      }
      return data;
    }

    async function loadCurrent() {
      try {
        const current = await api("/roi/current");
        adoptRoi(current, current.state);
        setStatus("Loaded.");
        return true;
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
        return false;
      }
    }

    async function nav(delta) {
      if (!beginForeground()) return;
      try {
        await api("/nav", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({delta})
        });
        await loadCurrent();
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        endForeground();
      }
    }

    function validateSavePoints() {
      const expectedCount = Array.isArray(payload?.labels) ? payload.labels.length : points.length;
      const allFinite = points.length > 0
        && points.length === expectedCount
        && points.every((point) => Array.isArray(point)
          && point.length >= 2
          && Number.isFinite(point[0])
          && Number.isFinite(point[1]));
      if (!allFinite) throw new Error("Place every missing landmark before saving this row.");
    }

    function saveStatusText(response, savedRoiIdx, saveMode) {
      const result = response?.result || {};
      const state = response?.state || response?.roi?.state || payload?.state || {};
      const count = stateCount(state, "unapplied_session_edit_count");
      if (saveMode === "checkpoint_v1" && result.saved === true && result.applied === false) {
        return `Checkpoint saved for ROI ${savedRoiIdx}. ${count} checkpoint${count === 1 ? "" : "s"} pending Apply; canonical keypoints were not changed.`;
      }
      if (saveMode === "immutable_delta_direct_v1") {
        return `Saved ROI ${savedRoiIdx} directly to the immutable delta. The immutable base was not changed.`;
      }
      return `Saved ROI ${savedRoiIdx}.` + mutationStatusSuffix(response);
    }

    async function save(advance) {
      if (!beginForeground()) return;
      const captured = captureLocalState();
      const savedRoiIdx = payload?.roi_idx;
      const saveMode = supportsCheckpointSave() || isDirectDeltaSave()
        ? payload?.state?.save_mode
        : null;
      try {
        validateSavePoints();
        const result = await api("/save", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({points, advance, target_token: payload?.state?.target_token})
        });
        if (result.roi && canAdoptResponseRoi(captured, result)) {
          adoptRoi(result.roi, result.state || result.roi.state);
        } else if (result.roi) {
          mergeCheckpointState(result.state || result.roi.state);
        } else if (canAdoptResponseRoi(captured, result)) {
          // Compatibility with servers predating folded Save responses.
          await loadCurrent();
        } else {
          mergeCheckpointState(result.state);
        }
        setStatus(saveStatusText(result, result.result?.roi_idx ?? savedRoiIdx, saveMode));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        endForeground();
      }
    }

    async function action(name) {
      if (!beginForeground()) return;
      const captured = captureLocalState();
      const savedRoiIdx = payload?.roi_idx;
      const saveMode = supportsCheckpointSave() || isDirectDeltaSave()
        ? payload?.state?.save_mode
        : null;
      try {
        const result = await api("/action", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({action: name, advance: true, target_token: payload?.state?.target_token})
        });
        if (result.roi && canAdoptResponseRoi(captured, result)) {
          adoptRoi(result.roi, result.state || result.roi.state);
        } else if (result.roi) {
          mergeCheckpointState(result.state || result.roi.state);
        } else if (canAdoptResponseRoi(captured, result)) {
          // Compatibility with servers predating folded action responses.
          await loadCurrent();
        } else {
          mergeCheckpointState(result.state);
        }
        const statusText = saveMode
          ? saveStatusText(result, result.result?.roi_idx ?? savedRoiIdx, saveMode)
          : "Applied " + name + "." + mutationStatusSuffix(result);
        setStatus(statusText);
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        endForeground();
      }
    }

    async function setReviewStatus() {
      if (hasPendingCheckpointWork()) {
        setStatus(checkpointGuardMessage(), true);
        return;
      }
      if (!beginForeground()) return;
      try {
        const reviewState = document.getElementById("review-state").value;
        const result = await api("/review-status", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({state: reviewState, target_token: payload?.state?.target_token})
        });
        await loadCurrent();
        setStatus(`Review state set to ${reviewState}.` + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        endForeground();
      }
    }

    async function completeTask() {
      if (hasPendingCheckpointWork()) {
        setStatus(checkpointGuardMessage(), true);
        return;
      }
      if (!beginForeground()) return;
      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/complete`, {method: "POST"});
        const data = await readApiPayload(response);
        if (!response.ok || !data.ok) throw apiFailure(response, data, "task_complete_failed");
        handleTaskCompletionSuccess(data);
      } catch (error) {
        showOperatorSupport(error, "task_complete_failed");
      } finally {
        endForeground();
      }
    }

    function newApplyId() {
      if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return window.crypto.randomUUID();
      }
      return `keypoint-apply-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    }

    function chooseApplyAttempt() {
      if (uncertainApplyAttempt) return uncertainApplyAttempt;
      const recovered = recoveredApplyAttempt();
      if (recovered) return recovered;
      const snapshotDigest = payload?.state?.checkpoint_snapshot_sha256;
      if (!snapshotDigest || payload?.state?.apply_available !== true) return null;
      return {applyId: newApplyId(), snapshotDigest: String(snapshotDigest)};
    }

    function applyStatusText(response, effectiveState=payload?.state) {
      const result = response?.result || {};
      const count = Number(result.applied_checkpoint_count
        ?? (Array.isArray(result.rows) ? result.rows.length : 0));
      const appliedCount = Number.isFinite(count) ? count : 0;
      const remaining = stateCount(effectiveState, "unapplied_session_edit_count");
      const replay = result.already_applied === true ? " (confirmed from the earlier request)" : "";
      return `Applied ${appliedCount} saved checkpoint${appliedCount === 1 ? "" : "s"} to canonical keypoints${replay}. ${remaining} pending Apply.`;
    }

    async function applyCheckpoints() {
      if (foregroundBusy || applyBusy) return;
      const attempt = chooseApplyAttempt();
      if (!attempt) {
        setStatus("No saved checkpoint snapshot is ready to apply.", true);
        refreshControls();
        return;
      }
      const captured = captureLocalState();
      applyBusy = true;
      refreshControls();
      setStatus("Applying saved checkpoints in the background. You can keep reviewing other rows.");
      try {
        const response = await api("/apply", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            apply_id: attempt.applyId,
            checkpoint_snapshot_sha256: attempt.snapshotDigest,
            target_token: captured.targetToken
          })
        });
        if (response.result?.apply_id !== attempt.applyId
            || response.result?.checkpoint_snapshot_sha256 !== attempt.snapshotDigest) {
          throw new Error("Apply response did not match the requested ID and checkpoint snapshot.");
        }
        uncertainApplyAttempt = null;
        if (response.roi && canAdoptResponseRoi(captured, response)) {
          adoptRoi(response.roi, response.state || response.roi.state);
        } else if (canMergeApplyState(captured)) {
          // Applying is intentionally a background operation. Only its
          // checkpoint lifecycle fields may update a row edited since launch.
          mergeCheckpointState(response.state);
        }
        setStatus(applyStatusText(response, payload?.state));
      } catch (error) {
        const failureData = error?.apiData;
        const freshSnapshotRequired = failureData?.apply_retry_disposition === "fresh_snapshot_required"
          && failureData?.safe_prewrite_rejection === true
          && failureData?.retain_apply_id === false;
        if (freshSnapshotRequired) {
          uncertainApplyAttempt = null;
          if (canMergeApplyState(captured)) mergeCheckpointState(failureData.state);
        } else {
          // The server may have committed before the response was lost.
          // Retrying the exact ID/digest pair makes that outcome idempotent.
          uncertainApplyAttempt = attempt;
        }
        showOperatorSupport(error, "session_request_failed");
        setStatus(freshSnapshotRequired
          ? "The saved checkpoint snapshot changed before Apply wrote anything. Start Apply again with the current snapshot."
          : "Apply response was not confirmed. Retry Apply to reuse the same request ID and snapshot.", true);
      } finally {
        applyBusy = false;
        refreshControls();
      }
    }

    function canvasPoint(event) {
      return viewport.canvasPoint(event);
    }

    function canvasToImage(x, y) {
      return viewport.canvasToImage(x, y);
    }

    function imageToCanvas(x, y) {
      return viewport.imageToCanvas(x, y);
    }

    function nearestPoint(x, y) {
      let best = -1;
      let bestD = Infinity;
      points.forEach((point, index) => {
        const dx = Number(point[0]) - x;
        const dy = Number(point[1]) - y;
        const d = dx * dx + dy * dy;
        if (d < bestD) {
          bestD = d;
          best = index;
        }
      });
      const hitRadius = Math.max(8, canvas.width / 28) / viewport.view.scale;
      return bestD <= Math.pow(hitRadius, 2) ? best : -1;
    }

    function setActivePoint(index) {
      if (foregroundBusy || !points.length) return;
      activePoint = Math.max(0, Math.min(points.length - 1, index));
      renderPoints();
      draw();
      const label = payload?.labels?.[activePoint] || String(activePoint + 1);
      setStatus(`Selected ${label}.`);
    }

    function cycleActivePoint(delta) {
      if (foregroundBusy || !points.length) return;
      const current = activePoint >= 0 ? activePoint : 0;
      setActivePoint((current + delta + points.length) % points.length);
    }

    function resetPoints() {
      if (foregroundBusy || !payload) return;
      points = decodeKeypoints(payload.points);
      activePoint = points.findIndex((p) => p.some((v) => !Number.isFinite(v)));
      localGeneration += 1;
      renderPoints();
      draw();
      setStatus("Reset points from current ROI.");
    }

    canvas.addEventListener("mousedown", (event) => {
      event.preventDefault();
      if (foregroundBusy) return;
      if (viewport.beginPan(event)) return;
      const [canvasX, canvasY] = canvasPoint(event);
      const [x, y] = canvasToImage(canvasX, canvasY);
      const nearest = nearestPoint(x, y);
      const placingMissing = activePoint >= 0 && points[activePoint].some((v) => !Number.isFinite(v));
      if (nearest >= 0 && !placingMissing) {
        activePoint = nearest;
      }
      dragging = activePoint >= 0;
      if (dragging) {
        points[activePoint] = [Math.max(0, Math.min(viewport.imageWidth - 1, x)), Math.max(0, Math.min(viewport.imageHeight - 1, y))];
        localGeneration += 1;
        renderPoints();
        draw();
      }
    });
    canvas.addEventListener("mousemove", (event) => {
      if (foregroundBusy) return;
      if (viewport.panMove(event)) return;
      const [canvasX, canvasY] = canvasPoint(event);
      if (!dragging || activePoint < 0) return;
      const [x, y] = canvasToImage(canvasX, canvasY);
      points[activePoint] = [Math.max(0, Math.min(viewport.imageWidth - 1, x)), Math.max(0, Math.min(viewport.imageHeight - 1, y))];
      localGeneration += 1;
      renderPoints();
      draw();
    });
    window.addEventListener("mouseup", () => { dragging = false; viewport.endPan(); });
    function selectPointRow(event) {
      if (foregroundBusy) return;
      if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
      const row = event.target.closest("[data-point-index]");
      if (!row) return;
      event.preventDefault();
      setActivePoint(Number(row.dataset.pointIndex));
    }
    document.getElementById("points").addEventListener("click", selectPointRow);
    document.getElementById("points").addEventListener("keydown", selectPointRow);
    canvas.addEventListener("wheel", viewport.handleWheel, {passive: false});
    window.addEventListener("keydown", (event) => {
      const targetTag = event.target?.tagName?.toLowerCase();
      if (targetTag === "input" || targetTag === "textarea" || targetTag === "select") return;
      if (foregroundBusy) {
        event.preventDefault();
        return;
      }
      if (event.key === "f" || event.key === "F") { event.preventDefault(); fitView(); return; }
      if (event.key === "n") { event.preventDefault(); nav(1); return; }
      if (event.key === "p") { event.preventDefault(); nav(-1); return; }
      if (event.key === "s") { event.preventDefault(); save(false); return; }
      if (event.key === "S") { event.preventDefault(); save(true); return; }
      if (event.key === "t" || event.key === "T") {
        event.preventDefault();
        showText = !showText;
        draw();
        setStatus(showText ? "Keypoint labels shown." : "Keypoint labels hidden.");
        return;
      }
      if (event.key === "[") { event.preventDefault(); cycleActivePoint(-1); return; }
      if (event.key === "]") { event.preventDefault(); cycleActivePoint(1); return; }
      const digit = Number.parseInt(event.key, 10);
      if (Number.isInteger(digit) && digit >= 1 && digit <= 9) {
        event.preventDefault();
        const index = digit - 1;
        if (index < points.length) setActivePoint(index);
        return;
      }
      if (event.key === "0") {
        event.preventDefault();
        if (points.length >= 10) setActivePoint(9);
        return;
      }
      if (event.key === "r" || event.key === "R") { event.preventDefault(); resetPoints(); return; }
      if (event.key === "x" || event.key === "X") { event.preventDefault(); action("mark_no_keypoints"); return; }
    });
    loadCurrent();
