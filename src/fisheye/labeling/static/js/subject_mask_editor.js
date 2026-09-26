    const sessionId = window.PALETTE_SUBJECT_MASK_SESSION_ID || "";
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let payload = null;
    let imageData = null;
    let mask = null;
    let loadedMask = null;
    let maskWidth = 0;
    let maskHeight = 0;
    let maskOverlayCanvas = document.createElement("canvas");
    let maskOverlayCtx = maskOverlayCanvas.getContext("2d");
    let maskOverlayDirty = true;
    let maskOverlayDirtyRect = null;
    let maskView = "overlay";
    let drawScheduled = false;
    let drawing = false;
    let lassoMode = false;
    let lassoDrawing = false;
    let lassoPoints = [];
    let lassoCursor = null;
    let cursorMaskPoint = null;
    let cursorShiftInvert = false;
    let tool = "paint";
    let brushSize = 8;
    let busyAction = false;
    let applyInFlight = false;
    let uncertainApplyId = null;
    let tailRefreshResult = null;
    let foregroundGeneration = 0;
    let effectsPollTimer = null;
    const effectsPollMs = 3000;
    const lassoMinPointStepPx = 2;
    let maskPieceInfo = null;
    let maskPiecesTimer = null;
    let bulkUndoMask = null;
    let bulkUndoLabel = "";

    // Connected pieces of a binary mask. Pixels that touch at an edge or a
    // corner (8-connectivity) are one piece, as in the server's
    // fragmented_subject_body_mask check, so the counts agree.
    function maskPieces(values, width, height) {
      const labels = new Int32Array(width * height);
      const sizes = [0];
      const boxes = [null];
      const stack = new Int32Array(width * height);
      let next = 0;
      for (let start = 0; start < values.length; start++) {
        if (!values[start] || labels[start]) continue;
        next += 1;
        let top = 0;
        stack[top++] = start;
        labels[start] = next;
        let size = 0, x0 = width, y0 = height, x1 = -1, y1 = -1;
        while (top > 0) {
          const idx = stack[--top];
          const x = idx % width, y = (idx - x) / width;
          size += 1;
          if (x < x0) x0 = x; if (x > x1) x1 = x;
          if (y < y0) y0 = y; if (y > y1) y1 = y;
          for (let dy = -1; dy <= 1; dy++) {
            const ny = y + dy;
            if (ny < 0 || ny >= height) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const nx = x + dx;
              if ((dx || dy) && nx >= 0 && nx < width) {
                const n = ny * width + nx;
                if (values[n] && !labels[n]) { labels[n] = next; stack[top++] = n; }
              }
            }
          }
        }
        sizes.push(size);
        boxes.push({x0, y0, x1, y1});
      }
      let main = 0;
      for (let id = 1; id < sizes.length; id++) if (sizes[id] > (sizes[main] || 0)) main = id;
      return {labels, sizes, boxes, main, count: sizes.length - 1};
    }

    // Fill every hole enclosed by the piece under (x, y). Background pixels
    // that cannot reach the image edge without crossing that piece (moving
    // edge-to-edge, 4-connectivity) are holes. Returns the pixels filled, or
    // -1 when (x, y) is not on the mask.
    function fillPieceHoles(values, width, height, x, y) {
      const at = y * width + x;
      if (!values[at]) return -1;
      const pieces = maskPieces(values, width, height);
      const piece = pieces.labels[at];
      const reached = new Uint8Array(width * height);
      const stack = new Int32Array(width * height);
      let top = 0;
      const seed = (idx) => {
        if (!reached[idx] && pieces.labels[idx] !== piece) { reached[idx] = 1; stack[top++] = idx; }
      };
      for (let xx = 0; xx < width; xx++) { seed(xx); seed((height - 1) * width + xx); }
      for (let yy = 0; yy < height; yy++) { seed(yy * width); seed(yy * width + width - 1); }
      while (top > 0) {
        const idx = stack[--top];
        const xx = idx % width, yy = (idx - xx) / width;
        if (xx > 0) seed(idx - 1);
        if (xx < width - 1) seed(idx + 1);
        if (yy > 0) seed(idx - width);
        if (yy < height - 1) seed(idx + width);
      }
      let filled = 0;
      for (let i = 0; i < values.length; i++) {
        if (!reached[i] && pieces.labels[i] !== piece) {
          if (!values[i]) filled += 1;
          values[i] = 1;
        }
      }
      return filled;
    }

    // Keep only the largest piece. Returns the pixels removed.
    function removeStrayPieces(values, width, height) {
      const pieces = maskPieces(values, width, height);
      let removed = 0;
      for (let i = 0; i < values.length; i++) {
        if (values[i] && pieces.labels[i] !== pieces.main) { values[i] = 0; removed += 1; }
      }
      return removed;
    }

    function setStatus(text, isError=false) {
      const node = document.getElementById("status");
      node.textContent = text;
      node.className = isError ? "status error" : "status";
      if (!isError) clearOperatorSupport();
    }

    function clearMutationSupportReference() {
      const button = document.getElementById("copy-mutation-support-reference");
      if (!button) return;
      button.hidden = true;
      button.textContent = "Copy support reference";
      if (button.dataset) {
        Object.keys(button.dataset).forEach((key) => delete button.dataset[key]);
      }
    }

    function updateNavButtons() {
      const prev = document.getElementById("nav-prev-button");
      const next = document.getElementById("nav-next-button");
      if (!prev || !next) return;
      const state = payload?.state || {};
      const total = Number(state.total || 0);
      const position = Number(state.position || 0);
      const noPayload = !payload || !Number.isFinite(total) || total <= 0;
      prev.disabled = busyAction || noPayload || position <= 0;
      next.disabled = busyAction || noPayload || position >= total - 1;
    }

    function setBusy(isBusy, text=null) {
      if (isBusy) foregroundGeneration += 1;
      busyAction = Boolean(isBusy);
      document.querySelectorAll("button, select, input").forEach((node) => {
        node.disabled = busyAction;
      });
      updateNavButtons();
      if (text) setStatus(text);
    }

    const viewport = createImageCanvasViewport(canvas, draw);
    canvas.style.cursor = "none";

    function decodeBytes(rawBase64) {
      const raw = atob(rawBase64);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      return bytes;
    }

    function decodeRawImage(image) {
      const bytes = decodeBytes(image.pixels);
      const h = image.shape[0];
      const w = image.shape[1];
      const c = image.shape.length >= 3 ? image.shape[2] : 1;
      const out = new ImageData(w, h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const dst = (y * w + x) * 4;
          if (c === 1) {
            const v = bytes[y * w + x];
            out.data[dst] = v; out.data[dst + 1] = v; out.data[dst + 2] = v;
          } else {
            const src = (y * w + x) * c;
            out.data[dst] = bytes[src];
            out.data[dst + 1] = bytes[src + 1] ?? bytes[src];
            out.data[dst + 2] = bytes[src + 2] ?? bytes[src];
          }
          out.data[dst + 3] = 255;
        }
      }
      return out;
    }

    function decodeMask(maskPayload) {
      const bytes = decodeBytes(maskPayload.pixels);
      maskHeight = maskPayload.shape[0];
      maskWidth = maskPayload.shape[1];
      mask = new Uint8Array(maskWidth * maskHeight);
      for (let i = 0; i < mask.length; i++) mask[i] = bytes[i] > 0 ? 1 : 0;
      loadedMask = mask.slice();
      forgetUndo();
      markMaskOverlayDirty();
    }

    function encodeMaskPayload() {
      let raw = "";
      const chunk = 8192;
      for (let i = 0; i < mask.length; i += chunk) {
        raw += String.fromCharCode.apply(null, mask.subarray(i, i + chunk));
      }
      return {
        shape: [maskHeight, maskWidth],
        dtype: "uint8",
        encoding: "base64_raw",
        pixels: btoa(raw)
      };
    }

    function schedulePieceRefresh() {
      // Recount once per frame, after this frame's edits.
      if (maskPiecesTimer !== null) return;
      maskPiecesTimer = true;
      window.requestAnimationFrame(() => {
        maskPiecesTimer = null;
        refreshPieces();
      });
    }

    function refreshPieces() {
      if (!mask || !maskWidth || !maskHeight) return;
      maskPieceInfo = maskPieces(mask, maskWidth, maskHeight);
      renderPieces();
      scheduleDraw();
    }

    function renderPieces() {
      const target = document.getElementById("mask-pieces");
      const button = document.getElementById("remove-stray-button");
      if (!target) return;
      const info = maskPieceInfo;
      const stray = info ? info.count - (info.count > 0 ? 1 : 0) : 0;
      if (button) button.disabled = stray <= 0;
      if (!info || info.count <= 1) {
        target.textContent = info && info.count === 1 ? "Mask is one piece." : "Mask is empty.";
        target.classList.toggle("warn", false);
        return;
      }
      const straySizes = [];
      for (let id = 1; id < info.sizes.length; id++) if (id !== info.main) straySizes.push(info.sizes[id]);
      straySizes.sort((a, b) => b - a);
      target.textContent = stray + " stray piece" + (stray === 1 ? "" : "s") + " outside the main body ("
        + straySizes.slice(0, 5).join(", ") + (straySizes.length > 5 ? ", …" : "") + " px), circled on the image.";
      target.classList.toggle("warn", true);
    }

    function rememberForUndo(label) {
      bulkUndoMask = mask.slice();
      bulkUndoLabel = label;
      const button = document.getElementById("undo-bulk-button");
      if (button) { button.disabled = false; button.textContent = "Undo " + label; }
    }

    function forgetUndo() {
      bulkUndoMask = null;
      bulkUndoLabel = "";
      const button = document.getElementById("undo-bulk-button");
      if (button) { button.disabled = true; button.textContent = "Undo"; }
    }

    function undoBulkEdit() {
      if (!bulkUndoMask || !mask || bulkUndoMask.length !== mask.length) return;
      mask.set(bulkUndoMask);
      const label = bulkUndoLabel;
      forgetUndo();
      markMaskOverlayDirty();
      scheduleDraw();
      setStatus("Undid " + label + " locally.");
    }

    function removeStrayPiecesAction() {
      if (!mask) return;
      rememberForUndo("stray-piece removal");
      const removed = removeStrayPieces(mask, maskWidth, maskHeight);
      if (!removed) { forgetUndo(); setStatus("No stray pieces to remove."); return; }
      markMaskOverlayDirty();
      scheduleDraw();
      setStatus("Removed " + removed + " stray px locally. Save to persist.");
    }

    function fillHolesAt(event) {
      const [mx, my] = maskPointFromEvent(event);
      rememberForUndo("hole fill");
      const filled = fillPieceHoles(mask, maskWidth, maskHeight, mx, my);
      if (filled < 0) { forgetUndo(); setStatus("Click on a mask piece to fill its holes.", true); return; }
      if (filled === 0) { forgetUndo(); setStatus("That piece has no holes."); return; }
      markMaskOverlayDirty();
      scheduleDraw();
      setStatus("Filled " + filled + " px of holes locally. Save to persist.");
    }

    function markMaskOverlayDirty(rect=null) {
      schedulePieceRefresh();
      const wasFullyDirty = maskOverlayDirty && maskOverlayDirtyRect === null;
      maskOverlayDirty = true;
      if (payload?.tail_crop_border) renderTailBorderStatus();
      if (!rect) {
        maskOverlayDirtyRect = null;
        return;
      }
      if (wasFullyDirty) return;
      const clipped = {
        x0: Math.max(0, Math.min(maskWidth, Math.floor(rect.x0))),
        y0: Math.max(0, Math.min(maskHeight, Math.floor(rect.y0))),
        x1: Math.max(0, Math.min(maskWidth, Math.ceil(rect.x1))),
        y1: Math.max(0, Math.min(maskHeight, Math.ceil(rect.y1)))
      };
      if (clipped.x1 <= clipped.x0 || clipped.y1 <= clipped.y0) return;
      if (maskOverlayDirtyRect === null) {
        maskOverlayDirtyRect = clipped;
        return;
      }
      maskOverlayDirtyRect = {
        x0: Math.min(maskOverlayDirtyRect.x0, clipped.x0),
        y0: Math.min(maskOverlayDirtyRect.y0, clipped.y0),
        x1: Math.max(maskOverlayDirtyRect.x1, clipped.x1),
        y1: Math.max(maskOverlayDirtyRect.y1, clipped.y1)
      };
    }

    function setMaskView(value) {
      maskView = value === "binary" ? "binary" : "overlay";
      document.getElementById("mask-view").value = maskView;
      markMaskOverlayDirty();
      scheduleDraw();
    }

    function rebuildMaskOverlay() {
      if (!mask || !maskWidth || !maskHeight) return;
      const binary = maskView === "binary";
      const resized = maskOverlayCanvas.width !== maskWidth || maskOverlayCanvas.height !== maskHeight;
      if (maskOverlayCanvas.width !== maskWidth) maskOverlayCanvas.width = maskWidth;
      if (maskOverlayCanvas.height !== maskHeight) maskOverlayCanvas.height = maskHeight;
      if (resized || !maskOverlayDirtyRect) {
        const overlay = new ImageData(maskWidth, maskHeight);
        for (let i = 0; i < mask.length; i++) {
          if (!mask[i] && !binary) continue;
          const dst = i * 4;
          const value = mask[i] ? 255 : 0;
          overlay.data[dst] = binary ? value : 0;
          overlay.data[dst + 1] = binary ? value : 200;
          overlay.data[dst + 2] = binary ? value : 148;
          overlay.data[dst + 3] = binary ? 255 : 118;
        }
        maskOverlayCtx.putImageData(overlay, 0, 0);
      } else {
        const x0 = maskOverlayDirtyRect.x0;
        const y0 = maskOverlayDirtyRect.y0;
        const w = maskOverlayDirtyRect.x1 - maskOverlayDirtyRect.x0;
        const h = maskOverlayDirtyRect.y1 - maskOverlayDirtyRect.y0;
        const overlay = new ImageData(w, h);
        for (let yy = 0; yy < h; yy++) {
          for (let xx = 0; xx < w; xx++) {
            const src = (y0 + yy) * maskWidth + (x0 + xx);
            if (!mask[src] && !binary) continue;
            const dst = (yy * w + xx) * 4;
            const value = mask[src] ? 255 : 0;
            overlay.data[dst] = binary ? value : 0;
            overlay.data[dst + 1] = binary ? value : 200;
            overlay.data[dst + 2] = binary ? value : 148;
            overlay.data[dst + 3] = binary ? 255 : 118;
          }
        }
        maskOverlayCtx.putImageData(overlay, x0, y0);
      }
      maskOverlayDirty = false;
      maskOverlayDirtyRect = null;
    }

    function scheduleDraw() {
      if (drawScheduled) return;
      drawScheduled = true;
      window.requestAnimationFrame(() => {
        drawScheduled = false;
        draw();
      });
    }

    function draw() {
      if (!imageData || !mask || !viewport.hasImage()) return;
      if (maskOverlayDirty) rebuildMaskOverlay();
      viewport.drawImage();
      viewport.drawCanvas(maskOverlayCanvas);
      drawStrayPieceRings();
      drawLassoOverlay();
      drawCursorOverlay();
    }

    function maskToCanvasPoint(mx, my) {
      return viewport.imageToCanvas(
        (Number(mx) + 0.5) * viewport.imageWidth / maskWidth,
        (Number(my) + 0.5) * viewport.imageHeight / maskHeight
      );
    }

    function drawStrayPieceRings() {
      const info = maskPieceInfo;
      if (!info || info.count <= 1) return;
      ctx.save();
      ctx.strokeStyle = "#ff4fd8";
      ctx.lineWidth = 2;
      for (let id = 1; id < info.boxes.length; id++) {
        if (id === info.main) continue;
        const box = info.boxes[id];
        const [cx, cy] = maskToCanvasPoint((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2);
        const [ex] = maskToCanvasPoint(box.x1 + 1, box.y1 + 1);
        // At least 10 screen pixels so a one-pixel piece is visible when zoomed out.
        const radius = Math.max(10, Math.abs(ex - cx) + 4);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
    }

    function drawLassoOverlay() {
      if (!lassoPoints.length || !maskWidth || !maskHeight) return;
      ctx.save();
      ctx.strokeStyle = "#fff176";
      ctx.fillStyle = "#fff176";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      lassoPoints.forEach((point, index) => {
        const [x, y] = maskToCanvasPoint(point[0], point[1]);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      if (lassoMode && lassoCursor) {
        const [x, y] = maskToCanvasPoint(lassoCursor[0], lassoCursor[1]);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      lassoPoints.forEach((point) => {
        const [x, y] = maskToCanvasPoint(point[0], point[1]);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    function drawCursorOverlay() {
      if (!cursorMaskPoint || !maskWidth || !maskHeight) return;
      const [x, y] = maskToCanvasPoint(cursorMaskPoint[0], cursorMaskPoint[1]);
      ctx.save();
      ctx.lineWidth = 2;
      if (lassoMode) {
        const size = 12;
        ctx.strokeStyle = "#fff176";
        ctx.beginPath();
        ctx.moveTo(x - size, y);
        ctx.lineTo(x + size, y);
        ctx.moveTo(x, y - size);
        ctx.lineTo(x, y + size);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = "#fff176";
        ctx.fill();
      } else if (tool === "fill") {
        ctx.strokeStyle = "#ff4fd8";
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y);
        ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10);
        ctx.stroke();
      } else {
        const radiusMask = Math.max(1, Math.round(brushSize * maskWidth / viewport.imageWidth));
        const radiusCanvas = Math.max(2, radiusMask * viewport.imageWidth * viewport.view.scale / maskWidth);
        const baseErase = tool === "erase";
        const erase = cursorShiftInvert ? !baseErase : baseErase;
        ctx.strokeStyle = erase ? "#d14a32" : "#00c894";
        ctx.fillStyle = erase ? "rgba(209, 74, 50, 0.12)" : "rgba(0, 200, 148, 0.12)";
        ctx.beginPath();
        ctx.arc(x, y, radiusCanvas, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }

    function renderSummary() {
      const state = payload.state || {};
      const componentReview = state.component_review_status || {};
      const completionGuard = state.component_review_completion_guard || {};
      const reviewState = componentReview.state || "pending";
      const pendingEffects = Number(state.pending_apply_effect_count || 0);
      const background = Boolean(state.apply_effects_background);
      const effectsStatus = state.apply_effects_status || {};
      const effectsLine = !(background && pendingEffects > 0) ? ""
        : effectsStatus.state === "failed"
        ? "<p class=\"status error\"><b>Background update failed</b> " + escapeSupportText(effectsStatus.reason || "") +
          " It will not retry automatically; the admin has been notified. Press Apply to retry it.</p>"
        : effectsStatus.state === "retrying"
        ? "<p class=\"status error\"><b>Background update failed</b> " + escapeSupportText(effectsStatus.reason || "") +
          " Retrying automatically.</p>"
        : "<p><b>Updating QC and tail versions…</b> Your mask edits are saved; you can keep working.</p>";
      const reviewWarning = pendingEffects > 0
        ? (background
          ? "<p>Approval and rejection wait for the background update. You can set <b>needs_review</b> now and complete the task.</p>"
          : "<p><b>Action needed</b> Finish the pending Apply before setting review status or completing this task.</p>")
        : (completionGuard.ready ? "" :
          "<p><b>Action needed</b> Set component review before completing this task.</p>");
      const savedOffer = state.tail_refresh || tailRefreshResult;
      const tailOffer = Number(savedOffer?.tail_refresh_mask_revision) === Number(state.edit_revision || 0) ? savedOffer : null;
      const tailTasks = Array.isArray(tailOffer?.tail_refresh_tasks)
        ? tailOffer.tail_refresh_tasks : [];
      const tailFailures = Array.isArray(tailOffer?.tail_refresh_failures)
        ? tailOffer.tail_refresh_failures : [];
      const tailSummary = tailOffer?.tail_refresh_status === "complete"
        ? "<p><b>Refreshed tail review</b> Mask revision " + Number(tailOffer.tail_refresh_mask_revision) + "; " + tailTasks.length + " new task(s); " +
          Number(tailOffer.tail_refresh_valid_rows || 0) + " valid row(s), " +
          Number(tailOffer.tail_refresh_training_eligible_rows || 0) + " training-eligible row(s), " +
          Number(tailOffer.tail_refresh_manual_point_count || 0) + " manual point(s) retained, " +
          tailFailures.length + " failure(s). " +
          (tailTasks.length ? "Find " + tailTasks.map((task) => escapeSupportText(task.task_id)).join(", ") +
            " in <a href=\"/my-work\">your task queue</a>." : "") + "</p>"
        : "";
      document.getElementById("summary").innerHTML =
        "<p><b>ROI</b> " + payload.roi_idx + " / <b>" + (payload.frame_index_domain === "legacy_training_sample_row" ? "source training row" : "frame") + "</b> " + (payload.frame_idx ?? "") + "</p>" +
        "<p><b>Position</b> " + (state.position + 1) + " of " + state.total + "</p>" +
        "<p><b>Component</b> " + payload.component_name + "</p>" +
        "<p><b>Run</b> " + payload.refined_run + "</p>" +
        "<p><b>Mask area</b> " + payload.mask_area_px + " px</p>" +
        "<p><b>Review</b> " + reviewState + "</p>" +
        "<p><b>Session edits</b> " + (state.unapplied_session_edit_count || 0) +
        (payload.session_checkpoint ? " (current ROI is checkpoint overlay)" : "") + "</p>" +
        "<p><b>QC</b> " + (state.qc_status || "pending") +
        (pendingEffects ? " (" + pendingEffects + " Apply effect(s) pending)" : "") + "</p>" +
        effectsLine + reviewWarning + tailSummary;
      updateReviewControls(background && pendingEffects > 0);
      scheduleEffectsPoll();
      const seekInput = document.getElementById("roi-seek-input");
      if (seekInput) seekInput.value = payload.roi_idx;
      renderTailBorderStatus();
    }

    function renderTailBorderStatus() {
      const panel = document.getElementById("tail-border-controls");
      if (!panel) return;
      const info = payload?.tail_crop_border;
      panel.hidden = !info;
      if (!info) return;
      const describeReason = (raw) => {
        const value = String(raw || "");
        if (value.includes("body_touches_crop_border")) return "Body mask reaches the crop edge (body_touches_crop_border)";
        if (value.includes("fragmented_subject_body_mask")) return "Body mask has disconnected regions (fragmented_subject_body_mask)";
        if (value.includes("snout_extension_too_long")) return "Snout extension exceeds the allowed limit (snout_extension_too_long)";
        if (value.includes("snout_outside_crop")) return "Snout lies outside the crop (snout_outside_crop)";
        if (value.includes("tail_station_outside_body")) return "A tail point lies outside the body mask (tail_station_outside_body)";
        if (value.includes("tail_tip_is_visible_crop_endpoint")) return "Tail points derived; the last point is the visible crop-edge endpoint";
        if (value === "tail_derived") return "Tail points derived from the current mask";
        if (value.includes("snout_projection")) return "Snout extension exceeds the allowed geometry limit (" + value + ")";
        return value || "not recorded";
      };
      const original = describeReason(info.original_queued_reason);
      const outcome = info.latest_outcome;
      const outcomeText = outcome
        ? outcome.status + " at mask revision " + outcome.mask_revision + ": " + describeReason(outcome.reason)
        : "No refreshed tail derivation for the current mask revision";
      const pending = info.pending_action?.action
        ? "Pending checkpoint: " + info.pending_action.action + "; use Apply saved edits to publish a new tail version."
        : info.checkpoint_state
          ? "Saved mask pixels await Apply; the latest derived result has not checked this checkpoint."
          : "No acceptance action pending.";
      const localPixelsChanged = mask && loadedMask && mask.some((value, index) => value !== loadedMask[index]);
      const accepted = localPixelsChanged
        ? "Unsaved painted pixels are not covered by the applied result. If the clipped endpoint is still acceptable, checkpoint acceptance for the edited mask."
        : info.accepted
        ? "Accepted for this exact body mask by " + info.acceptance.accepted_by +
          " at revision " + info.acceptance.accepted_at_mask_revision + "."
        : info.malformed_acceptance
          ? "Saved acceptance evidence is invalid: " + info.malformed_acceptance
          : (info.stale_acceptance ? "Prior acceptance no longer matches this body mask or row." : "Strict crop-border rule applies.");
      document.getElementById("tail-border-status").innerHTML =
        "<p><b>Original queued reason</b> " + escapeSupportText(original) + "</p>" +
        "<p><b>Current acceptance</b> " + escapeSupportText(accepted) + "</p>" +
        "<p><b>Latest applied outcome</b> " + escapeSupportText(outcomeText) + "</p>" +
        "<p><b>Checkpoint</b> " + escapeSupportText(pending) + "</p>" +
        (info.status_unavailable ? "<p><b>Status unavailable</b> Reload this ROI to verify the applied result.</p>" : "") +
        (localPixelsChanged ? "<p><b>Local paint</b> Unsaved pixels differ from the latest applied result. Save and Apply to update the derivation.</p>" : "");
    }

    async function api(path, options={}) {
      clearMutationSupportReference();
      const response = await fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/subject-mask" + path, options);
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
        payload = await api("/roi/current");
        imageData = decodeRawImage(payload.roi_image);
        const sizeChanged = viewport.imageWidth !== imageData.width || viewport.imageHeight !== imageData.height;
        viewport.setImageData(imageData, {resetView: sizeChanged});
        decodeMask(payload.mask);
        clearLasso(true);
        renderSummary();
        const border = payload?.tail_crop_border;
        loadTailBorderReason(border?.pending_action?.action === "accept"
          ? (border.pending_action.reason || "")
          : (border?.accepted ? (border.acceptance.reason || "") : ""));
        scheduleDraw();
        updateNavButtons();
        setStatus("Loaded.");
      } catch (error) {
        updateNavButtons();
        showOperatorSupport(error, "session_request_failed");
      }
    }

    async function nav(delta) {
      if (busyAction) return;
      const state = payload?.state || {};
      const total = Number(state.total || 0);
      const position = Number(state.position || 0);
      if (delta < 0 && position <= 0) {
        setStatus("Already at the first ROI.");
        updateNavButtons();
        return;
      }
      if (delta > 0 && total > 0 && position >= total - 1) {
        setStatus("Already at the last ROI.");
        updateNavButtons();
        return;
      }
      setBusy(true, delta < 0 ? "Loading previous ROI..." : "Loading next ROI...");
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
        setBusy(false);
      }
    }

    async function seekRoi() {
      if (busyAction) return;
      const input = document.getElementById("roi-seek-input");
      const rawValue = input ? String(input.value || "").trim() : "";
      const roiIdx = Number(rawValue);
      if (!rawValue || !Number.isInteger(roiIdx) || roiIdx < 0) {
        setStatus("Enter a non-negative integer ROI number.", true);
        return;
      }
      setBusy(true, "Loading ROI " + roiIdx + "...");
      try {
        await api("/nav", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({roi_idx: roiIdx})
        });
        await loadCurrent();
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        setBusy(false);
      }
    }

    async function save(advance, tailBorderAction=null) {
      if (busyAction) return;
      setBusy(true, advance ? "Checkpointing mask and advancing..." : "Checkpointing mask...");
      try {
        const result = await api("/save", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({mask: encodeMaskPayload(), advance, target_token: payload?.state?.target_token,
            ...(tailBorderAction ? {tail_crop_border_action: tailBorderAction} : {})})
        });
        await loadCurrent();
        setStatus("Checkpoint saved; area " + result.result.checkpoint_area_px + " px." + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        setBusy(false);
      }
    }

    const tailBorderPresetReason = "Only the tiny tail tip is clipped; visible tail is usable.";
    const tailBorderNotesSeparator = " Notes: ";

    function updateTailBorderReasonInput() {
      const custom = document.getElementById("tail-border-preset")?.value === "custom";
      const input = document.getElementById("tail-border-reason");
      const label = document.getElementById("tail-border-reason-label");
      if (label) label.textContent = custom ? "Custom reason (required)" : "Optional notes";
      if (input) {
        input.placeholder = custom ? "Why is the visible tail endpoint usable?" : "Optional details for this ROI";
        input.maxLength = custom ? 240 : 240 - tailBorderPresetReason.length - tailBorderNotesSeparator.length;
      }
    }

    function loadTailBorderReason(reason) {
      const preset = document.getElementById("tail-border-preset");
      const input = document.getElementById("tail-border-reason");
      if (!preset || !input) return;
      const prefix = tailBorderPresetReason + tailBorderNotesSeparator;
      const isPreset = !reason || reason === tailBorderPresetReason || reason.startsWith(prefix);
      preset.value = isPreset ? "slight-tip" : "custom";
      input.value = isPreset ? (reason.startsWith(prefix) ? reason.slice(prefix.length) : "") : reason;
      updateTailBorderReasonInput();
    }

    function saveTailBorder(action) {
      if (!payload?.tail_crop_border) return;
      const notes = String(document.getElementById("tail-border-reason")?.value || "").trim();
      const custom = document.getElementById("tail-border-preset")?.value === "custom";
      const reason = custom ? notes : tailBorderPresetReason + (notes ? tailBorderNotesSeparator + notes : "");
      if (action === "accept" && (reason.length < 3 || reason.length > 240)) {
        setStatus(custom ? "Enter a 3–240 character custom reason." : "Shorten the optional notes to fit the 240-character reason limit.", true);
        return;
      }
      return save(false, {action, reason: action === "accept" ? reason : ""});
    }

    async function refreshTailStatus(generation, roiIdx) {
      const result = await api("/roi/status");
      if (!payload || busyAction || foregroundGeneration !== generation || Number(payload.roi_idx) !== Number(roiIdx) || Number(result.roi_idx) !== Number(roiIdx)) return;
      payload.tail_crop_border = result.tail_crop_border;
      renderTailBorderStatus();
    }

    function newApplyId() {
      if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
      return "apply-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
    }

    function scheduleEffectsPoll() {
      const state = payload?.state || {};
      const owed = Boolean(state.apply_effects_background) && Number(state.pending_apply_effect_count || 0) > 0;
      if (!owed) {
        if (effectsPollTimer !== null) { clearTimeout(effectsPollTimer); effectsPollTimer = null; }
        return;
      }
      if (effectsPollTimer !== null) return;
      effectsPollTimer = setTimeout(async () => {
        effectsPollTimer = null;
        const generation = foregroundGeneration;
        try {
          const result = await api("/state");
          mergeApplyState(result.state, generation);
        } catch (_error) {
          // Keep polling; a transient state read failure is not an Apply failure.
        }
        scheduleEffectsPoll();
      }, effectsPollMs);
    }

    function mergeApplyState(state, generation) {
      if (!payload || !state || busyAction || foregroundGeneration !== generation) return;
      // Apply runs in the background. Never replace current pixels, ROI, token,
      // navigation or a newer foreground request's checkpoint state.
      for (const key of ["edit_revision", "unapplied_session_edit_count", "has_unapplied_session_edits",
        "pending_apply_effect_count", "resumable_apply_id", "qc_status", "qc_edit_revision",
        "component_review_completion_guard", "component_review_completion_ready", "tail_refresh",
        "apply_effects_background", "apply_effects_status"]) {
        if (Object.prototype.hasOwnProperty.call(state, key)) payload.state[key] = state[key];
      }
      renderSummary();
    }

    async function applySavedEdits() {
      if (busyAction || applyInFlight) return;
      applyInFlight = true;
      const generation = foregroundGeneration;
      // In background mode a queued Apply is finished by the server; only a
      // failed (refused) one is retried by reusing its apply_id.
      const backgroundOwned = Boolean(payload?.state?.apply_effects_background)
        && payload?.state?.apply_effects_status?.state !== "failed";
      const resumable = backgroundOwned ? null : payload?.state?.resumable_apply_id;
      const applyId = String(resumable || uncertainApplyId || newApplyId());
      setStatus("Applying saved edits and refreshing QC. You can continue editing other rows while this runs.");
      try {
        const result = await api("/apply", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({apply_id: applyId, target_token: payload?.state?.target_token})
        });
        uncertainApplyId = null;
        tailRefreshResult = result.result;
        mergeApplyState(result.state, generation);
        try { await refreshTailStatus(generation, payload?.roi_idx); } catch (_error) {
          if (payload?.tail_crop_border && foregroundGeneration === generation) {
            payload.tail_crop_border.status_unavailable = true;
            renderTailBorderStatus();
          }
        }
        renderSummary();
        const applied = result.result.applied_checkpoint_count || 0;
        const stale = result.result.stale_checkpoint_count || 0;
        const staleRows = Array.isArray(result.result.stale_rows) ? result.result.stale_rows : [];
        const before = result.result.edit_revision_before;
        const after = result.result.edit_revision_after;
        const remaining = Number(payload?.state?.unapplied_session_edit_count || 0);
        const pendingEffects = Number(payload?.state?.pending_apply_effect_count || 0);
        const qcComplete = payload?.state?.qc_status === "complete" || result.result.qc_status === "complete";
        const nextStep = result.result.effects === "queued"
          ? " QC and tail versions are updating in the background; you can keep working."
          : pendingEffects > 0
          ? " Follow-up checks are pending; use Apply again to finish them."
          : remaining > 0
          ? " " + remaining + " saved edit(s) still need applying."
          : qcComplete
          ? " Saved edits and QC are complete. You can now set review status or complete the task."
          : " No saved edits were applied.";
        const stalePreview = staleRows.slice(0, 12).join(", ");
        const staleSuffix = stale > 0
          ? " Skipped " + stale + " stale saved edit(s)" + (stalePreview ? " at ROI " + stalePreview : "") + "; revisit and save those ROI(s) again."
          : "";
        setStatus("Applied " + applied + " saved edit(s) to Zarr; revision " + before + " -> " + after + "." + staleSuffix + nextStep + mutationStatusSuffix(result));
      } catch (error) {
        uncertainApplyId = applyId;
        if (error?.operatorSupport?.error === "subject_mask_apply_effects_pending") {
          mergeApplyState(error.apiData?.state, generation);
          try { await refreshTailStatus(generation, payload?.roi_idx); } catch (_statusError) {
            if (payload?.tail_crop_border && foregroundGeneration === generation) {
              payload.tail_crop_border.status_unavailable = true;
              renderTailBorderStatus();
            }
          }
          setStatus("Mask pixels were applied; follow-up checks are pending. " + error.message + " Use Apply again to retry the same saved operation.", true);
        } else if (["previous_update_still_running", "previous_update_failed"].includes(error?.operatorSupport?.error)) {
          uncertainApplyId = null;  // nothing was claimed; the saved edits are kept
          mergeApplyState(error.apiData?.state, generation);
          setStatus(error.message + " Your saved edits are kept; press Apply again later.", true);
        } else {
          showOperatorSupport(error, "session_request_failed");
        }
      } finally {
        applyInFlight = false;
      }
    }

    // While a background update is owed only needs_review can be recorded;
    // approval and rejection wait for the update, so disable them visibly.
    function updateReviewControls(updating) {
      const select = document.getElementById("review-state");
      if (!select) return;
      for (const option of Array.from(select.options || [])) {
        const waits = updating && option.value !== "needs_review";
        option.disabled = waits;
        option.textContent = option.value + (waits ? " (after update)" : "");
      }
      if (updating && select.selectedOptions?.[0]?.disabled) select.value = "needs_review";
      select.title = updating
        ? "A background QC/tail update is running. Approval and rejection are available when it finishes."
        : "";
    }

    async function setReviewStatus() {
      if (busyAction) return;
      setBusy(true, "Setting component review status...");
      try {
        const reviewState = document.getElementById("review-state").value;
        const result = await api("/review-status", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({state: reviewState, target_token: payload?.state?.target_token})
        });
        await loadCurrent();
        setStatus(result?.deferred
          ? "Review state " + reviewState + " recorded; it is written once the background update finishes. You can complete the task now."
          : "Component review state set to " + reviewState + "." + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        setBusy(false);
      }
    }

    async function completeTask() {
      if (busyAction) return;
      setBusy(true, "Completing task...");
      try {
        clearMutationSupportReference();
        const response = await fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/complete", {method: "POST"});
        const data = await readApiPayload(response);
        if (!response.ok || !data.ok) throw apiFailure(response, data, "task_complete_failed");
        handleTaskCompletionSuccess(data);
      } catch (error) {
        showOperatorSupport(error, "task_complete_failed");
      } finally {
        setBusy(false);
      }
    }

    function setTool(nextTool) {
      tool = nextTool;
      document.getElementById("paint-button").classList.toggle("active", tool === "paint");
      document.getElementById("erase-button").classList.toggle("active", tool === "erase");
      document.getElementById("fill-holes-button")?.classList.toggle("active", tool === "fill");
      if (tool === "fill") setStatus("Fill holes: click a mask piece to fill the holes inside it.");
      scheduleDraw();
    }

    function toggleBrushMode() {
      setTool(tool === "erase" ? "paint" : "erase");
      setStatus("Brush mode: " + tool + ".");
    }

    function setBrushSize(nextSize) {
      brushSize = Math.max(1, Math.min(48, Number(nextSize) || 1));
      document.getElementById("brush-size").value = String(brushSize);
      document.getElementById("brush-label").textContent = String(brushSize);
      scheduleDraw();
    }

    function setLassoMode(enabled) {
      lassoMode = Boolean(enabled);
      lassoDrawing = false;
      lassoCursor = null;
      document.getElementById("lasso-button").classList.toggle("active", lassoMode);
      if (!lassoMode) lassoPoints = [];
      scheduleDraw();
      setStatus(lassoMode ? "Lasso mode enabled. Click or drag to add contour points." : "Lasso mode disabled.");
    }

    function toggleLassoMode() {
      setLassoMode(!lassoMode);
    }

    function clearLasso(quiet=false) {
      lassoPoints = [];
      lassoCursor = null;
      lassoDrawing = false;
      if (!quiet) {
        scheduleDraw();
        setStatus("Lasso contour cleared.");
      }
    }

    function undoLassoPoint() {
      if (!lassoPoints.length) return;
      lassoPoints.pop();
      scheduleDraw();
      setStatus("Removed last lasso point.");
    }

    function clearMask() {
      if (!mask) return;
      mask.fill(0);
      markMaskOverlayDirty();
      scheduleDraw();
      setStatus("Mask cleared locally. Save to persist.");
    }

    function canvasPoint(event) {
      return viewport.canvasPoint(event);
    }

    function pointerEvent(event) {
      return viewport.pointerEvent(event);
    }

    function maskPointFromEvent(event) {
      const point = pointerEvent(event);
      const [canvasX, canvasY] = canvasPoint(point);
      const [x, y] = viewport.canvasToImage(canvasX, canvasY);
      return [
        Math.max(0, Math.min(maskWidth - 1, Math.floor(x * maskWidth / viewport.imageWidth))),
        Math.max(0, Math.min(maskHeight - 1, Math.floor(y * maskHeight / viewport.imageHeight)))
      ];
    }

    function appendLassoPoint(point) {
      const candidate = [Number(point[0]), Number(point[1])];
      if (!lassoPoints.length) {
        lassoPoints.push(candidate);
        return true;
      }
      const last = lassoPoints[lassoPoints.length - 1];
      const dx = candidate[0] - last[0];
      const dy = candidate[1] - last[1];
      if ((dx * dx + dy * dy) < (lassoMinPointStepPx * lassoMinPointStepPx)) return false;
      lassoPoints.push(candidate);
      return true;
    }

    function fillLasso(invert=false, forcedValue=null) {
      if (!mask || lassoPoints.length < 3) {
        setStatus("Lasso fill requires at least 3 contour points.", true);
        return;
      }
      const fillValue = forcedValue === null ? (tool === "erase" ? 0 : 1) : Number(forcedValue);
      const lassoCanvas = document.createElement("canvas");
      lassoCanvas.width = maskWidth;
      lassoCanvas.height = maskHeight;
      const lassoCtx = lassoCanvas.getContext("2d");
      lassoCtx.fillStyle = "white";
      lassoCtx.beginPath();
      lassoPoints.forEach((point, index) => {
        if (index === 0) lassoCtx.moveTo(point[0], point[1]);
        else lassoCtx.lineTo(point[0], point[1]);
      });
      lassoCtx.closePath();
      lassoCtx.fill();
      const pixels = lassoCtx.getImageData(0, 0, maskWidth, maskHeight).data;
      for (let i = 0; i < mask.length; i++) {
        const inside = pixels[i * 4] > 0;
        if (invert ? !inside : inside) mask[i] = fillValue ? 1 : 0;
      }
      clearLasso(true);
      markMaskOverlayDirty();
      scheduleDraw();
      setStatus((invert ? "Lasso outside fill" : "Lasso fill") + " applied locally. Save to persist.");
    }

    function paintAt(event) {
      if (!mask) return;
      const [mx, my] = maskPointFromEvent(event);
      cursorMaskPoint = [mx, my];
      cursorShiftInvert = Boolean(event.shiftKey);
      const radius = Math.max(1, Math.round(brushSize * maskWidth / viewport.imageWidth));
      const baseErase = tool === "erase";
      const erase = event.shiftKey ? !baseErase : baseErase;
      const value = erase ? 0 : 1;
      let changed = false;
      const minX = Math.max(0, mx - radius);
      const maxX = Math.min(maskWidth - 1, mx + radius);
      const minY = Math.max(0, my - radius);
      const maxY = Math.min(maskHeight - 1, my + radius);
      for (let yy = minY; yy <= maxY; yy++) {
        for (let xx = minX; xx <= maxX; xx++) {
          const dx = xx - mx;
          const dy = yy - my;
          if (dx * dx + dy * dy <= radius * radius) {
            const idx = yy * maskWidth + xx;
            if (mask[idx] !== value) {
              mask[idx] = value;
              changed = true;
            }
          }
        }
      }
      if (changed) markMaskOverlayDirty({x0: minX, y0: minY, x1: maxX + 1, y1: maxY + 1});
      scheduleDraw();
    }

    function beginCanvasEdit(event) {
      event.preventDefault();
      if (viewport.beginPan(event)) {
        cursorMaskPoint = null;
        scheduleDraw();
        return;
      }
      cursorMaskPoint = maskPointFromEvent(event);
      cursorShiftInvert = Boolean(event.shiftKey);
      if (lassoMode) {
        lassoDrawing = true;
        const point = cursorMaskPoint;
        lassoCursor = point;
        appendLassoPoint(point);
        scheduleDraw();
        return;
      }
      if (tool === "fill") {
        fillHolesAt(event);
        return;
      }
      drawing = true;
      paintAt(event);
    }

    function moveCanvasEdit(event) {
      if (viewport.panMove(event)) return;
      cursorMaskPoint = maskPointFromEvent(event);
      cursorShiftInvert = Boolean(event.shiftKey);
      if (lassoMode) {
        event.preventDefault();
        const point = cursorMaskPoint;
        lassoCursor = point;
        if (lassoDrawing) appendLassoPoint(point);
        scheduleDraw();
        return;
      }
      if (drawing) paintAt(event);
      else scheduleDraw();
    }

    canvas.addEventListener("mousedown", beginCanvasEdit);
    canvas.addEventListener("mousemove", moveCanvasEdit);
    canvas.addEventListener("mouseleave", () => { cursorMaskPoint = null; lassoCursor = null; scheduleDraw(); });
    canvas.addEventListener("touchstart", beginCanvasEdit, {passive: false});
    canvas.addEventListener("touchmove", moveCanvasEdit, {passive: false});
    window.addEventListener("mouseup", () => { drawing = false; lassoDrawing = false; viewport.endPan(); });
    window.addEventListener("touchend", () => { drawing = false; lassoDrawing = false; viewport.endPan(); });
    canvas.addEventListener("wheel", viewport.handleWheel, {passive: false});
    window.addEventListener("keydown", (event) => {
      if (event.key === "Shift" && cursorMaskPoint) {
        cursorShiftInvert = true;
        scheduleDraw();
      }
    });
    window.addEventListener("keyup", (event) => {
      if (event.key === "Shift" && cursorMaskPoint) {
        cursorShiftInvert = false;
        scheduleDraw();
      }
    });
    window.addEventListener("keydown", (event) => {
      const targetTag = event.target?.tagName?.toLowerCase();
      if (targetTag === "input" || targetTag === "textarea" || targetTag === "select") return;
      if (event.key === "n") { event.preventDefault(); nav(1); return; }
      if (event.key === "p") { event.preventDefault(); nav(-1); return; }
      if (event.key === "s") { event.preventDefault(); save(false); return; }
      if (event.key === "S") { event.preventDefault(); save(true); return; }
      if ((event.ctrlKey || event.metaKey) && (event.key === "z" || event.key === "Z")) { event.preventDefault(); undoBulkEdit(); return; }
      if (event.key === "h" || event.key === "H") { event.preventDefault(); setTool("fill"); return; }
      if (event.key === "r" || event.key === "R") { event.preventDefault(); removeStrayPiecesAction(); return; }
      if (event.key === "b" || event.key === "B") { event.preventDefault(); setTool("paint"); return; }
      if (event.key === "x" || event.key === "X") { event.preventDefault(); toggleBrushMode(); return; }
      if (event.key === "[") { event.preventDefault(); setBrushSize(brushSize - 1); return; }
      if (event.key === "]") { event.preventDefault(); setBrushSize(brushSize + 1); return; }
      if (event.key === "v" || event.key === "V") { event.preventDefault(); toggleLassoMode(); return; }
      if (event.key === "u" || event.key === "U" || event.key === "Backspace") { event.preventDefault(); undoLassoPoint(); return; }
      if (event.key === "d" || event.key === "D") { event.preventDefault(); clearLasso(); return; }
      if (event.key === "f" || event.key === "F") {
        event.preventDefault();
        if (lassoMode) fillLasso(false, null);
        else viewport.fit();
        return;
      }
      if (event.key === "g" || event.key === "G") { event.preventDefault(); fillLasso(true, null); return; }
      if (event.key === "e" || event.key === "E") {
        event.preventDefault();
        if (lassoMode) fillLasso(true, 0);
        else setTool("erase");
        return;
      }
    });
    loadCurrent();
