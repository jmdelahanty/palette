    const sessionId = window.PALETTE_SUBJECT_MASK_SESSION_ID || "";
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let payload = null;
    let imageData = null;
    let mask = null;
    let maskWidth = 0;
    let maskHeight = 0;
    let maskOverlayCanvas = document.createElement("canvas");
    let maskOverlayCtx = maskOverlayCanvas.getContext("2d");
    let maskOverlayDirty = true;
    let maskOverlayDirtyRect = null;
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
    const lassoMinPointStepPx = 2;

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

    function markMaskOverlayDirty(rect=null) {
      const wasFullyDirty = maskOverlayDirty && maskOverlayDirtyRect === null;
      maskOverlayDirty = true;
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

    function rebuildMaskOverlay() {
      if (!mask || !maskWidth || !maskHeight) return;
      const resized = maskOverlayCanvas.width !== maskWidth || maskOverlayCanvas.height !== maskHeight;
      if (maskOverlayCanvas.width !== maskWidth) maskOverlayCanvas.width = maskWidth;
      if (maskOverlayCanvas.height !== maskHeight) maskOverlayCanvas.height = maskHeight;
      if (resized || !maskOverlayDirtyRect) {
        const overlay = new ImageData(maskWidth, maskHeight);
        for (let i = 0; i < mask.length; i++) {
          if (!mask[i]) continue;
          const dst = i * 4;
          overlay.data[dst] = 0;
          overlay.data[dst + 1] = 200;
          overlay.data[dst + 2] = 148;
          overlay.data[dst + 3] = 118;
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
            if (!mask[src]) continue;
            const dst = (yy * w + xx) * 4;
            overlay.data[dst] = 0;
            overlay.data[dst + 1] = 200;
            overlay.data[dst + 2] = 148;
            overlay.data[dst + 3] = 118;
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
      drawLassoOverlay();
      drawCursorOverlay();
    }

    function maskToCanvasPoint(mx, my) {
      return viewport.imageToCanvas(
        (Number(mx) + 0.5) * viewport.imageWidth / maskWidth,
        (Number(my) + 0.5) * viewport.imageHeight / maskHeight
      );
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
      const reviewWarning = completionGuard.ready ? "" :
        "<p><b>Action needed</b> Set component review before completing this task.</p>";
      document.getElementById("summary").innerHTML =
        "<p><b>ROI</b> " + payload.roi_idx + " / <b>frame</b> " + (payload.frame_idx ?? "") + "</p>" +
        "<p><b>Position</b> " + (state.position + 1) + " of " + state.total + "</p>" +
        "<p><b>Component</b> " + payload.component_name + "</p>" +
        "<p><b>Run</b> " + payload.refined_run + "</p>" +
        "<p><b>Mask area</b> " + payload.mask_area_px + " px</p>" +
        "<p><b>Review</b> " + reviewState + "</p>" +
        "<p><b>Session edits</b> " + (state.unapplied_session_edit_count || 0) +
        (payload.session_checkpoint ? " (current ROI is checkpoint overlay)" : "") + "</p>" +
        reviewWarning;
      const seekInput = document.getElementById("roi-seek-input");
      if (seekInput) seekInput.value = payload.roi_idx;
    }

    async function api(path, options={}) {
      clearMutationSupportReference();
      const response = await fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/subject-mask" + path, options);
      const data = await readApiPayload(response);
      if (!response.ok || !data.ok) throw apiFailure(response, data, "session_request_failed");
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

    async function save(advance) {
      if (busyAction) return;
      setBusy(true, advance ? "Checkpointing mask and advancing..." : "Checkpointing mask...");
      try {
        const result = await api("/save", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({mask: encodeMaskPayload(), advance, target_token: payload?.state?.target_token})
        });
        await loadCurrent();
        setStatus("Checkpoint saved; area " + result.result.checkpoint_area_px + " px." + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        setBusy(false);
      }
    }

    function newApplyId() {
      if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
      return "apply-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
    }

    async function applySavedEdits() {
      if (busyAction || applyInFlight) return;
      applyInFlight = true;
      setStatus("Applying saved edits to Zarr in the background. You can continue editing other rows while this runs.");
      try {
        const result = await api("/apply", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({apply_id: newApplyId(), target_token: payload?.state?.target_token})
        });
        await loadCurrent();
        const applied = result.result.applied_checkpoint_count || 0;
        const stale = result.result.stale_checkpoint_count || 0;
        const staleRows = Array.isArray(result.result.stale_rows) ? result.result.stale_rows : [];
        const before = result.result.edit_revision_before;
        const after = result.result.edit_revision_after;
        const remaining = Number(payload?.state?.unapplied_session_edit_count || 0);
        const nextStep = remaining > 0
          ? " " + remaining + " saved edit(s) still need applying."
          : " Saved edits are applied to Zarr. You can now set review status or complete the task.";
        const stalePreview = staleRows.slice(0, 12).join(", ");
        const staleSuffix = stale > 0
          ? " Skipped " + stale + " stale saved edit(s)" + (stalePreview ? " at ROI " + stalePreview : "") + "; revisit and save those ROI(s) again."
          : "";
        setStatus("Applied " + applied + " saved edit(s) to Zarr; revision " + before + " -> " + after + "." + staleSuffix + nextStep + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      } finally {
        applyInFlight = false;
      }
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
        setStatus("Component review state set to " + reviewState + "." + mutationStatusSuffix(result));
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
      scheduleDraw();
    }

    function toggleBrushMode() {
      setTool(tool === "paint" ? "erase" : "paint");
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
  