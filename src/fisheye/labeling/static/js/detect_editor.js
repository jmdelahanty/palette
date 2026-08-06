    const sessionId = window.PALETTE_DETECT_SESSION_ID || "";
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let payload = null;
    let detections = [];
    let selectedIndex = null;
    let dragStart = null;
    let dragTargetIndex = null;
    let dragMoved = false;
    let drawing = false;
    let lastImagePoint = null;

    function setStatus(text, isError=false) {
      const node = document.getElementById("status");
      node.textContent = text;
      node.className = isError ? "status error" : "status";
      if (!isError) clearOperatorSupport();
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

    function bboxDisplayTransform() {
      const t = payload?.bbox_display_transform || {};
      const contentX = Number.isFinite(Number(t.content_x)) ? Number(t.content_x) : 0;
      const contentY = Number.isFinite(Number(t.content_y)) ? Number(t.content_y) : 0;
      const contentW = Number.isFinite(Number(t.content_width)) && Number(t.content_width) > 0 ? Number(t.content_width) : viewport.imageWidth;
      const contentH = Number.isFinite(Number(t.content_height)) && Number(t.content_height) > 0 ? Number(t.content_height) : viewport.imageHeight;
      return {x: contentX, y: contentY, w: contentW, h: contentH};
    }

    function bboxToRect(box) {
      if (!box) return null;
      const display = bboxDisplayTransform();
      const cx = display.x + Number(box[0]) * display.w;
      const cy = display.y + Number(box[1]) * display.h;
      const w = Number(box[2]) * display.w;
      const h = Number(box[3]) * display.h;
      return {x: cx - w / 2, y: cy - h / 2, w, h};
    }

    function rectToBbox(rect) {
      const display = bboxDisplayTransform();
      const x0 = Math.max(display.x, Math.min(display.x + display.w, rect.x));
      const y0 = Math.max(display.y, Math.min(display.y + display.h, rect.y));
      const x1 = Math.max(display.x, Math.min(display.x + display.w, rect.x + rect.w));
      const y1 = Math.max(display.y, Math.min(display.y + display.h, rect.y + rect.h));
      const loX = Math.min(x0, x1);
      const hiX = Math.max(x0, x1);
      const loY = Math.min(y0, y1);
      const hiY = Math.max(y0, y1);
      const w = hiX - loX;
      const h = hiY - loY;
      if (w < 2 || h < 2) return null;
      return [
        (((loX + hiX) / 2) - display.x) / display.w,
        (((loY + hiY) / 2) - display.y) / display.h,
        w / display.w,
        h / display.h
      ];
    }

    function bboxSizeHint() {
      const hint = payload?.bbox_size_hint_norm || null;
      if (!hint) return null;
      const width = Number(hint.width_norm);
      const height = Number(hint.height_norm);
      if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return null;
      return {
        width: Math.min(1, width),
        height: Math.min(1, height),
        source: String(hint.source || "unknown")
      };
    }

    function imagePointFromEvent(event) {
      const point = viewport.pointerEvent(event);
      const [canvasX, canvasY] = viewport.canvasPoint(point);
      return viewport.canvasToImage(canvasX, canvasY);
    }

    function bboxCenterNormFromImagePoint(point) {
      const display = bboxDisplayTransform();
      const imageX = Array.isArray(point) ? Number(point[0]) : display.x + display.w / 2;
      const imageY = Array.isArray(point) ? Number(point[1]) : display.y + display.h / 2;
      const cx = (Math.max(display.x, Math.min(display.x + display.w, imageX)) - display.x) / display.w;
      const cy = (Math.max(display.y, Math.min(display.y + display.h, imageY)) - display.y) / display.h;
      return [cx, cy];
    }

    function placeTypicalBox() {
      if (!payload) {
        setStatus("Load a frame before placing a typical box.", true);
        return;
      }
      const hint = bboxSizeHint();
      if (!hint) {
        setStatus("No typical bbox size hint is available for this task.", true);
        return;
      }
      const [rawCx, rawCy] = bboxCenterNormFromImagePoint(lastImagePoint);
      const width = Math.max(0.001, Math.min(1.0, hint.width));
      const height = Math.max(0.001, Math.min(1.0, hint.height));
      const cx = Math.max(width / 2, Math.min(1.0 - width / 2, rawCx));
      const cy = Math.max(height / 2, Math.min(1.0 - height / 2, rawCy));
      detections.push({instance_key: null, bbox_norm: [cx, cy, width, height], class_id: 0});
      selectedIndex = detections.length - 1;
      draw();
      setStatus("Added typical " + (width * 100).toFixed(2) + "% x " + (height * 100).toFixed(2) + "% box from " + hint.source + ". Save to persist.");
    }

    function detectionAtPoint(point) {
      for (let index = detections.length - 1; index >= 0; index -= 1) {
        const rect = bboxToRect(detections[index].bbox_norm);
        if (!rect) continue;
        const x0 = Math.min(rect.x, rect.x + rect.w);
        const x1 = Math.max(rect.x, rect.x + rect.w);
        const y0 = Math.min(rect.y, rect.y + rect.h);
        const y1 = Math.max(rect.y, rect.y + rect.h);
        if (point[0] >= x0 && point[0] <= x1 && point[1] >= y0 && point[1] <= y1) return index;
      }
      return null;
    }

    function draw() {
      if (!payload) return;
      viewport.drawImage();
      detections.forEach((detection, index) => {
        const rect = bboxToRect(detection.bbox_norm);
        if (!rect) return;
        const [x0, y0] = viewport.imageToCanvas(rect.x, rect.y);
        const [x1, y1] = viewport.imageToCanvas(rect.x + rect.w, rect.y + rect.h);
        ctx.lineWidth = Math.max(2, canvas.width / 160);
        ctx.strokeStyle = index === selectedIndex ? "#f28f3b" : "#22d3ee";
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.fillStyle = index === selectedIndex ? "rgba(242,143,59,0.16)" : "rgba(34,211,238,0.08)";
        ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        ctx.fillStyle = index === selectedIndex ? "#f28f3b" : "#22d3ee";
        ctx.font = Math.max(12, canvas.width / 90) + "px sans-serif";
        const identity = detection.instance_key ? String(detection.instance_key).slice(-6) : "new";
        ctx.fillText(String(index + 1) + ":" + identity, x0 + 4, Math.max(14, y0 - 4));
      });
      const display = bboxDisplayTransform();
      if (display.x > 0 || display.y > 0 || display.w < viewport.imageWidth || display.h < viewport.imageHeight) {
        const [x0, y0] = viewport.imageToCanvas(display.x, display.y);
        const [x1, y1] = viewport.imageToCanvas(display.x + display.w, display.y + display.h);
        ctx.save();
        ctx.lineWidth = Math.max(1, canvas.width / 320);
        ctx.strokeStyle = "rgba(255,255,255,0.35)";
        ctx.setLineDash([8, 6]);
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
        ctx.restore();
      }
    }

    function renderSummary() {
      const state = payload.state || {};
      const status = payload.status || {};
      const hint = bboxSizeHint();
      const hintText = hint
        ? (hint.width * 100).toFixed(2) + "% x " + (hint.height * 100).toFixed(2) + "% (" + hint.source + ")"
        : "unavailable";
      document.getElementById("summary").innerHTML =
        "<p><b>Frame</b> " + payload.frame_idx + " / <b>detections</b> " + detections.length + "</p>" +
        "<p><b>Position</b> " + (state.position + 1) + " of " + state.total + "</p>" +
        "<p><b>Run</b> " + (state.refined_run || "") + "</p>" +
        "<p><b>Selected</b> " + (selectedIndex === null ? "none" : String(selectedIndex + 1)) + "</p>" +
        "<p><b>Status</b> " + (status.status_label || "") + " / " + (status.reason_label || "") + "</p>" +
        "<p><b>Frame label</b> " + (payload.frame_label_state || "unreviewed") + " / " + (payload.frame_label_reason || "none") + "</p>" +
        "<p><b>Typical box</b> " + hintText + "</p>";
    }

    async function api(path, options={}) {
      const response = await fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/detect" + path, options);
      const data = await readApiPayload(response);
      if (!response.ok || !data.ok) throw apiFailure(response, data, "session_request_failed");
      return data;
    }

    async function loadCurrent() {
      try {
        payload = await api("/frame/current");
        detections = Array.isArray(payload.detections)
          ? payload.detections.filter((item) => Array.isArray(item?.bbox_norm)).map((item) => ({
              instance_key: item.instance_key === null || item.instance_key === undefined ? null : String(item.instance_key),
              bbox_norm: item.bbox_norm.slice(),
              class_id: Number(item?.status?.class_id ?? item?.class_id ?? 0)
            }))
          : (payload.bbox_norm ? [{instance_key: null, bbox_norm: payload.bbox_norm.slice(), class_id: 0}] : []);
        selectedIndex = detections.length ? 0 : null;
        viewport.setImageData(decodeRawImage(payload.frame_image), {resetView: true});
        renderSummary();
        draw();
        setStatus("Loaded.");
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      }
    }

    async function nav(delta) {
      try {
        await api("/nav", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({delta})
        });
        await loadCurrent();
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      }
    }

    async function save(advance) {
      try {
        const result = await api("/save", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            detections: detections.map((item) => ({
              instance_key: item.instance_key,
              bbox_norm: item.bbox_norm,
              class_id: item.class_id
            })),
            advance,
            target_token: payload?.state?.target_token
          })
        });
        await loadCurrent();
        setStatus("Saved " + result.result.action + "." + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      }
    }

    async function markNegative() {
      if (detections.length !== 0) {
        setStatus("Remove and save retained detections before marking this frame negative.", true);
        return;
      }
      try {
        const result = await api("/mark-negative", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            reason: "subject_outside_dish",
            advance: true,
            target_token: payload?.state?.target_token
          })
        });
        const markedFrame = result.result.frame_idx;
        await loadCurrent();
        setStatus("Marked frame " + markedFrame + " negative (subject outside dish)." + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "negative_frame_error");
      }
    }

    function clearSelectedBox() {
      if (selectedIndex === null || selectedIndex < 0 || selectedIndex >= detections.length) {
        setStatus("Select a detection before removing it.", true);
        return;
      }
      detections.splice(selectedIndex, 1);
      selectedIndex = detections.length ? Math.min(selectedIndex, detections.length - 1) : null;
      draw();
      renderSummary();
      setStatus("Selected detection removed locally. Save to persist.");
    }

    async function completeTask() {
      try {
        const response = await fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/complete", {method: "POST"});
        const data = await readApiPayload(response);
        if (!response.ok || !data.ok) throw apiFailure(response, data, "task_complete_failed");
        handleTaskCompletionSuccess(data);
      } catch (error) {
        showOperatorSupport(error, "task_complete_failed");
      }
    }

    canvas.addEventListener("mousedown", (event) => {
      if (viewport.beginPan(event)) return;
      if (event.button !== 0) return;
      const p = imagePointFromEvent(event);
      lastImagePoint = p;
      dragStart = p;
      dragTargetIndex = detectionAtPoint(p);
      if (dragTargetIndex !== null) selectedIndex = dragTargetIndex;
      dragMoved = false;
      drawing = true;
      draw();
      renderSummary();
    });
    canvas.addEventListener("mousemove", (event) => {
      if (viewport.panMove(event)) return;
      const p = imagePointFromEvent(event);
      lastImagePoint = p;
      if (!drawing || !dragStart) return;
      const nextBox = rectToBbox({x: dragStart[0], y: dragStart[1], w: p[0] - dragStart[0], h: p[1] - dragStart[1]});
      if (!nextBox) return;
      dragMoved = true;
      if (dragTargetIndex === null) {
        detections.push({instance_key: null, bbox_norm: nextBox, class_id: 0});
        dragTargetIndex = detections.length - 1;
        selectedIndex = dragTargetIndex;
      } else {
        detections[dragTargetIndex].bbox_norm = nextBox;
      }
      draw();
    });
    window.addEventListener("mouseup", () => {
      if (drawing && dragMoved) {
        renderSummary();
        setStatus("Detection collection changed locally. Save to persist.");
      }
      drawing = false;
      dragStart = null;
      dragTargetIndex = null;
      dragMoved = false;
      viewport.endPan();
    });
    canvas.addEventListener("wheel", viewport.handleWheel, {passive: false});
    window.addEventListener("keydown", (event) => {
      const targetTag = event.target?.tagName?.toLowerCase();
      if (targetTag === "input" || targetTag === "textarea" || targetTag === "select") return;
      if (event.key === "n") { event.preventDefault(); nav(1); return; }
      if (event.key === "p") { event.preventDefault(); nav(-1); return; }
      if (event.key === "s") { event.preventDefault(); save(false); return; }
      if (event.key === "x" || event.key === "X") { event.preventDefault(); markNegative(); return; }
      if (event.key === "t" || event.key === "T") { event.preventDefault(); placeTypicalBox(); return; }
      if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); clearSelectedBox(); return; }
      if (event.key === "f" || event.key === "F") { event.preventDefault(); viewport.fit(); return; }
    });
    loadCurrent();
