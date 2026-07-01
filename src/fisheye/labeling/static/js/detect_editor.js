    const sessionId = window.PALETTE_DETECT_SESSION_ID || "";
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let payload = null;
    let bbox = null;
    let dragStart = null;
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
      bbox = [cx, cy, width, height];
      draw();
      setStatus("Placed typical " + (width * 100).toFixed(2) + "% x " + (height * 100).toFixed(2) + "% box from " + hint.source + ". Save to persist.");
    }

    function draw() {
      if (!payload) return;
      viewport.drawImage();
      const rect = bboxToRect(bbox);
      if (rect) {
        const [x0, y0] = viewport.imageToCanvas(rect.x, rect.y);
        const [x1, y1] = viewport.imageToCanvas(rect.x + rect.w, rect.y + rect.h);
        ctx.lineWidth = Math.max(2, canvas.width / 160);
        ctx.strokeStyle = "#f28f3b";
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
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
        "<p><b>Frame</b> " + payload.frame_idx + " / <b>row</b> " + payload.row_idx + "</p>" +
        "<p><b>Position</b> " + (state.position + 1) + " of " + state.total + "</p>" +
        "<p><b>Run</b> " + (state.refined_run || "") + "</p>" +
        "<p><b>Status</b> " + (status.status_label || "") + " / " + (status.reason_label || "") + "</p>" +
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
        bbox = payload.bbox_norm ? payload.bbox_norm.slice() : null;
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
          body: JSON.stringify({bbox_norm: bbox, advance, target_token: payload?.state?.target_token})
        });
        await loadCurrent();
        setStatus("Saved " + result.result.action + "." + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      }
    }

    function clearBox() {
      bbox = null;
      draw();
      setStatus("Box cleared locally. Save to persist.");
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
      drawing = true;
    });
    canvas.addEventListener("mousemove", (event) => {
      if (viewport.panMove(event)) return;
      const p = imagePointFromEvent(event);
      lastImagePoint = p;
      if (!drawing || !dragStart) return;
      bbox = rectToBbox({x: dragStart[0], y: dragStart[1], w: p[0] - dragStart[0], h: p[1] - dragStart[1]});
      draw();
    });
    window.addEventListener("mouseup", () => { drawing = false; dragStart = null; viewport.endPan(); });
    canvas.addEventListener("wheel", viewport.handleWheel, {passive: false});
    window.addEventListener("keydown", (event) => {
      const targetTag = event.target?.tagName?.toLowerCase();
      if (targetTag === "input" || targetTag === "textarea" || targetTag === "select") return;
      if (event.key === "n") { event.preventDefault(); nav(1); return; }
      if (event.key === "p") { event.preventDefault(); nav(-1); return; }
      if (event.key === "s") { event.preventDefault(); save(false); return; }
      if (event.key === "t" || event.key === "T") { event.preventDefault(); placeTypicalBox(); return; }
      if (event.key === "f" || event.key === "F") { event.preventDefault(); viewport.fit(); return; }
    });
    loadCurrent();
  