    const sessionId = window.PALETTE_KEYPOINT_SESSION_ID || "";
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let payload = null;
    let points = [];
    let activePoint = -1;
    let dragging = false;
    let showText = true;
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
        return `<div class="point-row"><b><span style="display:inline-block;width:0.75em;height:0.75em;border-radius:999px;background:${color};margin-right:0.4em;border:1px solid rgba(0,0,0,.24);"></span>${marker}${label}</b><span>${Number.isFinite(x) ? x.toFixed(1) : "nan"}, ${Number.isFinite(y) ? y.toFixed(1) : "nan"}</span></div>`;
      }).join("");
      document.getElementById("points").innerHTML = rows;
    }

    function renderSummary() {
      const state = payload.state || {};
      document.getElementById("summary").innerHTML = `
        <p><b>ROI</b> ${payload.roi_idx} / <b>frame</b> ${payload.frame_idx}</p>
        <p><b>Position</b> ${state.position + 1} of ${state.total}</p>
        <p><b>Run</b> ${state.refined_run || ""}</p>
        <p><b>Reason</b> ${payload.reason || ""}</p>
      `;
    }

    async function api(path, options={}) {
      const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/keypoints${path}`, options);
      const data = await readApiPayload(response);
      if (!response.ok || !data.ok) throw apiFailure(response, data, "session_request_failed");
      return data;
    }

    async function loadCurrent() {
      try {
        payload = await api("/roi/current");
        points = payload.points.map((p) => [Number(p[0]), Number(p[1])]);
        activePoint = -1;
        prepareImageSurface();
        renderSummary();
        renderPoints();
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
          body: JSON.stringify({points, advance, target_token: payload?.state?.target_token})
        });
        await loadCurrent();
        setStatus(`Saved ROI ${result.result.roi_idx}.` + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      }
    }

    async function action(name) {
      try {
        const result = await api("/action", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({action: name, advance: true, target_token: payload?.state?.target_token})
        });
        await loadCurrent();
        setStatus("Applied " + name + "." + mutationStatusSuffix(result));
      } catch (error) {
        showOperatorSupport(error, "session_request_failed");
      }
    }

    async function setReviewStatus() {
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
      }
    }

    async function completeTask() {
      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/complete`, {method: "POST"});
        const data = await readApiPayload(response);
        if (!response.ok || !data.ok) throw apiFailure(response, data, "task_complete_failed");
        handleTaskCompletionSuccess(data);
      } catch (error) {
        showOperatorSupport(error, "task_complete_failed");
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
      if (!points.length) return;
      activePoint = Math.max(0, Math.min(points.length - 1, index));
      renderPoints();
      draw();
      const label = payload?.labels?.[activePoint] || String(activePoint + 1);
      setStatus(`Selected ${label}.`);
    }

    function cycleActivePoint(delta) {
      if (!points.length) return;
      const current = activePoint >= 0 ? activePoint : 0;
      setActivePoint((current + delta + points.length) % points.length);
    }

    function resetPoints() {
      if (!payload) return;
      points = payload.points.map((point) => [Number(point[0]), Number(point[1])]);
      activePoint = -1;
      renderPoints();
      draw();
      setStatus("Reset points from current ROI.");
    }

    canvas.addEventListener("mousedown", (event) => {
      event.preventDefault();
      if (viewport.beginPan(event)) return;
      const [canvasX, canvasY] = canvasPoint(event);
      const [x, y] = canvasToImage(canvasX, canvasY);
      const nearest = nearestPoint(x, y);
      if (nearest >= 0) {
        activePoint = nearest;
      }
      dragging = activePoint >= 0;
      if (dragging) {
        points[activePoint] = [Math.max(0, Math.min(canvas.width - 1, x)), Math.max(0, Math.min(canvas.height - 1, y))];
        renderPoints();
        draw();
      }
    });
    canvas.addEventListener("mousemove", (event) => {
      if (viewport.panMove(event)) return;
      const [canvasX, canvasY] = canvasPoint(event);
      if (!dragging || activePoint < 0) return;
      const [x, y] = canvasToImage(canvasX, canvasY);
      points[activePoint] = [Math.max(0, Math.min(viewport.imageWidth - 1, x)), Math.max(0, Math.min(viewport.imageHeight - 1, y))];
      renderPoints();
      draw();
    });
    window.addEventListener("mouseup", () => { dragging = false; viewport.endPan(); });
    canvas.addEventListener("wheel", viewport.handleWheel, {passive: false});
    window.addEventListener("keydown", (event) => {
      const targetTag = event.target?.tagName?.toLowerCase();
      if (targetTag === "input" || targetTag === "textarea" || targetTag === "select") return;
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
  