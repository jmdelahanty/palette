    const sessionId = window.PALETTE_VIDEO_DETECT_SESSION_ID || "";
    const video = document.getElementById("video");
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");
    let payload = null;
    let bbox = null;
    let dragStart = null;
    let drawing = false;
    let currentMediaUrl = "";

    function setStatus(text, isError=false) {
      const node = document.getElementById("status");
      node.textContent = text;
      node.className = isError ? "status error" : "status";
      if (!isError) clearOperatorSupport();
    }


    function syncCanvas() {
      const width = video.videoWidth || (payload ? payload.media_width : 640) || 640;
      const height = video.videoHeight || (payload ? payload.media_height : 480) || 480;
      canvas.width = width;
      canvas.height = height;
      draw();
    }

    function bboxToRect(box) {
      if (!box) return null;
      const cx = Number(box[0]) * canvas.width;
      const cy = Number(box[1]) * canvas.height;
      const w = Number(box[2]) * canvas.width;
      const h = Number(box[3]) * canvas.height;
      return {x: cx - w / 2, y: cy - h / 2, w, h};
    }

    function rectToBbox(rect) {
      const x0 = Math.max(0, Math.min(canvas.width, rect.x));
      const y0 = Math.max(0, Math.min(canvas.height, rect.y));
      const x1 = Math.max(0, Math.min(canvas.width, rect.x + rect.w));
      const y1 = Math.max(0, Math.min(canvas.height, rect.y + rect.h));
      const loX = Math.min(x0, x1);
      const hiX = Math.max(x0, x1);
      const loY = Math.min(y0, y1);
      const hiY = Math.max(y0, y1);
      const w = hiX - loX;
      const h = hiY - loY;
      if (w < 2 || h < 2) return null;
      return [((loX + hiX) / 2) / canvas.width, ((loY + hiY) / 2) / canvas.height, w / canvas.width, h / canvas.height];
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const rect = bboxToRect(bbox);
      if (!rect) return;
      ctx.lineWidth = Math.max(2, canvas.width / 180);
      ctx.strokeStyle = "#f28f3b";
      ctx.fillStyle = "rgba(242, 143, 59, 0.16)";
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    }

    function renderSummary() {
      const state = payload.state || {};
      const status = payload.status || {};
      const editable = Boolean(state.editable);
      document.getElementById("save-button").disabled = !editable;
      document.getElementById("save-next-button").disabled = !editable;
      document.getElementById("summary").innerHTML =
        "<p><b>Parent frame</b> " + payload.parent_frame_index + " / <b>source frame</b> " + payload.source_frame_index + "</p>" +
        "<p><b>Position</b> " + (state.position + 1) + " of " + state.total + "</p>" +
        "<p><b>Mode</b> " + (state.mode || "") + " / <b>editable</b> " + editable + "</p>" +
        "<p><b>Run</b> " + (payload.refined_run_name || "") + "</p>" +
        "<p><b>Status</b> " + (status.status_label || "") + " / " + (status.reason_label || "") + "</p>";
    }

    async function api(path, options={}) {
      const response = await fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/detect-analysis" + path, options);
      const data = await readApiPayload(response);
      if (!response.ok || !data.ok) throw apiFailure(response, data, "session_request_failed");
      return data;
    }

    async function loadCurrent() {
      try {
        payload = await api("/frame/current");
        bbox = payload.bbox_norm ? payload.bbox_norm.slice() : null;
        if (payload.media_url && payload.media_url !== currentMediaUrl) {
          currentMediaUrl = payload.media_url;
          video.src = currentMediaUrl;
        }
        const seek = Number(payload.video_time_s || 0);
        if (Number.isFinite(seek)) video.currentTime = seek;
        renderSummary();
        syncCanvas();
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
      setStatus("Box cleared locally. Save to persist if this task is editable.");
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

    function canvasPoint(event) {
      const rect = canvas.getBoundingClientRect();
      return [
        (event.clientX - rect.left) * canvas.width / rect.width,
        (event.clientY - rect.top) * canvas.height / rect.height
      ];
    }

    canvas.addEventListener("mousedown", (event) => {
      dragStart = canvasPoint(event);
      drawing = true;
    });
    canvas.addEventListener("mousemove", (event) => {
      if (!drawing || !dragStart) return;
      const p = canvasPoint(event);
      bbox = rectToBbox({x: dragStart[0], y: dragStart[1], w: p[0] - dragStart[0], h: p[1] - dragStart[1]});
      draw();
    });
    window.addEventListener("mouseup", () => { drawing = false; dragStart = null; });
    window.addEventListener("resize", syncCanvas);
    video.addEventListener("loadedmetadata", syncCanvas);
    window.addEventListener("keydown", (event) => {
      if (event.key === "n") nav(1);
      if (event.key === "p") nav(-1);
      if (event.key === "s") save(false);
    });
    loadCurrent();
  