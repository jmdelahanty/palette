
    function createImageCanvasViewport(canvas, drawCallback, options={}) {
      const ctx = canvas.getContext("2d");
      const imageSurface = document.createElement("canvas");
      const view = {scale: 1, offsetX: 0, offsetY: 0};
      const minScale = Number(options.minScale || 1);
      const maxScale = Number(options.maxScale || 16);
      let panning = false;
      let panLast = [0, 0];
      let cursorBeforePan = "";

      function clamp(value, minValue, maxValue) {
        return Math.max(minValue, Math.min(maxValue, value));
      }

      function hasImage() {
        return imageSurface.width > 0 && imageSurface.height > 0;
      }

      function constrain() {
        if (!hasImage()) return;
        view.scale = clamp(view.scale, minScale, maxScale);
        const maxOffsetX = Math.max(0, imageSurface.width - canvas.width / view.scale);
        const maxOffsetY = Math.max(0, imageSurface.height - canvas.height / view.scale);
        view.offsetX = clamp(view.offsetX, 0, maxOffsetX);
        view.offsetY = clamp(view.offsetY, 0, maxOffsetY);
      }

      function fit(redraw=true) {
        view.scale = 1;
        view.offsetX = 0;
        view.offsetY = 0;
        if (redraw) drawCallback();
      }

      function setImageData(imageData, config={}) {
        const resetView = Boolean(config.resetView);
        const sizeChanged = imageSurface.width !== imageData.width || imageSurface.height !== imageData.height;
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        imageSurface.width = imageData.width;
        imageSurface.height = imageData.height;
        imageSurface.getContext("2d").putImageData(imageData, 0, 0);
        if (resetView || sizeChanged) fit(false);
        else constrain();
      }

      function drawCanvas(sourceCanvas) {
        if (!hasImage()) return;
        constrain();
        ctx.drawImage(
          sourceCanvas,
          view.offsetX,
          view.offsetY,
          canvas.width / view.scale,
          canvas.height / view.scale,
          0,
          0,
          canvas.width,
          canvas.height
        );
      }

      function drawImage() {
        if (!hasImage()) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.imageSmoothingEnabled = false;
        drawCanvas(imageSurface);
      }

      function canvasPoint(event) {
        const rect = canvas.getBoundingClientRect();
        return [
          (event.clientX - rect.left) * canvas.width / rect.width,
          (event.clientY - rect.top) * canvas.height / rect.height
        ];
      }

      function pointerEvent(event) {
        return event.touches && event.touches.length ? event.touches[0] : event;
      }

      function canvasToImage(x, y) {
        return [view.offsetX + x / view.scale, view.offsetY + y / view.scale];
      }

      function imageToCanvas(x, y) {
        return [(x - view.offsetX) * view.scale, (y - view.offsetY) * view.scale];
      }

      function shouldPan(event) {
        return event.button === 1 || (event.buttons & 4) === 4;
      }

      function beginPan(event) {
        if (!shouldPan(event)) return false;
        event.preventDefault();
        panLast = canvasPoint(pointerEvent(event));
        panning = true;
        cursorBeforePan = canvas.style.cursor;
        canvas.style.cursor = "grabbing";
        return true;
      }

      function panMove(event) {
        if (!panning) return false;
        event.preventDefault();
        const point = canvasPoint(pointerEvent(event));
        view.offsetX -= (point[0] - panLast[0]) / view.scale;
        view.offsetY -= (point[1] - panLast[1]) / view.scale;
        panLast = point;
        constrain();
        drawCallback();
        return true;
      }

      function endPan() {
        if (panning) {
          canvas.style.cursor = cursorBeforePan;
        }
        panning = false;
      }

      function handleWheel(event) {
        event.preventDefault();
        if (!hasImage()) return;
        const [canvasX, canvasY] = canvasPoint(pointerEvent(event));
        const [beforeX, beforeY] = canvasToImage(canvasX, canvasY);
        const factor = event.deltaY < 0 ? 1.2 : 1 / 1.2;
        view.scale = clamp(view.scale * factor, minScale, maxScale);
        view.offsetX = beforeX - canvasX / view.scale;
        view.offsetY = beforeY - canvasY / view.scale;
        constrain();
        drawCallback();
      }

      canvas.addEventListener("auxclick", (event) => {
        if (event.button === 1) event.preventDefault();
      });

      return {
        ctx,
        view,
        imageSurface,
        hasImage,
        setImageData,
        drawImage,
        drawCanvas,
        canvasPoint,
        pointerEvent,
        canvasToImage,
        imageToCanvas,
        fit,
        constrain,
        beginPan,
        panMove,
        endPan,
        handleWheel,
        get imageWidth() { return imageSurface.width; },
        get imageHeight() { return imageSurface.height; }
      };
    }
