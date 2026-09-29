// Size an editor canvas to the largest box that fits its stage at the
// canvas's own aspect ratio. Only CSS size changes: the canvas width/height
// attributes (pixel resolution) stay owned by the editor, and pointer mapping
// already scales by the displayed box (image_canvas_viewport canvasPoint).
(function () {
  function fitCanvasToStage(canvas, stage, margin) {
    const pad = Number.isFinite(margin) ? margin : 16;
    function fit() {
      const w = canvas.width;
      const h = canvas.height;
      const boxW = stage.clientWidth - pad * 2;
      const boxH = stage.clientHeight - pad * 2;
      if (!(w > 0 && h > 0 && boxW > 0 && boxH > 0)) return;
      const scale = Math.min(boxW / w, boxH / h);
      canvas.style.width = Math.floor(w * scale) + "px";
      canvas.style.height = Math.floor(h * scale) + "px";
    }
    new ResizeObserver(fit).observe(stage);
    new MutationObserver(fit).observe(canvas, { attributes: true, attributeFilter: ["width", "height"] });
    fit();
    return fit;
  }
  window.fitCanvasToStage = fitCanvasToStage;
})();
