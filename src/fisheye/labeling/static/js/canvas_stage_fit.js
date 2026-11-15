// Size an editor canvas (or video) to the largest box that fits its stage at
// the element's own aspect ratio. Only CSS size changes: the canvas
// width/height attributes (pixel resolution) stay owned by the editor, and
// pointer mapping already scales by the displayed box (canvasPoint).
(function () {
  function naturalSize(element) {
    if (typeof HTMLVideoElement !== "undefined" && element instanceof HTMLVideoElement) {
      return [element.videoWidth, element.videoHeight];
    }
    return [element.width, element.height];
  }

  function fitCanvasToStage(element, stage, margin) {
    const pad = Number.isFinite(margin) ? margin : 16;
    function fit() {
      const [w, h] = naturalSize(element);
      const boxW = stage.clientWidth - pad * 2;
      const boxH = stage.clientHeight - pad * 2;
      if (!(w > 0 && h > 0 && boxW > 0 && boxH > 0)) return;
      const scale = Math.min(boxW / w, boxH / h);
      element.style.width = Math.floor(w * scale) + "px";
      element.style.height = Math.floor(h * scale) + "px";
    }
    new ResizeObserver(fit).observe(stage);
    new MutationObserver(fit).observe(element, { attributes: true, attributeFilter: ["width", "height"] });
    // A video's natural size is known once its metadata loads.
    element.addEventListener("loadedmetadata", fit);
    element.addEventListener("resize", fit);
    fit();
    return fit;
  }
  window.fitCanvasToStage = fitCanvasToStage;
})();
