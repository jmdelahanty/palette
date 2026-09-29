// Keypoint colours by anatomical category, derived from landmark names so any
// pose schema works. The category sets the hue family and each point gets its
// own shade; a left/right pair shares a colour, drawn solid on the left and as
// a hollow ring on the right. Unrecognised names fall back to a fixed palette.
(function () {
  const CATEGORY_ORDER = ["head", "snout", "tail", "fin", "other"];
  const CATEGORY_NAMES = { head: "Head", snout: "Snout", tail: "Tail", fin: "Fins", other: "Other" };
  const TAIL_DARK = [122, 58, 0];
  const TAIL_LIGHT = [255, 179, 92];
  const FALLBACK = ["#e4572e", "#1479ff", "#00a86b", "#f2a541", "#9b5de5", "#00bbf9", "#f15bb5", "#6c757d"];

  function hex(rgb) {
    return "#" + rgb.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
  }

  function side(name) {
    if (/(^|_)left(_|$)/.test(name)) return "left";
    if (/(^|_)right(_|$)/.test(name)) return "right";
    return "";
  }

  function category(name) {
    if (name.startsWith("snout")) return "snout";
    if (name.startsWith("tail")) return "tail";
    if (name.includes("fin")) return "fin";
    if (name.startsWith("eye") || name === "swim_bladder" || name.startsWith("head")) return "head";
    return "other";
  }

  function keypointStyles(labels) {
    const names = (labels || []).map((label) => String(label || "").toLowerCase());
    const tailIndices = names.map((n, i) => (category(n) === "tail" ? i : -1)).filter((i) => i >= 0);
    let fallback = 0;
    return names.map((name, index) => {
      const cat = category(name);
      const s = side(name);
      let color;
      if (cat === "tail") {
        const t = tailIndices.length > 1 ? tailIndices.indexOf(index) / (tailIndices.length - 1) : 0;
        color = hex(TAIL_DARK.map((d, k) => d + (TAIL_LIGHT[k] - d) * t));
      } else if (cat === "head") {
        color = name.startsWith("eye") ? "#8a6cf0" : "#4b2fb0";
      } else if (cat === "snout") {
        color = "#d6336c";
      } else if (cat === "fin") {
        color = name.includes("tip") ? "#6fb1ff" : "#1558b0";
      } else {
        color = FALLBACK[fallback++ % FALLBACK.length];
      }
      return { category: cat, categoryName: CATEGORY_NAMES[cat], side: s, hollow: s === "right", color };
    });
  }

  function keypointGroups(styles) {
    return CATEGORY_ORDER.map((cat) => ({
      category: cat,
      name: CATEGORY_NAMES[cat],
      indices: styles.map((st, i) => (st.category === cat ? i : -1)).filter((i) => i >= 0),
    })).filter((group) => group.indices.length > 0);
  }

  window.keypointStyles = keypointStyles;
  window.keypointGroups = keypointGroups;
})();
