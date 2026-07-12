"use strict";

const STATE = {
  options: null,
  summary: null,
};

const COLORS = ["#217a68", "#2f6fbd", "#b7791f", "#7a5aa6", "#248d92", "#b54848"];

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    const details = payload.details ? `: ${payload.details}` : "";
    throw new Error(`Request failed (${response.status})${details}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "";
  }
  const number = Number(value);
  if (Math.abs(number) >= 1000) {
    return number.toLocaleString(undefined, { maximumFractionDigits: 0 });
  }
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function setHealth(ok, text) {
  const pill = document.getElementById("health-pill");
  pill.textContent = text;
  pill.className = ok ? "pill ok" : "pill error";
}

function tableHtml(rows, columns) {
  if (!rows || rows.length === 0) {
    return '<div class="empty">No rows.</div>';
  }
  const visibleColumns = columns && columns.length ? columns : Object.keys(rows[0]);
  let html = "<table><thead><tr>";
  for (const column of visibleColumns) {
    html += `<th>${escapeHtml(column.label || column.key || column)}</th>`;
  }
  html += "</tr></thead><tbody>";
  for (const row of rows) {
    html += "<tr>";
    for (const column of visibleColumns) {
      const key = column.key || column;
      const value = row[key];
      const numeric = typeof value === "number";
      const text = numeric ? fmt(value) : value === null || value === undefined ? "" : String(value);
      html += `<td class="${numeric ? "numeric" : ""}">${escapeHtml(text)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  return html;
}

function populateSelect(selectId, options, valueKey, labelKey) {
  const select = document.getElementById(selectId);
  select.innerHTML = "";
  for (const option of options) {
    const value = option[valueKey];
    const label = option[labelKey];
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    select.appendChild(node);
  }
}

function groupedBarsSvg(rows, options) {
  const width = Math.max(760, (options.groups.length || 1) * 210);
  const height = options.height || 300;
  const margin = { top: 24, right: 24, bottom: 56, left: 64 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const values = rows.map((row) => Number(row.value)).filter((value) => Number.isFinite(value));
  let minValue = Math.min(0, ...values);
  let maxValue = Math.max(0, ...values);
  if (minValue === maxValue) {
    minValue -= 1;
    maxValue += 1;
  }
  const groupW = plotW / Math.max(1, options.groups.length);
  const seriesCount = Math.max(1, options.series.length);
  const barW = Math.min(32, (groupW * 0.72) / seriesCount);
  const y = (value) =>
    margin.top + plotH - ((Number(value || 0) - minValue) / Math.max(1e-9, maxValue - minValue)) * plotH;
  const zeroY = y(0);

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  svg += `<line x1="${margin.left}" y1="${zeroY}" x2="${width - margin.right}" y2="${zeroY}" stroke="#9aa5b1" />`;
  svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = minValue + ((maxValue - minValue) * tick) / 4;
    const yy = y(value);
    svg += `<line x1="${margin.left - 4}" y1="${yy}" x2="${width - margin.right}" y2="${yy}" stroke="#e1e6ee" />`;
    svg += `<text x="${margin.left - 8}" y="${yy + 4}" text-anchor="end" fill="#637080" font-size="11">${fmt(value, 2)}</text>`;
  }

  const rowByKey = new Map();
  for (const row of rows) {
    rowByKey.set(`${row[options.groupKey]}|${row[options.seriesKey]}`, row);
  }

  options.groups.forEach((group, groupIndex) => {
    const groupX = margin.left + groupIndex * groupW + groupW * 0.14;
    const labelX = margin.left + groupIndex * groupW + groupW / 2;
    svg += `<text x="${labelX}" y="${height - 22}" text-anchor="middle" fill="#637080" font-size="11">${escapeHtml(group)}</text>`;
    options.series.forEach((series, seriesIndex) => {
      const row = rowByKey.get(`${group}|${series.key}`);
      const value = row ? Number(row.value || 0) : 0;
      const x = groupX + seriesIndex * barW + seriesIndex * 4;
      const yy = y(value);
      const barY = Math.min(yy, zeroY);
      const h = Math.abs(zeroY - yy);
      svg += `<rect x="${x}" y="${barY}" width="${barW}" height="${Math.max(0, h)}" fill="${COLORS[seriesIndex % COLORS.length]}">`;
      svg += `<title>${escapeHtml(group)} ${escapeHtml(series.label)}: ${fmt(value)}</title></rect>`;
    });
  });

  const legendX = margin.left;
  options.series.forEach((series, index) => {
    const x = legendX + index * 150;
    svg += `<rect x="${x}" y="6" width="10" height="10" fill="${COLORS[index % COLORS.length]}" />`;
    svg += `<text x="${x + 15}" y="15" fill="#637080" font-size="11">${escapeHtml(series.label)}</text>`;
  });
  svg += "</svg>";
  return svg;
}

function histogramSvg(rows) {
  const width = 760;
  const height = 250;
  const margin = { top: 26, right: 24, bottom: 42, left: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const finiteRows = rows.filter((row) => row.bin_center_mm !== null && row.pooled_density !== null);
  if (!finiteRows.length) {
    return '<div class="empty">No histogram rows.</div>';
  }
  const minX = Math.min(...finiteRows.map((row) => Number(row.bin_left_mm)));
  const maxX = Math.max(...finiteRows.map((row) => Number(row.bin_right_mm)));
  const maxY = Math.max(1e-9, ...finiteRows.map((row) => Number(row.pooled_density || 0)));
  const x = (value) => margin.left + ((Number(value) - minX) / Math.max(1e-9, maxX - minX)) * plotW;
  const y = (value) => margin.top + plotH - (Number(value || 0) / maxY) * plotH;

  const series = new Map();
  for (const row of finiteRows) {
    const key = `${row.window_label} chaser ${row.chaser_index}`;
    if (!series.has(key)) {
      series.set(key, []);
    }
    series.get(key).push(row);
  }

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  svg += `<line x1="${margin.left}" y1="${margin.top + plotH}" x2="${width - margin.right}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  svg += `<text x="${margin.left}" y="${height - 8}" fill="#637080" font-size="11">Distance (mm)</text>`;
  svg += `<text x="8" y="${margin.top + 10}" fill="#637080" font-size="11">Density</text>`;
  let idx = 0;
  for (const [label, points] of series.entries()) {
    points.sort((a, b) => Number(a.distance_bin_index) - Number(b.distance_bin_index));
    const path = points
      .map((row, pointIndex) => {
        const cmd = pointIndex === 0 ? "M" : "L";
        return `${cmd}${x(row.bin_center_mm).toFixed(2)},${y(row.pooled_density).toFixed(2)}`;
      })
      .join(" ");
    const color = COLORS[idx % COLORS.length];
    svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"><title>${escapeHtml(label)}</title></path>`;
    const lx = margin.left + idx * 142;
    svg += `<line x1="${lx}" y1="10" x2="${lx + 12}" y2="10" stroke="${color}" stroke-width="2" />`;
    svg += `<text x="${lx + 17}" y="14" fill="#637080" font-size="11">${escapeHtml(label)}</text>`;
    idx += 1;
  }
  svg += "</svg>";
  return svg;
}

function metricHistogramSvg(rows, xLabel) {
  const width = 760;
  const height = 250;
  const margin = { top: 26, right: 24, bottom: 42, left: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const finiteRows = rows.filter((row) => row.bin_center !== null && row.pooled_density !== null);
  if (!finiteRows.length) {
    return '<div class="empty">No histogram rows.</div>';
  }
  const minX = Math.min(...finiteRows.map((row) => Number(row.bin_left)));
  const maxX = Math.max(...finiteRows.map((row) => Number(row.bin_right)));
  const maxY = Math.max(1e-9, ...finiteRows.map((row) => Number(row.pooled_density || 0)));
  const x = (value) => margin.left + ((Number(value) - minX) / Math.max(1e-9, maxX - minX)) * plotW;
  const y = (value) => margin.top + plotH - (Number(value || 0) / maxY) * plotH;

  const series = new Map();
  for (const row of finiteRows) {
    const key = String(row.window_label || "epoch");
    if (!series.has(key)) {
      series.set(key, []);
    }
    series.get(key).push(row);
  }

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  svg += `<line x1="${margin.left}" y1="${margin.top + plotH}" x2="${width - margin.right}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  svg += `<text x="${margin.left}" y="${height - 8}" fill="#637080" font-size="11">${escapeHtml(xLabel || "Metric")}</text>`;
  svg += `<text x="8" y="${margin.top + 10}" fill="#637080" font-size="11">Density</text>`;
  let idx = 0;
  for (const [label, points] of series.entries()) {
    points.sort((a, b) => Number(a.bin_index) - Number(b.bin_index));
    const path = points
      .map((row, pointIndex) => {
        const cmd = pointIndex === 0 ? "M" : "L";
        return `${cmd}${x(row.bin_center).toFixed(2)},${y(row.pooled_density).toFixed(2)}`;
      })
      .join(" ");
    const color = COLORS[idx % COLORS.length];
    svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"><title>${escapeHtml(label)}</title></path>`;
    const lx = margin.left + idx * 142;
    svg += `<line x1="${lx}" y1="10" x2="${lx + 12}" y2="10" stroke="${color}" stroke-width="2" />`;
    svg += `<text x="${lx + 17}" y="14" fill="#637080" font-size="11">${escapeHtml(label)}</text>`;
    idx += 1;
  }
  svg += "</svg>";
  return svg;
}

function speedDistanceSvg(rows) {
  const width = 900;
  const height = 300;
  const margin = { top: 28, right: 28, bottom: 48, left: 64 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const finiteRows = rows.filter(
    (row) =>
      row.distance_bin_center_mm !== null &&
      row.pooled_mean_speed_mm_s !== null &&
      Number(row.pooled_speed_sample_count || 0) > 0
  );
  if (!finiteRows.length) {
    return '<div class="empty">No speed-distance rows with valid speed samples.</div>';
  }
  const minX = Math.min(...finiteRows.map((row) => Number(row.distance_bin_left_mm)));
  const maxX = Math.max(...finiteRows.map((row) => Number(row.distance_bin_right_mm)));
  const maxY = Math.max(1e-9, ...finiteRows.map((row) => Number(row.pooled_mean_speed_mm_s || 0)));
  const x = (value) => margin.left + ((Number(value) - minX) / Math.max(1e-9, maxX - minX)) * plotW;
  const y = (value) => margin.top + plotH - (Number(value || 0) / maxY) * plotH;

  const series = new Map();
  for (const row of finiteRows) {
    const key = `${row.window_label} chaser ${row.chaser_index}`;
    if (!series.has(key)) {
      series.set(key, []);
    }
    series.get(key).push(row);
  }

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  svg += `<line x1="${margin.left}" y1="${margin.top + plotH}" x2="${width - margin.right}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = (maxY * tick) / 4;
    const yy = y(value);
    svg += `<line x1="${margin.left - 4}" y1="${yy}" x2="${width - margin.right}" y2="${yy}" stroke="#e1e6ee" />`;
    svg += `<text x="${margin.left - 8}" y="${yy + 4}" text-anchor="end" fill="#637080" font-size="11">${fmt(value, 2)}</text>`;
  }
  svg += `<text x="${margin.left}" y="${height - 10}" fill="#637080" font-size="11">Distance to chaser (mm)</text>`;
  svg += `<text x="8" y="${margin.top + 10}" fill="#637080" font-size="11">Speed (mm/s)</text>`;

  let idx = 0;
  for (const [label, points] of series.entries()) {
    points.sort((a, b) => Number(a.distance_bin_index) - Number(b.distance_bin_index));
    const color = COLORS[idx % COLORS.length];
    const path = points
      .map((row, pointIndex) => {
        const cmd = pointIndex === 0 ? "M" : "L";
        return `${cmd}${x(row.distance_bin_center_mm).toFixed(2)},${y(row.pooled_mean_speed_mm_s).toFixed(2)}`;
      })
      .join(" ");
    svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2"><title>${escapeHtml(label)}</title></path>`;
    for (const row of points) {
      const radius = Math.min(5, 2 + Math.log10(Math.max(1, Number(row.pooled_speed_sample_count || 1))));
      svg += `<circle cx="${x(row.distance_bin_center_mm).toFixed(2)}" cy="${y(row.pooled_mean_speed_mm_s).toFixed(2)}" r="${radius.toFixed(2)}" fill="${color}" opacity="0.86">`;
      svg += `<title>${escapeHtml(label)} | ${fmt(row.distance_bin_center_mm, 1)} mm | ${fmt(row.pooled_mean_speed_mm_s, 2)} mm/s | n=${fmt(row.pooled_speed_sample_count, 0)}</title></circle>`;
    }
    const lx = margin.left + (idx % 4) * 190;
    const ly = 10 + Math.floor(idx / 4) * 16;
    svg += `<line x1="${lx}" y1="${ly}" x2="${lx + 12}" y2="${ly}" stroke="${color}" stroke-width="2" />`;
    svg += `<text x="${lx + 17}" y="${ly + 4}" fill="#637080" font-size="11">${escapeHtml(label)}</text>`;
    idx += 1;
  }
  svg += "</svg>";
  return svg;
}

function heatmapColor(value) {
  const t = Math.max(0, Math.min(1, Number(value || 0)));
  const stops = [
    [247, 251, 255],
    [111, 164, 204],
    [181, 72, 72],
  ];
  const scaled = t * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const local = scaled - idx;
  const a = stops[idx];
  const b = stops[idx + 1];
  const rgb = a.map((channel, channelIndex) =>
    Math.round(channel + (b[channelIndex] - channel) * local)
  );
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function bearingDistanceHeatmapSvg(rows) {
  const finiteRows = rows.filter(
    (row) =>
      row.distance_bin_center_mm !== null &&
      row.bearing_bin_center_deg !== null &&
      row.pooled_count !== null
  );
  if (!finiteRows.length) {
    return '<div class="empty">No bearing-distance rows.</div>';
  }

  const aggregate = new Map();
  for (const row of finiteRows) {
    const key = `${Number(row.distance_bin_center_mm)}|${Number(row.bearing_bin_center_deg)}`;
    const existing = aggregate.get(key) || {
      distance: Number(row.distance_bin_center_mm),
      bearing: Number(row.bearing_bin_center_deg),
      count: 0,
    };
    existing.count += Number(row.pooled_count || 0);
    aggregate.set(key, existing);
  }
  const cells = [...aggregate.values()];
  const distances = [...new Set(cells.map((row) => row.distance))].sort((a, b) => a - b);
  const bearings = [...new Set(cells.map((row) => row.bearing))].sort((a, b) => a - b);
  const cellByKey = new Map(cells.map((row) => [`${row.distance}|${row.bearing}`, row]));
  const width = 760;
  const height = 270;
  const margin = { top: 22, right: 22, bottom: 48, left: 70 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const cellW = plotW / Math.max(1, bearings.length);
  const cellH = plotH / Math.max(1, distances.length);
  const maxCount = Math.max(1, ...cells.map((row) => row.count));

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  svg += `<text x="${margin.left}" y="14" fill="#637080" font-size="11">Bearing relative to fish heading (deg)</text>`;
  svg += `<text x="8" y="${margin.top + 12}" fill="#637080" font-size="11">Distance</text>`;
  distances.forEach((distance, distanceIndex) => {
    bearings.forEach((bearing, bearingIndex) => {
      const row = cellByKey.get(`${distance}|${bearing}`);
      const count = row ? row.count : 0;
      const x = margin.left + bearingIndex * cellW;
      const y = margin.top + (distances.length - distanceIndex - 1) * cellH;
      svg += `<rect x="${x}" y="${y}" width="${Math.max(0, cellW - 1)}" height="${Math.max(0, cellH - 1)}" fill="${heatmapColor(count / maxCount)}">`;
      svg += `<title>${fmt(distance)} mm, ${fmt(bearing)} deg: ${fmt(count, 0)}</title></rect>`;
    });
  });
  bearings.forEach((bearing, index) => {
    if (index % Math.max(1, Math.ceil(bearings.length / 8)) !== 0) {
      return;
    }
    const x = margin.left + index * cellW + cellW / 2;
    svg += `<text x="${x}" y="${height - 24}" text-anchor="middle" fill="#637080" font-size="10">${fmt(bearing, 0)}</text>`;
  });
  distances.forEach((distance, index) => {
    if (index % Math.max(1, Math.ceil(distances.length / 6)) !== 0) {
      return;
    }
    const y = margin.top + (distances.length - index - 0.5) * cellH + 4;
    svg += `<text x="${margin.left - 8}" y="${y}" text-anchor="end" fill="#637080" font-size="10">${fmt(distance, 1)}</text>`;
  });
  svg += `<line x1="${margin.left}" y1="${margin.top + plotH}" x2="${width - margin.right}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotH}" stroke="#9aa5b1" />`;
  svg += "</svg>";
  return svg;
}

function stableHash(text) {
  let hash = 0;
  const input = String(text || "");
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
  }
  return hash;
}

function pathFromPoints(points, xKey, yKey, xScale, yScale) {
  return points
    .map((row, index) => {
      const command = index === 0 ? "M" : "L";
      return `${command}${xScale(row[xKey]).toFixed(2)},${yScale(row[yKey]).toFixed(2)}`;
    })
    .join(" ");
}

function finiteNumbers(values) {
  return values.map((value) => Number(value)).filter((value) => Number.isFinite(value));
}

function extentWithZero(values, padFraction = 0.08) {
  const finite = finiteNumbers(values);
  if (!finite.length) {
    return [-1, 1];
  }
  let minValue = Math.min(0, ...finite);
  let maxValue = Math.max(0, ...finite);
  if (minValue === maxValue) {
    minValue -= 1;
    maxValue += 1;
  }
  const pad = (maxValue - minValue) * padFraction;
  return [minValue - pad, maxValue + pad];
}

function linearScale(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin || 1;
  return (value) => rangeMin + ((Number(value) - domainMin) / span) * (rangeMax - rangeMin);
}

function craSpecificitySvg(data) {
  if (!data || !data.available) {
    return '<div class="empty">No CRA specificity rows found.</div>';
  }
  const width = 920;
  const rowH = 250;
  const height = 560;
  const margin = { top: 46, right: 24, bottom: 34, left: 50 };
  const slopeW = 230;
  const dotW = 300;
  const gap = 34;
  const xAgg = margin.left;
  const xBenign = xAgg + slopeW + gap;
  const xSpec = xBenign + slopeW + gap;
  const distanceRows = data.distance_slope_rows || [];
  const occRows = data.occupancy_index_slope_rows || [];
  const distanceDomain = extentWithZero(
    distanceRows.flatMap((row) => [row.pre_value, row.post_value]),
    0.06
  );
  const occDomain = extentWithZero(
    occRows.flatMap((row) => [row.pre_value, row.post_value]),
    0.08
  );

  function slopePanel(rows, role, title, domain, panelX, panelY, color, unitLabel) {
    const y = linearScale(domain[0], domain[1], panelY + rowH - 62, panelY + 28);
    const preX = panelX + 48;
    const postX = panelX + slopeW - 34;
    let svg = `<text x="${panelX}" y="${panelY + 14}" fill="#182026" font-size="12" font-weight="700">${escapeHtml(title)}</text>`;
    svg += `<line x1="${panelX + 30}" y1="${y(0).toFixed(2)}" x2="${panelX + slopeW - 10}" y2="${y(0).toFixed(2)}" stroke="#b8c0cc" stroke-dasharray="4 4" />`;
    svg += `<line x1="${preX}" y1="${panelY + rowH - 52}" x2="${postX}" y2="${panelY + rowH - 52}" stroke="#9aa5b1" />`;
    svg += `<text x="${preX}" y="${panelY + rowH - 34}" text-anchor="middle" fill="#637080" font-size="11">pre</text>`;
    svg += `<text x="${postX}" y="${panelY + rowH - 34}" text-anchor="middle" fill="#637080" font-size="11">post</text>`;
    const roleRows = rows.filter((row) => row.object_role === role);
    for (const row of roleRows) {
      const pre = Number(row.pre_value);
      const post = Number(row.post_value);
      if (!Number.isFinite(pre) || !Number.isFinite(post)) {
        continue;
      }
      svg += `<line x1="${preX}" y1="${y(pre).toFixed(2)}" x2="${postX}" y2="${y(post).toFixed(2)}" stroke="${color}" stroke-width="1.3" opacity="0.42">`;
      svg += `<title>${escapeHtml(row.recording_id)}: ${fmt(pre)} -> ${fmt(post)}</title></line>`;
      svg += `<circle cx="${preX}" cy="${y(pre).toFixed(2)}" r="2.8" fill="${color}" opacity="0.68" />`;
      svg += `<circle cx="${postX}" cy="${y(post).toFixed(2)}" r="2.8" fill="${color}" opacity="0.68" />`;
    }
    svg += `<text x="${panelX}" y="${panelY + rowH - 8}" fill="#637080" font-size="10">${escapeHtml(unitLabel)}</text>`;
    return svg;
  }

  function dotPanel(rows, valueKey, stat, title, panelX, panelY, color) {
    const values = rows.map((row) => row[valueKey]);
    const domain = extentWithZero(values, 0.12);
    const x = linearScale(domain[0], domain[1], panelX + 18, panelX + dotW - 16);
    const zeroX = x(0);
    const baseY = panelY + rowH / 2;
    let svg = `<text x="${panelX}" y="${panelY + 14}" fill="#182026" font-size="12" font-weight="700">${escapeHtml(title)}</text>`;
    svg += `<line x1="${zeroX.toFixed(2)}" y1="${panelY + 28}" x2="${zeroX.toFixed(2)}" y2="${panelY + rowH - 56}" stroke="#637080" stroke-dasharray="4 4" />`;
    svg += `<line x1="${panelX + 18}" y1="${panelY + rowH - 52}" x2="${panelX + dotW - 16}" y2="${panelY + rowH - 52}" stroke="#9aa5b1" />`;
    svg += `<text x="${zeroX + 4}" y="${panelY + rowH - 36}" fill="#637080" font-size="10">0</text>`;
    for (const row of rows) {
      const value = Number(row[valueKey]);
      if (!Number.isFinite(value)) {
        continue;
      }
      const jitter = ((stableHash(`${row.recording_id}|${valueKey}`) % 100) / 100 - 0.5) * 88;
      svg += `<circle cx="${x(value).toFixed(2)}" cy="${(baseY + jitter).toFixed(2)}" r="3.8" fill="${color}" opacity="0.72">`;
      svg += `<title>${escapeHtml(row.recording_id)}: ${fmt(value)}</title></circle>`;
    }
    if (stat && stat.ci_low !== null && stat.ci_low !== undefined && stat.ci_high !== null && stat.ci_high !== undefined) {
      const ciY = panelY + 34;
      svg += `<line x1="${x(stat.ci_low).toFixed(2)}" y1="${ciY}" x2="${x(stat.ci_high).toFixed(2)}" y2="${ciY}" stroke="${color}" stroke-width="3" />`;
      if (stat.median_difference !== null && stat.median_difference !== undefined) {
        svg += `<circle cx="${x(stat.median_difference).toFixed(2)}" cy="${ciY}" r="4.2" fill="${color}" />`;
      }
    }
    svg += `<text x="${panelX}" y="${panelY + rowH - 10}" fill="#637080" font-size="10">n=${fmt(stat && stat.n, 0)}, p=${fmt(stat && stat.p_value, 4)}, median=${fmt(stat && stat.median_difference, 3)}</text>`;
    return svg;
  }

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  svg += `<text x="${margin.left}" y="20" fill="#182026" font-size="14" font-weight="700">CRA confirmatory specificity contrasts</text>`;
  svg += `<text x="${margin.left}" y="38" fill="#637080" font-size="11">Distance specificity is primary; occupancy index is the phase-relative quadrant companion.</text>`;
  svg += slopePanel(distanceRows, "aggressive", "Distance: aggressive", distanceDomain, xAgg, margin.top, "#b54848", "median fish-object distance (mm)");
  svg += slopePanel(distanceRows, "inert", "Distance: inert", distanceDomain, xBenign, margin.top, "#2f6fbd", "same y-axis as aggressive");
  svg += dotPanel(data.distance_specificity_rows || [], "specificity_distance", data.distance_specificity_statistics, "Distance specificity", xSpec, margin.top, "#182026");
  const occY = margin.top + rowH + 18;
  svg += slopePanel(occRows, "aggressive", "Occupancy index: aggressive", occDomain, xAgg, occY, "#b54848", "object quadrant - mean(other quadrants)");
  svg += slopePanel(occRows, "inert", "Occupancy index: inert", occDomain, xBenign, occY, "#2f6fbd", "same y-axis as aggressive");
  svg += dotPanel(data.occupancy_index_specificity_rows || [], "occupancy_index_specificity", data.occupancy_index_specificity_statistics, "Occupancy specificity", xSpec, occY, "#182026");
  svg += "</svg>";
  return svg;
}

function quadrantOccupancyDensitySvg(data) {
  if (!data || !data.available || !data.rows || !data.rows.length) {
    return '<div class="empty">No CRA quadrant occupancy export found.</div>';
  }
  const phases = (data.phases || []).slice().sort((a, b) => Number(a.phase_axis_index) - Number(b.phase_axis_index));
  const width = 920;
  const panelH = 320;
  const margin = { top: 44, right: 34, bottom: 28, left: 124 };
  const plotW = width - margin.left - margin.right;
  const stripH = 142;
  const densityTopOffset = 184;
  const densityH = 96;
  const height = margin.top + phases.length * panelH + margin.bottom;
  const x = (value) => margin.left + Math.max(0, Math.min(1, Number(value || 0))) * plotW;
  const chanceX = x(data.chance || 0.25);
  const stat = data.statistics || {};

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  svg += `<text x="${margin.left}" y="20" fill="#182026" font-size="13" font-weight="700">Descriptive gross quadrant relocation</text>`;
  svg += `<text x="${margin.left}" y="38" fill="#637080" font-size="11">Chaser-only pre/post is descriptive: n=${fmt(stat.n, 0)}, p=${fmt(stat.p_value, 4)}, median delta=${fmt(stat.median_difference, 3)}, rank-biserial=${fmt(stat.effect_size, 3)}</text>`;

  for (let phaseIndex = 0; phaseIndex < phases.length; phaseIndex += 1) {
    const phase = phases[phaseIndex];
    const phaseLabel = phase.phase_label;
    const panelTop = margin.top + phaseIndex * panelH;
    const stripTop = panelTop + 30;
    const densityTop = panelTop + densityTopOffset;
    const rows = data.rows.filter((row) => row.phase_label === phaseLabel);
    const quadrantRows = (data.quadrant_rows || []).filter((row) => row.phase_label === phaseLabel);
    const quadrants = [...new Map(rows.map((row) => [row.quadrant_id, row])).values()].sort(
      (a, b) => Number(a.display_order) - Number(b.display_order)
    );
    const rowY = new Map();
    quadrants.forEach((row, index) => {
      rowY.set(row.quadrant_id, stripTop + 22 + index * 30);
    });

    svg += `<text x="${margin.left}" y="${panelTop + 17}" fill="#182026" font-size="13" font-weight="700">${escapeHtml(phaseLabel)}</text>`;
    svg += `<line x1="${margin.left}" y1="${stripTop - 10}" x2="${width - margin.right}" y2="${stripTop - 10}" stroke="#d9dee7" />`;
    svg += `<line x1="${margin.left}" y1="${stripTop + stripH}" x2="${width - margin.right}" y2="${stripTop + stripH}" stroke="#d9dee7" />`;
    svg += `<line x1="${chanceX}" y1="${stripTop - 8}" x2="${chanceX}" y2="${densityTop + densityH}" stroke="#637080" stroke-dasharray="4 4" />`;
    svg += `<text x="${chanceX + 5}" y="${stripTop - 13}" fill="#637080" font-size="10">chance 0.25</text>`;

    for (const quadrant of quadrants) {
      const yy = rowY.get(quadrant.quadrant_id);
      svg += `<text x="${margin.left - 12}" y="${yy + 4}" text-anchor="end" fill="#364252" font-size="11">${escapeHtml(quadrant.quadrant_id)}</text>`;
      svg += `<line x1="${margin.left}" y1="${yy}" x2="${width - margin.right}" y2="${yy}" stroke="#edf0f4" />`;
    }

    for (const row of rows) {
      const value = Number(row.occupancy_fraction);
      if (!Number.isFinite(value)) {
        continue;
      }
      const baseY = rowY.get(row.quadrant_id);
      const jitter = ((stableHash(`${row.recording_id}|${phaseLabel}|${row.quadrant_id}`) % 100) / 100 - 0.5) * 13;
      const color = row.is_chaser_quadrant ? "#b54848" : "#6b7280";
      const opacity = row.is_chaser_quadrant ? 0.9 : 0.42;
      svg += `<circle cx="${x(value).toFixed(2)}" cy="${(baseY + jitter).toFixed(2)}" r="${row.is_chaser_quadrant ? 4.2 : 3.4}" fill="${color}" opacity="${opacity}">`;
      svg += `<title>${escapeHtml(row.recording_id)} ${escapeHtml(phaseLabel)} ${escapeHtml(row.quadrant_id)}: ${fmt(value)}</title></circle>`;
    }

    for (const row of quadrantRows) {
      const yy = rowY.get(row.quadrant_id);
      const mean = Number(row.mean);
      if (!Number.isFinite(mean)) {
        continue;
      }
      const sem = Number(row.sem || 0);
      const color = Number(row.chaser_recording_count || 0) > 0 ? "#b54848" : "#334155";
      svg += `<line x1="${x(Math.max(0, mean - sem)).toFixed(2)}" y1="${yy - 9}" x2="${x(Math.min(1, mean + sem)).toFixed(2)}" y2="${yy - 9}" stroke="${color}" stroke-width="2" />`;
      svg += `<circle cx="${x(mean).toFixed(2)}" cy="${yy - 9}" r="3.5" fill="${color}"><title>mean=${fmt(mean)}, SEM=${fmt(sem)}</title></circle>`;
    }

    const densityRows = (data.density_rows || []).filter((row) => row.phase_label === phaseLabel);
    const maxDensity = Math.max(1e-9, ...densityRows.map((row) => Number(row.density || 0)));
    const yDensity = (value) => densityTop + densityH - (Number(value || 0) / maxDensity) * densityH;
    svg += `<line x1="${margin.left}" y1="${densityTop + densityH}" x2="${width - margin.right}" y2="${densityTop + densityH}" stroke="#9aa5b1" />`;
    svg += `<text x="${margin.left - 12}" y="${densityTop + 12}" text-anchor="end" fill="#637080" font-size="10">KDE</text>`;
    for (const [seriesRole, color, fillOpacity] of [
      ["non_chaser", "#334155", 0.16],
      ["chaser", "#b54848", 0.24],
    ]) {
      const points = densityRows
        .filter((row) => row.series_role === seriesRole)
        .sort((a, b) => Number(a.x) - Number(b.x));
      if (!points.length) {
        continue;
      }
      const linePath = pathFromPoints(points, "x", "density", x, yDensity);
      const areaPath = `${linePath} L${x(points[points.length - 1].x).toFixed(2)},${(densityTop + densityH).toFixed(2)} L${x(points[0].x).toFixed(2)},${(densityTop + densityH).toFixed(2)} Z`;
      svg += `<path d="${areaPath}" fill="${color}" opacity="${fillOpacity}" />`;
      svg += `<path d="${linePath}" fill="none" stroke="${color}" stroke-width="${seriesRole === "chaser" ? 2.2 : 1.8}"><title>${escapeHtml(seriesRole)} density</title></path>`;
    }
    if (phase.phase_kind === "post") {
      svg += `<text x="${x(0.08)}" y="${densityTop + 18}" fill="#b54848" font-size="18" font-weight="700">#</text>`;
    }
    svg += `<rect x="${margin.left}" y="${panelTop + 2}" width="10" height="10" fill="#b54848" />`;
    svg += `<text x="${margin.left + 16}" y="${panelTop + 11}" fill="#637080" font-size="11">chaser quadrant</text>`;
    svg += `<rect x="${margin.left + 142}" y="${panelTop + 2}" width="10" height="10" fill="#334155" opacity="0.45" />`;
    svg += `<text x="${margin.left + 158}" y="${panelTop + 11}" fill="#637080" font-size="11">non-chaser quadrants</text>`;
  }

  if (phases.length >= 2 && stat.p_value !== null && stat.p_value !== undefined) {
    const bracketX = width - margin.right - 6;
    const y1 = margin.top + densityTopOffset + densityH / 2;
    const y2 = margin.top + panelH + densityTopOffset + densityH / 2;
    svg += `<path d="M${bracketX - 16},${y1.toFixed(2)} H${bracketX} V${y2.toFixed(2)} H${bracketX - 16}" fill="none" stroke="#b54848" stroke-width="1.6" />`;
    svg += `<text x="${bracketX - 20}" y="${((y1 + y2) / 2 - 4).toFixed(2)}" text-anchor="end" fill="#b54848" font-size="11">paired</text>`;
    svg += `<text x="${bracketX - 20}" y="${((y1 + y2) / 2 + 10).toFixed(2)}" text-anchor="end" fill="#b54848" font-size="11">p=${fmt(stat.p_value, 4)}</text>`;
  }

  svg += `<text x="${margin.left}" y="${height - 8}" fill="#637080" font-size="11">Time spent in quadrant (normalized occupancy)</text>`;
  svg += "</svg>";
  return svg;
}

function nearFieldCurvesSvg(data) {
  if (!data || !data.available) {
    return '<div class="empty">No CRA near-field curve export found.</div>';
  }
  const width = 920;
  const height = 520;
  const margin = { top: 44, right: 24, bottom: 34, left: 58 };
  const panelW = 395;
  const panelH = 190;
  const gapX = 50;
  const gapY = 44;
  const roles = ["aggressive", "inert"];
  const phaseColors = { pre_static: "#2f6fbd", post_static: "#b54848", pre: "#2f6fbd", post: "#b54848" };
  const radialRows = data.radial_rows || [];
  const cdfRows = data.cdf_rows || [];
  const radialX = extentWithZero(radialRows.map((row) => row.radial_bin_center_mm), 0.02);
  radialX[0] = Math.max(0, radialX[0]);
  const radialY = extentWithZero(radialRows.map((row) => row.mean), 0.08);
  radialY[0] = Math.max(0, radialY[0]);
  const cdfX = extentWithZero(cdfRows.map((row) => row.distance_threshold_mm), 0.02);
  cdfX[0] = Math.max(0, cdfX[0]);
  const cdfY = [0, 1];

  function curvePanel(rows, role, title, panelX, panelY, xKey, yKey, xDomain, yDomain, yLabel) {
    const x = linearScale(xDomain[0], xDomain[1], panelX + 44, panelX + panelW - 12);
    const y = linearScale(yDomain[0], yDomain[1], panelY + panelH - 36, panelY + 22);
    let svg = `<text x="${panelX}" y="${panelY + 12}" fill="#182026" font-size="12" font-weight="700">${escapeHtml(title)}</text>`;
    svg += `<line x1="${panelX + 44}" y1="${panelY + panelH - 36}" x2="${panelX + panelW - 12}" y2="${panelY + panelH - 36}" stroke="#9aa5b1" />`;
    svg += `<line x1="${panelX + 44}" y1="${panelY + 22}" x2="${panelX + 44}" y2="${panelY + panelH - 36}" stroke="#9aa5b1" />`;
    svg += `<text x="${panelX + 44}" y="${panelY + panelH - 16}" fill="#637080" font-size="10">${fmt(xDomain[0], 1)}</text>`;
    svg += `<text x="${panelX + panelW - 12}" y="${panelY + panelH - 16}" text-anchor="end" fill="#637080" font-size="10">${fmt(xDomain[1], 1)} mm</text>`;
    svg += `<text x="${panelX + 2}" y="${panelY + 34}" fill="#637080" font-size="10">${escapeHtml(yLabel)}</text>`;
    const roleRows = rows.filter((row) => row.object_role === role);
    const phases = [...new Set(roleRows.map((row) => row.phase_label))].sort();
    for (const phaseLabel of phases) {
      const points = roleRows
        .filter((row) => row.phase_label === phaseLabel && Number.isFinite(Number(row[yKey])))
        .sort((a, b) => Number(a[xKey]) - Number(b[xKey]));
      if (!points.length) {
        continue;
      }
      const color = phaseColors[phaseLabel] || COLORS[phases.indexOf(phaseLabel) % COLORS.length];
      const path = pathFromPoints(points, xKey, yKey, x, y);
      svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2.2">`;
      svg += `<title>${escapeHtml(phaseLabel)} ${escapeHtml(role)}</title></path>`;
      for (const point of points) {
        svg += `<circle cx="${x(point[xKey]).toFixed(2)}" cy="${y(point[yKey]).toFixed(2)}" r="2.5" fill="${color}" opacity="0.7" />`;
      }
    }
    svg += `<rect x="${panelX + panelW - 126}" y="${panelY + 4}" width="9" height="9" fill="#2f6fbd" />`;
    svg += `<text x="${panelX + panelW - 112}" y="${panelY + 12}" fill="#637080" font-size="10">pre</text>`;
    svg += `<rect x="${panelX + panelW - 74}" y="${panelY + 4}" width="9" height="9" fill="#b54848" />`;
    svg += `<text x="${panelX + panelW - 60}" y="${panelY + 12}" fill="#637080" font-size="10">post</text>`;
    return svg;
  }

  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  svg += `<text x="${margin.left}" y="20" fill="#182026" font-size="14" font-weight="700">Near-field fish-level mean curves</text>`;
  svg += `<text x="${margin.left}" y="38" fill="#637080" font-size="11">Area-normalized radial density and small-distance CDF, aggregated by fish before plotting.</text>`;
  for (let roleIndex = 0; roleIndex < roles.length; roleIndex += 1) {
    const role = roles[roleIndex];
    const x0 = margin.left + roleIndex * (panelW + gapX);
    svg += curvePanel(radialRows, role, `Radial density: ${role}`, x0, margin.top, "radial_bin_center_mm", "mean", radialX, radialY, "density/mm2");
    svg += curvePanel(cdfRows, role, `Distance CDF: ${role}`, x0, margin.top + panelH + gapY, "distance_threshold_mm", "mean", cdfX, cdfY, "P(d < x)");
  }
  svg += "</svg>";
  return svg;
}

async function loadHealth() {
  try {
    const payload = await fetchJson("/healthz");
    setHealth(payload.ok, payload.ok ? "Healthy" : "Unhealthy");
  } catch (error) {
    setHealth(false, "Error");
  }
}

async function loadSummary() {
  const payload = await fetchJson("/api/export/summary");
  const summary = payload.summary;
  STATE.summary = summary;
  document.getElementById("export-line").textContent =
    `${summary.export_run_id} | ${summary.collection ? summary.collection.collection_id : "no collection"}`;
  document.getElementById("summary-recordings").textContent = fmt(summary.source_recording_count, 0);
  document.getElementById("summary-spatial").textContent = fmt(
    summary.row_counts_by_table.chaser_epoch_spatial_occupancy_zones,
    0
  );
  document.getElementById("summary-chaser").textContent = fmt(
    summary.row_counts_by_table.chaser_epoch_distance_summary,
    0
  );
  document.getElementById("summary-epoch-speed").textContent = fmt(
    summary.row_counts_by_table.chaser_epoch_behavior_summary,
    0
  );
  document.getElementById("summary-speed-distance").textContent = fmt(
    summary.row_counts_by_table.chaser_speed_distance_bins,
    0
  );
  document.getElementById("summary-bout-histogram").textContent = fmt(
    summary.row_counts_by_table.chaser_epoch_bout_histogram,
    0
  );
  document.getElementById("summary-ibi-histogram").textContent = fmt(
    summary.row_counts_by_table.chaser_epoch_inter_bout_interval_histogram,
    0
  );
  document.getElementById("summary-histogram").textContent = fmt(
    summary.row_counts_by_table.chaser_epoch_distance_histogram,
    0
  );
  document.getElementById("summary-cra").textContent = fmt(
    summary.row_counts_by_table.chaser_cra_primary_endpoint_summary,
    0
  );
  document.getElementById("summary-cra-object-phase").textContent = fmt(
    summary.row_counts_by_table.chaser_cra_primary_endpoint_object_phase,
    0
  );
  document.getElementById("summary-cra-quadrant").textContent = fmt(
    summary.row_counts_by_table.chaser_cra_quadrant_occupancy,
    0
  );
  document.getElementById("summary-cra-near-field").textContent = fmt(
    summary.row_counts_by_table.chaser_cra_near_field_summary,
    0
  );
  document.getElementById("summary-cra-near-field-object-phase").textContent = fmt(
    summary.row_counts_by_table.chaser_cra_near_field_object_phase,
    0
  );
  document.getElementById("summary-egocentric").textContent = fmt(
    summary.row_counts_by_table.chaser_egocentric_epoch_summary,
    0
  );
  document.getElementById("summary-egocentric-histogram").textContent = fmt(
    summary.row_counts_by_table.chaser_egocentric_distance_bearing_histogram,
    0
  );
  document.getElementById("summary-diagnostics").textContent = fmt(summary.diagnostics_count, 0);
  document.getElementById("summary-statistics").textContent =
    summary.statistics && summary.statistics.available ? fmt(summary.statistics.row_count, 0) : "none";
}

async function loadOptions() {
  const payload = await fetchJson("/api/options");
  const options = payload.options;
  STATE.options = options;
  populateSelect("spatial-metric", options.spatial_metrics, "metric", "label");
  populateSelect("chaser-metric", options.chaser_metrics, "metric", "label");
  populateSelect("epoch-speed-metric", options.epoch_speed_metrics || [], "metric", "label");
  populateSelect("epoch-bout-hist-metric", options.epoch_bout_histogram_metrics || [], "metric", "label");
  populateSelect("cra-metric", options.cra_object_phase_metrics, "metric", "label");
  populateSelect("cra-near-field-metric", options.cra_near_field_object_phase_metrics || [], "metric", "label");
  populateSelect("egocentric-metric", options.egocentric_metrics, "metric", "label");
  const hist = document.getElementById("hist-window");
  for (const item of options.windows) {
    const node = document.createElement("option");
    node.value = item.window_label;
    node.textContent = item.window_label;
    hist.appendChild(node);
  }
  const speedDistanceChaser = document.getElementById("speed-distance-chaser");
  for (const chaser of options.chasers || []) {
    const node = document.createElement("option");
    node.value = String(chaser);
    node.textContent = `chaser ${chaser}`;
    speedDistanceChaser.appendChild(node);
  }
}

async function loadSpatial() {
  const metric = document.getElementById("spatial-metric").value || "time_s";
  const valueMode = document.getElementById("spatial-value-mode").value || "auto";
  const payload = await fetchJson(
    `/api/chaser/spatial?metric=${encodeURIComponent(metric)}&value_mode=${encodeURIComponent(valueMode)}`
  );
  const data = payload.spatial;
  const groups = [...new Set(data.rows.map((row) => row.window_label))];
  const series = [...new Map(data.rows.map((row) => [row.zone_id, { key: row.zone_id, label: row.zone_label }])).values()];
  document.getElementById("spatial-meta").textContent = `${data.metric_label} | ${data.value_mode}`;
  document.getElementById("spatial-chart").innerHTML = groupedBarsSvg(data.rows, {
    groups,
    series,
    groupKey: "window_label",
    seriesKey: "zone_id",
    height: 330,
  });
  document.getElementById("spatial-table").innerHTML = tableHtml(data.rows, [
    { key: "window_label", label: "Epoch" },
    { key: "zone_label", label: "Zone" },
    { key: "value", label: "Value" },
    { key: "recording_count", label: "N" },
    { key: "sum", label: "Sum" },
    { key: "mean", label: "Mean" },
    { key: "std_dev", label: "Std dev" },
    { key: "sem", label: "SEM" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
}

async function loadChaserSummary() {
  const metric = document.getElementById("chaser-metric").value || "p50_distance_mm";
  const stat = document.getElementById("chaser-stat").value || "mean";
  const payload = await fetchJson(
    `/api/chaser/distance-summary?metric=${encodeURIComponent(metric)}&stat=${encodeURIComponent(stat)}`
  );
  const data = payload.chaser_summary;
  const groups = [...new Set(data.rows.map((row) => row.window_label))];
  const series = [...new Map(data.rows.map((row) => [row.chaser_index, { key: row.chaser_index, label: `chaser ${row.chaser_index}` }])).values()];
  document.getElementById("chaser-meta").textContent = `${data.metric_label} | ${data.stat}`;
  document.getElementById("chaser-chart").innerHTML = groupedBarsSvg(data.rows, {
    groups,
    series,
    groupKey: "window_label",
    seriesKey: "chaser_index",
    height: 260,
  });
  document.getElementById("chaser-table").innerHTML = tableHtml(data.rows, [
    { key: "window_label", label: "Epoch" },
    { key: "chaser_index", label: "Chaser" },
    { key: "value", label: "Value" },
    { key: "recording_count", label: "N" },
    { key: "mean", label: "Mean" },
    { key: "median", label: "Median" },
    { key: "std_dev", label: "Std dev" },
    { key: "sem", label: "SEM" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
}

async function loadHistogram() {
  const windowLabel = document.getElementById("hist-window").value;
  const params = new URLSearchParams();
  if (windowLabel) {
    params.set("window_label", windowLabel);
  }
  const payload = await fetchJson(`/api/chaser/distance-histogram?${params.toString()}`);
  const rows = payload.histogram.rows;
  const seriesCount = new Set(rows.map((row) => `${row.window_label}|${row.chaser_index}`)).size;
  document.getElementById("hist-meta").textContent = `${rows.length} bins | ${seriesCount} series`;
  document.getElementById("histogram-chart").innerHTML = histogramSvg(rows);
}

async function loadEpochSpeed() {
  const metric = document.getElementById("epoch-speed-metric").value || "mean_speed_mm_s";
  const stat = document.getElementById("epoch-speed-stat").value || "mean";
  const payload = await fetchJson(
    `/api/chaser/epoch-behavior?metric=${encodeURIComponent(metric)}&stat=${encodeURIComponent(stat)}`
  );
  const data = payload.epoch_speed;
  if (!data.available) {
    document.getElementById("epoch-speed-meta").textContent = data.message || "No epoch speed export";
    document.getElementById("epoch-speed-chart").innerHTML =
      '<div class="empty">No epoch speed export found for this cohort.</div>';
    document.getElementById("epoch-speed-table").innerHTML = "";
    return;
  }
  const groups = [...new Set(data.rows.map((row) => row.window_label))];
  const rows = data.rows.map((row) => ({ ...row, series: data.stat }));
  document.getElementById("epoch-speed-meta").textContent =
    `${data.metric_label} | ${data.stat} | ${data.source_label || "unknown_source"} | ${
      data.summary_source || "computed_from_export_rows"
    }`;
  document.getElementById("epoch-speed-chart").innerHTML = groupedBarsSvg(rows, {
    groups,
    series: [{ key: data.stat, label: data.stat }],
    groupKey: "window_label",
    seriesKey: "series",
    height: 260,
  });
  document.getElementById("epoch-speed-table").innerHTML = tableHtml(data.rows, [
    { key: "window_label", label: "Epoch" },
    { key: "value", label: "Value" },
    { key: "recording_count", label: "N" },
    { key: "mean", label: "Mean" },
    { key: "median", label: "Median" },
    { key: "std_dev", label: "Std dev" },
    { key: "sem", label: "SEM" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
}

async function loadSpeedDistance() {
  const windowLabel = document.getElementById("hist-window").value;
  const chaserValue = document.getElementById("speed-distance-chaser").value;
  const params = new URLSearchParams();
  if (windowLabel) {
    params.set("window_label", windowLabel);
  }
  if (chaserValue) {
    params.set("chaser_index", chaserValue);
  }
  const payload = await fetchJson(`/api/chaser/speed-distance?${params.toString()}`);
  const data = payload.speed_distance;
  if (!data.available) {
    document.getElementById("speed-distance-meta").textContent = data.message || "No speed-distance export";
    document.getElementById("speed-distance-chart").innerHTML =
      '<div class="empty">No speed-distance export found for this cohort.</div>';
    document.getElementById("speed-distance-table").innerHTML = "";
    return;
  }
  const seriesCount = new Set(data.rows.map((row) => `${row.window_label}|${row.chaser_index}`)).size;
  const sampleCount = data.rows.reduce((total, row) => total + Number(row.pooled_speed_sample_count || 0), 0);
  document.getElementById("speed-distance-meta").textContent =
    `${fmt(data.rows.length, 0)} bins | ${fmt(seriesCount, 0)} series | ${fmt(sampleCount, 0)} samples`;
  document.getElementById("speed-distance-chart").innerHTML = speedDistanceSvg(data.rows);
  document.getElementById("speed-distance-table").innerHTML = tableHtml(
    data.rows.filter((row) => Number(row.pooled_speed_sample_count || 0) > 0).slice(0, 80),
    [
      { key: "window_label", label: "Epoch" },
      { key: "chaser_index", label: "Chaser" },
      { key: "distance_bin_center_mm", label: "Dist mm" },
      { key: "pooled_mean_speed_mm_s", label: "Speed mm/s" },
      { key: "pooled_speed_sample_count", label: "Samples" },
      { key: "recording_count", label: "N rec" },
      { key: "recording_mean_speed_mm_s", label: "Rec mean" },
      { key: "recording_sem", label: "Rec SEM" },
    ]
  );
}

async function loadEpochBoutHistogram() {
  const metric = document.getElementById("epoch-bout-hist-metric").value || "bout_path_length_mm";
  const windowLabel = document.getElementById("hist-window").value;
  const params = new URLSearchParams();
  params.set("metric", metric);
  if (windowLabel) {
    params.set("window_label", windowLabel);
  }
  const payload = await fetchJson(`/api/chaser/epoch-bout-histogram?${params.toString()}`);
  const data = payload.epoch_bout_histogram;
  if (!data.available) {
    document.getElementById("epoch-bout-hist-meta").textContent =
      data.message || "No bout histogram export";
    document.getElementById("epoch-bout-hist-chart").innerHTML =
      '<div class="empty">No bout histogram export found for this cohort.</div>';
    document.getElementById("epoch-bout-hist-table").innerHTML = "";
    return;
  }
  const totalCount = data.rows.reduce((total, row) => total + Number(row.pooled_count || 0), 0);
  document.getElementById("epoch-bout-hist-meta").textContent =
    `${data.metric_label} | ${fmt(data.rows.length, 0)} bins | ${fmt(totalCount, 0)} bouts`;
  document.getElementById("epoch-bout-hist-chart").innerHTML =
    metricHistogramSvg(data.rows, data.metric_label);
  document.getElementById("epoch-bout-hist-table").innerHTML = tableHtml(
    data.rows.filter((row) => Number(row.pooled_count || 0) > 0).slice(0, 80),
    [
      { key: "window_label", label: "Epoch" },
      { key: "bin_center", label: "Bin center" },
      { key: "pooled_count", label: "Count" },
      { key: "pooled_fraction", label: "Fraction" },
      { key: "pooled_density", label: "Density" },
      { key: "recording_count", label: "N rec" },
    ]
  );
}

async function loadEpochInterBoutIntervalHistogram() {
  const windowLabel = document.getElementById("hist-window").value;
  const params = new URLSearchParams();
  if (windowLabel) {
    params.set("window_label", windowLabel);
  }
  const payload = await fetchJson(
    `/api/chaser/epoch-inter-bout-interval-histogram?${params.toString()}`
  );
  const data = payload.epoch_inter_bout_interval_histogram;
  if (!data.available) {
    document.getElementById("epoch-ibi-hist-meta").textContent =
      data.message || "No inter-bout interval histogram export";
    document.getElementById("epoch-ibi-hist-chart").innerHTML =
      '<div class="empty">No inter-bout interval histogram export found for this cohort.</div>';
    document.getElementById("epoch-ibi-hist-table").innerHTML = "";
    return;
  }
  const totalCount = data.rows.reduce((total, row) => total + Number(row.pooled_count || 0), 0);
  document.getElementById("epoch-ibi-hist-meta").textContent =
    `${data.metric_label} | ${fmt(data.rows.length, 0)} bins | ${fmt(totalCount, 0)} intervals`;
  document.getElementById("epoch-ibi-hist-chart").innerHTML =
    metricHistogramSvg(data.rows, data.metric_label);
  document.getElementById("epoch-ibi-hist-table").innerHTML = tableHtml(
    data.rows.filter((row) => Number(row.pooled_count || 0) > 0).slice(0, 80),
    [
      { key: "window_label", label: "Epoch" },
      { key: "bin_center", label: "Bin center" },
      { key: "pooled_count", label: "Count" },
      { key: "pooled_fraction", label: "Fraction" },
      { key: "pooled_density", label: "Density" },
      { key: "recording_count", label: "N rec" },
    ]
  );
}

async function loadCraObjectPhase() {
  const metric = document.getElementById("cra-metric").value || "median_distance_mm";
  const stat = document.getElementById("cra-stat").value || "mean";
  const payload = await fetchJson(
    `/api/chaser/cra-object-phase?metric=${encodeURIComponent(metric)}&stat=${encodeURIComponent(stat)}`
  );
  const data = payload.cra_object_phase;
  const groups = [...new Set(data.rows.map((row) => row.phase_label))];
  const series = [...new Map(data.rows.map((row) => [row.object_role, { key: row.object_role, label: row.object_role }])).values()];
  document.getElementById("cra-meta").textContent = `${data.metric_label} | ${data.stat}`;
  document.getElementById("cra-chart").innerHTML = groupedBarsSvg(data.rows, {
    groups,
    series,
    groupKey: "phase_label",
    seriesKey: "object_role",
    height: 260,
  });
  document.getElementById("cra-table").innerHTML = tableHtml(data.rows, [
    { key: "phase_label", label: "Phase" },
    { key: "object_role", label: "Role" },
    { key: "raw_color_hex", label: "Raw color" },
    { key: "object_quadrant_label", label: "Object quadrant" },
    { key: "value", label: "Value" },
    { key: "recording_count", label: "N" },
    { key: "mean", label: "Mean" },
    { key: "median", label: "Median" },
    { key: "std_dev", label: "Std dev" },
    { key: "sem", label: "SEM" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
}

async function loadCraSummary() {
  const payload = await fetchJson("/api/chaser/cra-summary");
  const data = payload.cra_summary;
  document.getElementById("cra-summary-meta").textContent = `${fmt(data.row_count, 0)} recordings`;
  document.getElementById("cra-summary-table").innerHTML = tableHtml(data.metrics, [
    { key: "metric_label", label: "Metric" },
    { key: "n", label: "N" },
    { key: "mean", label: "Mean" },
    { key: "median", label: "Median" },
    { key: "std_dev", label: "Std dev" },
    { key: "sem", label: "SEM" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
  document.getElementById("cra-recording-table").innerHTML = tableHtml(data.rows || [], [
    { key: "recording_id", label: "Recording" },
    { key: "endpoint_status", label: "Status" },
    { key: "aggressive_color", label: "Agg color" },
    { key: "inert_color", label: "Inert color" },
    { key: "delta_agg", label: "dAgg mm" },
    { key: "delta_inert", label: "dInert mm" },
    { key: "specificity_distance", label: "Spec dist" },
    { key: "delta_occ_agg", label: "dOcc agg" },
    { key: "delta_occ_inert", label: "dOcc inert" },
    { key: "specificity_occupancy", label: "Spec occ" },
    { key: "frac_tracking_dropout_pre", label: "Drop pre" },
    { key: "frac_tracking_dropout_post", label: "Drop post" },
    { key: "pre_aggressive_quadrant", label: "Pre agg quad" },
    { key: "post_aggressive_quadrant", label: "Post agg quad" },
    { key: "source_cra_primary_endpoint_path", label: "Source component" },
  ]);
}

async function loadCraSpecificity() {
  const payload = await fetchJson("/api/chaser/cra-specificity");
  const data = payload.cra_specificity;
  if (!data.available) {
    document.getElementById("cra-specificity-meta").textContent = data.message || "No CRA specificity rows";
    document.getElementById("cra-specificity-chart").innerHTML =
      '<div class="empty">No CRA specificity rows found for this cohort.</div>';
    document.getElementById("cra-specificity-table").innerHTML = "";
    return;
  }
  const distStat = data.distance_specificity_statistics || {};
  const occStat = data.occupancy_index_specificity_statistics || {};
  document.getElementById("cra-specificity-meta").textContent =
    `${fmt(data.recording_count, 0)} recordings | distance p=${fmt(distStat.p_value, 4)} | occupancy-index p=${fmt(occStat.p_value, 4)}`;
  document.getElementById("cra-specificity-chart").innerHTML = craSpecificitySvg(data);
  document.getElementById("cra-specificity-table").innerHTML = tableHtml(
    (data.distance_specificity_rows || []).map((row) => ({
      ...row,
      occupancy_index_specificity: (data.occupancy_index_specificity_rows || []).find(
        (occRow) => occRow.recording_id === row.recording_id
      )?.occupancy_index_specificity,
    })),
    [
      { key: "recording_id", label: "Recording" },
      { key: "specificity_distance", label: "Distance spec" },
      { key: "delta_agg", label: "dAgg mm" },
      { key: "delta_inert", label: "dInert mm" },
      { key: "occupancy_index_specificity", label: "Occ-index spec" },
    ]
  );
}

async function loadCraStatistics() {
  const payload = await fetchJson("/api/chaser/statistics?metric_family=cra_primary_endpoint");
  const data = payload.statistics;
  if (!data.available || !data.rows.length) {
    document.getElementById("cra-statistics-meta").textContent = data.message || "No CRA statistics";
    document.getElementById("cra-statistics-table").innerHTML =
      '<div class="empty">No CRA statistics export found for this cohort.</div>';
    return;
  }
  document.getElementById("cra-statistics-meta").textContent =
    `${data.stats_run_id} | ${fmt(data.row_count, 0)} rows | ${data.source_export_run_id}`;
  document.getElementById("cra-statistics-table").innerHTML = tableHtml(data.rows, [
    { key: "metric_name", label: "Metric" },
    { key: "contrast_name", label: "Contrast" },
    { key: "paired_unit_count", label: "N paired" },
    { key: "mean_difference", label: "Mean delta" },
    { key: "median_difference", label: "Median delta" },
    { key: "ci_low", label: "CI low" },
    { key: "ci_high", label: "CI high" },
    { key: "effect_size", label: "Rank-biserial" },
    { key: "p_value", label: "p" },
    { key: "q_value", label: "q" },
    { key: "test_method", label: "Test" },
    { key: "status", label: "Status" },
  ]);
}

async function loadCraQuadrantDensity() {
  const payload = await fetchJson("/api/chaser/cra-quadrant-occupancy-density");
  const data = payload.cra_quadrant_occupancy_density;
  if (!data.available) {
    document.getElementById("cra-quadrant-density-meta").textContent = data.message || "No CRA quadrant table";
    document.getElementById("cra-quadrant-density-chart").innerHTML =
      '<div class="empty">No CRA quadrant occupancy export found for this cohort.</div>';
    document.getElementById("cra-quadrant-density-table").innerHTML = "";
    document.getElementById("cra-quadrant-density-paired-table").innerHTML = "";
    return;
  }
  const stat = data.statistics || {};
  document.getElementById("cra-quadrant-density-meta").textContent =
    `${fmt(data.recording_count, 0)} recordings | descriptive chaser-only p=${fmt(stat.p_value, 4)} | KDE bandwidth ${fmt(data.kde.bandwidth, 3)}`;
  document.getElementById("cra-quadrant-density-chart").innerHTML = quadrantOccupancyDensitySvg(data);
  document.getElementById("cra-quadrant-density-table").innerHTML = tableHtml(data.quadrant_rows || [], [
    { key: "phase_label", label: "Phase" },
    { key: "quadrant_id", label: "Quadrant" },
    { key: "recording_count", label: "N" },
    { key: "chaser_recording_count", label: "N chaser" },
    { key: "mean", label: "Mean" },
    { key: "sem", label: "SEM" },
    { key: "median", label: "Median" },
    { key: "std_dev", label: "Std dev" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
  document.getElementById("cra-quadrant-density-paired-table").innerHTML = tableHtml(data.paired_rows || [], [
    { key: "recording_id", label: "Recording" },
    { key: "pre_chaser_quadrant", label: "Pre chaser quadrant" },
    { key: "post_chaser_quadrant", label: "Post chaser quadrant" },
    { key: "pre_chaser_quadrant_occ", label: "Pre occ" },
    { key: "post_chaser_quadrant_occ", label: "Post occ" },
    { key: "delta_chaser_quadrant_occ", label: "Delta" },
    { key: "pre_tracking_dropout_fraction", label: "Drop pre" },
    { key: "post_tracking_dropout_fraction", label: "Drop post" },
  ]);
}

async function loadCraNearFieldCurves() {
  const payload = await fetchJson("/api/chaser/cra-near-field-curves");
  const data = payload.cra_near_field_curves;
  if (!data.available) {
    document.getElementById("cra-near-field-curves-meta").textContent = data.message || "No near-field curves";
    document.getElementById("cra-near-field-curves-chart").innerHTML =
      '<div class="empty">No CRA near-field curve export found for this cohort.</div>';
    document.getElementById("cra-near-field-curves-table").innerHTML = "";
    return;
  }
  document.getElementById("cra-near-field-curves-meta").textContent =
    `${fmt(data.recording_count, 0)} recordings | ${fmt(data.radial_rows.length, 0)} radial bins | CDF <= ${fmt(data.max_cdf_threshold_mm, 1)} mm`;
  document.getElementById("cra-near-field-curves-chart").innerHTML = nearFieldCurvesSvg(data);
  document.getElementById("cra-near-field-curves-table").innerHTML = tableHtml(
    (data.cdf_rows || []).filter((row) => row.distance_threshold_mm <= data.max_cdf_threshold_mm).slice(0, 80),
    [
      { key: "phase_label", label: "Phase" },
      { key: "object_role", label: "Role" },
      { key: "distance_threshold_mm", label: "d mm" },
      { key: "recording_count", label: "N" },
      { key: "mean", label: "Mean CDF" },
      { key: "sem", label: "SEM" },
      { key: "median", label: "Median" },
    ]
  );
}

async function loadCraNearFieldObjectPhase() {
  const metric = document.getElementById("cra-near-field-metric").value || "near_zone_occupancy_fraction";
  const stat = document.getElementById("cra-near-field-stat").value || "mean";
  const payload = await fetchJson(
    `/api/chaser/cra-near-field-object-phase?metric=${encodeURIComponent(metric)}&stat=${encodeURIComponent(stat)}`
  );
  const data = payload.cra_near_field_object_phase;
  const groups = [...new Set(data.rows.map((row) => row.phase_label))];
  const series = [...new Map(data.rows.map((row) => [row.object_role, { key: row.object_role, label: row.object_role }])).values()];
  document.getElementById("cra-near-field-meta").textContent = `${data.metric_label} | ${data.stat}`;
  document.getElementById("cra-near-field-chart").innerHTML = groupedBarsSvg(data.rows, {
    groups,
    series,
    groupKey: "phase_label",
    seriesKey: "object_role",
    height: 260,
  });
  document.getElementById("cra-near-field-table").innerHTML = tableHtml(data.rows, [
    { key: "phase_label", label: "Phase" },
    { key: "object_role", label: "Role" },
    { key: "raw_color_hex", label: "Raw color" },
    { key: "value", label: "Value" },
    { key: "recording_count", label: "N" },
    { key: "mean", label: "Mean" },
    { key: "median", label: "Median" },
    { key: "std_dev", label: "Std dev" },
    { key: "sem", label: "SEM" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
}

async function loadCraNearFieldSummary() {
  const payload = await fetchJson("/api/chaser/cra-near-field-summary");
  const data = payload.cra_near_field_summary;
  document.getElementById("cra-near-field-summary-meta").textContent = `${fmt(data.row_count, 0)} recordings`;
  document.getElementById("cra-near-field-summary-table").innerHTML = tableHtml(data.metrics, [
    { key: "metric_label", label: "Metric" },
    { key: "n", label: "N" },
    { key: "mean", label: "Mean" },
    { key: "median", label: "Median" },
    { key: "std_dev", label: "Std dev" },
    { key: "sem", label: "SEM" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
  document.getElementById("cra-near-field-recording-table").innerHTML = tableHtml(data.rows || [], [
    { key: "recording_id", label: "Recording" },
    { key: "endpoint_status", label: "Status" },
    { key: "aggressive_color", label: "Agg color" },
    { key: "inert_color", label: "Inert color" },
    { key: "approach_p05_specificity", label: "P05 spec" },
    { key: "approach_p10_specificity", label: "P10 spec" },
    { key: "nearzone_occ_specificity", label: "NZ occ spec" },
    { key: "nearzone_entry_rate_specificity", label: "Entry spec" },
    { key: "thigmotaxis_frac_pre", label: "Thig pre" },
    { key: "thigmotaxis_frac_post", label: "Thig post" },
    { key: "frac_tracking_dropout_pre", label: "Drop pre" },
    { key: "frac_tracking_dropout_post", label: "Drop post" },
    { key: "geometry_status", label: "Geometry" },
    { key: "arena_shape", label: "Arena" },
    { key: "source_cra_near_field_path", label: "Source component" },
  ]);
}

async function loadCraNearFieldStatistics() {
  const payload = await fetchJson("/api/chaser/statistics?metric_family=cra_near_field");
  const data = payload.statistics;
  if (!data.available || !data.rows.length) {
    document.getElementById("cra-near-field-statistics-meta").textContent = data.message || "No near-field statistics";
    document.getElementById("cra-near-field-statistics-table").innerHTML =
      '<div class="empty">No CRA near-field statistics export found for this cohort.</div>';
    return;
  }
  document.getElementById("cra-near-field-statistics-meta").textContent =
    `${data.stats_run_id} | ${fmt(data.row_count, 0)} rows | ${data.source_export_run_id}`;
  document.getElementById("cra-near-field-statistics-table").innerHTML = tableHtml(data.rows, [
    { key: "metric_name", label: "Metric" },
    { key: "contrast_name", label: "Contrast" },
    { key: "paired_unit_count", label: "N paired" },
    { key: "mean_difference", label: "Mean delta" },
    { key: "median_difference", label: "Median delta" },
    { key: "ci_low", label: "CI low" },
    { key: "ci_high", label: "CI high" },
    { key: "effect_size", label: "Rank-biserial" },
    { key: "p_value", label: "p" },
    { key: "q_value", label: "q" },
    { key: "test_method", label: "Test" },
    { key: "status", label: "Status" },
  ]);
}

async function loadEgocentricSummary() {
  const metric = document.getElementById("egocentric-metric").value || "mean_alignment_cos";
  const stat = document.getElementById("egocentric-stat").value || "mean";
  const payload = await fetchJson(
    `/api/chaser/egocentric-summary?metric=${encodeURIComponent(metric)}&stat=${encodeURIComponent(stat)}`
  );
  const data = payload.egocentric_summary;
  const groups = [...new Set(data.rows.map((row) => row.window_label))];
  const series = [...new Map(data.rows.map((row) => [row.chaser_index, { key: row.chaser_index, label: `chaser ${row.chaser_index}` }])).values()];
  document.getElementById("egocentric-meta").textContent = `${data.metric_label} | ${data.stat}`;
  document.getElementById("egocentric-chart").innerHTML = groupedBarsSvg(data.rows, {
    groups,
    series,
    groupKey: "window_label",
    seriesKey: "chaser_index",
    height: 260,
  });
  document.getElementById("egocentric-table").innerHTML = tableHtml(data.rows, [
    { key: "window_label", label: "Epoch" },
    { key: "chaser_index", label: "Chaser" },
    { key: "value", label: "Value" },
    { key: "recording_count", label: "N" },
    { key: "mean", label: "Mean" },
    { key: "median", label: "Median" },
    { key: "std_dev", label: "Std dev" },
    { key: "sem", label: "SEM" },
    { key: "min", label: "Min" },
    { key: "max", label: "Max" },
  ]);
}

async function loadEgocentricHistogram() {
  const windowLabel = document.getElementById("hist-window").value;
  const params = new URLSearchParams();
  if (windowLabel) {
    params.set("window_label", windowLabel);
  }
  const payload = await fetchJson(`/api/chaser/egocentric-histogram?${params.toString()}`);
  const rows = payload.histogram.rows;
  const seriesCount = new Set(rows.map((row) => `${row.window_label}|${row.chaser_index}`)).size;
  document.getElementById("egocentric-hist-meta").textContent = `${rows.length} bins | ${seriesCount} series`;
  document.getElementById("egocentric-histogram-chart").innerHTML = bearingDistanceHeatmapSvg(rows);
  document.getElementById("egocentric-histogram-table").innerHTML = tableHtml(rows.slice(0, 80), [
    { key: "window_label", label: "Epoch" },
    { key: "chaser_index", label: "Chaser" },
    { key: "distance_bin_center_mm", label: "Dist mm" },
    { key: "bearing_bin_center_deg", label: "Bearing" },
    { key: "pooled_count", label: "Count" },
    { key: "pooled_probability", label: "Prob" },
  ]);
}

async function loadRecordings() {
  const payload = await fetchJson("/api/chaser/recordings");
  const rows = payload.recordings.rows;
  document.getElementById("recording-meta").textContent = `${payload.recordings.row_count} recordings`;
  document.getElementById("recording-table").innerHTML = tableHtml(rows, [
    { key: "recording_id", label: "Recording" },
    { key: "pre_event_coverage_pct", label: "Pre cov" },
    { key: "training_event_coverage_pct", label: "Train cov" },
    { key: "post_event_coverage_pct", label: "Post cov" },
    { key: "pre_event_chaser_0_p50_mm", label: "Pre c0 p50" },
    { key: "pre_event_chaser_1_p50_mm", label: "Pre c1 p50" },
    { key: "post_event_chaser_0_p50_mm", label: "Post c0 p50" },
    { key: "post_event_chaser_1_p50_mm", label: "Post c1 p50" },
    { key: "pre_event_mean_speed_mm_s", label: "Pre speed" },
    { key: "training_event_mean_speed_mm_s", label: "Train speed" },
    { key: "post_event_mean_speed_mm_s", label: "Post speed" },
    { key: "cra_endpoint_status", label: "CRA status" },
    { key: "cra_delta_agg_mm", label: "CRA dAgg" },
    { key: "cra_delta_inert_mm", label: "CRA dInert" },
    { key: "cra_specificity_distance_mm", label: "CRA spec dist" },
    { key: "cra_delta_occ_agg", label: "CRA dOcc agg" },
    { key: "cra_specificity_occupancy", label: "CRA spec occ" },
    { key: "pre_event_chaser_0_alignment", label: "Pre c0 align" },
    { key: "pre_event_chaser_1_alignment", label: "Pre c1 align" },
    { key: "post_event_chaser_0_alignment", label: "Post c0 align" },
    { key: "post_event_chaser_1_alignment", label: "Post c1 align" },
  ]);
}

async function loadStatistics() {
  const payload = await fetchJson("/api/chaser/statistics");
  const data = payload.statistics;
  if (!data.available) {
    document.getElementById("statistics-meta").textContent = data.message || "No statistics run";
    document.getElementById("statistics-table").innerHTML = '<div class="empty">No statistics export found for this cohort.</div>';
    return;
  }
  document.getElementById("statistics-meta").textContent =
    `${data.stats_run_id} | ${fmt(data.row_count, 0)} rows | ${data.source_export_run_id}`;
  document.getElementById("statistics-table").innerHTML = tableHtml(data.rows, [
    { key: "metric_family", label: "Family" },
    { key: "metric_name", label: "Metric" },
    { key: "contrast_name", label: "Contrast" },
    { key: "group", label: "Group" },
    { key: "paired_unit_count", label: "N paired" },
    { key: "mean_a", label: "Mean A" },
    { key: "mean_b", label: "Mean B" },
    { key: "mean_difference", label: "Delta" },
    { key: "ci_low", label: "CI low" },
    { key: "ci_high", label: "CI high" },
    { key: "effect_size", label: "Effect" },
    { key: "p_value", label: "p" },
    { key: "q_value", label: "q" },
    { key: "test_method", label: "Test" },
    { key: "status", label: "Status" },
    { key: "skip_reason", label: "Skip reason" },
  ]);
}

async function loadProvenance() {
  const payload = await fetchJson("/api/chaser/provenance");
  document.getElementById("provenance-meta").textContent = payload.provenance.summary.manifest_path;
  document.getElementById("provenance-json").textContent = JSON.stringify(payload.provenance, null, 2);
}

async function refreshAll() {
  await Promise.all([
    loadHealth(),
    loadSummary(),
    loadSpatial(),
    loadChaserSummary(),
    loadHistogram(),
    loadEpochSpeed(),
    loadSpeedDistance(),
    loadEpochBoutHistogram(),
    loadEpochInterBoutIntervalHistogram(),
    loadCraObjectPhase(),
    loadCraSummary(),
    loadCraSpecificity(),
    loadCraStatistics(),
    loadCraQuadrantDensity(),
    loadCraNearFieldCurves(),
    loadCraNearFieldObjectPhase(),
    loadCraNearFieldSummary(),
    loadCraNearFieldStatistics(),
    loadEgocentricSummary(),
    loadEgocentricHistogram(),
    loadStatistics(),
    loadRecordings(),
    loadProvenance(),
  ]);
}

function bindControls() {
  for (const id of ["spatial-metric", "spatial-value-mode"]) {
    document.getElementById(id).addEventListener("change", loadSpatial);
  }
  for (const id of ["chaser-metric", "chaser-stat"]) {
    document.getElementById(id).addEventListener("change", loadChaserSummary);
  }
  for (const id of ["epoch-speed-metric", "epoch-speed-stat"]) {
    document.getElementById(id).addEventListener("change", loadEpochSpeed);
  }
  document.getElementById("epoch-bout-hist-metric").addEventListener("change", loadEpochBoutHistogram);
  for (const id of ["cra-metric", "cra-stat"]) {
    document.getElementById(id).addEventListener("change", loadCraObjectPhase);
  }
  for (const id of ["cra-near-field-metric", "cra-near-field-stat"]) {
    document.getElementById(id).addEventListener("change", loadCraNearFieldObjectPhase);
  }
  for (const id of ["egocentric-metric", "egocentric-stat"]) {
    document.getElementById(id).addEventListener("change", loadEgocentricSummary);
  }
  document.getElementById("hist-window").addEventListener("change", () => {
    loadHistogram();
    loadSpeedDistance();
    loadEpochBoutHistogram();
    loadEpochInterBoutIntervalHistogram();
    loadEgocentricHistogram();
  });
  document.getElementById("speed-distance-chaser").addEventListener("change", loadSpeedDistance);
  document.getElementById("refresh-button").addEventListener("click", refreshAll);
}

async function init() {
  try {
    await loadOptions();
    bindControls();
    await refreshAll();
  } catch (error) {
    setHealth(false, "Error");
    document.getElementById("export-line").textContent = String(error);
  }
}

init();
