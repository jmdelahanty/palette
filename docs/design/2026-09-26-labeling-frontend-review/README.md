# Labeling web app: front-end review and simplification plan

- **Status:** proposed; three decisions are needed (see the end)
- **Owner:** labeling/Apply work (session palette-12, for the user); last reviewed 2026-09-26
- **Evidence:** four parallel read-only reviews of main at 2d182a06, covering UX and visual design, JavaScript structure, server-rendered HTML and payloads, and serving/security/styling. Where measurements were needed, they were taken against a copy of the store on a private port.

## What we have

| Layer | Size | Notes |
|---|---:|---|
| `labeling/` Python | 64,632 lines | `web.py` alone is 12,623 lines |
| JavaScript | ~6,100 lines | only 2,834 of them are in `static/js`; ~1,150 are inline in admin templates and ~2,150 sit inside Python strings (`web_personal_renderers.py`) |
| CSS | ~2,520 lines inline | 20 `:root` blocks, 88 hex colours (with drifted near-duplicates), 3 font stacks, and one 44-line stylesheet |
| Templating | `@@key@@` string replacement | the `.j2` files are not Jinja |
| Serving | all JS/CSS inlined into every page, `no-store`, no gzip | the full editor is re-sent on every navigation |
| Framework | Flask is already a dependency and serves 33 routes | the stdlib handler still owns `/`, signed links, and ~22 session and ~14 admin routes |

**Payload sizes** (8 datasets, 10 tasks):

| Response | Size |
|---|---:|
| `/my-datasets` page | 120 KB |
| `/api/me/datasets` | 295 KB (788 keys) |
| `/api/me/tasks` | 291 KB |
| `/api/admin/summary` | 984 KB |

- About 87–90% of the keys feed only copy-paste diagnostic text, or nothing at all.
- The per-row "support reference" is built eagerly in the browser: 26.7 KB per row, about 748 KB of hidden DOM text.
- `datasets` and `dataset_queue` are the same 109 KB array.
- 72 top-level fields are flat copies of nested fields.

**Code built for a campus launch but idle in solo use:** the handoff bundles, rosters, launch bundles, operator-evidence records, validation reports and batch plans, about 18,000 lines. The web request handler references none of it; it is reachable only through CLI subcommands that no script calls. About 10,000 lines of tests cover it. That includes ~645 `assert "field" in html` lines that pin field names in page JavaScript.

## Findings a labeler feels, ranked

1. **n / p / Go to ROI silently discard unsaved edits.** There is no unsaved indicator, no confirmation and no `beforeunload` handler.
2. **Hotkeys fire with Ctrl/Cmd/Alt held.**
   - Ctrl+R (reload) removes stray pieces; Ctrl+X marks a row as having no keypoints.
   - Keypoint `x` acts instantly.
   - The same key means different things in the two editors (`r`, `x`), and `f`/`e` change meaning in lasso mode.
   - *The modifier part is fixed in #224.*
3. **The review-status dropdown defaults to "approved"** and never shows the row's current state, so one click approves a row.
4. **The mask editor can undo only fills and stray removal.** Brush strokes and "Clear local" cannot be undone, and "Clear local" sits next to "Complete task".
5. **Success looks like an error.** Every successful save shows a red "operator support" box, which teaches the labeler to ignore red.
6. **Queue pages are dominated by internal text:** policy paragraphs, `zarr_use=`, "authorization contract: ready", and internal field names.
7. **The image is not the main element.** A 4.2rem heading and a session banner sit above the canvas, the canvas has no height cap, and the status line is below the fold.
8. **Controls are a flat grid:** 18 buttons in 3 colours, mixing navigation, tools, save, Apply and Complete. "Complete" is styled as a warning.
9. **Terminology is inconsistent:** Save, Checkpoint, Apply, "Apply saved edits to Zarr", "persist", "exclusive canonical writer".
10. **Disabled buttons don't say why,** and the mask page has no disabled style at all.
11. **Hotkeys are hard to discover:** one run-on help paragraph, and panning needs a middle mouse button.
12. **The two editors and the queue pages use three different palettes and fonts.**

**Blocking packaging bug:** the wheel's package-data leaves out `templates/**/*.html.j2`. An installed (non-editable) deployment would fail on every session and admin page. The CI wheel smoke test never imports `fisheye.labeling`, so it misses this.

## Recommended direction

- **No build step now.** Use native ES modules served as static files, and vendor Preact + htm (about 5 KB, no `unsafe-eval`) only for DOM panels: queues, summaries and dashboards.
- **Canvas and editing cores stay plain, imperative JS modules.** This is the right tool for pixel and point editing.
- **Rejected alternatives:**
  - *Alpine / petite-vue:* they need `unsafe-eval`, and they fit canvas state poorly.
  - *htmx:* it pushes rendering into `web.py`.
  - *Vite:* it adds a node toolchain and bundle-in-CI decisions that are too big for one maintainer and about 6k lines.
- **One stylesheet** (`static/css/palette.css`) with design tokens, and per-page CSS only for real layout differences.
- **One design system:**
  - Colours: base (bg/surface/ink/ink-2/line), one accent, status pairs (ok, pending, busy, danger).
  - Spacing: 4/8/12/16/24/32. Radius: 6 for controls, 10 for cards.
  - Type: a system-ui scale of 12/14/16/20/24.
  - Components: buttons (primary/secondary/ghost/danger with `:disabled` and `:focus-visible`), a segmented tool picker, `<kbd>` keycaps, status chips, a pinned status bar, a toast with Undo, and a collapsible "Operator details".
- **Editor layout:** a 48px top bar (queue ◀ · recording · ROI n/N · Unsaved chip · ?), the canvas capped to the viewport, a right panel grouped as Tools / View / Row review, and a pinned bottom bar (◀ Prev · Save · Save+Next · Next ▶ · status · "3 saved · Publish" · Complete). Both editors use the same frame.
- **Two verbs only:** Save (a row) and Publish (the current Apply).

## Staged plan (each stage is small PRs; the app works after every step)

| Stage | What | Kind | Estimated effect |
|---|---|---|---|
| **P0 fixes** | #224 hotkeys; package `**/*.j2` plus an installed-wheel render smoke; unsaved-edit guard on n/p; review control shows the current state (no default approve); success is not red | bug fixes | removes the labeler risks |
| **P1 subtract** | replace per-row support text with a ~15-field reference plus a lazy admin endpoint; drop the duplicated `dataset_queue` and the flat copies; trim 403/409 bodies; pin the ~100 keys the JS reads with a keyset test | behaviour-preserving (the tests change from string presence to behaviour) | −1,000 JS lines, −750 KB DOM, `/api/me/*` from ~290 KB to ~60 KB |
| **P2 serve** | a `/static/` route with content-hashed URLs and long caching; move the two personal pages' HTML/CSS/JS out of Python into files (byte-identical first); dedupe the security-header dict | behaviour-preserving | −2,700 Python lines |
| **P3 style** | `palette.css` and tokens; migrate templates, then renderers; editor layout redesign | UI change | ~2,500 inline CSS lines → one sheet |
| **P4 JS core** | `core/` modules (api, status, busy, apply, hotkeys, image, bbox); editors migrate one at a time (video_detect → detect → keypoint → mask) with ported tests; Preact+htm for the queue and admin panels; 84 inline handlers → 0 | refactor | ~6,100 → ~5,200 JS lines; one tested Apply and error path |
| **P5 harden** | CSP `script-src 'self'; style-src 'self'` (versioned: supersedes recorded header evidence); finish moving routes to Flask and retire the stdlib handler | enforcement | real XSS protection for 51 `innerHTML` sites |
| **P6 decide** | the ~18k-line handoff/launch/evidence CLI family | your decision | −18k source, −10k test lines if removed |

## Decisions needed

1. **Framework:** native ES modules plus vendored Preact+htm, no build step (recommended), or a Vite build?
2. **The campus-launch CLI family (P6):** freeze and move it into a separate package outside the served app, or delete it now and rebuild a smaller version when campus hosting is real?
3. **CSP tightening (P5):** OK to supersede the header evidence recorded by the current operator-validation contract?
