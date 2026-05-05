# Palette Manim animations

Manim Community animations that visualize the geometry behind Palette's
eye-angle pipeline. Built primarily as a teaching artifact for the v5
refactor (`eye_angle_analysis.v5`) — the silent-failure mode that motivated
it, the canonical Bianco-style resolution that fixes it, and the vergence
math that consumes the output.

This README is a handoff for future agents working on this animation. Read
it before changing geometry, scene structure, or rendering setup.

## File structure

```
apps/manim/
├── README.md             ← this file
└── gaze_flip_rule.py     ← all scenes + GazeBase shared infrastructure
```

The companion teaching doc is `docs/eye_axis_half_plane_margin.md` — the
text-form explanation of the `|m̂·f̂|` margin concept that Scene 4 visualizes.

## What's built

### Scenes (in `gaze_flip_rule.py`)

| Scene class          | What it shows |
|----------------------|---------------|
| `Scene1Setup`        | Definitions card → fish anatomy → heading-from-swim-bladder-to-eye-midpoint derivation → `l̂` axis → half-plane shading |
| `Scene2Typical`      | Minor-axis flip rule resolving correctly through typical larval gaze; fitter-sign toggle to demonstrate invariance |
| `Scene3Failure`      | Eye rotates past 90°; rule silently flips gaze 180° from truth, with green-vs-red arrow contrast and "180° error" label |
| `Scene4MajorAxis`    | Bianco-style resolution on the *major* axis with T/N endpoint labels and live `\|m̂·f̂\|` margin readout |
| `Scene5Boundary`     | Boundary-jitter contrast: small fit noise near the half-plane boundary causes minor-axis-resolved gaze to flicker by 180° |
| `Scene6Vergence`     | Both eyes converging symmetrically with monocular pink/blue cones, green binocular overlap, and live Bianco-style vergence math |
| `MasterCorrectScene` | "Correct way" sequence: definitions → setup → major-axis-clean → vergence (skips failure scenes) |
| `MasterScene`        | Full sequence including failure modes; useful for the "why we needed v5" story |

### Two flavors of major-axis presentation

- `play_major_axis` (used by `Scene4MajorAxis` and `MasterScene`) tears down
  the failing minor-axis dynamics from the preceding scenes, then introduces
  the canonical method as a contrast.
- `play_major_axis_clean` (used by `MasterCorrectScene`) introduces the
  canonical method directly without first showing the failure machinery, and
  rotates the eye from rest into a converged position so the viewer sees
  the rotation that produces the gaze direction.

If you need to teach the canonical method without the failure-mode setup,
use `play_major_axis_clean`. Otherwise use `play_major_axis`.

## Rendering

### Environment

Manim Community 0.19.1 is installed in the `manim` conda env. Use the
absolute path (no shell activation required):

```bash
~/miniconda3/envs/manim/bin/manim -ql --disable_caching apps/manim/gaze_flip_rule.py Scene1Setup
```

LaTeX is **not** installed and `MathTex` will fail. All math labels in this
file use `Text` with unicode combining circumflex (`f̂`, `l̂`, `ĝ`, `m̂`) so
the file renders without any system dependency. **Do not switch to `MathTex`
without first installing a TeX distribution** (`texlive-latex-base
texlive-latex-extra` on Debian/Ubuntu).

### Quick render commands

```bash
# 480p15 quick smoke test (~few seconds per scene)
~/miniconda3/envs/manim/bin/manim -ql --disable_caching apps/manim/gaze_flip_rule.py Scene1Setup

# 720p30 final quality (~30-90 seconds per scene)
~/miniconda3/envs/manim/bin/manim -pqm apps/manim/gaze_flip_rule.py MasterCorrectScene
```

Output lands at `media/videos/gaze_flip_rule/{480p15,720p30}/<SceneName>.mp4`.

### Viewing over SSH/tmux

The author uses Ghostty on Mac and an HHMI workstation. Inline-in-terminal
playback (`mpv --vo=tct`, `--vo=caca`) is unreliable through tmux. Two paths
that work:

1. **scp the mp4 to the local machine and play in QuickTime/etc.**
2. **HTTP server + SSH port forward:**
   ```bash
   # On the workstation (in a tmux pane):
   cd media/videos/gaze_flip_rule/480p15
   python3 -m http.server 8765 --bind 127.0.0.1
   ```
   ```bash
   # On the Mac (in a separate terminal):
   ssh -L 8765:localhost:8765 -N delahantyj@<workstation>
   # Then open http://localhost:8765/ in any browser.
   ```

The HTTP path is preferred during iteration because re-rendering overwrites
the mp4 and the user just hard-refreshes the browser. Browsers cache mp4 by
default; use Cmd-Shift-R to force a re-fetch.

## Conventions and key design decisions

Read these before changing geometry. A lot of pieces are coupled.

### Coordinate conventions

- `+y` is forward (rostral / up the page).
- `-x` is anatomical left.
- `theta` (the rotation tracker for the left eye) is measured **CCW from +y**.
  - `gaze_unit(θ) = (-sin θ, cos θ)`
  - `major_unit(θ) = (cos θ, sin θ)`
  - Left eye at rest: `θ = 90°` → gaze points anatomical left.
  - Right eye at rest: `θ = -90°` → gaze points anatomical right.
  - Nasal rotation: left eye `θ` *decreases* from 90°; right eye `θ`
    *increases* from -90°.
- Body-frame signed angles are computed via `atan2(left_component,
  forward_component)` and have **positive = anatomical left** for both eyes.
  This is the v5 stored convention.
- Bianco eye-frame nasal angles (positive = nasal for each eye specifically)
  are derived as `left_nasal = -left_major_signed`, `right_nasal =
  +right_major_signed`. Used in Scene 6.

### Color palette

| Constant                  | Hex        | Used for |
|---------------------------|------------|---------|
| `BODY_COLOR`              | `#3a3a3a`  | fish outline, anatomical eyes |
| `FRAME_COLOR`             | `#1f6feb`  | body frame `f̂`, `l̂`, midpoint dot |
| `TRUTH_COLOR`             | `#27ae60`  | ground-truth gaze (green) |
| `ERROR_COLOR`             | `#e74c3c`  | wrongly resolved gaze (red) |
| `RAW_COLOR`               | `#9aa0a6`  | raw ellipse-axis candidates |
| `MAJOR_AXIS_COLOR`        | `#ff8c00`  | TN axis line, T/N labels |
| `LEFT_CONE_COLOR`         | `#f5b8b3`  | left eye monocular cone |
| `RIGHT_CONE_COLOR`        | `#a3c8ff`  | right eye monocular cone |
| `BINOCULAR_COLOR`         | `#9be59c`  | binocular overlap polygon |
| `HALFPLANE_FILL_FORWARD`  | `#d8ecff`  | pale blue forward half-plane shade |
| `HALFPLANE_FILL_BACKWARD` | `#ffdcdc`  | pale red backward half-plane shade |
| `BACKGROUND_COLOR`        | `#f7f7f7`  | scene background |

### Geometry

- Fish body polygon has a broadened, rounded head (widest at `y=1.10`) and
  an elongated near-parallel tail stem ending at `y=-3.95`.
- Eyes positioned at `(±0.45, 0.95)` with aspect ratio ~2.8:1
  (`ELLIPSE_MAJOR=0.78`, `ELLIPSE_MINOR=0.28`). Tall ellipses.
- Half-plane line passes through the eye level (`HALFPLANE_Y =
  EYE_LEFT_CENTER[1] = 0.95`).
- The static eye dimensions match the fit ellipse at default lateral gaze
  (width = `ELLIPSE_MINOR`, height = `ELLIPSE_MAJOR`) so the swap from
  static eye → rotating fit ellipse is visually continuous at `theta=90°`.

### Theta ranges (biologically realistic per Bianco 2011)

- `THETA_TYPICAL_START = 55°` — comfortable mid-convergence (~35° nasal).
- `THETA_TYPICAL_END = 85°` — near rest, just inside forward half-plane.
- `THETA_FAILURE_END = 125°` — ~35° past lateral, biologically extreme but
  plausible.
- `THETA_MAJOR_END = 125°` — same as failure end; major-axis resolution is
  still robust here.
- `THETA_BOUNDARY_LOW/HIGH = 85°/95°` — boundary-jitter range.

## Animation patterns to reuse

### `always_redraw` opacity gotcha

`always_redraw(fn)` rebuilds the mobject every frame from the factory `fn`.
Each rebuild produces a *new* mobject at full opacity, so `FadeOut(m)` does
not work — the FadeOut sets opacity, but the next frame the factory runs
and resets it. **Two patterns** are used in this file:

1. **Snapshot pattern** (used in `play_vergence` transition): take a
   `mob.copy()` and `clear_updaters()` before fading. The static copy fades
   cleanly; remove the original `always_redraw` instance.
   ```python
   snap = m.copy()
   snap.clear_updaters()
   self.remove(m)
   self.add(snap)
   self.play(FadeOut(snap), run_time=0.6)
   ```
2. **Opacity-tracker pattern** (used for the cone reveal in Scene 6):
   factory reads a `ValueTracker` and multiplies the fill opacity by it.
   Animate the tracker from 0→1 to fade in.
   ```python
   sector = AnnularSector(
       ...,
       fill_opacity=0.22 * float(self.verg_cone_alpha.get_value()),
   )
   # Later:
   self.play(self.verg_cone_alpha.animate.set_value(1.0), run_time=1.5)
   ```

### Invisible mobject swap

To transition between two `always_redraw` representations of the same
visual object (e.g., `fit_ellipse` driven by `theta_tracker` →
`verg_eye_left` driven by `convergence_tracker`), animate them to a
matching state, then `self.remove(old)` and `self.add(new)` in one frame.
The viewer sees no change because both produce identical pixels at the
matching state. This is what `play_vergence`'s Phase 3 does at
`theta=90°` / `convergence=0`.

### Sectors

`Sector` does not accept `outer_radius` directly (it's set internally).
Use `AnnularSector(inner_radius=0, outer_radius=...)` for pie-slice cones.

### Z-index for layering

Half-plane shading: `z=-10` (deepest background).
Cones: `z=-3`. Binocular polygon: `z=-2` (over cones, behind fish/eyes).
Default mobjects (eyes, arrows): `z=0`.
Math readout / titles: default; rendered above everything else by add order.

## Things to be careful about

### Re-render after shared changes

If you edit *anything* used by multiple scenes — `FISH_BODY_POINTS`,
`EYE_*_CENTER`, the body-frame axes, the definitions card, the half-plane
shading, anything in `build_objects` — you must re-render every scene that
uses it. Easy to forget.

The simplest sanity check is to re-render `MasterCorrectScene` (covers
Scenes 1 + 4-clean + 6 — most of the visual surface). If failure-mode
scenes also need updating, re-render `MasterScene` and the individual
Scenes 2/3/5 standalones.

### Scenes 2-5 standalones don't show the heading derivation

The heading-derivation step (`_play_heading_derivation`) is called from
inside `play_setup(animated=True)`. Standalone scenes 2/3/4/5 use
`play_setup(animated=False)` for speed, so they skip it. Only the masters
and `Scene1Setup` show the derivation. Don't be confused by the difference.

### Master scene transitions

`MasterCorrectScene`'s transition from major-axis-clean → vergence uses a
three-phase animation in `play_vergence`:
1. Rotate the left eye back to rest.
2. Fade out only the labels that have no Scene 6 equivalent (snapshot
   `always_redraw` mobjects first).
3. Invisibly swap `fit_ellipse → verg_eye_left`, `major_axis_line →
   verg_left_major`, `tn_endpoints → verg_left_tn`, `eye_right →
   verg_eye_right`.

If you add new always_redraw mobjects to `play_major_axis_clean`, you'll
likely need to add them to `fade_redraws` in `play_vergence`.

### Vergence readout uses the input directly

`_make_verg_readout` reads `self.convergence_tracker.get_value()` directly
rather than computing nasal angles from the resolved major axis. The
displayed values are correct (geometrically equivalent) but not produced
by the actual pipeline math. If the user asks "are these real angles?",
the honest answer is: yes, the values are real; no, they're not computed
through the major-axis resolution. See the comment in `_make_verg_readout`.
A future enhancement could refactor it to compute through the pipeline.

### LaTeX absence

This was discovered during the first render. If you ever need MathTex,
install `texlive-latex-base` first. Do not assume LaTeX is available.

### Background color must be set per scene

`self.camera.background_color = BACKGROUND_COLOR` is set inside
`build_objects()`, *not* via `config.background_color`. Don't move it to
module scope unless you want to affect imports.

## Open ideas for future work

In rough priority order:

1. **Re-render `MasterScene`** (full sequence) with all the recent changes
   (heading derivation, long tail, clean transitions, slow cone reveal).
   Last time it was rendered, several of these were missing.
2. **Audit Scenes 2-5 visual cleanups.** They were given a final pass for
   layout (legend position, error label position) but not as carefully as
   the canonical-flow scenes. Some may have stale mobject layouts after
   the eye geometry / fish polygon changes.
3. **Compute the vergence readout through the pipeline.** Replace the
   direct tracker read with an `atan2` on the resolved major-axis vector
   so the math mirrors `eye_angle_analysis.py:_signed_angle_from_body_axes`.
   ~10 lines.
4. **Chaser-tracking demo (potential Scene 7).** Uses `gaze_xy` from the v5
   eye-angle pipeline and a chaser position from `track_kinematics_runs`
   to compute per-eye alignment with `acos(dot(gaze_xy, eye_to_chaser))`.
   Would visualize "is this eye tracking the chaser?" with a green/red
   alignment indicator. The math is in the implementation prompt I drafted
   in chat earlier (search "chaser-tracking" in the conversation).
5. **Boundary-scene audio cues.** The 180° flicker in Scene 5 is a strong
   visual story but could be even more striking with a "click" sample on
   each flip. Manim supports `self.add_sound("path.wav")`.
6. **Performance.** Each frame, several `Text` mobjects are rebuilt from
   f-strings inside `always_redraw` factories. At 1080p60 this might lag.
   If it does, switch to `DecimalNumber` or `Integer` mobjects which are
   designed for live-updating values.
7. **Render captions / narration text.** All current narration is on-screen
   text. If the user wants real narration, Manim supports synced audio.

## Useful repo references

- v5 refactor of the eye-angle pipeline: `src/fisheye/analysis/eye_angle_analysis.py`
- v5 schema reference: `src/fisheye/docs/eye_angle_conventions.md`
- Margin-metric teaching doc (companion to Scene 4):
  `docs/eye_axis_half_plane_margin.md`
- Tests for the resolution algorithm:
  `tests/unit/fisheye/test_eye_angle_axis_resolution.py`

## Working effectively with the user

The user (Jeremy) iterates rapidly on visual details — expect rounds of
"move that label," "make this bigger," "fade that out." When making such
edits:

- Render a low-quality smoke test (`-ql`) and grab a single frame with
  `ffmpeg -y -sseof -1 -i <mp4> -frames:v 1 /tmp/check.png` to verify the
  change before reporting.
- For Scene 6 / vergence frames specifically, the meaningful frame is
  partway through the rotation, not the last frame. Use `ffmpeg -ss <N>`
  to grab a mid-frame.
- After several rounds of refinement on one scene, ask the user if they
  want all scenes re-rendered, or just the one being iterated.
- The user prefers terse, complete-sentence summaries of what changed and
  what to refresh in their browser.
