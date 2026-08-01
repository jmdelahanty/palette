# Rig Characterization — Measured 2026-07-24

<!-- contract-meta
status: measurement_record
last_verified: 2026-07-24
purpose: Empirical characterization of the acquisition rig's sensor noise, illumination stability, encoder behaviour, and colour-range semantics, for use by acquisition-side work.
-->

Every number here was measured on real store data, not inferred from specifications.
Full derivation and the exact commands are in
[`storage_and_rig_conversation_2026-07-24.md`](storage_and_rig_conversation_2026-07-24.md).
Storage-tiering consequences are in
[`../raw_video_storage_tiering_proposal.md`](../raw_video_storage_tiering_proposal.md).

Primary reference recordings:
- `2026-07-21T19-38-32Z_arena_1_Batman` (`pc`-tagged era)
- `2026-06-21T18-18-31Z_arena_1_GoodCopBadCop` (`tv`-tagged era)

---

## 1. Stream configuration (as recorded)

From `raw/external_recorder_supervisor_plan.json` and
`derived/*/\*_summary.json`:

| | Master (`cams/`) | Crop (`derived/external_crop_recorder/`) |
|---|---|---|
| Resolution | `4512×4512` | `256×256` (GoodCopBadCop), `384×384` (Batman) |
| Codec | HEVC | HEVC |
| Preset | **`p1`** (fastest) | **`p7`** (slowest) |
| Rate control | fixed `bitrate_bps: 150000000` | `vbr`, **`tuning: lossless`**, `quality_value: 20` |
| Frame rate | 100 fps | 100 fps |
| Frames / duration | 139,877 / 1398.77 s | identical timeline |
| Delivered size | ~32.5 GB mean (n=153) | ~9.4 GB mean (n=120) |

Derived quantities:

- Master bit density: 150 Mbps ÷ (4512² × 100) = **0.0737 bits/pixel**. Very low —
  typical "high quality" video is 0.1–0.5 bpp.
- Uncompressed equivalent: 4512² × 1.5 B × 100 fps = **24.4 Gbps**, so the master is
  already at **163:1**.
- Crop lossless cost: 32 Mbps against a 78.6 Mbps raw rate = **2.4:1**. That is simply
  what lossless costs; there is no quality knob to turn.

**Mono semantics confirmed empirically.** Chroma planes are constant: `UV min = max =
128` in every frame examined. Camera intensity lives in the NV12 `Y` plane exactly as
`roi_pixel_contract.py` asserts. ffprobe reports decoded frames as `yuv420p`/`yuvj420p`,
which is a post-decode memory-layout artefact, not a semantic change.

---

## 2. Colour range — store is split at 2026-07-02

| Stream | `tv` | `pc` | Cutover |
|---|---|---|---|
| Master | 100 | 53 | last `tv` = **2026-07-01**, first `pc` = **2026-07-02** |
| Crop | 72 | 48 | same day |

**The runtime tagging fix changed the tag only, not the pixels.** Verified by decoding
raw `Y` planes via PyAV (`frame.planes[0]`, no swscale) on same-protocol recordings one
day either side:

| File | Tag | `Y` min/max | px < 16 | px > 235 |
|---|---|---|---|---|
| 2026-07-01 arena_1 | `tv` | 8 / 255 | 2,251,752 | 126,061 |
| 2026-07-02 arena_1 | `pc` | 8 / 255 | 2,289,839 | 35,098 |

Both are genuine full-range mono8. Post-cutover files are correctly self-describing;
pre-cutover files carry a `tv` **mislabel** over full-range data.

Palette is safe across the boundary only because `roi_pixel_contract.py` performs
`read_direct_y_plane_without_decoder_range_remap`. **Any tool that honours the tag
behaves differently either side of the cutover.**

---

## 3. Sensor noise — the headline rig number

Measured on `2026-07-21T19-38-32Z_arena_1_Batman`, frames 131–144 (a run where
`crop_x`/`crop_y` are constant, so the lossless crop and the master cover *identical
sensor pixels*). Background corners only; fish is crop-centred.

| Source | Background temporal σ |
|---|---|
| **Lossless crop — true sensor** | **4.279 grey levels** |
| **P1 master — same photons** | **1.502 grey levels** |
| Ratio | **0.35** |

### The P1 encoder is a temporal denoiser

It **removes ~65% of the sensor noise**. At 0.0737 bits/pixel it cannot afford to code
noise, so it smooths it away. Two consequences:

1. Any theory that "150 Mbps is being wasted encoding sensor noise" is **wrong**. The
   recurring bit cost is not noise; it is the static spatial texture plus the residual
   noise the encoder does retain.
2. **The two pixel sources differ in noise by 2.85×** on the same photons. Keypoint and
   mask stages read the lossless crop (σ≈4.3) via `CropImageSource`; YOLO detection runs
   `pynvvc_nv12_rgb` on master pixels (σ≈1.5). Any model trained on one and applied to
   the other carries an unaccounted domain shift. **Audit this independently.**

### Physical origin

σ ≈ 4.3 grey levels at background luma ≈ 190 implies roughly **2,000 photoelectrons**
per pixel per frame — consistent with **photon shot noise** at the ≤10 ms exposure
forced by 100 fps.

Supporting structure measurements (on the master, i.e. post-denoise):

| Property | Measured | Interpretation |
|---|---|---|
| σ vs luma, bins 170→256 | 2.752, 2.721, 2.783 (flat) | not √-scaling *in the denoised master*; test properly on lossless data |
| Spatial autocorrelation | r=+0.24 @1 px, +0.07 @2 px, ~0 @≥4 px | nearly white, mildly correlated by optics/MTF and in-loop filters |
| Block-grid variance spread | 1.21 (8 px) → 1.47 (64 px) | weak block structure; compression contribution is minor |
| Fixed-pattern spatial residual | **7.99 grey levels** | static — costs bits once in the I-frame, then predicted free |

### Levers

Shot noise scales as √N, so the only real fix is **more photons**: brighter IR
illumination, wider aperture, or binning. 2×2 binning gives 4× photons and halves
relative noise but takes the 153 px fish to 76 px — adequate for detection and heading,
marginal for tail keypoints. **Confirm whether exposure is already at the 10 ms ceiling
before anything else.**


### Exposure / gain / illumination trade — measured on lossless data

Measured on the Batman lossless crop, frames 130-144 (true sensor pixels, no encoder
involvement):

| Quantity | Measured |
|---|---|
| Clipping: px ≥ 245 | **0.000 %** (nothing at 255 at all) |
| Background luma | 207.7 DN, temporal σ 4.450 |
| Fish luma | 165.0 DN, temporal σ 3.983 |
| Fish/background contrast | **42.8 DN** |
| **Contrast-to-noise ratio** | **10.1** |
| σ/√luma, bins 170-200 and 200-225 | 0.3087, 0.3086 |
| Implied gain | **g ≈ 10.5 e⁻ per DN** |
| Implied full-scale well | **≈ 2,677 e⁻** → SNR at full scale ≈ 52 |

**Shot-noise limited, confirmed.** σ/√L is constant to four significant figures across
luma bins, so σ ∝ √N holds. (The earlier "flat vs luma" result came from the *denoised*
master and was misleading — always test this on lossless data.)

**Trading light intensity for exposure time is a null trade.** Shot-noise SNR = √N and
N ∝ intensity × exposure, so halving the light and doubling the exposure leaves SNR
exactly unchanged. It is not a null trade for the *science*, because exposure buys motion
blur. At roughly 35 px/mm (153 px fish ≈ 4.4 mm):

| Exposure | Blur at 100 mm/s (body) | Blur at 500 mm/s (tail tip) |
|---|---|---|
| 50 µs | 0.18 px | 0.88 px |
| 1 ms | 3.5 px | 17.5 px |
| 5 ms | 17.5 px | 88 px |

**The 50 µs exposure is protecting the escape-response kinematics.** Do not lengthen it.

**The actual limiter is gain, not photons.** Only ~2,677 e⁻ maps to DN 255. If the
sensor's true full well is 10,000-30,000 e⁻, the ADC is clipping long before the pixel
well fills, so bright light is being discarded rather than converted to electrons. Since
`σ_DN at full scale = 255/√N_max`, raising the usable well raises SNR directly:

| Effective well | σ at DN 255 |
|---|---|
| 2,677 e⁻ (current) | 4.93 DN |
| 10,000 e⁻ | 2.55 DN |
| 30,000 e⁻ | 1.47 DN |

Correct order of operations: **lower analog gain → raise illumination to refill toward
DN 255 → keep exposure at 50 µs.** Bright light, short exposure, low gain is the optimum
for a shot-noise-limited fast-motion rig.

Diagnostic to tell ADC clipping from true well saturation: at fixed illumination, halve
the gain. If the exposure needed to reach DN 255 roughly doubles, you were gain/ADC
limited and headroom exists. If it barely moves, you are at true well saturation and the
remaining levers are 2×2 binning (4× well, 2× SNR, fish 153 → 76 px) or a different
sensor.

**Contrast may be the bigger lever than photons.** CNR is only 10.1, and that is driven
by a fish/background contrast of just 42.8 DN — the useful signal occupies DN 165-208,
about **17% of the available range**. Larval zebrafish are semi-transparent, so contrast
is set by illumination *geometry*, not intensity: scaling exposure scales fish and
background together and barely improves the difference. Dark-field, oblique, or
polarized illumination changes the ratio. Doubling contrast to ~85 DN would take CNR to
~20 — equivalent to a 4× increase in photon count, at no cost in exposure time.


### Photon transfer curve — the definitive gain measurement

Per-pixel temporal variance vs temporal mean, on lossless crop data (frames 130-144).
Shot noise predicts `var = (L − B)/g`, so a straight line whose slope gives the gain and
whose x-intercept gives the black level.

| Luma bin | n px | mean L | var | σ |
|---|---|---|---|---|
| 176-183 | 5,398 | 180.31 | 17.897 | 4.230 |
| 183-190 | 19,777 | 187.06 | 18.476 | 4.298 |
| 190-197 | 40,024 | 193.68 | 19.184 | 4.380 |
| 197-204 | 43,302 | 200.30 | 19.793 | 4.449 |
| 204-211 | 24,594 | 206.95 | 20.595 | 4.538 |
| 211-218 | 7,774 | 213.63 | 21.107 | 4.594 |

Linear fit: **r = +0.9988**

| Fitted quantity | Value |
|---|---|
| Gain | **g = 10.12 e⁻ per DN** |
| Black level (x-intercept) | **B = −0.4 DN** (i.e. no pedestal) |
| Electrons at background DN 207.7 | **2,106 e⁻** |
| **Full-scale capacity (DN 255)** | **2,584 e⁻** |
| **SNR ceiling at full scale** | **51** (≈ 2% noise) |

The near-perfect linearity and zero intercept confirm **shot-noise-dominated operation
with no black-level offset**, so DN is directly proportional to collected electrons.

### What "gain = 256" does and does not tell you

`256` is almost certainly a Q8 fixed-point register where 256 = 1.0× — a common
industrial-camera convention, and the natural reading. **But unity analog gain does not
imply that the sensor's full well maps to DN 255.** The electrons-per-DN in delivered
Mono8 is set by the sensor's conversion gain and ADC span, not by the gain register.

Measured full-scale span is only **2,584 e⁻**. Two very different explanations, with
opposite remedies:

- **(A) Analog gain above unity is compressing the range.** Then lowering gain buys
  electrons and SNR. *But if 256 is already unity, unity is typically the floor and there
  is nothing to lower* — this is why the earlier "just lower the gain" advice may not
  apply.
- **(B) The sensor is in a high-conversion-gain / low-full-well readout mode.** Then
  2,584 e⁻ *is* the well, the exposure is correctly set, and the rig is at its ceiling
  rather than misconfigured. The negligible fitted read-noise floor is weakly consistent
  with this (HCG modes trade well depth for low read noise), though the intercept is
  extrapolated from a narrow luma range and should not be leaned on.

**This also explains the reported saturation at 50 µs.** A 2,584 e⁻ well fills very
quickly under bright illumination. Background sits at DN 208 = 2,106 e⁻ = **81% of full
scale with zero clipping** — the exposure is well chosen. The rig is not badly set up; it
is operating near the ceiling of a small well.

**Decisive experiment:** repeat this photon transfer curve at two or three gain settings
(e.g. 128, 256, 512), on lossless captures.

- If `g` scales inversely with the register (512 → g≈5, 128 → g≈20), gain is genuinely
  analog and sub-unity settings, if available, convert surplus light into electrons.
- If `g` barely moves, the register is digital-only and lowering it gains nothing. Look
  instead for a conversion-gain / full-well / dynamic-range mode (vendor names vary:
  "Gain Mode", "Conversion Gain", "Dual Gain", "Well Depth").

**Mono8 output is not the bottleneck — ruled out.** At g = 10.12, one DN ≈ 10 electrons
and σ ≈ 4.5 DN, so 8-bit quantisation noise (1/√12 ≈ 0.29 DN) sits ~15× below the shot
noise. Capturing deeper would not help SNR.

**Remaining levers, in order of cost:**

1. **Illumination geometry for contrast** (CNR 10.1 is contrast-limited, not
   photon-limited — see above). Cheapest and largest expected gain.
2. **A high-full-well / low-conversion-gain readout mode**, if the camera offers one.
3. **2×2 binning** — 4× electrons, **2× SNR**, at 153 → 76 px on the fish.
4. More light only helps *after* (2) or (3); with the present 2,584 e⁻ span it just
   clips sooner.


### Where contrast actually lives, and why geometry beats brightness

Using `g = 10.12 e⁻/DN` from the photon transfer curve, with background at 204.9 DN
(2,073 e⁻, background shot noise 45.5 e⁻):

| Population (percentile of fish region) | DN | Δ DN | Signal e⁻ | **CNR** |
|---|---|---|---|---|
| darkest 0.5% — pigment / eye | 51.1 | 153.8 | 1,556 | **34.2** |
| 0.5-2% | 125.0 | 79.8 | 808 | **17.7** |
| 2-6% — body | 175.6 | 29.3 | 296 | **6.5** |
| 6-15% — body / tail | 185.4 | 19.5 | 197 | **4.3** |
| 15-30% — thin tail / fin | 190.5 | 14.4 | 146 | **3.2** |

**The eyes are easy; the tail is marginal.** Pigment sits at CNR 34, but the thin tail
and fins sit at **CNR 3-4** — which is exactly the fragile consumer for tail splines and
mask boundaries. Any fidelity argument about this rig should be made about the tail, not
about the average.

**Well accounting** at a 2,581 e⁻ capacity:

| | e⁻ | % of well |
|---|---|---|
| Establishing the bright background | **2,073** | **80%** |
| Carrying the fish signal (mean Δ) | 404 | 16% |

**Brightfield CNR = Δe⁻ / √N_background**, so the featureless bright background sets the
noise floor. Two consequences:

1. **CNR scales only as √N in brightfield.** Lifting the tail from CNR 3.2 to CNR 10
   needs (10/3.2)² ≈ **10× more electrons** — a ~25,000 e⁻ well. Unreachable at the
   current 2,581 e⁻ span. 2×2 binning gets 4× (CNR 6.4); 4×4 gets 16× (CNR 12.8) but
   takes the fish to 38 px.
2. **Darkfield escapes the limit rather than raising it.** If direct light misses the
   aperture, the background falls to (say) 50 e⁻ with noise 7 e⁻, and a tail scattering
   150 e⁻ gives CNR ≈ 21 versus 3.2 today. The win is not more signal — it is deleting
   the background shot noise *and* freeing the 80% of well currently spent on it.

### Camera-side vs lighting-side levers

**Electronic settings cannot create contrast.** Gain, gamma, black level, and any digital
"contrast" control rescale signal and noise together, and CNR is invariant under
monotonic rescaling. Camera menus label this "contrast"; it is not.

**Detection-path optics can, and these are genuinely camera-side:**

- **Aperture / f-number — the main one.** For a transparent, refracting and scattering
  specimen, contrast depends on collection angle: stopping down rejects rays the fish
  deflected, so the fish darkens and contrast rises. This is darkfield physics applied at
  the lens. It costs photons, but there is a surplus — stop down, then raise illumination
  to refill the well. Same electrons, better contrast, plus more depth of field so the
  fish stays in focus across dish depth.
- **Illumination wavelength — CLOSED for this rig.** Confirmed at **850 nm**. Silicon
  absorbed-fraction is `1 − exp(−d/L)` with L ≈ 9 µm at 780 nm, 19 µm at 850 nm, 55 µm at
  940 nm, so 850 → 780 nm would be worth roughly **1.8×** in electrons for a 5 µm
  depletion. **But this buys no SNR here**, because the rig is *well*-limited, not
  photon-limited — higher QE only reaches the same 2,584 e⁻ ceiling with less
  illumination, and there is already surplus light. An earlier version of this document
  claimed wavelength was the largest available win; that was wrong. **Stay at 850 nm**
  (it is also safely clear of zebrafish visual sensitivity, ~>750 nm).
- **Cross-polarization.** Rejects specular reflections from the meniscus and dish wall
  (the likely source of localized saturation). Muscle is also weakly birefringent, so
  cross-polarized detection can generate signal from the tail specifically — where
  contrast is worst. Weak signal, but aimed at the right target.
- **Telecentric lens.** The limit case of stopping down: a narrow parallel collection
  bundle maximizes sensitivity to refraction.

**Illumination geometry is still the biggest lever** (darkfield or oblique), for the
quantitative reason above. The cost is that darkfield **inverts polarity** — fish bright
on dark — so existing detectors, masks, and training data would all need revisiting.
Given the domain-shift concern already noted in §3, treat that as a real migration, not a
knob. Oblique / partial darkfield is the cheaper intermediate.


### Optics: 100 mm macro geometry, and the diffraction ceiling at 850 nm

Object scale ≈ 28.7 µm/px (from a 153 px fish ≈ 4.4 mm — **verify against
`fisheye.shared.arena_geometry`**, this estimate propagates into everything below).
Object field ≈ 4512 × 28.7 µm ≈ **130 mm**. With a 100 mm lens:

| Sensor pitch | Sensor width | Magnification m | Working distance ≈ f(1+1/m) | Bellows factor (1+m) |
|---|---|---|---|---|
| 2.5 µm | 11.3 mm | 0.087 | ~1.25 m | 1.09 |
| 3.45 µm | 15.6 mm | 0.120 | ~0.93 m | 1.12 |
| 5.0 µm | 22.6 mm | 0.174 | ~0.68 m | 1.17 |

So this is **low magnification, not true macro range** — the lens is earning its place
through flat field and low distortion, which is the right reason for quantitative work
(distortion would corrupt the mm-per-pixel calibration across the field). The bellows
factor inflates the effective f-number only ~10-17%, so it barely moves the diffraction
budget.

**Diffraction is the binding optical constraint.** Airy diameter = 2.44·λ·N = 2.07·N µm at
850 nm:

| Marked f-number | Airy Ø | ≈ pixels at 3.45 µm pitch |
|---|---|---|
| f/2.8 | 5.8 µm | 1.7 |
| f/4 | 8.3 µm | 2.4 |
| f/5.6 | 11.6 µm | 3.4 |
| f/8 | 16.6 µm | 4.8 |

A 100 mm macro typically opens to f/2.8, where the Airy spot is already ~1.7 px on a
3.45 µm sensor. **The system is close to diffraction-limited wide open at 850 nm**, so
stopping down for contrast is a small and quickly self-defeating lever — it blurs the
thin tail, which is exactly the feature with the least contrast to spare. Optimise
**edge gradient ÷ noise** (currently 28.6 DN/px ÷ 4.5 DN ≈ **6.4 σ per pixel**), not
contrast alone.

**Verify focus under 850 nm illumination, not visible light.** Lens chromatic focus shift
puts the NIR focal plane behind the visible one; focusing in visible (or with an IR-cut
filter fitted) then imaging at 850 nm leaves the system defocused, which would soften the
tail edge and depress the figure of merit above.


### Diffraction / depth-of-field trade, computed (850 nm)

For a 3.45 µm pitch at m = 0.120. "Tail" frequency taken as a ~3 px feature
(0.167 cyc/px); Nyquist is 0.5 cyc/px. DOF is total object depth **in air** for an
acceptable blur of 1 or 2 px.

| f/N | MTF @ tail | MTF @ Nyquist | Airy (px) | DOF 1px | DOF 2px |
|---|---|---|---|---|---|
| f/2.0 | 0.895 | 0.690 | 1.20 | 1.07 mm | 2.15 mm |
| f/2.8 | 0.854 | 0.570 | 1.68 | 1.50 mm | 3.01 mm |
| f/4.0 | 0.791 | 0.399 | 2.40 | 2.15 mm | 4.29 mm |
| f/5.6 | 0.709 | 0.197 | 3.37 | 3.01 mm | 6.01 mm |
| f/8.0 | 0.589 | 0.002 | 4.81 | 4.29 mm | 8.59 mm |
| f/11 | 0.444 | 0.000 | 6.61 | 5.90 mm | 11.81 mm |

**Correction to an earlier version of this document**, which warned that stopping down
would blur the tail and was therefore barely available. The computation does not support
that. **MTF at the tail frequency degrades gracefully** — 0.85 at f/2.8, 0.79 at f/4,
0.71 at f/5.6 — while it is *Nyquist-scale* detail that collapses (0.57 → 0.20 over the
same range). Going f/2.8 → f/5.6 costs about 17% of tail contrast transfer and **doubles
depth of field**. On these numbers that is likely a good trade, not a bad one.

**Depth of field is the more likely binding constraint.** At f/2.8 the in-focus slab is
only ~1.5 mm in air, or roughly **2 mm of real water depth** (axial distances viewed
through an air-water interface are compressed by ≈ n = 1.33, so real depth ≈ n × air
DOF). A fish free-swimming in a column of several mm will leave the in-focus slab
intermittently.

**And with a single top-down camera there is no z measurement**, so defocus is an
*unmeasured nuisance variable* — there is no way to know which frames were affected. This
is a candidate contributor to the confidence variance in §4b (sd 0.065 near centre rising
to 0.126 in the outer annulus).

**The cheapest fix is probably shallower water, not more depth of field.** Matching the
water column to the in-focus slab removes the problem at no optical cost, whereas buying
DOF by stopping down costs tail MTF and light.

⚠ The 3.45 µm pitch and 28.7 µm/px object scale remain **estimates**. The tail width has
since been measured — see below — and the table above uses the *wrong* value; the corrected
version follows.

### Calibration targets — what is worth buying, in priority order

1. **A calibration grid / checkerboard — highest value, and not a resolution target.**
   It pins the object scale (µm/px) and measures distortion across the field. This
   directly bears on the known `arena_geometry` discrepancy (nominal experimental area
   ~3 mm off, which once silently inverted a thigmotaxis result) and it validates the
   28.7 µm/px figure that every calculation in this document rests on.
2. **A slanted-edge target for MTF (ISO 12233) — better than USAF for this purpose.**
   Yields the full quantitative MTF curve from a single image with sub-pixel precision, so
   MTF can be read at the tail's actual spatial frequency instead of at a threshold.
   Repeat per f-number and per defocus step to map the whole trade surface. A chrome-on-
   glass knife edge suffices.
3. **USAF 1951 — useful, but limited.** It is a *threshold* test (find the smallest
   resolvable group), so it is subjective and reports limiting resolution only. It is also
   a ~100% contrast target, whereas the fish tail is ~6% contrast (14.4 DN on a 205 DN
   background) — high-contrast limiting resolution does not predict low-contrast
   detectability. Fine as a configuration-to-configuration sanity check.
4. **A tilted depth-of-field target — genuinely relevant**, since DOF looks like the real
   constraint, and it shows the whole defocus envelope in one frame.

**Two NIR gotchas that will invalidate the measurement if missed:**

- **Buy chrome-on-glass, not printed film.** Chrome is opaque at 850 nm; printing inks are
  frequently semi-transparent in the NIR, which destroys target contrast in a way that
  looks like an optics problem.
- **Measure through the actual water column and dish bottom**, and under the real 850 nm
  illumination. A target in air at visible wavelengths gets both the focal plane
  (chromatic shift) and the axial scale (refraction) wrong.

**Acceptance criterion: use edge gradient ÷ noise, not visual sharpness.** The rig sits at
28.6 DN/px ÷ 4.5 DN ≈ **6.4 σ per pixel** of edge crossing today. Maximise that over the
depth range the fish actually occupies — it is the quantity that sets mask-boundary and
tail-spline precision, which is what the science consumes.

---

## 4b. Field non-uniformity and a position-dependent detection gradient

### Illumination falloff inside the dish

Full-frame block-mean analysis (dish edge detected at r ≈ 2450 px):

| Radius (px) | Luma | vs centre | Relative CNR (∝√L) |
|---|---|---|---|
| 50 | 238.0 | 0.0 | 1.000 |
| 1050 | 225.2 | −12.8 | 0.973 |
| 1550 | 213.6 | −24.5 | 0.947 |
| 1950 | 199.0 | −39.0 | 0.914 |
| 2150 | 171.8 | −66.2 | 0.850 |

**Centre-to-dish-edge falloff = −65 DN (−27%)**, implying **~15% worse CNR at the wall**
than at the centre. The profile is smooth and monotonic; a quadratic fit leaves a centre
excess of only +6.4 DN, which is within the modelling error of a quadratic — **no
evidence of a localized IR hotspot**.

### Detection confidence degrades with radius

Across **546,186 detected frames** in 4 Batman recordings (`crop_meta.csv`):

| Radius (px) | n | Mean confidence | Relative luma |
|---|---|---|---|
| 0-400 | 9,942 | 0.7593 | 1.000 |
| 800-1200 | 46,414 | 0.7617 | 0.946 |
| 1200-1600 | 147,575 | 0.7846 | 0.897 |
| 1600-1900 | 127,864 | 0.7519 | 0.897 |
| 1900-2100 | 148,768 | **0.6806** | 0.836 |
| 2100-2400 | 48,061 | **0.6094** | 0.722 |

Pearson r(radius, confidence) = **−0.337**. Inner (r<1200) mean 0.761 vs outer (r>1900)
0.663 — a **−12.8%** degradation. And the fish spends **36% of its time at r>1900**
against 13.5% inside r<1200: thigmotaxis puts the animal where detection is worst.

### But illumination is probably NOT the main cause

**The shapes do not match.** Illumination falls smoothly and monotonically from the
centre (already −10% by r=1550), whereas confidence is *flat* out to r≈1600 and then
collapses. A smooth photometric gradient would produce a smooth confidence decline. The
sharp onset instead points to a **geometric threshold**:

- **Image-border effects** — the frame edge is at r=2256 along the axes, and CNNs lose
  context near borders.
- **Dish wall, meniscus, and wall reflections** — occlusion and mirror images.
- **Posture and defocus near the wall** — wall-parallel postures, or a different optical
  depth through the meniscus.

**What stands regardless of cause:** there is a position-dependent detection-quality
gradient that correlates with **wall proximity** — a variable used directly in the
thigmotaxis and chase-mediation analyses. Given that a prior speed-noise-floor artefact
already killed a false avoidance result, treat this as the same class of hazard.

### Recommended disentangling tests

1. **Flat-field correct** (empty-dish reference, divide) and re-measure confidence vs
   radius. If the trend persists, it is geometric, not photometric. Note that
   flat-fielding changes pixel values and therefore has pixel-contract and
   training/inference-consistency implications.
2. Re-test against **distance-to-wall** rather than radius, and against
   **distance-to-frame-border** separately — they are confounded here but separable with
   the right binning.
3. Check whether **mask area and tail-spline residuals** show the same radial trend; those
   are more sensitive than detection confidence.
4. Physically flatten the illumination — extend the illuminated area well beyond the dish
   so the dish sits in the flat centre of the profile.

---

## 4. Illumination stability — clean

Whole-frame mean luma over 200 frames (2 s):

| Metric | Value |
|---|---|
| Mean | 179.007 |
| Standard deviation | **0.0145 grey levels** |
| Peak-to-peak drift | **0.093 grey levels** |
| Frame-to-frame Δ, sd | 0.0135 |
| Dominant FFT components | amplitude ≈ 0.006 (noise floor) — **no flicker** |

Illumination is not a source of encoding cost or measurement variance. This was the
most likely suspect and it is ruled out.

---

## 5. Fish signal properties

| Property | Measured |
|---|---|
| Detection bbox | ~**153 × 121 px** |
| Fish-region contrast (p99 − p1) | **120 grey levels** |
| Silhouette edge steepness (p99.5 of gradient) | **28.6 grey levels / pixel** |
| Fish fraction of frame | 256²/4512² = **0.32%** |
| Crop centring | crop centre == detection centre (verified) |

The edge-steepness figure is the conversion from intensity error to geometry error:
`edge displacement (px) = RMS intensity error ÷ 28.6`. Use it to translate any encoder
or denoiser change into apparent mask-boundary motion.

---

## 6. Encoder behaviour (RTX A6000)

### Preset sweep — preset is not the lever

60 s / 6000-frame segment, `-rc vbr -cq 28`:

| Preset | Wall | Output | vs source | Pipeline fps |
|---|---|---|---|---|
| `p3` | 120.3 s | 716.3 MB | 1.57× | 49.9 |
| `p5` | 194.5 s | 678.2 MB | 1.66× | 30.9 |
| `p7` | 393.8 s | 678.6 MB | 1.66× | 15.2 |

`p7` produced a file **0.06% larger** than `p5` for 2.2× the time. Decode-only baseline
via `hevc_cuvid`: 33.05 s (**182 fps**). **All presets run below the 100 fps acquisition
rate**, which is why `p1` is necessary at capture.

### `-cq` is broken in this ffmpeg — use `-rc constqp`

`-rc vbr -cq N -b:v 0` at N=20 and N=24 produced **byte-identical output**
(710,721,538 bytes). Everything clamped near 90–95 Mbps.

### Constant-QP ladder (600 frames, `p5`, fish-region scored over n=25)

| Encode | Output | vs source | Mbps | Fish dB | σ | Fish MAE | Edge shift |
|---|---|---|---|---|---|---|---|
| source `p1` | 112.5 MB | 1.00× | 150.0 | — | — | — | — |
| `qp22` | 93.7 MB | 1.20× | 125.0 | 42.49 | 2.33 | 1.48 | 0.067 px |
| `qp26` | 47.0 MB | **2.39×** | 62.7 | 41.11 | 2.38 | 1.73 | 0.078 px |
| `qp30` | 16.8 MB | **6.69×** | 22.4 | 39.15 | 1.96 | 2.18 | 0.098 px |
| `qp34` | 6.8 MB | 16.52× | 9.1 | 38.12 | 2.37 | 2.46 | 0.111 px |
| `x265 crf22` | 41.5 MB | 2.71× | 55.3 | 40.35 | 2.10 | 1.89 | 0.086 px |

HEVC QP: step size doubles every **+6 QP**; observed file size falls ~2–2.8× per +4 QP
because more coefficients round to zero. Encode time is **independent of QP** (~30 fps
at `p5`) → ~78 min per full recording, ~198 GPU-hours for a 153-recording backlog.

**NVENC beats x265 by ~1.3×** at matched fish quality, is 2.3× faster, and has no range
trap. Stay on NVENC. `--input-csp i400` monochrome remains untested.

---

## 7. Traps — all of these were hit in practice

1. **`-hwaccel cuda` NVDEC fails at 4512×4512** with `CUDA_ERROR_INVALID_VALUE` /
   "No decoder surfaces left", then silently falls back to software decode *plus
   swscale*. Use **`-c:v hevc_cuvid`** explicitly.
2. **`-fps_mode` does not exist in ffmpeg 4.4.6** (the `/opt/orange/lib/ffmpeg-nvidia`
   build). Use `-vsync 0`.
3. **x265 range trap.** Neither `-color_range pc` nor `-x265-params range=full` prevents
   a full→limited compression. Measured output: `Y` 25–240, zero samples below 16, mean
   shifted 9.2 levels, **still tagged `pc`** — mislabelled in the dangerous direction.
   Fix: `-vf "scale=in_range=full:out_range=full"`. Verified to restore `Y` 10–255.
   The NVENC path needed no workaround and was byte-clean throughout.
4. **Verification tooling hits the same trap.** A naive
   `ffmpeg -i IN -f rawvideo -pix_fmt yuv420p -` returns `Y` 9–255 for a `tv` file but
   `Y` 23–235 for a `pc` file — swscale compresses the correctly-tagged one. Always read
   raw decoded planes (PyAV `frame.planes[0]` or PyNvVideoCodec). This produced a table
   that looked exactly like data corruption which had not occurred.
5. **PSNR needs ≥20 frames.** Per-frame fish PSNR varies with σ ≈ 2.3 dB, so n=4 gives
   ±1.2 dB standard error — enough to invert adjacent QP rankings and manufacture a
   spurious non-monotonicity.
6. **Whole-frame PSNR is only 0.5–1.2 dB above fish-region PSNR** here, so it is less
   misleading than expected — but still score the region you care about.
7. **Measure noise in a representative region.** An early σ estimate of 0.74 grey levels
   came from a dark corner and understated the real background noise by ~4×.

---

## 8. Reference commands

Range-safe NVENC transcode:

```bash
ffmpeg -y -v warning -c:v hevc_cuvid -i IN.mp4 -an \
  -c:v hevc_nvenc -preset p5 -tune hq -rc constqp -qp 26 \
  -color_range pc -vsync 0 OUT.mp4
```

Range-safe x265 transcode (if ever needed):

```bash
/usr/bin/ffmpeg -y -v error -i IN.mp4 -an \
  -c:v libx265 -preset medium -crf 26 \
  -vf "scale=in_range=full:out_range=full" \
  -pix_fmt yuv420p -color_range pc \
  -x265-params "range=full:log-level=error" -vsync 0 OUT.mp4
```

Trustworthy plane read (no swscale):

```python
import av, numpy as np
c = av.open(path); s = c.streams.video[0]
for i, fr in enumerate(c.decode(s)):
    p = fr.planes[0]
    y = np.frombuffer(bytes(p), dtype=np.uint8).reshape(fr.height, p.line_size)[:, :fr.width]
    break
```

Transfer rate, measured: 26.12 GB PRFS → local in **41.75 s = 625 MB/s** (~5 Gbps on a
10 GbE link).

---

## 9. Open questions for acquisition work

1. **Is exposure already at the 10 ms ceiling?** If not, raising it is the cheapest SNR
   win available. If it is, more light is the only lever.
2. **Separate shot from read noise properly** — run an illumination ramp and check
   whether σ ∝ √luma, measured on **lossless** data. The flat profile observed here was
   on the denoised master over a narrow luma range and cannot settle it.
3. **The 2.85× noise mismatch between pixel sources** (§3) — does it degrade any model
   trained on one and run on the other?
4. **QP maps driven by realtime detections.** `qpDeltaMap` is reachable (the recorder is
   a direct NVENC SDK integration — `NvEncLockBitstream`, `NvEncUnmapInputResource`,
   `NvEncEncodePicture` all appear in the timing CSVs). Cautions: the map for frame *N*
   must exist before submit (crop stream `enqueue_age_p95_ms` ≈ 19.7 ms against a 10 ms
   budget, so it must be predictive from *N−1* with a padded ROI); on
   `has_detection == 0` it must fall back to uniform QP; it must **never block** the
   encoder submit path, because the master stream is currently independent of detection
   liveness and should stay that way. Note also that it protects the same pixels the
   lossless crop already protects and fails on the same frames. A **static** map
   (coarse outside the dish, from `fisheye.shared.arena_geometry`) is risk-free, and the
   chaser dot is a better dynamic target since its position is known a priori.
5. **Retagging pre-cutover files to `pc`** would remove the split-brain in §2. Metadata
   rewrite only — never a pixel conversion.
