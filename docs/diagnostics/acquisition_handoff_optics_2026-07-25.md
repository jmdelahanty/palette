# Acquisition Handoff — Optics, Sensor, and Illumination Findings

<!-- contract-meta
status: handoff
last_verified: 2026-07-25
purpose: Self-contained brief for acquisition-side work, summarising measured rig characteristics, corrections to earlier claims, and open actions.
-->

Self-contained brief. Full detail in
[`rig_characterization_2026-07-24.md`](rig_characterization_2026-07-24.md); complete
derivation and commands in
[`storage_and_rig_conversation_2026-07-24.md`](storage_and_rig_conversation_2026-07-24.md).

Rig: 4512×4512 mono, 100 fps, 850 nm illumination, 100 mm macro lens, HEVC dual-stream
capture (lossy `p1` full frame at fixed 150 Mbps + **lossless** `p7` fish crop).

---

## 1. Measured — trust these

### Fish geometry (refined subject masks, GoodCopBadCop arena_1, n=40 frames)

| Quantity | Value |
|---|---|
| Body length along principal axis | **127 px** |
| Body area | 2,535 px |
| Body maximum width | **36.1 px** |
| **Tail width, last 1/10 of length** | **11.9 ± 4.1 px** |
| Tail width, last 2/10 | 14.0 px |

Width profile head → tail (10 stations of ~13 px): 28.6, 35.3, 35.7, 31.5, 27.5, 24.0,
22.7, 20.8, 16.1, 11.9 px. Cross-checks against the acquisition recorder's own Batman
detection box (138 × 127 px).

### Sensor — shot-noise limited, small well

Photon transfer curve (per-pixel temporal variance vs mean, on **lossless** crop data):

| Fitted quantity | Value |
|---|---|
| Linearity | **r = +0.9988** |
| Gain | **g = 10.12 e⁻ / DN** |
| Black level (x-intercept) | **−0.4 DN** (no pedestal) |
| Electrons at background DN 207.7 | 2,106 e⁻ |
| **Full-scale capacity (DN 255)** | **≈ 2,584 e⁻** |
| **SNR ceiling at full scale** | **51** (≈2% noise) |

σ/√L is constant to four significant figures across luma bins — genuinely shot-noise
dominated. Background sits at DN 208 = **81% of full scale with zero clipping**: the
exposure is well chosen.

### Noise, and what the encoder does to it

| Source | Background temporal σ |
|---|---|
| Lossless crop (true sensor) | **4.279 DN** |
| P1 master (identical sensor pixels) | **1.502 DN** |

**The P1 encoder removes ~65% of the sensor noise** — at 0.0737 bits/pixel it cannot
afford to code it. Consequence: the two pixel sources differ in noise by **2.85×**, and
keypoint/mask stages read the lossless crop while YOLO detection reads master pixels.
That is an unaccounted domain shift worth auditing.

### Illumination

- **Temporally excellent:** whole-frame mean luma σ = **0.0145 DN** over 2 s,
  peak-to-peak drift 0.093 DN, no periodic component above the noise floor.
- **Spatially not:** **−27% centre-to-dish-edge falloff** (238 → 172 DN), smooth and
  monotonic. No localized IR hotspot (centre excess +6.4 DN is within quadratic
  modelling error).

### Contrast — the tail is the weak link

Background 207.7 DN, fish 165.0 DN, contrast 42.8 DN, overall CNR 10.1. Broken out:

| Population | Δ DN | Signal e⁻ | **CNR** |
|---|---|---|---|
| pigment / eye (darkest 0.5%) | 153.8 | 1,556 | **34.2** |
| body (2-6%) | 29.3 | 296 | **6.5** |
| thin tail / fin (15-30%) | 14.4 | 146 | **3.2** |

Well accounting: **80% of the 2,584 e⁻ capacity is spent establishing the bright
background**; only ~404 e⁻ carries fish signal. Brightfield CNR = Δe⁻/√N_background, so
the background you do not care about sets the noise floor.

### A position-dependent detection gradient

546,186 detected frames, 4 Batman recordings:

| Radius (px) | Mean YOLO confidence |
|---|---|
| 0-400 | 0.759 |
| 1200-1600 | 0.785 |
| 1900-2100 | **0.681** |
| 2100-2400 | **0.609** |

r(radius, confidence) = **−0.337**. Inner (r<1200) 0.761 vs outer (r>1900) 0.663 —
**−12.8%**. The fish spends **36% of its time at r>1900** versus 13.5% inside r<1200, so
thigmotaxis puts the animal where detection is worst.

**Illumination is probably not the main cause.** Illumination falls smoothly from the
centre (already −10% by r=1550) while confidence is *flat* to r≈1600 then collapses. The
sharp onset points to geometry: image-border effects (frame edge at r=2256), the dish wall
and meniscus with its reflections, or wall-parallel posture and defocus. **Either way there
is a detection-quality gradient correlated with wall proximity — a variable used directly
in thigmotaxis and chase-mediation analysis.**

### MEASURED system sharpness — softer than physics requires

1,446 edge profiles from 10 frames of the **lossless** crop, sampled along the gradient
normal to the fish silhouette and aligned on the sub-pixel 50% crossing. Encoder softening
is excluded by construction.

| Quantity | Measured |
|---|---|
| 10-90% edge rise | **3.45 px** |
| LSF FWHM | **3.50 px** |

| Frequency | **Measured MTF** | Diffraction f/2.8 | Diffraction f/8 | Pixel sinc |
|---|---|---|---|---|
| 0.042 cyc/px — tail scale (12 px) | **0.862** | 0.963 | 0.895 | 0.997 |
| 0.125 cyc/px — 4 px feature | **0.490** | 0.890 | 0.690 | 0.974 |
| 0.25 cyc/px — 2 px edge | **0.033** | 0.782 | 0.399 | 0.900 |
| 0.5 cyc/px — Nyquist | **0.006** | 0.570 | 0.002 | 0.637 |

**A 3.5 px edge is 2-3× more blur than diffraction alone would produce at any sensible
f-number.** Encoder softening is ruled out (lossless source) and the fish was near-static
(minimal motion blur), so the residual is **defocus, aberrations, or the object's own edge
softness**. Defocus is the prime suspect and **focus verification under 850 nm is the
cheapest test.**

⚠ This measurement gives (system MTF) × (object edge spectrum). A semi-transparent tapering
fish with a fin fold is not a knife edge, so its intrinsic softness inflates the apparent
blur by an unknown amount. **Separating the two requires a known step — the empirical
argument for a chrome-on-glass slanted-edge target.**

### Consequence: mask boundaries are blur-limited, not noise-limited

Boundary localization ≈ (edge width) / CNR:

| Feature | CNR | Approx boundary precision |
|---|---|---|
| Body | 10.1 | ~0.35 px |
| Thin tail / fin | 3.2 | **~1.1 px** |

Tail outlines are uncertain at roughly the **1 px** level, and MTF at the 2 px scale is only
0.033 — so this is dominated by **blur, not photon noise**. Improving tail-spline precision
needs *sharpness* (focus, aberrations), not more light.

Side effect: the extra apparent edge motion from `qp26` transcoding was 0.078 px against a
0.35-1.1 px baseline — negligible in quadrature, which strengthens the transcode case.

### Optics trade at 850 nm

Diffraction MTF (Airy Ø = 2.07·N µm) and depth of field, for an assumed 3.45 µm pitch and
m = 0.120. DOF is total object depth **in air**; real water depth ≈ n × that ≈ 1.33×.

| f/N | MTF @ tail (12 px) | MTF @ 2 px edge | MTF @ Nyquist | DOF (1px) |
|---|---|---|---|---|
| f/2.8 | **0.963** | 0.782 | 0.570 | 1.50 mm |
| f/4.0 | **0.948** | 0.690 | 0.399 | 2.15 mm |
| f/5.6 | **0.927** | 0.570 | 0.197 | 3.01 mm |
| f/8.0 | **0.896** | 0.399 | 0.002 | 4.29 mm |
| f/11 | **0.857** | 0.209 | 0.000 | 5.90 mm |

At 12 px the tail is nowhere near Nyquist and **tolerates stopping down comfortably**
(MTF 0.90 even at f/8). **Depth of field is the binding constraint**, not resolution: at
f/2.8 the in-focus slab is only ~2 mm of real water depth, and with a single top-down
camera there is **no z measurement**, so defocus is an unmeasured nuisance variable.

---

## 2. Corrections — do not chase these, they were wrong

| Earlier claim | Status |
|---|---|
| "Tail is ~3 px wide" | **Wrong — measured 11.9 px.** Was an assumed tail thickness divided by an estimated object scale. |
| "Stopping down will blur the tail, so it is barely available" | **Wrong.** MTF at the tail's own scale is 0.90 at f/8. Stopping down for DOF is safe. |
| "Shorter IR wavelength is the biggest win" | **Closed.** 850 → 780 nm is worth ~1.8× in electrons, but the rig is *well*-limited, not photon-limited, so it buys no SNR. **Stay at 850 nm.** |
| "Just lower the analog gain" | **Probably not available.** Gain register 256 is almost certainly Q8 unity, and unity is typically the floor. |
| "Sensor noise σ ≈ 0.74 DN" | **Wrong region.** That was a dark corner. Real background noise is **4.28 DN** on lossless data. |
| "Refined masks contain ~28% spurious satellite blobs" | **Retracted.** Artefact of naively thresholding raw probability masks at 0.5. Refined masks give median **1** component per frame, largest-component area fraction **1.000**. |
| "150 Mbps is wasted encoding sensor noise" | **Wrong.** The encoder *removes* 65% of the noise. Recurring bit cost is static spatial texture (~8 DN residual), not noise. |
| The diffraction MTF table read as a prediction | **It is a ceiling, not a prediction.** MTF multiplies through the chain (diffraction × pixel × defocus × aberrations), and the *measured* system MTF is well below the diffraction curve. |

---

## 3. Still unverified — please supply or measure

These are assumptions that propagate into the MTF/DOF table and the object-scale figures:

1. **Sensor pixel pitch** (assumed 3.45 µm) — from sensor width ÷ 4512.
2. **Object scale** (assumed 28.7 µm/px, from assuming a 127-153 px fish is ~4.4 mm).
3. **Magnification** (0.120) and working distance (~0.93 m) — derived from the above.
4. **Current f-number.**
5. **Exposure time** — reported as 50 µs; confirm, and confirm whether it is at the
   100 fps ceiling.
6. **Water depth**, and how much of the column the fish uses.
7. **Gain register semantics** — is 256 analog unity, and is sub-unity available?

---

## 4. Recommended actions, highest value first

1. **Replace eye-based focusing with a quantitative focus sweep.** Focus is already set on
   the live 850 nm stream, so chromatic shift is *not* the issue — but visual judgement on a
   possibly-downsampled preview cannot resolve the difference between 1.2 px and 3.5 px of
   blur. Use a static high-contrast target under water at the fish plane, step the focus
   parameter, and maximise a sharpness metric computed on **full-resolution** frames. See
   "Focus procedure and tank depth" above.
2. **Set aperture to f/4-f/5.6.** The tank is ~3 mm deep; f/5.6 gives ~4 mm of water-depth
   DOF (full coverage) for a ~4% tail-MTF cost, and the light loss is free given the
   illumination surplus.
3. **Look for a high-full-well / low-conversion-gain readout mode.** The 2,584 e⁻
   full-scale span is the root constraint behind both the SNR ceiling (51) and the
   apparent saturation at 50 µs. A dual-conversion-gain sensor in HCG mode would explain
   it exactly; switching to LCG could multiply the well several-fold for free.
4. **Run a photon transfer curve at 2-3 gain settings** (e.g. 128 / 256 / 512), on
   lossless captures. If `g` scales inversely with the register, gain is genuinely analog
   and there is headroom. If `g` barely moves, it is digital-only and lowering it gains
   nothing. Method: per-pixel temporal variance vs mean; slope = 1/g, x-intercept = black
   level.
5. **Do not lengthen exposure.** Shot-noise SNR = √N and N ∝ intensity × exposure, so
   trading light for time is a null trade — but exposure buys motion blur. At ~35 px/mm,
   50 µs gives 0.18 px of blur at 100 mm/s versus 3.5 px at 1 ms. The short exposure is
   protecting the escape-response kinematics.
6. **Water depth already matches the in-focus slab** at ~3 mm — no action needed, but keep
   it in mind if the arena is ever deepened.
7. **Score any aperture or focus change on edge gradient ÷ noise** (currently
   28.6 DN/px ÷ 4.5 DN ≈ 6.4 σ per pixel), not on visual sharpness.
8. **Flatten the illumination** — extend the illuminated area well beyond the dish so the
   dish sits in the flat centre of the profile. Also consider flat-field correction, noting
   it changes pixel values and so has pixel-contract and training/inference-consistency
   implications.
9. **Investigate the radial confidence gradient.** Flat-field first, then re-measure. Test
   against distance-to-wall and distance-to-frame-border *separately* — they are confounded
   in the current analysis. Also check whether mask area and tail-spline residuals show the
   same trend; they are more sensitive than detection confidence.
10. **Consider darkfield or oblique illumination for tail CNR.** In brightfield CNR ∝ √N, so
   lifting the tail from CNR 3.2 to 10 needs ~10× more electrons — unreachable at
   2,584 e⁻. Darkfield sidesteps it by deleting the background shot noise and freeing the
   80% of well currently spent on background. **Cost: polarity inverts (fish bright on
   dark), so detectors, masks, and training data all need revisiting.** Treat as a
   migration, not a knob. Oblique/partial darkfield is the cheaper intermediate.

### Calibration targets

- **A calibration grid / checkerboard is the highest-value purchase** — it pins object
  scale (µm/px) and distortion across the field. Bears directly on the known
  `arena_geometry` discrepancy (nominal experimental area ~3 mm off, which once silently
  inverted a thigmotaxis result), and validates the 28.7 µm/px figure everything else
  rests on.
- **A slanted-edge target (ISO 12233) beats USAF 1951** for this purpose: full quantitative
  MTF curve from one image, readable at the tail's actual spatial frequency. A
  chrome-on-glass knife edge suffices.
- **USAF 1951 is a threshold test** at ~100% contrast, whereas the tail is ~6% contrast —
  high-contrast limiting resolution does not predict low-contrast detectability. Fine as a
  config-to-config sanity check.
- **A tilted DOF target is genuinely relevant** given that DOF is the binding constraint.
- **Buy chrome-on-glass, not printed film** — printing inks are frequently semi-transparent
  at 850 nm, which destroys target contrast in a way that mimics an optics fault.
- **Measure through the actual water column and dish bottom, under real 850 nm
  illumination.** A target in air at visible wavelengths gets both the focal plane
  (chromatic shift) and the axial scale (refraction) wrong.
