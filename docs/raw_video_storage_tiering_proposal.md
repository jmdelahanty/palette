# Raw Video Storage Tiering Proposal

<!-- contract-meta
status: proposal
last_verified: 2026-07-24
purpose: Decide where raw acquisition video, derived crops, proxies, and analysis products live across PRFS / NRS / Nearline.
-->

## The immediate problem

`/groups/johnson` is **65 TB, 94% full, 4.0 TB free.**

Measured footprint of `johnsonlab/jeremy/recordings` on 2026-07-24:

| Artifact | Count | Total | Mean each |
|---|---|---|---|
| Master acquisition MP4 (`cams/`) | 153 | **4.97 TB** | 32.5 GB |
| Crop MP4 (`derived/external_crop_recorder/`) | 120 | **1.13 TB** | 9.4 GB |
| Analysis Zarr (`zarr/`) | 153 | **0.36 TB** | 2.34 GB |

Total measured: **6.46 TB**.

Acquisition rate, from directory dates: **60 recordings/month** in each of June and
July 2026. At the measured means that is **~2.6 TB/month, ~31 TB/year.**

**Runway: 4.0 TB free ÷ 2.6 TB/month ≈ 6 weeks.** Around early September 2026 the
lab's shared PRFS volume fills, and this project's video is what fills it. That
outage lands on everyone in the lab, not just this project.

This is the part that makes the decision urgent rather than philosophical. The
question is not "should we start putting raw video on backed-up storage" — 4.97 TB
of raw video is *already* on PRFS. The only open question is whether it leaves on
a plan or leaves during an emergency.

Current alternate tiers are provisioned far too small and will need a formal
request to SciComp:

- `/nrs/johnson` — 5.0 TB total, 4.8 TB free
- `/nearline/johnson` — 1.0 TB total, 997 GB free

## The measured encoding facts

Master video: `4512x4512`, HEVC `yuv420p`, 100 fps, **150 Mbps**, ~23.3 min/recording.

- Raw equivalent: 4512² × 1.5 B/px × 100 fps = **24.4 Gbps**
- Stored at 150 Mbps → already **163:1 compression**

Crop video: `256x256`, HEVC, 100 fps, **32 Mbps**, same duration.

- Raw equivalent: 256² × 1.5 B × 100 fps = 78.6 Mbps
- Stored at 32 Mbps → **2.4:1 compression**

Encoder settings, read from the acquisition records:

| | `external_recorder_supervisor_plan.json` / `_summary.json` |
|---|---|
| Master | `codec: hevc`, **`preset: p1`**, `bitrate_bps: 150000000`, `max_bitrate_bps: 150000000` |
| Crop | `codec: hevc`, **`preset: p7`**, **`tuning: lossless`**, `rate_control_mode: vbr` |

This is a deliberate and well-designed dual-stream capture: a lossy full-frame
context stream that must keep up in realtime, plus a **truly lossless** crop cut from
live YOLO detections for maximum detail. Two consequences follow, and they point in
opposite directions.

### The crop is lossless by design — that is not a tunable

`tuning: lossless` at `preset: p7` means 32 Mbps is simply what lossless costs for
256×256 at 100 fps (2.4:1 against the 78.6 Mbps raw rate). There is no CRF to raise.
My earlier framing of this as "over-encoded" was wrong twice over: it is not a bitrate
setting, and the stream is doing double duty as both the fine-detail visualization
substrate and — per
[`acquisition_crop_video_roi_provider_plan.md`](acquisition_crop_video_roi_provider_plan.md)
— as `CropImageSource` **model input pixels** for keypoint and mask inference.

The only open question is whether *visually* lossless (CQ ~12-16) would serve both
roles as well as true lossless. That is a legitimate measurement, worth perhaps 3-5×
on 1.13 TB, but it is a real scientific tradeoff and not free.

### The master is on `p1` at a fixed bitrate — this is the biggest lever in the store

`p1` is NVENC's **fastest and least efficient** preset, correctly chosen because
full-frame encoding must sustain 100 fps at 20 MP in realtime. But two things make
the resulting file far larger than its quality warrants:

1. **`p1` vs `p7`** costs roughly 20-30% bitrate at equal quality on its own.
2. **`bitrate_bps` is a fixed 150 Mbps budget, not a quality target.** A near-static
   arena with one small fish does not *need* 150 Mbps — the encoder spends its whole
   allowance because it was told to. A constant-quality (CQ/VBQ) encode on content
   this compressible allocates dramatically fewer bits.

**Archival is not realtime-constrained**, so an offline transcode can spend far more
time per frame. This is the broadcast pattern: a fast camera codec at acquisition,
transcoded to an efficient mezzanine/archive codec afterwards.

### Measured transcode sweep (2026-07-24)

Benchmarked on an RTX A6000. Preset sweep on a 60 s / 6000-frame stream-copied segment
of `2026-07-21T19-38-32Z_arena_1_Batman` (4512×4512, `pc`-tagged, 150 Mbps); quality
ladders on the first 600 frames of the same segment (112.5 MB source).

#### Preset is not the lever

| Preset | Wall (6000 fr) | Output | vs source | Pipeline fps |
|---|---|---|---|---|
| `p3` | 120.3 s | 716.3 MB | 1.57× | 49.9 |
| `p5` | 194.5 s | 678.2 MB | 1.66× | 30.9 |
| `p7` | 393.8 s | 678.6 MB | 1.66× | 15.2 |

`p7` produced a file **0.06% larger than `p5`** while taking 2.2× longer. There is no
reason to go past `p5`. Decode-only baseline was 33.05 s (182 fps). All presets run
below the 100 fps acquisition rate, which independently confirms `p1` is necessary at
capture time.

#### `-cq` silently does nothing — use `-rc constqp`

`-rc vbr -cq N -b:v 0` at N = 20 and N = 24 produced **byte-identical output**
(710,721,538 bytes both times). The constant-quality target was not being applied;
everything clamped near 90-95 Mbps. **An earlier version of this memo reported "1.6×
is all that is available" — that conclusion was an artefact of this broken rate
control.** `-rc constqp -qp N` works correctly and spans a wide range.

#### Quality ladder, scored in the fish neighbourhood

Fish located per-frame by temporal differencing; metrics over a 384×384 box around it.
**Scored over 25 frames** — an earlier 4-frame version of this table was too noisy to
rank encodes (per-frame σ ≈ 2.3 dB, so n=4 gives ±1.2 dB standard error) and produced a
spurious non-monotonicity.

| Encode | Output | vs source | Mbps | **Fish dB** (n=25) | σ | Fish MAE | RMS err | Edge shift |
|---|---|---|---|---|---|---|---|---|
| source `p1` | 112.5 MB | 1.00× | 150.0 | — | — | — | — | — |
| `nvenc qp22` | 93.7 MB | 1.20× | 125.0 | **42.49** | 2.33 | 1.48 | 1.91 | 0.067 px |
| `nvenc qp26` | 47.0 MB | **2.39×** | 62.7 | **41.11** | 2.38 | 1.73 | 2.24 | 0.078 px |
| `nvenc qp30` | 16.8 MB | **6.69×** | 22.4 | **39.15** | 1.96 | 2.18 | 2.81 | 0.098 px |
| `nvenc qp34` | 6.8 MB | 16.52× | 9.1 | **38.12** | 2.37 | 2.46 | 3.17 | 0.111 px |
| `x265 crf22` | 41.5 MB | 2.71× | 55.3 | **40.35** | 2.10 | 1.89 | 2.45 | 0.086 px |

Monotonic, as it should be. All NVENC outputs preserved `Y` 9-255 and `color_range=pc`.

#### Physical interpretation

Measured in the fish region of the source: contrast (p99−p1) = **120 grey levels**,
silhouette edge steepness (p99.5 of gradient) = **28.6 grey levels/pixel**. Dividing
RMS error by edge steepness gives the apparent displacement of the silhouette — the
"Edge shift" column. At `qp26` the fish outline moves **0.078 px on average**; even
`qp34` is 0.111 px. Worst-case single-pixel errors (22-30 levels) correspond to ~0.8-1.0
px of local edge displacement.

#### Two corrections to earlier versions of this memo

1. **These encodes are NOT at the sensor noise floor.** Measured directly: consecutive
   source frames differ by σ = **0.74 grey levels** in a static background region, which
   corresponds to **50.74 dB**. Every encode here sits at 38-42 dB, i.e. **8-13 dB
   worse than the floor**. They are losing real signal, not merely discarding noise.
   The earlier "PSNR saturates at the noise floor" claim was wrong.

2. **NVENC's advantage over x265 is ~1.3×, not 2.5×.** Interpolating the NVENC curve to
   `x265 crf22`'s 40.35 dB gives roughly 31.5 MB against x265's 41.5 MB. NVENC still
   wins, and is 2.3× faster with no range trap, so the recommendation to stay on NVENC
   stands — but the margin is modest, not dramatic.

**This is the limit of what encoder metrics can establish.** The sweep bounds the
available savings; it cannot certify fidelity. Only the pipeline can. Score `qp26` and
`qp30` on **mask boundary stability and tail-spline agreement**, over **escape epochs
specifically** rather than average frames — masks and tail splines are the fragile
consumers, and "detection still works" is weak evidence.

**Operational cost.** Encode time is independent of QP (~30 fps at `p5`), so a full
1398.77 s recording takes ~78 min of GPU time; the 153-recording backlog is
**~198 GPU-hours (~8 days on one A6000)**, ~78 GPU-hours/month ongoing.

## The conceptual error to avoid

> "If I have to downsample for viewing anyway, I should just acquire at low
> resolution."

No. Acquisition resolution and *service* resolution are independent decisions, and
collapsing them destroys the dataset.

- **100 fps is scientifically load-bearing.** Escape responses and C-starts are
  10-20 ms events. At 30 fps a C-start is one or two frames. Do not touch the frame
  rate.
- **20 MP is load-bearing, but only in a 256×256 neighborhood.** It exists so a
  small fish in a large arena still has enough pixels for tail keypoints and mask
  quality. Drop to 5 MP and the fish is 128×128 — probably fine for centroid and
  heading, marginal for pose.
- **The other 99.7% of the frame is arena background at a resolution nobody needs.**

So: acquire high, because you get exactly one chance at each animal on each day.
Serve low, because eyes do not resolve 20 MP at 100 fps and no display shows it.
Archive the master, because the only reasons to keep it — re-cropping after a
tracker failure, a different ROI, a second animal, a reanalysis you have not thought
of yet — are *archival* needs, not *online* needs.

## Classify by cost-to-recreate, not by size

This is the principle that makes the tiering fall out, and the one that will make
sense to a budget holder.

| Class | Cost to recreate | Examples | Tier |
|---|---|---|---|
| **Irreplaceable — physical** | Impossible at any price. That fish, that day. | Master acquisition MP4 (P1 HEVC) | **Nearline** archive + **NRS** serving copy |
| **Irreplaceable — higher fidelity than the master** | Impossible. Cannot be regenerated from the master at all. | **Lossless crop MP4** | **Nearline** archive + **NRS** serving copy |
| **Irreplaceable — human labor** | Months of a person. | Annotations, review corrections, curated training sets, the registry | **PRFS** (backed up) |
| **Human-facing, tiny** | Minutes of GPU per recording | Proxies, zoom-pyramid renditions | **PRFS** / **NRS** |
| **Expensive but derivable** | Compute time only, given master + crop + code | Analysis Zarr, detections, model outputs | **NRS** (not backed up — acceptable) |

> **Correction to an earlier version of this memo.** It placed crops in the
> "derivable" class, on NRS, treated as a regenerable cache. **That was wrong and
> dangerous.** The crops are cut *at acquisition* from live YOLO detections and are
> **lossless**, while the full-frame master is **lossy P1 HEVC**. The crop therefore
> contains strictly *more* information about the fish than the corresponding region
> of the master. It cannot be regenerated from the master — not with more compute,
> not ever. For the region that matters most scientifically, **the crop is the
> highest-fidelity record that exists.** It must be archived as primary data.

There is no pristine original anywhere in this system. The sensor data was discarded
at acquisition and the full-frame record has *always* been lossy P1 HEVC. The job is
to preserve what exists, not to imagine a lossless master that never was.

The load-bearing consequence: **once NRS holds only things derivable from an archived
master plus versioned code, "not backed up" stops being frightening.** Losing the
volume would cost compute time, not science. That discipline converts a scary cheap
tier into a free one. The mirror image is why the Synology is dangerous — not because
it is cheap disk, but because it holds the *irreplaceable* class. As a rig-side
landing buffer it is fine.

## The viewing substrate

### Why the current proxy fails

[`review_proxy_video_contract.md`](review_proxy_video_contract.md) specifies 1024×1024
H.264 proxies, display-only, regenerable. Four of 157 recordings have them, built for
the clip-review labeling app. Labelers report poor quality, and the arithmetic agrees:

- Source 4512×4512 → proxy 1024×1024 is a **0.227× scale**
- A fish detection box measures **153×121 px** natively (`crop_meta.csv`)
- In the proxy that fish is **35×27 pixels**

You cannot annotate a tail bend, an eye, or a body midline on 35 pixels. The design
error is structural: **a whole-frame downsample spends the entire pixel budget on
arena background to serve a task that only cares about a ~256 px neighbourhood.**

### The actual requirement is a zoom ladder

The stated need is to **pan and zoom the full frame while the fish crosses the arena,
plus the lossless crop for fine detail.** The fine-detail half is already solved — the
lossless crop *is* the detail view. The gap is that nothing bridges "whole arena at
1024" and "native resolution somewhere in 20 MP", and no single downsample level can,
because the fish goes everywhere.

The established answer is a **multiscale zoom ladder with tiled range requests** —
the gigapixel-image pattern (Neuroglancer, OME-Zarr, IIIF) and, for video, DASH-SRD
spatial tiling.

| Level | Content | Fetched when | Approx size/recording |
|---|---|---|---|
| L0 overview | 564² or 1024² whole frame | always; scrubbing, context | ~0.1-0.5 GB |
| L1 mid | 2256² whole frame | moderate zoom | ~1-2 GB |
| L2 native, **tiled** | 4512² as 4×4 grid of 1128² tiles | only viewport tiles | viewport-limited |
| Lossless crop | 256² native (exists today) | fish detail, labeling, model input | 9.4 GB |

The player always holds L0, swaps to L1 on zoom, and fetches only the L2 tiles under
the viewport. The user gets "full resolution, zoom anywhere" without transferring 32 GB.

**Cost, honestly.** A pyramid stores ~1.33× the pixel count (1 + ¼ + 1⁄16), and
independent tile encoding at L2 forfeits cross-tile prediction for perhaps another
10-30% — call it **~1.5-1.7× the master's size**. That is an increase, not a saving.
But the measured transcode (2.4× at `qp26`) more than pays for it: `2.39 ÷ 1.7 ≈ 1.4`,
so a fully zoomable pyramid still lands smaller than today's un-zoomable P1 master.

Sequence the expensive rung last. L0 and L1 are cheap and close most of the gap, and
the lossless crop already covers the fish. L2 native tiles are only needed to inspect
something that is *not* the fish at native resolution — the chaser dot, an arena
artifact, a second animal. Build L0/L1 first and add L2 where users actually hit the wall.

### Overlay alignment

Mapping overlays to any pyramid level is a single power-of-two scale, and onto the
lossless crop a pure integer translation — `crop_meta.csv` records per-frame
`crop_x`/`crop_y`. Both are cleaner than the current 0.227× non-dyadic scale, which
quantises overlay positions.

### Serving tier

Tiled range-request serving needs masters **online and range-readable**, which means
**NRS**, not PRFS. That does not conflict with the supervisor's position: NRS is the
cheap non-backed-up tier and is the right home for a serving copy whose archive of
record sits on Nearline.

## How this is done elsewhere

- **Film / broadcast post-production.** Camera original negative → ingest →
  immediately generate proxies → editors never touch the original → originals to
  LTO tape, two copies, one offsite. Same problem shape, forty-year-old solved
  answer.
- **Autonomous vehicles.** Petabyte-scale multi-camera logs go to object storage in
  archive tiers behind a metadata catalog. "Scenario extraction" pulls interesting
  segments (hard braking, near-miss) into a hot tier for training. The overwhelming
  majority of raw is never read again but is retained for rare-event queries.
- **Astronomy (Rubin/LSST, SDSS).** Raw exposures on tape; calibrated data products
  served online; a versioned "data release" is the science-usable artifact.
- **Genomics** — the instructive one, because the field *changed its mind*. Everyone
  originally kept raw intensity files; the community measured, decided they were not
  worth their storage, and discarded them in favour of FASTQ/BAM/CRAM. CRAM is
  explicitly lossy (reference-based, binned quality scores) and was validated
  empirically before adoption. That is the precedent for "measure whether the lossy
  version changes your answer, then commit."
- **Janelia internally.** The light-sheet and EM groups solved this with multiscale
  pyramids served to Neuroglancer, raw on archive. `/groups/ahrens` is mounted on
  this very machine. That is a precedent to cite and, more usefully, a person to ask.

## Proposed tiering

```
Rig / Synology     landing buffer only; deleted after verified copy + checksum
      |
      v  (measured: 26.1 GB master copied in 41.75 s = 625 MB/s over 10 GbE)
Cluster transcode  P1 150 Mbps -> nvenc p5 constqp (validated) ; ~78 min/recording
      |            + generate L0/L1 zoom-ladder renditions
      |
      +--> NRS  /nrs/johnson        SERVING + WORKING SET
      |                             transcoded master, zoom ladder, lossless crops,
      |                             analysis Zarr, model outputs
      |                             range-readable; 100 Gb to cluster; not backed up
      |
      +--> Nearline /nearline/johnson   ARCHIVE OF RECORD, write-once, checksummed
                                        untouched P1 master + lossless crops
                                        (both irreplaceable; neither derivable)

PRFS  /groups/johnson    annotations + registry + published Zarr + small proxies
                         irreplaceable human labor; backed up
```

**Both** primary streams go to Nearline. The lossless crop is not a cache.

On the transfer concern: a 26.1 GB master copied from PRFS to local disk in **41.75 s
(625 MB/s)**, measured. Roughly 3 minutes for all four arena cameras, once per
recording. Not a bottleneck — and it only stays that way because the proxy/crop
architecture avoids re-reading masters.

## Retention policy — including what gets deleted

Proposing deletions is what makes the retention asks credible.

**Delete now:**
- `.bak` sidecar MP4s (~43 GB seen in a single recording)
- `__h5_context_quarantine` directories for superseded recordings
- Failed/misconfigured recordings that QC marked unusable

**Transcode (after pipeline validation, never before):**
- P1 masters → `nvenc p5 -rc constqp`, measured 2.4× at `qp26` / 6.7× at `qp30`

**Retain forever, Nearline — both primary streams:**
- Any P1 master backing a publication, figure, or training set
- **The lossless crop for the same recording** — not derivable from the master

**Retain 2 years, then review:** everything else.

### What this does to the irreplaceable-data budget

Per recording the irreplaceable payload is master + lossless crop:
**32.5 + 9.4 ≈ 42 GB**, i.e. **~2.6 TB/month, ~30 TB/year** that must be archived.

| Scenario | Master | Per recording | Per year | Status |
|---|---|---|---|---|
| Today | 32.5 GB | 42 GB | ~30 TB | measured |
| Master at **2.39×** (`qp26`, 40.8 dB fish) | 13.6 GB | 23 GB | **~16.5 TB** | measured; needs pipeline validation |
| Master at **6.69×** (`qp30`, 39.2 dB fish) | 4.9 GB | 14.3 GB | **~10.3 TB** | measured; needs pipeline validation |

A validated transcode plausibly cuts the archival obligation **~1.8-2.9×**. Pitch the
conservative `qp26` point (41.1 dB in the fish region, 0.078 px mean edge displacement).

**Do not pre-spend this in the Nearline request.** Size on ~30 TB/year until pipeline
validation passes.

### Where the background noise actually comes from (measured)

The rationale above assumed "150 Mbps is being spent encoding sensor noise." **That is
not what is happening.** Measured on `2026-07-21T19-38-32Z_arena_1_Batman`, frames
131-144 (a run where `crop_x`/`crop_y` are constant, so the lossless crop and the master
cover identical sensor pixels):

| Source | Background temporal σ |
|---|---|
| **Lossless crop** (true sensor) | **4.279 grey levels** |
| **P1 master** (same photons) | **1.502 grey levels** |
| Ratio | **0.35** |

**The P1 encode is acting as a temporal denoiser, discarding ~65% of the sensor noise.**
At 0.074 bits/pixel it cannot afford to code the noise, so it smooths it away. The bits
are not going into noise — they are going into the static spatial texture (measured
spatial residual ≈ 8 grey levels) and the residual noise it does retain.

Ruled out as causes of background cost:
- **Illumination flicker.** Whole-frame mean luma over 2 s: σ = 0.0145 grey levels,
  peak-to-peak drift 0.093, no periodic component above the noise floor. Rock stable.
- **Fixed-pattern noise.** Static, so it costs bits once in the I-frame and is perfectly
  predicted thereafter.
- **The encoder.** It removes noise rather than adding it (above).

Sensor σ ≈ 4.3 grey levels at luma ≈ 190 is consistent with **photon shot noise** at
roughly 2,000 photoelectrons — expected for a ≤10 ms exposure forced by the 100 fps
rate. That is fundamental: the only real fix is more photons (brighter illumination,
wider aperture, or binning at a spatial-resolution cost).

### Consequence: the two pixel sources have different noise levels

The lossless crop carries σ ≈ 4.3; the master carries σ ≈ 1.5 — a **2.85× difference on
the same photons.** Any stage trained on one and run on the other sees a domain shift.
This matters because `CropImageSource` reads the lossless crop for keypoint/mask work
while YOLO detection runs `pynvvc_nv12_rgb` on full-frame master pixels. Worth auditing
independently of the storage question.

## Pixel-contract implications of transcoding

Transcoding interacts directly with the contracts in
[`video_pixel_model_input_contract.md`](video_pixel_model_input_contract.md) and
`src/fisheye/shared/roi_pixel_contract.py`. What survives and what does not:

### Survives

- `uint8`, `[N,H,W]`, C order, zero padding outside frame bounds.
- Orange mono semantics — camera intensity in the NV12 `Y` plane, neutral `UV` —
  provided the transcode stays `yuv420p` and never round-trips through RGB.
- The `pynvvc_luma` decode path, which reads the `Y` plane directly and by contract
  performs **no decoder range remap**.
- Frame geometry, `roi_coordinates_full`, `crop_x`/`crop_y`, `center_rounding`, and
  the reversible model-input transforms — all downstream of pixel values.

### Does not survive, and must be planned for

1. **Exact pixel values.** Lossy→lossy requantisation shifts sample values slightly.
   No contract promises bit-exactness, but every previously computed detection, pose,
   and mask was computed on the *original* bytes. Re-running against a transcode will
   not reproduce them exactly. That is a reproducibility boundary and must be recorded
   as one.

2. **The `color_range` tag — the live landmine, and the store is split.**

   Measured across the whole store on 2026-07-24:

   | Stream | `tv` | `pc` | Cutover |
   |---|---|---|---|
   | Master (`cams/`) | 100 | 53 | last `tv` = **2026-07-01**, first `pc` = **2026-07-02** |
   | Crop (`external_crop_recorder/`) | 72 | 48 | same boundary, same day |

   The runtime tagging fix landed **2026-07-02** and applied to both streams
   simultaneously.

   **Verified: the fix changed the tag only, not the pixels.** Decoding the raw `Y`
   plane via PyAV — no swscale, no format conversion — on same-protocol recordings
   one day either side of the cutover:

   | File | Tag | `Y` min/max | px < 16 | px > 235 |
   |---|---|---|---|---|
   | 2026-07-01 arena_1 | `tv` | 8 / 255 | 2,251,752 | 126,061 |
   | 2026-07-02 arena_1 | `pc` | 8 / 255 | 2,289,839 | 35,098 |

   Both hold genuine **full-range mono8**. So post-cutover files are now correctly
   self-describing, and pre-cutover files carry a `tv` **mislabel** over full-range
   data. Palette is safe across the boundary *only* because
   `roi_pixel_contract.py` reads the `Y` plane directly and performs
   `read_direct_y_plane_without_decoder_range_remap`.

   **Any tool that honours the tag behaves differently on either side of
   2026-07-02.** This was demonstrated accidentally while producing the table above.
   A naive extraction — `ffmpeg -i IN -f rawvideo -pix_fmt yuv420p -`, with no
   explicit output range — returned:

   - `tv` file → `Y` 9-255 (input tagged limited, output defaulted limited: identity)
   - `pc` file → `Y` **23-235**, zero samples outside 16-235 (input tagged full,
     output defaulted limited: **swscale silently compressed full → limited**)

   Same command, same camera, one day apart, ~15% contrast compression on one of
   them and none on the other. Nothing errored, and the output looked entirely
   plausible. Read naively, that table is fabricated evidence of data corruption that
   never happened.

   Two consequences for any transcode:

   - **A single command applied store-wide will do different things to pre- and
     post-cutover recordings.** Range handling must be pinned explicitly per file
     against the *contract* (full-range mono8), never inherited from the container tag.
   - **Verification tooling is subject to the identical trap.** Any parity check must
     read raw decoded planes (PyAV `frame.planes[0]` or PyNvVideoCodec), never a
     convenience path that may invoke swscale.

   For pre-cutover files a transcode is also the natural moment to correct the tag to
   `pc` — but **set the tag without converting.** `-color_range pc` on the encoder
   writes VUI metadata; `scale=in_range=…:out_range=…` rescales samples. Confusing
   the two is the bug.

3. **Frame count and ordering.** Every join key in the system — `recording_frame_id`,
   `local_frame_id`, crop rows, Zarr frame axes — assumes an exact frame timeline.
   Encode with `-fps_mode passthrough` (`-vsync 0`), no `-r`, no fps filter, and
   assert `nb_frames` is identical (139877 in the reference recording) before
   accepting any output.

4. **Frame indexes and GOP structure.** `_keyframe.json`, `*_gop_routing.csv`, and the
   registry's `source_recording_frame_index_path` all describe the *original* packet
   layout. A transcode changes keyframe placement and byte offsets, so every cached
   frame index must be rebuilt. Correctness is recoverable; forgetting the step is not.

5. **Provenance identity.** Content hashes are recorded per artifact. A transcode
   changes the file hash, so it must be registered as a **new derived artifact with
   its own identity and a link to its source**, never as a silent in-place
   replacement. Otherwise the provenance chain asserts something false.

6. **Full-frame detection is a model input, not just a viewing artifact.** Per
   `video_pixel_model_input_contract.md`, YOLO detection runs `pynvvc_nv12_rgb` over
   **full-frame** tensors. Transcoding the master therefore changes future detection
   inputs. Validation must include full-frame detection parity, not only crop pose.

### The x265 range trap (measured)

The x265 path applies a full→limited range compression that **`-color_range pc` does
not prevent**, and `-x265-params range=full` does not prevent either. Measured output
was `Y` **25-240 with zero samples below 16**, mean shifted 9.2 levels — while still
*tagged* `pc`. Data limited-range, container claiming full-range: mislabelled in the
more dangerous direction, and invisible without a raw plane check.

It also made all three CRF levels score an identical ~25 dB, which reads as uniform
catastrophic quality loss rather than as a range bug. **That table looked like evidence
of data corruption and was not.**

The fix is an explicit identity scale:

```
-vf "scale=in_range=full:out_range=full" -pix_fmt yuv420p -color_range pc
```

Verified to restore `Y` 10-255 with mean matching source to 0.02. The NVENC path
(`hevc_cuvid` → `hevc_nvenc -color_range pc`) needed no workaround and was byte-clean
throughout — one more reason to prefer it.

### Hard scoping rule

**Never transcode the lossless crops.** Re-encoding a lossless stream to anything
lossy destroys the exact property that makes it primary, irreplaceable data. Only the
already-lossy P1 master is a transcode candidate.

### Acceptance test before any transcode is trusted

1. `nb_frames`, `width`, `height`, `pix_fmt` identical to source.
2. Decode both, compare `Y` planes numerically — mean absolute difference, max
   difference, and Y-only PSNR. A range-shift bug shows up immediately as a large
   systematic offset rather than small scattered quantisation noise.
3. Spot-check `Y` min/max per frame. If the source spans roughly 0-255 and the output
   is pinned near 16-235 (or vice versa), a range conversion fired — stop.
4. Full-frame detection parity and crop pose/mask parity through the existing pipeline.
5. Only then register the transcode as a derived artifact. Keep the P1 original on
   Nearline regardless.

### This is optional, and should be sequenced last

The tiering plan does not depend on the transcode. Storing P1 masters as-is on
Nearline solves the capacity problem on its own; the transcode is a cost optimisation
worth a measured 2.4-6.7×. Given the contract sensitivity, **do the tiering first and
treat the transcode as a follow-on project** with its own validation gate.

That said, walking into the meeting able to say "the transcode measurement is done and
shows 2.4× available at 0.078 px mean silhouette displacement" is a stronger
position than "I need more storage" — provided it is presented as measured-but-not-yet-
validated, which is exactly what it is.

## The arguments that actually move a budget holder

1. **Concede the expensive tier immediately.** Do not ask for masters on PRFS. The
   supervisor is right that 30 TB/year of raw video does not belong on the charged,
   backed-up tier. Saying so first buys the credibility to insist on the rest.
2. **Lead with the runway, not the principle.** 6-7 weeks to a full lab volume. This
   is happening regardless of what anyone decides. Be the person who prevented a
   lab-wide outage.
3. **Separate the three asks, because they land on three different budgets.**
   - **PRFS (charged, backed up):** annotations, registry, published Zarr, small
     proxies — **under 0.5 TB, growing slowly.** This is the only expensive ask and
     it is nearly nothing.
   - **Nearline (cheap, cold):** the irreplaceable archive, both primary streams —
     **~30 TB/year.** Size the request on this; a transcode saving is measured at
     2.4-6.7× but is not yet pipeline-validated, so do not pre-spend it.
   - **NRS (cheap, not backed up):** serving copies, zoom ladder, analysis Zarr,
     model outputs. Sized for the active working set, not all of history.

   Presented this way, the supervisor is not being asked to put raw video on the
   expensive tier at all — which is the position he already holds. The large asks are
   both for cheap tiers.
4. **Quantify reacquisition honestly.** 153 recordings represents more than a year of
   rig time and protocol-counted animals, and it is not reproducible — that fish, that
   day, that stimulus. The comparison is never "storage vs. zero." It is "storage vs.
   the fraction of a person-year the data represents." Nearline is typically 5-20×
   cheaper than primary storage; **get the actual Janelia $/TB/year figures from
   SciComp before the meeting** — this memo deliberately does not guess at them, and
   the argument is much weaker without them.
5. **Bus factor.** The Synology has a bus factor of one. When the person who set it up
   leaves, nobody will know how it is organized, what is on it, or that the drives need
   replacing. Institutional storage has a bus factor of the institution. This argument
   is about the PI's own long-term interest — their lab's data outliving their trainees
   — and it is usually the one that lands.
6. **Storage is a publication cost, not an IT cost.** Check the lab's NIH Data
   Management and Sharing obligations and HHMI's open-science commitments. If data
   supporting a published figure lives only on an unmonitored, unbacked Synology, that
   is a real risk to the publication record. Reframing the line item from
   "infrastructure" to "cost of publishing" changes who owns the budget.

## Non-negotiables regardless of which tiers are chosen

- **Checksum at ingest**, store the hash in the registry (provenance content hashes
  already exist), and verify on a schedule.
- **Do a test restore quarterly.** An archive nobody has ever restored from is not an
  archive.
- **Never delete a master until the archived copy's checksum has been verified**, not
  merely until the copy command exited zero.
