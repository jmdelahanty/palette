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

Per-pixel bit density:

| | bits/pixel stored |
|---|---|
| Master (4512²) | 0.074 |
| Crop (256²) | 4.88 |

**The crop is stored at 66× the bit density of the master.** It covers 0.32% of the
frame's pixels but consumes 18% of the video bytes. At 2.4:1 it is close to
uncompressed.

That is *not* automatically waste. Per
[`acquisition_crop_video_roi_provider_plan.md`](acquisition_crop_video_roi_provider_plan.md),
`CropImageSource` reads these crop videos directly as **model input pixels** for
keypoint and subject-mask inference. Near-lossless encoding is a defensible choice
for a model input, and compression artifacts at 256×256 would land directly on the
pixels the pose network reads. So the correct move is the measurement in
[The measurement that buys the biggest multiplier](#the-measurement-that-buys-the-biggest-multiplier)
— quantify pose drift across a CRF ladder, then decide — not simply lowering the
bitrate. The 256×256 analysis crop may legitimately stay near-lossless. The *display*
artifact is a separate object with separate requirements, below.

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

This is the principle that makes the tiering fall out, and it is the one that will
make sense to a budget holder.

| Class | Cost to recreate | Examples | Tier |
|---|---|---|---|
| **Irreplaceable — physical** | Impossible at any price. That fish, that day. | Master acquisition MP4 | **Nearline** (cold, checksummed, write-once) |
| **Irreplaceable — human labor** | Months of a person. | Manual annotations, review corrections, curated training sets, the registry | **PRFS** (backed up) |
| **Human-facing, tiny** | Minutes of GPU per recording | Review proxies | **PRFS** (backed up) |
| **Expensive but derivable** | Compute time only, given master + code | Analysis Zarr, detections, crops, model outputs | **NRS** (not backed up — acceptable) |

The load-bearing consequence: **once NRS holds only things that are derivable from
an archived master plus versioned code, "not backed up" stops being frightening.**
You could lose the entire NRS volume and lose compute time, not science. That
discipline is what converts a scary cheap tier into a free one.

The mirror image is why the Synology is genuinely dangerous: it is not dangerous
because it is cheap disk, it is dangerous because it holds the *irreplaceable*
class. As a rig-side landing buffer it is completely fine.

## The viewing substrate: follow-crop, not whole-frame downsample

### Why the current proxy fails

[`review_proxy_video_contract.md`](review_proxy_video_contract.md) specifies
1024×1024 H.264 proxies, display-only, regenerable, overlays scaled by the frontend.
Four of 157 recordings have them materialized, built for the clip-review labeling
app. Labelers report the visual quality is poor, and the arithmetic says they are
right:

- Source 4512×4512 → proxy 1024×1024 is a **0.227× scale**
- A fish detection box measures **153×121 px** natively (`crop_meta.csv`)
- In the proxy that fish is **35×27 pixels**

You cannot annotate a tail bend, an eye, or a body midline on 35 pixels. This is not
a matter of preference; the artifact is unfit for the task.

The design error is structural: **a whole-frame downsample spends the entire pixel
budget on arena background in order to serve a task that only cares about a ~256 px
neighborhood.** It starves the one region anyone actually looks at.

### The fix costs nothing

Replace the 1024×1024 *downsample of the whole frame* with a 1024×1024
**native-resolution cutout that follows the fish**.

| | Fish size on screen | Frame dims | Encode cost |
|---|---|---|---|
| Current whole-frame proxy | 35×27 px | 1024×1024 | baseline |
| Native follow-crop | **153×121 px** | 1024×1024 | **identical** |

Same dimensions, same bitrate, same bandwidth, same player. **4.4× linear / 19×
areal improvement in the only region that matters**, purely by choosing which pixels
to spend the budget on. It also gives roughly 20 mm of surrounding arena at full
detail — wall proximity, the chaser dot, neighbouring fish — which the 256×256
analysis crop is far too tight to show, and which is a large part of why people ask
for the full frame.

Overlay alignment gets *easier*, not harder. `crop_meta.csv` already records
per-frame `crop_x`/`crop_y`, so a native cutout is a pure integer translation with no
scale factor and no resampling. The current proxy forces every overlay through a
0.227× scale that quantises positions.

Keep the follow-crop at **100 fps** — the proxy contract requires preserving frame
count, FPS, and the frame-index timeline, and frame-accurate labeling of escape
responses needs every frame.

### Sizing

1024×1024 HEVC at 100 fps, display quality (~6-10 Mbps), 23.3 min:

- **~1-2 GB per recording**
- **~150-300 GB for the entire 153-recording history**, versus 4.97 TB of masters
- **A 20-30× reduction**

This is larger than a naive 30 fps whole-frame proxy would be. That is the honest
cost of actually fixing the quality complaint, and it is still a rounding error
against the masters.

Add a **small 512×512 whole-frame context proxy** (~100 MB/recording) for the "where
is the fish in the arena / did the tracker follow the right animal" question. At that
zoom nobody needs detail, so it can be cheap.

### The three viewing artifacts

| Artifact | Resolution | Purpose | Tier |
|---|---|---|---|
| Context proxy | 512×512 whole-frame | Where in the arena; tracker QA | PRFS |
| **Follow-crop** | **1024×1024 native cutout** | **Labeling and review substrate** | **PRFS** |
| Analysis crop | 256×256 native | Model input pixels | NRS |

### On "users want the original resolution"

That request is ambiguous between two very different things:

1. *"I need native pixels on the fish."* — cheap, and fully solved by the follow-crop.
2. *"I need arbitrary zoom anywhere in the full 20 MP frame."* — expensive.

The current proxy fails (1), and users experiencing that failure will describe it as
(2). **Ship the follow-crop and re-ask before buying the expensive interpretation.**

Note also that (2) is partly illusory: no monitor displays 4512×4512, so "viewing the
original resolution" always means viewing a window into it. The follow-crop *is* that
window, chosen automatically.

If genuine arbitrary zoom is still required after testing, the established answer is
a **multiscale pyramid with tiled range requests** (Neuroglancer / OME-Zarr / IIIF).
Nobody scrubs at 100 fps while zoomed to native resolution, so the practical hybrid
is: play the follow-crop, and when the user pauses and zooms, fetch that single
frame's native tiles on demand. That requires masters to be online and
range-readable — which is an argument for **NRS**, not PRFS, and therefore does not
conflict with the supervisor's position at all.

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
      v  (one push per recording, ~30-45 s per 32.5 GB master over 10 GbE)
NRS   /nrs/johnson       working set: crops, analysis Zarr, model outputs
      |                  cluster reads at 100 Gb; not backed up, by design
      v
Nearline /nearline/johnson   master MP4, write-once, checksummed, cold
                             retrieved only for re-crop / reanalysis

PRFS  /groups/johnson    proxies + annotations + registry + published Zarr
                         small, irreplaceable, backed up
```

On the 10 GbE concern: a 32.5 GB master moves in **~30-45 s** at realistic 10 GbE
throughput, ~3 minutes for all four arena cameras. Once per recording. That is not
a bottleneck. It would only become one if masters were read repeatedly — which is
precisely what the proxy + crop architecture eliminates. The orchestration change is
also smaller than it looks: per-stage bsub jobs and the completion-marker runner do
not change, only the store root does, and
[`recording_store_relocation_components.md`](recording_store_relocation_components.md)
already enumerates the pointers that need rewriting.

## Retention policy — including what gets deleted

Proposing deletions is what makes the retention asks credible. This is not a
rhetorical concession; these are real recoverable bytes.

**Delete now:**
- `.bak` sidecar MP4s (e.g. `*_update_timing.csv`-adjacent `.mp4.bak`, ~43 GB seen
  in one recording alone)
- `__h5_context_quarantine` directories for recordings already superseded
- Failed/misconfigured recordings that QC marked unusable

**Re-encode:**
- Crop videos, from 32 Mbps to a measured quality target (see below)

**Retain forever, Nearline:**
- Any master backing a publication, a figure, or a training set

**Retain 2 years, then review:**
- Everything else

## The measurement that buys the biggest multiplier

Before the budget conversation, run this — it is directly in scope for the existing
pipeline and it is the strongest possible opening move.

1. Take ~10 representative recordings.
2. Re-encode masters at a ladder of bitrates (150 → 80 → 40 → 20 Mbps) and crops at
   a ladder of CRF values.
3. Run the existing detection and pose pipeline on original vs re-encoded.
4. Compare keypoint coordinates, mask IoU, and bout/escape metrics.
5. Adopt the lowest setting whose drift sits below the human annotation noise floor.

A static scene with one small fish is dominated by sensor noise, and sensor noise is
what eats bits. A light temporal denoise before encode plus a CRF-based target
plausibly cuts the master 2-4× and the crop 10-30× without touching a scientifically
relevant pixel. But it must be *measured against the pipeline*, not eyeballed — the
genomics precedent is the model here.

Walking into the meeting with "I already cut the projected storage bill by 4× and
here is the validation showing pose error is unchanged" is a fundamentally different
conversation from "I need more storage."

## The arguments that actually move a budget holder

1. **Concede the expensive tier immediately.** Do not ask for masters on PRFS. The
   supervisor is right that 30 TB/year of raw video does not belong on the charged,
   backed-up tier. Saying so first buys the credibility to insist on the rest.
2. **Lead with the runway, not the principle.** 6-7 weeks to a full lab volume. This
   is happening regardless of what anyone decides. Be the person who prevented a
   lab-wide outage.
3. **Show how small the backed-up ask is.** Follow-crops (~150-300 GB) + context
   proxies (~15 GB) + analysis Zarr (**358 GB measured**) + annotations and registry
   is on the order of **0.7-0.9 TB today**, growing at roughly **2 TB/year** — against
   31 TB/year for masters.
   The expensive ask is nearly nothing; the large ask is for the *cheap* tier.
   Note this is the ask *after* fixing the labeling quality complaint, not a version
   that trades user experience for budget.
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
