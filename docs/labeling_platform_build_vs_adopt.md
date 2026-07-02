# Labeling platform: build vs. adopt (webKnossos / Roboflow evaluation)

<!-- contract-meta
status: proposed
created: 2026-07-02
owner: jeremy
audience: HPC administrators + maintainer decision
related: docs/palette_cli_narrow_waist_design.md,
         docs/web_labeling_modularization_plan.md,
         docs/identity_lineage_staleness_review.md
-->

## The question

HPC admins have asked whether an existing web labeling platform (webKnossos named;
Roboflow as a commercial comparator) could serve as a universal labeler for bounding
boxes, keypoints, and segmentation masks — instead of maintaining Palette's bespoke
labeling web app for campus multi-user deployment.

## Reframe: annotation surface vs. data-management backbone

Palette does two separable things:

1. **Annotation** — the labeling/review UI. Replaceable.
2. **Data management** — the registry, run-versioned lineage, provenance, completion
   markers, pipeline integration. Domain-specific research infrastructure. **No
   off-the-shelf tool does this, and it is the genuinely valuable part.**

Every candidate tool (webKnossos, Roboflow, CVAT, Label Studio) does #1, not #2. So the
decision is **never** "replace Palette." It is: *can an external tool be the annotation
surface while Palette remains the source-of-truth backbone, joined by an import/export
bridge?* Any evaluation that loses this frame produces a bad decision in either
direction.

Corollary to bring to the admins: the reason no product does "zarr labeling + data
management for animal models" is that the data-management half is bespoke research
infrastructure — the part correctly built in-house. What can plausibly be *stopped* is
building the annotation UI.

## Decisive facts established 2026-07-02

1. **The campus already runs webKnossos extensively** for a harder problem class, with
   users and workflows in place. This answers spike questions 4 (SSO — almost certainly
   already wired to campus Okta) and 5 (admin willingness to deploy/maintain — already
   done). **The two largest non-technical adopt risks are therefore zero.** Posture
   shifts from "evaluate build vs adopt" to **"adopt-leaning; the remaining question is
   the technical bridge."** The entire Palette-app *productization* track (homegrown
   auth, multi-user hardening, self-hosted deployment) can likely be retired rather than
   built.
2. **Palette's zarr is a custom layout, not OME-NGFF** (no `multiscales`/OME axes;
   confirmed by grep). webKnossos ingests OME-Zarr/NGFF or its own wkw. Therefore a
   **Palette-zarr → OME-NGFF conversion/streaming layer exists in any adopt scenario**,
   and it re-touches the pixel/decode contract work in flight (presenting pixels to
   webKnossos requires the range/grayscale contract to be correct first — see the
   silent-wrong-data slice).

**What is NOT solved by campus adoption:** the round-trip (annotations back into Palette
runs with lineage/provenance) and the named-pose-keypoint mismatch. These are unchanged
and remain the real work.

## webKnossos — fit assessment

Confidence: directional. Exact 2026 capabilities (keypoint model, zarr flavor, SSO
options) require hands-on verification, not assertion.

### Likely strong fit
- **Correct tool class.** Purpose-built to annotate data too large for the browser
  (streamed, chunked, multiscale) — the 20MP problem, solved at connectomics scale.
  Palette's flat-ROI-cache is a partial reinvention of this.
- **Segmentation masks.** Multi-label volumetric/2D mask painting is core competence,
  likely better than anything built in-house.
- **Multi-user + deployment, from the admin's chair.** A maintained open-source project
  (scalable minds; active; documented) with real team/task management. Admins typically
  prefer deploying a community tool over a bespoke researcher app. SSO likely aligns
  with campus Okta far more cheaply than adding OIDC to the Palette app.

### Real mismatches
- **Custom zarr, not OME-NGFF** (above) — conversion layer required regardless.
- **Named pose keypoints are a semantic mismatch.** webKnossos skeletons are
  node/edge trees (built for neuron tracing). Palette pose is *named landmarks against
  a schema* (left_eye, swim_bladder, …). Forcible but lossy; the weakest-fit of the
  three annotation types. Bbox and masks map cleanly; keypoints do not.
- **Round-trip is the hard part — and it is where Palette's value lives.** Getting
  annotations back *in* as a proper Palette run (lineage, provenance, completion
  markers, `source_crop_row_ids`) is a bridge owned and maintained in perpetuity. The
  annotation is easy; re-attaching it to the provenance model is not.
- **2D+time video is not its sweet spot** (optimized for 3D volumes). Works, not native.

## Roboflow — assessment

Commercial; does bbox/keypoint/segmentation well; strong ML dataset tooling. But: no
zarr concept, closed-source (cannot be extended to Palette's data model), and hosted
SaaS raises data-governance questions for campus/HHMI data. Useful as a
capability comparator, not a serious adopt candidate given the zarr and
extensibility constraints.

## Recommendation

Adopt-leaning. The decision is largely made toward webKnossos as the annotation
front-end; the work is now a focused **bridge** spike, staged by annotation type
because the three types differ sharply in difficulty.

### The bridge is three bridges, not one

| Type | webKnossos representation | Palette target | Difficulty |
|---|---|---|---|
| Segmentation masks | segmentation layer (labeled voxels) | `subject_mask_runs` | **Medium** — voxels round-trip; work is mapping segment IDs → component channels (subject_body/eye_left/…) + re-attaching lineage |
| Bounding boxes | bbox objects | `detect_runs` | **Easy** — coordinates only |
| Named pose keypoints | skeleton (node/edge trees) | `keypoint_runs` (named schema) | **Hard** — semantic mismatch; may stay in a light Palette tool |

**Start with masks.** Highest value (aligns with the active SAM3 teacher-label work),
webKnossos's strongest suit, and it proves the hard half of the round-trip. Then bbox.
Decide keypoints last — they may not move at all.

### Integration pattern

Prefer **data-in-place streaming** (webKnossos reads Palette imagery as a remote
OME-Zarr dataset) over export/upload batches, if webKnossos's remote-dataset support
fits. Either way an OME-NGFF view/adapter over Palette imagery is required.

### Focused spike — decisive questions (4 & 5 already answered by campus adoption)

1. **Existing campus export workflow (fastest, highest-value).** Ask the campus
   webKnossos power users *how they get annotations out* — the export format they
   already use IS the round-trip input for the bridge. This may shortcut most of the
   design.
2. **Mask round-trip (the crux).** Thinnest possible: one Palette recording → OME-NGFF
   → webKnossos → annotate one subject mask → export → import as a `subject_mask_run`
   with lineage/provenance intact. What breaks?
3. **Conversion + pixel contract.** Cost of Palette-zarr → OME-NGFF for one recording;
   does the range/grayscale contract survive (depends on the silent-wrong-data slice)?
4. **Keypoint degradation.** How lossy is named-landmark pose as skeletons — decide
   whether keypoints move to webKnossos or stay in a light Palette surface.

### Strongest plausible future

webKnossos as the multi-user annotation front-end · Palette as the provenance/pipeline
backbone · an OME-NGFF bridge (masks first, then bbox). Sheds labeling-UI maintenance
AND the auth/deployment/hosting burden entirely (campus already operates it). Hinges on
the mask round-trip being tractable — the spike's core.

## Impact on current web track

The residual Palette web app is **not a labeler** — it is the provenance/status/review
dashboard webKnossos does not provide (run status, dataset queues, registry-integrated
review, QC). The strangler should port *those survivors*, not the annotation editors.

| Work | Disposition |
|---|---|
| Flask strangler — admin/dashboard/status/review routes | **Continue** — these survive as the data-management surface |
| Flask strangler — session annotation editors (keypoint/mask/detect) | **Stop investing** — webKnossos replaces these; do not port them |
| Homegrown Okta/SSO auth | **Drop** — webKnossos owns campus auth |
| Multi-user concurrency hardening (labeling writes) | **Drop for annotation**; revisit only for any residual dashboard mutations |
| Self-hosted deployment contract | **Drop** — campus already hosts webKnossos |
| Production decision record sign-off | **Retire** — was gating a labeling app that won't ship as the labeler |
| OME-NGFF bridge (masks → bbox → keypoints?) | **New primary web-track work** |
