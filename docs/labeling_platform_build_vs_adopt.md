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

## Decisive fact established 2026-07-02

**Palette's zarr is a custom layout, not OME-NGFF** (no `multiscales`/OME axes;
confirmed by grep). Web bioimage tools ingest OME-Zarr/NGFF (or tool-native formats
like webKnossos-wrap). Therefore a **Palette-zarr → OME-NGFF conversion layer exists in
any adopt scenario**, and it re-touches the pixel/decode contract work currently in
flight (materializing pixels for the external tool requires the range/grayscale
contract to be correct first — see the silent-wrong-data slice).

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

Take it seriously; evaluate as a **bridge**, time-boxed, before committing either way.

- **Do NOT stop the Flask strangler.** It improves code retained regardless (the
  provenance-integrated review/QC surfaces webKnossos will never provide), and it is
  cheap, reversible, and incremental.
- **DO pause the heavier *productization* slices** — homegrown Okta auth, multi-user
  concurrency hardening, the deployment contract — until the webKnossos question
  resolves. These are the exact burdens an adopted tool might absorb; bolting SSO onto
  the app right before adopting a tool with SSO built in is the avoidable mistake.

### Two-week evaluation spike — decisive questions

1. **Annotation coverage.** Can it represent all three types for 2D fish frames? How
   badly do named pose keypoints degrade into skeletons?
2. **Conversion cost.** Real cost of Palette-zarr → OME-NGFF for one recording; does the
   pixel contract survive the round-trip?
3. **Round-trip (the crux).** Build the thinnest possible export→re-import for *one*
   mask back into a Palette run with lineage intact. What breaks?
4. **SSO.** Does its auth integrate with campus Okta? (Cheapest question — the admins
   answer it.)
5. **Ownership.** Will the admins actually deploy and maintain it? (A hard input, not a
   technicality.)

### Strongest plausible future

webKnossos as the multi-user annotation front-end · Palette as the provenance/pipeline
backbone · an OME-NGFF bridge between them. Sheds labeling-UI maintenance and the
auth/deployment burden; keeps the irreplaceable part. Hinges entirely on the round-trip
(#3) being tractable — which the spike exists to determine before the bet is placed.

## Impact on current web track

| Work | Disposition |
|---|---|
| Flask strangler (route-by-route) | **Continue** — code kept regardless |
| Homegrown Okta/SSO auth | **Pause** pending decision |
| Multi-user concurrency hardening | **Pause** pending decision |
| Deployment contract with admins | **Reframe** — now includes "which app are we deploying" |
| Production decision record sign-off | **Hold** — may apply to a different platform |
