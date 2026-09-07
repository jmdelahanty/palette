# Cross-protocol behavior space — a generic, updatable embedding of how fish respond

<!-- contract-meta
version: 1
status: design
last_verified: 2026-09-07
implementation: not-started
-->

**Scope:** how to build one comparable, versioned description of fish
behavior across every stimulus protocol in the store (chaser variants,
moving gratings in open and closed loop, concentric gratings, looming,
sleepyfish baselines, and protocols not yet run), such that adding a
recording updates the picture without refitting it, and such that
"states" and their transitions (the Sankey view) mean the same thing from
one experiment to the next.

**Non-goals:** this doc does not pick a clustering algorithm for bout
syllables, does not define the storage contract for the embedding
artifacts, and does not replace the per-protocol analytics
(`stimulus_response*.py`, the chaser component family,
`baseline_strategy/`). It defines the layers those feed into and the rules
that keep them comparable.

**Origin:** generalizes the goodbatbadbat strategy-state analysis
(`docs/diagnostics/strategy_state_analysis_2026-09-01.md`) whose
explorer-to-punctuated Sankey is currently a GMM fitted to one cohort of
~80 fish on one rig. That result cannot be compared to GoodCopBadCop,
batman, or redscare because each of those would get its own fit and its
own axes.

---

## 1. The problem, stated precisely

A dimensionality reduction such as PCA has two parts that get conflated:

| Part | What it is | Comparable across datasets? |
|---|---|---|
| Feature contract | the named, unit-bearing quantities going in (bout rate, IBI tail mass, tortuosity, occupancy entropy, OMR gain, escape latency) | yes, if defined identically upstream |
| Fitted basis | the scaler and loadings that come out of a fit | no: axes rotate, flip sign, and reorder with the sample |

Almost every "PCAs are not comparable" problem is a fitted-basis problem.
The remedy is to separate the two explicitly: version the feature contract,
fit the basis once on a designated reference set, freeze it, and project
new data into it. That is the pattern behind brain atlases, MoSeq reference
models, and reference-mapped single-cell embeddings. Refitting becomes a
deliberate, versioned event with a drift check, not something that happens
every time a recording lands.

The second problem is that not every feature is meaningful in every
protocol. "Distance to the chaser" has no value in a grating experiment.
"OMR gain" has no value in a chaser experiment. So one flat space is wrong
too. Section 3 defines the hierarchy that resolves this.

The third problem is batch effects. Rig, arena, camera, frame rate, and
lighting all shift feature distributions. Left alone, the leading axes of
any pooled embedding are rig axes, and any cross-protocol "state" is
partly a rig label. Section 5 covers harmonization and the test that
catches this.

## 2. Stimulus ontology: how to think about "all kinds of experiments"

The field's working answer (Marques et al. 2018; Johnson et al. 2020;
Stytra, Štih et al. 2019) is to describe stimuli by their **geometry and
feedback structure**, not by the experiment name. Two experiments with
different names but the same geometry share a response-feature contract.
Two experiments with the same name but different geometry do not.

Proposed stimulus classes for this store:

| Class | Geometry | Feedback | Store examples | Canonical stimulus frame |
|---|---|---|---|---|
| `point_object` | a localized object with position and velocity | open loop (parked dot) or closed loop (pursuit) | goodbatbadbat, batman, GoodCopBadCop, redscare, prey-like dots | fish-centred: distance, bearing, approach velocity of the object |
| `flow_field` | a whole-field translation | open loop (fixed grating) or closed loop (velocity gain) | moving gratings, OMR | direction-centred: heading relative to flow, flow speed |
| `radial_field` | expansion or contraction about a point | open loop | concentric gratings, looming | centre-centred: angular size, expansion rate, distance to focus |
| `luminance` | global intensity change | open loop | dark flash, phototaxis (not yet run) | none (scalar) |
| `none` | no stimulus | n/a | sleepyfish, every pre epoch | none |

Each class carries three things:

1. **A stimulus frame.** The coordinate system in which fish behavior is
   expressed relative to the stimulus. For `point_object` it is the
   fish-to-object vector. For `flow_field` it is heading relative to flow
   direction. This is the part that makes "response" well defined.
2. **A response-feature contract.** The stimulus-relative features that
   exist only within that class. Twin excess and escape latency for
   `point_object`; OMR gain, turn bias, and bout-to-flow alignment for
   `flow_field`; response latency versus angular-size threshold for
   `radial_field`.
3. **An event alignment.** What "onset" means. Realized chase-motion start
   for chasers, grating motion start for OMR, threshold angular size for
   looming.

Closed loop is a modifier on the class, not a class of its own. It changes
what the response features mean (gain and adaptation become primary; see
Portugues & Engert 2011, Ahrens et al. 2012) but not the stimulus frame.

The experiment-level differences that remain after classification (dot
colour, arena side, gap schedule, grating spatial frequency) are
**protocol parameters**, recorded as covariates, never used to define a
space. GoodCopBadCop's red-versus-blue confound and the CRA quadrant
side confound are both examples of a protocol parameter that was silently
doing the work of a stimulus class.

## 3. Three levels of space, not one

```
L0  locomotor atlas            stimulus-blind features        one space, all protocols, all rigs
L1  class response space       stimulus-relative features     one space per stimulus class
L2  experiment residue         protocol-specific features     no pooling, no shared basis
```

### L0: the locomotor atlas

Stimulus-blind features describe the fish regardless of what was
displayed: bout rate, IBI distribution and long-pause tail, bout duration,
path length, peak speed, net displacement, tortuosity, occupancy entropy,
wall preference, progression-versus-dwelling. These are exactly the
`DECODER_FEATURE_COLUMNS` of the strategy-state module plus the
`baseline_strategy/` phenotype dimensions.

They are meaningful in **every epoch of every protocol**, including every
pre epoch. That makes the union of pre epochs across the store the natural
reference set: it is large, it is stimulus-free by construction, and every
future protocol contributes to it.

L0 is the only level at which cross-protocol states and Sankey transitions
are legitimate. "Explorer" and "punctuated" become fixed regions of a
frozen space, so the same fish region means the same thing in a chaser
post epoch and a grating post epoch.

### L1: class response spaces

Stimulus-relative features are defined by the class's stimulus frame and
are comparable only within the class. One frozen basis per class, fitted
on the class's pooled recordings across all protocols and rigs that share
it. `point_object` pools every chaser variant. `flow_field` pools every
grating variant, open and closed loop, with loop mode as a covariate.

Within-fish baseline normalization is mandatory here: response features
are expressed relative to that fish's own pre epoch (or its virtual twin,
for `point_object`), which removes fish- and rig-level offsets before any
fitting.

### L2: experiment residue

Whatever one implementation adds that nothing else has. Never pooled,
never given a shared basis. If a feature starts appearing in a second
protocol, it is promoted to L1 by adding it to that class's contract with
a version bump.

### The cross-class bridge (optional, later)

There is one further abstraction people use to compare responses **across**
classes: the peri-event response profile. For any class with an event
alignment, describe the response by a small, class-agnostic set of
quantities: onset latency, peak magnitude in the fish's own units, decay
time constant, and habituation slope across repeated events (Randlett et
al. 2019 for dark-flash habituation is the template). This gives a low-
dimensional "how fast, how much, how persistent, how much it wears off"
description that a looming escape and a chaser escape can share. It is
worth building only after L0 and at least two L1 spaces exist.

## 4. The frozen-basis protocol

For each of L0 and each L1 class:

1. **Reference set.** A named, immutable list of recording IDs and epochs,
   stratified so no single cohort dominates (equal per-cohort weights or a
   capped per-cohort sample). Recorded with the export digests of its
   inputs.
2. **Harmonization step** (section 5) applied before fitting.
3. **Fit.** Standardizer plus linear basis. PCA by default. A supervised
   linear alternative (LDA on a designed contrast) is acceptable if the
   contrast is declared and its leakage test passes.
4. **Freeze.** Persist scaler, loadings, explained variance, feature
   contract version, harmonization parameters, reference-set digest, and
   fit code digest as one versioned artifact (`basis_v<N>`).
5. **Project.** Every recording, past and future, is projected into
   `basis_v<N>`. Projection is pure and cheap. This is the "summarily
   updated" operation.
6. **Refit only on a version bump.** Triggers: a feature-contract change,
   a new rig, a new stimulus class contributing to L0, or a drift check
   failing. A refit produces `basis_v<N+1>`; both stay readable, and every
   downstream result names the basis it used.
7. **Drift check between versions.** Procrustes alignment or principal
   subspace angles between old and new loadings on the shared reference
   recordings. Report it; do not silently accept a rotated space.

What not to use for the basis: UMAP and t-SNE. They are non-parametric,
lack a principled out-of-sample projection, and their layouts are not
stable across fits. They are fine for looking at a frozen linear space
afterwards.

## 5. Harmonization and the tests that keep you honest

Nuisance sources known in this store: rig, arena index, camera, fps
(the clipped-vs-full-rate split), lighting, arena geometry, colour-range
tag era, and cohort date. Three practices, applied in order:

1. **Within-fish baseline normalization** for L1 (already the strategy-
   state and twin-null pattern). Pre epoch or virtual twin as the fish's
   own reference.
2. **Nuisance residualization** for L0. Regress each feature on the
   declared nuisance covariates using pre epochs only, keep residuals, or
   z-score per rig against that rig's pooled pre-epoch distribution.
   Which covariates are residualized is part of the basis version.
3. **The leakage test** (acceptance criterion for any basis). Using pre
   epochs only, train a leave-recording-out classifier to predict rig,
   then protocol, from the embedding. Near-chance performance passes.
   Above-chance means the space carries batch structure and the basis is
   rejected. This is the cross-protocol analogue of the leave-recording-out
   decoder already in use, and it should run in CI-style whenever a basis
   is refit.

Validation for any claim built on the space moves from
leave-recording-out to **leave-protocol-out**: a state or transition that
only appears when its own protocol is in the fit is a protocol artifact.

## 6. Epoch alignment across implementations

Protocols with the same class differ in step layout. Comparison happens at
the level of **semantic epoch role**, not step ID. The composite semantic
epoch export that landed on main (PR #158) already does this for chasers;
the same role vocabulary (baseline, stimulus, probe, recovery, with trial
indices for repeated events and a continuous flag for closed-loop
episodes) should be the only thing L0 and L1 code ever reads. Step IDs and
protocol-specific timings stay in the per-protocol layer.

## 7. The bout-vocabulary path

The dominant approach in fish and rodent behavior works one level below
per-epoch summary statistics. Individual bouts are clustered into a fixed
vocabulary of bout types (Marques et al. 2018 for larval zebrafish;
Mearns et al. 2020 for hunting; MoSeq, Wiltschko et al. 2015; B-SOiD,
Hsu and Yttri 2021; MotionMapper, Berman et al. 2014). Any epoch of any
protocol is then a usage histogram over the same syllables, transitions
between syllables give Markov structure, and condition-to-condition flow is
the Sankey view by construction.

This is the same frozen-basis idea applied at the bout level, and it is
the natural successor to L0 once L0 is stable. Centroid bout kinematics
are enough to start; the `subject_shape` midline data extends it to
tail-shape syllables later. It is deliberately **not** step one, because
the syllable vocabulary inherits every batch-effect problem of section 5
and is harder to audit than a linear space on named features.

### 7.1 Two candidate vocabularies: Megabouts versus keypoint-MoSeq

Both tools are frozen bases in the sense of section 4. They differ in who
fits the basis and on what.

| | Megabouts (Orger lab) | keypoint-MoSeq (Datta lab) |
|---|---|---|
| Unit | segmented swim bout | HMM state over keypoint trajectories |
| Basis origin | pretrained classifier on the Marques et al. 2018 bout map (13 larval zebrafish bout types) | unsupervised AR-HMM fitted by us on our own reference set |
| Supervision | supervised, fixed vocabulary | unsupervised, vocabulary discovered from data |
| Cross-lab comparability | yes, by construction | no, our vocabulary only |
| Extensible | no | yes, refit is a version bump |
| Inputs | tail angle and/or centroid trajectory; freely swimming or head-restrained | egocentrically aligned keypoints, PCA-preprocessed, explicit keypoint noise model |
| Timescale assumption | discrete bouts separated by pauses (fish-native) | continuous motion with soft transitions (rodent-native) |
| Batch-effect handling | inherited from the training set; our rig is out-of-domain until checked | our responsibility: reference set, drift checks, leakage test |
| Status in Palette | first producer of `analysis/bout_classification_runs/` (`docs/bout_classification_runs_contract.md`) | none |

**Where each fits in this design.** Both sit at L0. Neither touches the
stimulus ontology, the L1 class response spaces, semantic-epoch alignment,
or leave-protocol-out validation; those layers wrap whichever vocabulary
is chosen. A syllable vocabulary is a feature contract, not the design.

**Recommendation: Megabouts first.** It is fish-native, already has a
landing contract in the repo, and gives a vocabulary comparable across
every protocol in the store and across labs without our owning a
reference set. Two checks gate its use as the atlas:

1. **Domain check.** Verified against the paper (Jouary et al., bioRxiv
   2024.09.14.613078, eLife reviewed preprint 107859) on 2026-09-07:
   - Training data: 1.95 million bouts from 108 larvae at **5 to 7 dpf**,
     recorded at 700 fps. No juvenile or adult fish.
   - Supported input range: 20 to 700 Hz per the README; the paper's
     "tail+trajectory" configuration is described as 100 to 2000 fps.
   - Accuracy versus frame rate (downsampling plus tail-masking
     augmentation): balanced accuracy 89.1% at 700 fps with full
     tracking; the paper states performance **plateaus at around
     120 fps**; trajectory-only at 60 fps gives 71.2% (chance is 7.7%).
   - eLife assessment: significance "useful", evidence "incomplete";
     reviewers flag that generalization across datasets, noise
     sensitivity, and the sources of misclassification are unclear.

   Our 100 fps sits at the floor of the tail+trajectory range and just
   below the reported plateau, so tail-inclusive classification should be
   usable but not at headline accuracy, and the reviewer concern about
   cross-dataset generalization is exactly our situation (different rig,
   possibly different age). At 100 fps a fast C-start lasts roughly one
   frame, so short- versus long-latency C-start and O-bend discrimination
   will rest on trajectory rather than tail shape. Acceptance: the
   per-type label distribution on a sample of hand-vetted bouts matches
   expectation, and the confidence distribution does not collapse on the
   fast-escape classes. Fish age relative to the 5 to 7 dpf training
   cohort must be recorded as a covariate, and any cohort outside it is
   out-of-domain until checked.
2. **Leakage test** (section 5) applied to per-epoch syllable-usage
   histograms on pre epochs: rig and protocol must not be decodable.

**When keypoint-MoSeq earns its cost.** Only if Megabouts labels are
poorly matched to what the data show after the domain check, or if
tail-shape syllables from the `subject_shape` midlines are wanted that
the fixed thirteen cannot express. Fitting it means owning everything in
section 4 for a non-linear model, and its rodent-shaped timescale prior
means it will spend capacity rediscovering bout boundaries that the
segmentation already provides. If tried, fit it on pre-segmented bouts
rather than raw continuous trajectories.

**What neither replaces.** The linear L0 atlas on named features (phase 1)
remains the auditable first space. A syllable histogram is a second,
richer feature contract that feeds the same frozen-basis and leakage
machinery once the linear atlas has proven the harmonization works.

## 8. Phased plan

Each phase has an acceptance criterion; none requires the previous
diagnostic branches to merge first, though phase 2 benefits from them.

**Phase 0 — feature contracts.** Promote the stimulus-blind feature list
to a versioned contract in the validated-behavior export, and write one
response-feature contract per stimulus class in section 2, starting with
`point_object` (exists in pieces) and `flow_field` (exists in
`stimulus_response_omr.py`). Acceptance: every feature has a name, unit,
source module, and class tag; nothing reads step IDs.

**Phase 1 — L0 atlas v1.** Reference set = stratified pre epochs across
all protocols and rigs currently in the store. Residualize rig and fps.
Fit, freeze, project. Acceptance: leakage test at chance for rig and
protocol; explained-variance and loadings reported; the strategy-state
explorer/punctuated axis is recoverable as a direction in the space.

**Phase 2 — cross-protocol replication.** Re-express the goodbatbadbat
Sankey in L0 v1 and run the same transition analysis on GoodCopBadCop,
batman, and redscare. Acceptance: a leave-protocol-out statement of
whether the explorer-to-punctuated conversion replicates, and whether the
earlier single-cohort states survive the shared basis. This is the real
scientific payoff and also the test of whether the earlier cohort result
was a rig artifact.

**Phase 3 — L1 spaces.** `point_object` v1 pooled across chaser variants
with the twin-null baseline; `flow_field` v1 pooled across grating
variants with loop mode as covariate. Acceptance: leakage test per class;
within-fish normalization enforced by the contract.

**Phase 4 — bout vocabulary.** Section 7, Megabouts first via the existing
`bout_classification_runs` contract, gated by the 100 fps domain check and
the L0 leakage test reused on syllable-usage histograms. keypoint-MoSeq
only on the triggers in section 7.1.

## 9. Decisions recorded

- **Linear frozen basis over non-parametric embeddings** for anything that
  must be comparable or updatable. Revisit if a parametric embedding with
  a stable out-of-sample projection becomes worth its audit cost.
- **Stimulus class by geometry and feedback, not by experiment name.**
  New protocols must declare a class before they get response analytics.
- **Closed loop is a modifier, not a class.**
- **Cross-protocol states live only in L0.** No Sankey across protocols is
  drawn from an L1 or L2 space.
- **Leakage test is the acceptance gate** for every basis version.
- **Reference sets are immutable and digest-bound**, following the
  validated-behavior receipt conventions.

## 10. Open questions

- Whether the store's earliest recordings (colour-range tag era, /nvme1
  survivors) belong in a reference set at all, or only get projected.
- Equal-cohort weighting versus capped sampling for the reference set;
  the choice changes what the leading axes mean.
- Whether L0 should include wall-relative features (arena geometry differs
  by rig and these are the most rig-sensitive features in the current
  list).
- How to represent closed-loop gain adaptation in the `flow_field`
  contract so that it is comparable to open-loop OMR gain.
- Whether the peri-event bridge (section 3) is worth building before a
  second L1 class exists.
- Megabouts' per-class accuracy at 100 fps for the fast-escape classes is
  not reported in the paper; it needs measuring on our hand-vetted bouts.
  Cohort ages relative to the 5 to 7 dpf training data still need
  tabulating from the registry.

## References (starting points, not a survey)

- Marques, Lackner, Félix, Orger 2018, *Current Biology*: structure of the
  zebrafish locomotor repertoire across many stimulus classes.
- Johnson et al. 2020, *Current Biology* (Engert lab): probabilistic models
  of larval zebrafish behavior on many scales.
- Mearns, Donovan, Fernandes, Semmelhack, Baier 2020, *Current Biology*:
  deconstructing hunting behavior into bout types.
- Wiltschko et al. 2015, *Neuron*: MoSeq. Hsu and Yttri 2021, *Nature
  Communications*: B-SOiD. Berman et al. 2014, *J. R. Soc. Interface*:
  MotionMapper.
- Portugues and Engert 2011; Ahrens et al. 2012, *Nature*: closed-loop OMR
  gain adaptation.
- Randlett et al. 2019, *Current Biology*: dark-flash habituation as a
  peri-event response profile.
- Štih, Petrucco, Kist, Portugues 2019, *PLoS Computational Biology*:
  Stytra, a stimulus-class-oriented protocol framework.
- Jouary, Silva, Laborde, Mata, Marques, Collins, Peterson, Machens,
  Orger 2024, bioRxiv 10.1101/2024.09.14.613078 (eLife reviewed preprint
  107859): Megabouts, pretrained transformer bout classifier onto the
  Marques 2018 bout map; docs at megabouts.ai, code at
  github.com/orger-lab/megabouts.
- Weinreb et al. 2024, *Nature Methods*: keypoint-MoSeq.
