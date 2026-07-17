# GLM-HMM sample-vs-avoid state analysis — agent prompt reference

Reference for handing an agent the "does the latent behavioral-state
structure differ between learners and non-learners" analysis on the
GoodCopBadCop chaser data (or any input-output behavioral dataset with a
policy hypothesis). It captures both the prompt to hand the agent and the
reasoning behind its shape, so the choices are reviewable rather than
buried in a notebook.

## Prerequisites and current state (2026-07-16)

This analysis is **not ready to run.** A sweep of the repo's statistical
machinery and the 2026-07-14 GoodCopBadCop diagnostics shows the cohort
comparison this doc is built around does not yet exist, and three things must
be settled before a GLM-HMM is worth fitting.

1. **The learner/non-learner split does not exist and is not defined.** Every
   quantitative result in the repo is a within-fish pre→post paired contrast
   on one pooled cohort (32 fish, 8 sessions). There is no learner/non-learner
   — or any between-group — split in the data, and the stats engine
   (`src/fisheye/group_statistics/`) implements only paired within-recording
   epoch contrasts (bootstrap CI, Wilcoxon, sign-flip permutation, BH-FDR),
   with no two-sample test. The split is a fill-in-the-blank in this doc's own
   skeleton (`[learner/non-learner labels AND how derived]`). It must be
   *defined* before it can be tested.

2. **The natural definition is circular and confounded.** The obvious way to
   label a "learner" is by post-training avoidance of the red object — but
   that is exactly the emission variable the GLM-HMM compares (the circularity
   this doc already flags), and it is confounded with colour and position:
   aggressive = red `#ff0000` at fixed positions in 32/32 recordings, inert =
   blue at different fixed positions in 32/32
   (`docs/diagnostics/goodcopbadcop_cohort_results_2026-07-14.md:135-142`). A
   split derived from the behavior inherits the design confound. Counter-
   balancing colour/position, or deriving the label from something independent
   of the emissions, is a hard prerequisite.

3. **The avoidance is both innate and (marginally) learned — the hypothesis is
   live, but the baseline is large.** In the corrected 8–16 mm band analysis,
   red-object avoidance steering is already significant *before* any chase
   (pre +0.267 mm, p=0.006) AND increases with training (pre→post +0.422,
   parametric p=0.050, session-RE p=0.041, Wilcoxon p=0.14), while the inert
   control stays flat (p=0.36) and post aggressive-vs-inert is p=0.0023
   (`...goodcopbadcop_cohort_results_2026-07-14.md:104-133`). So the group ×
   epoch handoff this doc hypothesizes has real, if marginal, support — it is
   not a phantom. But it sits on a significant innate baseline, the learned
   increment is exactly at the edge of significance (and fails the Wilcoxon),
   and the increase cannot be attributed to red-ness specifically because the
   inert control differs in both colour and position. The model must separate
   a marginal learned change from a large pre-existing bias.

**Infrastructure gap.** Neither the hierarchical partial-pooling GLM-HMM nor
the group-label permutation null has any foundation in the codebase: no
pooling (clustering is aggregate-to-recording-then-pair, no mixed model), no
between-group comparison, no HMM/latent-state code, and even the bout-level
cluster bootstrap is documented but unimplemented
(`src/fisheye/analysis/chaser_bout_response.py:835`). Both the model and the
group-comparison machinery would be built from scratch.

**Sequencing.** The defensible phase-one result already exists and is
within-fish pre→post (escape 8.5×, p=2.2e-04; thigmotaxis +0.138, session-RE
p=0.040; the band-localized steering increase above). The GLM-HMM is
downstream of two unbuilt prerequisites — a non-circular cohort definition and
counterbalanced data — not downstream of a modeling gap. Build it when a
defensible split exists, not before.

## Why the prompt is scoped this way

An agent handed "build a behavioral state-analysis framework" will produce
something that *looks* rigorous and confirms whatever you point it at. It
has no domain knowledge about what makes the result meaningful and no stake
in the analysis being wrong. It will pick a reasonable method, fit it, find
states, and hand you a plot. The prompt's job is to supply the two things
the agent structurally lacks: the **domain constraints** that define the
states correctly, and the **adversarial scaffolding** that could disconfirm
the hypothesis.

A prompt scoped to the *claim and its disconfirmation* produces a defensible
result. A prompt scoped to the *method* produces infrastructure that then
needs its own justification. Same agent, same data — the difference is
whether you handed it the scientific decision or the toolbox.

Three domain facts the agent gets wrong by default, and which invert or
inflate the result if unstated:

1. **States are a policy, not a magnitude.** The latent state is a *mapping*
   from the robot's position to the fish's response (orient/approach/sample
   vs turn-away/avoid), NOT how much the fish is moving. Left unspecified the
   agent defines states on locomotor magnitude or bout rate, and a
   hyperactive fleeing non-learner gets labeled an "explorer," inverting the
   result. This points at the method: a **GLM-HMM** (input-output HMM, à la
   Ashwood et al. 2022), where each latent state carries its own mapping from
   inputs (robot distance, robot bearing relative to heading) to action.

2. **A "state" only means something if it persists.** "Non-learners are
   biased toward an avoidance state" is only distinct from "non-learners
   respond more to proximity" if the state carries across bouts *beyond* what
   instantaneous distance explains. That persistence is the transition term.
   So the transition structure is not gravy — it is where the hypothesis
   lives, and it must be isolated and tested, not assumed.

3. **The paradigm is learning, so the transitions are non-stationary.** A
   vanilla GLM-HMM assumes stationary transition probabilities. But the
   mechanism is a trajectory: the learner *hands off* from a sampling state
   to a persistent avoidance state as it learns; the non-learner is
   avoid-locked from the start. A stationary fit averages over that
   within-session handoff and smears the exact signature you are after. So
   epoch (pre/post conditioning) is not a data-contract footnote — it is the
   axis the claim lives on. State the hypothesis as a **group × epoch
   interaction**: do learners show a pre→post increase in avoid-state
   occupancy/persistence that non-learners do not? A static between-group
   contrast can return null *even if the learning effect is real*, because it
   averaged each learner's own before and after. Fit pre and post epochs
   separately and compare the change, or put a within-session covariate on
   the transition matrix — do not collapse the epoch.

## The model ladder (the core design choice)

Do not pose "GLM-HMM vs a static threshold" — distance is already a covariate
*inside* the GLM-HMM, so the threshold is not a fair competitor. Instead fit
a ladder and compare on cross-validated held-out log-likelihood, folds held
out **across fish**:

- **Model 0 — marginal:** bout-type base rates, no inputs, no states.
- **Model 1 — 1-state GLM:** a single policy mapping inputs to actions, no
  latent states, no transitions. This captures everything a proximity
  threshold captures, so it is the **champion to beat**, not a hygiene null.
- **Model 2 — GLM-HMM:** each latent state is its own input→action policy;
  the transition matrix is the object the hypothesis lives in.

**The primary result is the held-out xLL gap between Model 1 and Model 2.**
That gap *is* the measurement of how much bout history / state persistence
matters. If it is ~0, there is no latent structure beyond proximity-tracking
and the persistence claim fails — that is a real, reportable outcome.

This matters here specifically because the project's own history warns
against elaborate latent-state machinery: on the chase data, k-means missed
the escape effect and a plain speed threshold found it (see
`docs/diagnostics/goodcopbadcop_escape_pursuit_2026-07-14.md`). But k-means
is a *weak* comparator for two independent reasons — it is memoryless (no
transitions) and unconditioned (no inputs). The threshold beat it via
input-conditioning, which is the **GLM** contribution, not the history
contribution. The 1-state GLM absorbs that, so the ladder isolates the one
open question — does bout history add signal — instead of relitigating
k-means.

## Phase gate: is this the analysis to run now?

Before standing up hierarchical numpyro machinery, settle two facts, because
the prior that this returns a *small* primary gap is real (a speed threshold
already found the escape effect — the fancy model may just characterize what
you can already state).

- **Report bouts-per-fish first, as a go/no-go.** Ashwood-scale GLM-HMMs were
  fit on tens of thousands of trials per animal. A few hundred bouts per fish
  may cap you at two states and distance-only inputs *regardless of the
  heading gate*. If the count is small, the honest move may be to not fit the
  GLM-HMM at all and instead run the group × epoch comparison on a readout you
  already trust (next).

- **Run the cheap timing check before the expensive state model.** Bout rate
  and inter-bout interval *are* the readout of how often the fish samples, and
  bout-indexing the emission sequence throws that timing away (it treats bouts
  as evenly spaced). A non-learner may not differ in per-bout *policy* at all;
  it may differ in how *often* it acts. Check whether bout rate or IBI alone —
  and its pre→post change — separates the groups before committing to a
  latent-state story. If a scalar rate separates them, that is the simplest
  honest description; the latent-state model must then *beat* it, not replace
  it. This is the Model-1-champion logic applied to the time axis instead of
  the input axis. If you want timing inside the model, that is a semi-Markov
  formulation where dwell time is fit rather than assumed geometric.

- **Apply the "within ___ because otherwise ___" test to the persistence
  claim itself.** What decision in the paper changes based on whether
  avoidance is a persistent *state* versus proximity-tracking? If a biological
  claim genuinely turns on it, build it. If the threshold/escape result
  (`goodcopbadcop_escape_pursuit`) is already your defensible claim and this
  would *characterize its structure*, this is phase two — and running it first
  is how a result you can already state gets held hostage to a hard model that
  most likely tells you "proximity explains it."

## The input-autocorrelation trap

Because the robot moves smoothly, distance-to-robot is autocorrelated across
consecutive bouts. So even a memoryless GLM produces autocorrelated
predictions, and an HMM fit on top can misattribute that to sticky
behavioral states when it is really just the input being smooth in time.
Apparent state persistence can be a smooth-input artifact. The Model-1 GLM
(which contains the input) is the control; the GLM-HMM only earns its
transition structure by beating that GLM on held-out data **and** surviving a
block-shuffle of the bout sequence.

There is a deeper, *causal* version of the same trap. The robot pursues, so
distance and bearing are not exogenous — they are co-determined by the fish's
own last action (it avoided, so distance grew). A GLM-HMM treats inputs as
given, but here the input is partly the fish's own output fed back through a
closed loop, which manufactures both apparent coupling and apparent
persistence in a way block-shuffling emissions will not fully catch if the
input is regenerated from behavior. The robot's *commanded* trajectory is
exogenous; the *relative* geometry the fish perceives is not. The remedy is
subtler than "log the command signal and use it as the input" — the fish
responds to relative geometry, not to a command it cannot see, so the command
is not a drop-in cleaner regressor. It is an **instrument**: use it to measure
how much of the relative-geometry variance is command-driven versus
behavior-driven. Name this as a distinct hazard from the smooth-input one, and
log the command signal separately if the rig allows.

## Heading is the load-bearing fragile input

`bearing relative to heading` requires a trustworthy per-bout heading, and
orientation from masks/midline is exactly the shaky signal in this repo
(see `docs/keypoint_heading_validity_todo.md` and the snout-centerline
artifacts). If heading is noisy the policy is measured with error and states
smear. Validating heading quality is a hard gate in data-validation, not a
"units" footnote; fall back to distance-only inputs if heading is unreliable.

## Three nulls

The first two test whether states/transitions exist at all; the third tests
the group claim itself, and the doc originally omitted it. A large primary xLL
gap is fully compatible with *identical* transitions across groups, so
clearing the gate does not license reporting a group difference.

- **Input-shuffle** (permute inputs vs emissions): tests whether the GLM
  *coupling* is real.
- **Block / circular-shuffle of the bout sequence** (preserves marginal state
  occupancy, destroys temporal order): tests whether the *transition*
  structure carries information in general.
- **Group-label permutation** (permute learner/non-learner labels across fish,
  re-estimate the contrast): the null for the actual hypothesis. Choose the
  single contrast statistic *in advance* — one of dwell time, occupancy, or a
  named transition probability, in its group × epoch form — then build its
  null distribution by permutation. Pre-registering the statistic matters: a
  permutation null over a freely chosen statistic still p-hacks across the
  garden of forking paths. This is also the right tool for the small n, since
  it is distribution-free.

## Other things agents silently guess

- **Pooling.** With ~10 learners and ~10 non-learners, fitting 20 independent
  HMMs and t-testing a parameter is underpowered and courts pseudoreplication.
  Use partial pooling — a global fit with a shared prior and per-fish
  posterior (Ashwood-style global-then-individual). Note that turnkey
  hierarchical GLM-HMM is **not** first-class in dynamax/ssm, so either
  implement global-fit + per-fish MAP under a shared prior explicitly, or
  build the hierarchical model in numpyro/PyMC. Name the choice; don't let the
  agent hand-roll the weak thing silently.
- **State number by selection, not fiat.** Cross-validated xLL held out across
  fish; show the selection curve.
- **The binding power constraint is the group contrast**, not the fish count
  and not the per-fish fit. The learner/non-learner transition-probability
  comparison has ~10 effective points per arm. State that honestly.
- **Label circularity.** If learner/non-learner labels were derived from a
  behavior correlated with approach/avoidance, the group result is partly
  built-in. Require a check that the labels are independent of the
  emission/state variables, and a flag if not.
- **Output in the claim's terms** — per-group sampling-state dwell time,
  occupancy, and sample↔avoid transition probabilities with uncertainty; not
  a classifier accuracy. Template: Ashwood et al. 2022.

## Workflow

Build in stages; the first artifact is an assumptions-and-decisions log,
written *before* any modeling code, with every neuroscience/statistics choice
flagged for review and versioned. That log — not the model — is what a
committee interrogates.

1. Assumptions-and-decisions log.
2. Data load + validate, **including the heading-quality gate**, plus the
   phase gate — report bouts-per-fish and the scalar rate/IBI group ×
   epoch check — then STOP for review. If the rate check already separates
   the groups and bouts-per-fish is thin, that review is where you decide
   whether the GLM-HMM is worth building at all.
3. Models 0–1 + sanity plots.
4. Model 2 + pooling.
5. Three nulls (including group-label permutation) + group × epoch
   statistics.

Flag every assumption rather than deciding it silently.

## The prompt skeleton

```
Scientific question: As fish learn, does a learner HAND OFF from an
object-sampling state to a persistent avoidance state, while a non-learner
stays avoid-locked from the start? State this as a group x epoch
interaction: do learners show a pre->post increase in avoid-state
occupancy/persistence that non-learners do not? Two words are load-bearing.
PERSISTENT: the claim is only meaningful if the state carries across bouts
BEYOND what instantaneous proximity to the robot explains — a fish that
avoids when the robot is near is tracking distance, not in an "avoidance
state". LEARNED: the mechanism is a within-session trajectory, so a static
between-group contrast can come back null EVEN IF the learning effect is
real, because it averages each learner's own before and after. Fit pre and
post epochs separately and compare the change (or put a within-session
covariate on the transition matrix); do not collapse the epoch.

Data: [array shapes, units, one row = one bout, per-fish grouping,
pre/post-conditioning epochs, learner/non-learner labels AND how derived].
One observation = one bout; the bout sequence per fish is the emission
sequence.

Inputs to the policy: robot distance and robot bearing relative to fish
heading. HEADING IS A LOAD-BEARING, FRAGILE MEASUREMENT in this dataset
(mask/midline-derived, known orientation artifacts). Before any modeling,
validate the per-bout heading: report how it was derived, its coverage,
and an error estimate. If heading is unreliable, say so and fall back to
distance-only inputs rather than modeling on a corrupted covariate.

Before any model — phase gate: report bouts-per-fish (this may cap state
count and input dimensionality at 2 states / distance-only REGARDLESS of
the heading gate; Ashwood-scale fits used tens of thousands of trials per
animal). Then check whether bout RATE or inter-bout interval alone — and
its pre->post change — separates the groups. Bout-indexing the emission
sequence discards timing; if a scalar rate separates the groups, that is
the honest description and the latent-state model must BEAT it, not replace
it. If bouts-per-fish is small, prefer running the group x epoch comparison
on the rate/escape readout you already trust over fitting the GLM-HMM.

Model ladder — fit in this order, compare on cross-validated held-out
log-likelihood with folds held out ACROSS fish:
  0. Marginal baseline: bout-type base rates, no inputs, no states.
  1. 1-state GLM: a single logistic/multinomial policy mapping inputs
     (distance, bearing) to actions (approach vs avoid turn; hunt- vs
     escape-type bout). NO latent states, NO transitions. This is the
     model to beat — it already captures everything a proximity threshold
     captures, so it isolates the value of everything above it.
  2. GLM-HMM (use [resolved library choice, see below]): each latent
     state is its OWN input->action policy. States are defined by that
     policy, NOT by locomotor magnitude. The transition matrix is the
     object the hypothesis lives in.

The PRIMARY RESULT is the held-out xLL gap between model 1 and model 2.
That gap is the measurement of how much bout history / state persistence
matters. Report it explicitly with uncertainty. If it is ~0, there is no
latent-state structure beyond proximity-tracking and the persistence claim
fails — report that outcome plainly, do not proceed to group comparisons
as if the states were real.

Pooling: fit the GLM-HMM across all 20 fish with partial pooling — a
global fit with a shared prior and per-fish posterior parameters (the
Ashwood-style global-then-individual scheme), NOT 20 independent models.
Note: turnkey hierarchical GLM-HMM is not first-class in dynamax/ssm, so
either (a) implement global-fit + per-fish MAP under a shared prior
explicitly, or (b) build the hierarchical model in numpyro/PyMC. State
which you chose and why. Flag if the library forces a compromise.

Rigor requirements:
- State-number selection: cross-validated xLL held out across fish; show
  the curve. Do not fix the count by fiat.
- THREE nulls: the first two test whether states/transitions exist; the
  third tests the group claim (a large primary gap is compatible with
  IDENTICAL transitions across groups, so the gap does not license a group
  difference):
    * Input-shuffle (permute inputs vs emissions): is the GLM coupling real.
    * Block/circular-shuffle of the bout SEQUENCE (preserves marginal state
      occupancy, destroys temporal order): does the transition structure
      carry information in general.
    * Group-label permutation (permute learner/non-learner across fish,
      re-estimate the contrast): the null for the hypothesis itself. Pick
      ONE contrast statistic in advance (dwell time, occupancy, or a named
      transition probability, in its group x epoch form); permuting a
      freely-chosen statistic still p-hacks.
- Input-autocorrelation confound: because the robot moves smoothly,
  distance-to-robot is autocorrelated across bouts, so apparent state
  persistence can be an artifact of a smooth input rather than real
  behavioral hysteresis. The model-1 GLM (which contains the input) is
  the control for this; the GLM-HMM only earns its transition structure
  by beating that GLM on held-out data AND surviving the block-shuffle.
- Closed-loop / endogenous-input confound (distinct from the above and
  deeper): the robot pursues, so distance/bearing are co-determined by the
  fish's own last action, and block-shuffling emissions will not fully
  catch the resulting apparent persistence. The relative geometry the fish
  perceives is endogenous; the robot's COMMANDED trajectory is exogenous.
  Do NOT swap the command in as the input (the fish cannot see it) — use it
  as an instrument to measure how much relative-geometry variance is
  command- vs behavior-driven. Log the command signal separately if the rig
  allows.
- Label circularity: check that the learner/non-learner labels are
  independent of the emission/state variables being compared. If the
  labels were derived from a behavior correlated with approach/avoidance,
  flag the circularity — do not report the group difference as if clean.
- Power: the binding constraint is the 10-vs-10 GROUP contrast of
  transition probabilities, not the fish count and not the per-fish fit.
  State the effective n for that contrast and its power limits honestly.

Outputs, in the claim's own terms (template: Ashwood et al. 2022): per-
group AND per-epoch (pre/post) sampling-state dwell time, occupancy, and
sample->avoid vs avoid->sample transition probabilities, with uncertainty.
The headline number is the group x epoch CHANGE, not the static group
difference. NOT a classifier accuracy.

Workflow: build in stages. First artifact is an assumptions-and-decisions
log (before any modeling code) — every neuroscience and statistics choice
flagged for my review, versioned. Then: data load + validate (INCLUDING
heading-quality gate AND the phase gate — bouts-per-fish + scalar rate/IBI
group x epoch check) and STOP for my review. Then models 0-1 + sanity
plots. Then model 2 + pooling. Then the three nulls + group x epoch stats.
Flag every assumption rather than deciding it silently.
```

## References

- Ashwood et al. 2022, *Nature Neuroscience* — GLM-HMM behavioral states;
  the canonical method and the endpoint-figure template.
- `docs/diagnostics/goodcopbadcop_escape_pursuit_2026-07-14.md` — the
  escape-rate result a threshold found and k-means missed; the reason the
  ladder makes the GLM-HMM earn its complexity.
- `docs/diagnostics/goodcopbadcop_cohort_results_2026-07-14.md` — cohort
  n and the learner/non-learner framing.
- `docs/keypoint_heading_validity_todo.md` — heading reliability, the
  load-bearing fragile input.
- `docs/goodcopbadcop_avoidance_readout_survey.md` — companion doc: which
  avoidance *measurement axes* to use before modeling; argues the freeze-curve
  and tail-kinematics readouts are the higher-value immediate move, and the
  GLM-HMM is downstream of a non-circular split.
