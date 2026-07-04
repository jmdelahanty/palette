# Apptainer deployment — vision & feasibility (DEFERRED)

<!-- contract-meta
status: deferred
created: 2026-07-04
owner: jeremy
picks-up-after: provenance enforcement (Slice 2) landed; repo structural cleanup further along
related: docs/interface_and_execution_strategy.md,
         docs/provenance_finalization_enforcement_design.md,
         docs/palette_cli_narrow_waist_design.md
-->

## Status: deferred, on purpose

Package Palette as an **Apptainer** image so colleagues, the cluster, and eventually the
autonomous executor run a *versioned image* instead of a live git checkout — removing the
"tell everyone to check their git status" problem. This is a real, convergent piece of the
architecture, not a side-quest — but it should wait until the repo is in a good place
(provenance enforcement landed, the API arc done, utils reorg further along). Captured now
so the reasoning isn't lost.

## Why (and why Apptainer, not Docker)

The problem: today people check out the GitHub repo and must manage git state (right
commit, not dirty, up to date) to run the pipeline. That's a reproducibility and
deployment smell. A versioned container fixes it: the code is baked in at a known commit,
so there's nothing to babysit — you run a tag.

Apptainer (not Docker) because the deployment target is shared HPC: rootless, no daemon,
scheduler-friendly (`bsub`), `--nv` GPU passthrough, host bind-mounts. Docker is a
non-starter on shared clusters (needs root/daemon).

## The three convergences (why this isn't a side-quest)

1. **It solves the provenance cluster git-state problem at the root.** The provenance work
   (`provenance_finalization_enforcement_design.md`) flags that the rsync'd `/groups`
   checkout can be dirty/ambiguous, so the captured `git_sha` on the production path may be
   a lie. A container makes the code *immutable at a known build-time SHA* — no checkout to
   drift, no dirty tree. Provenance *records* which code ran; the container *guarantees* it.
   Together: "trustworthy and deployable" instead of either alone.
2. **The `palette` CLI makes the container clean.** A container wrapping a coherent entry
   point (`apptainer run palette.sif detect <rec>`) is far better than one wrapping "cd
   into the repo and run `python -m fisheye.utils.<script>`." The narrow-waist work gave
   the container a small, real surface to expose.
3. **It's the deployment unit for autonomous execution.** The interface-strategy doc's
   "runs itself on rig transfer" executor wants to run inside a versioned image the HPC
   admins can deploy and maintain as a known quantity — which is what admins prefer.

## Feasibility (honest)

**Near-term achievable.** A `.def` from a CUDA base that installs `environment.yml` (already
curated with per-pin rationale — most of the hard containerization work is done), pip-installs
fisheye at a commit, exposes `palette`, with `--nv` and `/groups` + `/nvme1` + registry
bind-mounts. CPU/waist paths work almost immediately.

**The real wrinkles (this is where the work is — bounded, standard, not research):**
- **decord's custom build** against the ffmpeg 4.4 ABI (the submodule) must be reproduced
  inside the image.
- **TensorRT engines are GPU-arch-specific** — carry the runtime in the image, but engine
  building may still need to happen per-node or be built-and-cached (an engine built for one
  arch won't run on another). Fiddliest piece.
- **Image size** — CUDA + TensorRT + torch is multi-GB. Manageable.
- **Data vs code separation** — code+env in the image; recordings and the registry SQLite
  stay on host bind-mounts. Mostly already the model.

## Recommended phasing (when picked up)

1. **Recon/design pass** (like the other big pieces) — map the native stack (decord, ffmpeg,
   CUDA, TensorRT), the bind-mount set, the entry points, the tagging scheme.
2. **CPU-only image first** — runs `palette status`/`plan` and CPU paths. De-risks the whole
   shape before fighting the GPU stack.
3. **GPU image** — `--nv`, decord custom build, TensorRT runtime + engine-cache strategy.
4. **Versioned tags** (`palette_<date>.sif` or SHA-tagged) + a build/publish process.
5. **Wire to the autonomous executor** as its deployment substrate, when that exists.

## Preconditions before starting (the "good place")

- Provenance enforcement (Slice 2) landed — so the image's baked SHA feeds a real
  finalization gate.
- API arc complete (done) — so `palette` is the clean entry point.
- utils reorg further along (ideally the `system.py`/helper relocations and the
  `fisheye.utils` retirement) — so the image doesn't bake in the script sprawl.

## Employability note (for the maintainer)

"Containerized a GPU CV pipeline for reproducible HPC deployment" is a concrete,
resume-legible line that research-software-engineering and MLOps roles hire for directly.
Recognizing that "check your git status" is a deployment smell and containerization is the
fix is exactly the systems judgment those roles value. Worth doing partly for that reason,
when the time comes.
