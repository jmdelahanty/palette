# Container Packaging and Distribution Design
<!-- contract-meta
status: active-design
last_updated: 2026-07-12
-->

## Decision

Palette will begin container deployment with a CPU-only analytics application
image suitable for FileGlancer and direct Apptainer execution. Container
recipes remain in the Palette repository so code, package metadata, tests,
launch contracts, and image tags can change atomically at one Palette commit.

The initial distribution target is the maintainer's personal GitHub Container
Registry namespace:

```text
ghcr.io/jmdelahanty/palette-analytics:<immutable-tag-or-digest>
```

SciComp organization publishing rights are not required for this bootstrap.
Moving or mirroring a validated image into an institutional namespace remains
a later deployment decision.

## Why containers

The current `scripts/py` launcher is a development and workstation contract.
It searches the invoking user's home directory for a Conda environment named
`palette-py311`, or uses `PALETTE_PYTHON`. That is reliable for the current
maintainer but is not a portable shared-application contract.

A versioned container provides:

- installed Palette code at a known revision;
- a tested Python and native dependency environment;
- no per-user Palette checkout or Conda setup;
- an immutable application filesystem;
- direct compatibility with FileGlancer's Apptainer service runner;
- a path toward later CPU and GPU pipeline distribution.

The recording Zarrs, analytics exports, registries, caches, and report outputs
remain host data supplied through explicit bind mounts. They are not baked into
the image.

## Repository organization

Multiple container recipes may live in this repository. Separate repositories
are not required merely because images have different dependency sets.

Target layout:

```text
containers/
  README.md
  analytics/
    Dockerfile
    smoke_test.sh
  pipeline-cpu/
    Dockerfile
    smoke_test.sh
  pipeline-gpu/
    Dockerfile
    smoke_test.sh
```

Only `containers/analytics` is part of the first implementation. The CPU and
GPU pipeline directories should be added only when their runtime contracts are
ready.

One image should correspond to one materially different runtime:

- `palette-analytics`: Marimo, Plotly, Parquet access, group statistics,
  visualization contracts, and report inspection; no CUDA or model stack;
- `palette-pipeline-cpu`: future ingestion and CPU analysis workflows;
- `palette-pipeline-gpu`: future Torch/CUDA detection, segmentation, training,
  and inference workflows.

Apptainer supports multiple SCIF applications within one image, but that is not
a reason to combine a small analytics web application with a large GPU
pipeline. Multiple entry points are useful when they share substantially the
same runtime; materially different runtimes should remain separate images.

Container definitions should move to a separate repository only if ownership,
release cadence, build credentials, security policy, or cross-project reuse
becomes independent of Palette. None of those conditions currently applies.

## OCI image as the first distribution format

The first image will be built as an OCI image and published to GHCR. FileGlancer
uses `docker://` container references, and Apptainer can pull an OCI image,
convert it to SIF, and reuse the result from its per-user cache.

The source repository must not commit generated `.sif` files. Git stores the
Dockerfile, dependency locks, labels, tests, and workflow. GHCR or another
registry stores built OCI images; Apptainer stores its local converted image in
the user cache.

Every published production reference should use:

- an immutable commit-derived tag, such as `sha-<git-sha>`;
- preferably the OCI digest in the FileGlancer manifest;
- a pinned base-image digest;
- OCI labels for source repository, revision, version, and creation metadata;
- no floating `latest` reference in a production runnable.

OCI-to-SIF conversion can produce different outer SIF metadata on separate
pulls even when the OCI filesystem is fixed. The OCI digest remains the initial
content authority. Exact byte-identical SIF distribution through ORAS may be
considered later, but the inspected FileGlancer container runner currently
assumes `docker://` inputs.

## Personal GHCR bootstrap

The current Git remote is:

```text
git@github.com:jmdelahanty/palette.git
```

GitHub Actions in that repository can publish to the personal package
namespace with the workflow-provided `GITHUB_TOKEN`. The workflow needs
`contents: read` and `packages: write`; it does not need a SciComp credential or
a maintainer-created registry password.

The first workflow will be manual-only. A push to an ordinary branch must not
publish an image implicitly. Automatic publication from version tags may be
added after the build and smoke-test contract is stable.

The first public reference is expected to resemble:

```text
docker://ghcr.io/jmdelahanty/palette-analytics:sha-<git-sha>
```

Public visibility is the simplest FileGlancer bootstrap because execution
nodes can pull anonymously. A private package would require a deliberate
registry-credential distribution design.

## Packaging boundary

The runtime user should invoke a stable installed command rather than
`scripts/py` or a source-relative Marimo notebook path:

```text
palette-analytics-explorer serve ...
```

The command should start the fixed read-only Marimo application and accept the
same host, port, token, export-root, and optional registry arguments in Conda,
wheel, OCI, and Apptainer execution. Preserving that argument contract lets the
short-term runner and production container differ without changing application
behavior.

Current packaging gaps are:

- `apps/marimo/group_analytics_explorer.py` is outside the `src/` package tree
  and is not included by the current wheel;
- Marimo is present in `environment.yml` but not declared by `pyproject.toml`;
- Plotly is currently a development extra rather than a deployable application
  dependency;
- the base `palette` dependency set is broad and not yet separated into a
  minimal analytics runtime;
- the group query layer still imports selected modules from Palette analysis
  and group-statistics packages.

For the first image, installing the full CPU Palette wheel is acceptable. It is
larger than the eventual minimum but avoids prematurely splitting scientific
code. A later `palette-analytics` distribution or narrower dependency base
should be considered only after the viewer/query boundary stabilizes.

## Development and production stages

### Short-term source runner

Use the current Palette checkout and `scripts/py` to validate application
behavior. This requires the maintainer's `palette-py311` environment and is not
the shared deployment artifact.

### Intermediate installed environment

A shared read-only Conda environment containing a non-editable Palette wheel
could run the same installed command without a user checkout. This is a valid
fallback but not the preferred FileGlancer distribution.

### Production analytics image

A pinned CPU-only OCI image contains the installed command, Palette package,
Marimo, Plotly, and the required tabular/scientific dependencies. FileGlancer
runs it through Apptainer and bind-mounts explicitly selected host data.

### Future pipeline images

CPU and GPU pipeline images should reuse the packaging and provenance
conventions established by the analytics image. They are not prerequisites for
the first FileGlancer application.

## Local tooling status

At the start of the container discussion, the Palette workstation had none of
Apptainer, Singularity, Docker, Podman, or Buildah on `PATH`. It is Ubuntu
24.04.4 LTS on `x86_64`.

Apptainer was then installed by the maintainer. The following runtime smoke
completed successfully on 2026-07-12:

```bash
apptainer exec docker://alpine:latest cat /etc/alpine-release
```

This verifies that the workstation can execute Apptainer and pull/convert an
OCI image from a public registry. The earlier failed `docker://apline` command
was only a spelling error and is not an environment failure.

Docker, Podman, and Buildah are not required for the initial workflow. GitHub
Actions can build and publish the OCI image remotely; the workstation needs
Apptainer only to pull and test the resulting artifact.

## Build and validation sequence

Application behavior is a packaging gate. Complete the acceptance criteria in
[Group Analytics Marimo Application Design](group_analytics_marimo_application_design.md)
before freezing the application into an image.

1. Settle and validate the reactive, read-only Marimo dataset experience.
2. Package `palette-analytics-explorer` as an installed command.
3. Declare the deployable Marimo/Plotly analytics dependencies.
4. Add `containers/analytics/Dockerfile` with pinned inputs and provenance
   labels.
5. Add a container smoke test covering imports, CLI help, application static
   validation, and a small read-only analytics fixture.
6. Add a manually dispatched GitHub Actions workflow with `contents: read` and
   `packages: write`.
7. Build and publish `ghcr.io/jmdelahanty/palette-analytics` with an immutable
   tag and capture its digest.
8. Pull the exact digest with local Apptainer and run the smoke test against
   mounted fixture data.
9. Reference the digest from Palette's FileGlancer `runnables.yaml`.
10. Validate FileGlancer launch, token authentication, dataset switching,
   WebSockets, and cancellation.
11. Add pipeline containers only after the analytics path is stable.

## Open decisions

- Which Python/base OCI image should seed `palette-analytics`?
- Should the first image install the full CPU Palette wheel or a temporary
  analytics-only requirements projection?
- Which exact versions of Marimo, Plotly, PyArrow, NumPy, and Pandas should be
  locked?
- Should GHCR publication use a public package immediately or remain private
  for the first build?
- What small fixture can validate Parquet querying without shipping research
  data in the image?
- Where should image build attestations, SBOMs, and vulnerability results be
  retained?
- What shared export and registry paths will FileGlancer mount in production?
