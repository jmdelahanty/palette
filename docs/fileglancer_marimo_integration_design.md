# FileGlancer and Marimo Integration Design
<!-- contract-meta
status: design-decision
last_updated: 2026-07-12
-->

## Decision

Palette's group analytics explorer will integrate with FileGlancer as a
FileGlancer **service app** declared by a Palette-owned `runnables.yaml`.
FileGlancer will authorize the shared filesystem inputs, start and stop the
service, allocate resources, and publish an authenticated URL. A fixed Palette
Marimo application will provide the reactive analytics-export selector and
visualizations.

Palette will not model its analytics exports as FileGlancer OME-Zarr viewers.
The viewer-capability interface is intentionally focused on OME-Zarr metadata
and one dataset URL, while Palette reports and group analytics operate over an
indexed export containing multiple Parquet tables and potentially many source
recordings.

The first deployment is read-only. Report generation remains a later,
explicitly authorized capability with a separately validated output location.

The image organization, personal GHCR bootstrap, packaging boundary, and
successful local Apptainer smoke are defined in
[Container Packaging and Distribution Design](container_packaging_and_distribution_design.md).

## Repository evidence

This decision follows read-only inspection on 2026-07-12 of:

- `JaneliaSciComp/fileglancer`, local commit
  `630d6512a9d159d64edc0e173cce7a0ffa82ac70` from 2026-07-08;
- `JaneliaSciComp/fg-interactive-apps`, local commit
  `ae15e9f0281334e809456ca84ea69fb081caf37e` from 2026-07-06.

No files were changed in either repository during the inspection.

FileGlancer's relevant contracts are:

- React frontend, FastAPI backend, per-user workers, and local or cluster job
  execution (`fileglancer/README.md` and `fileglancer/worker_pool.py`);
- recursive discovery of Git repositories containing `runnables.yaml`
  (`fileglancer/apps/manifest.py`);
- immutable app and code commit snapshots (`fileglancer/apps/jobs.py`);
- `job` and `service` entry points with typed parameters, resource defaults,
  Conda or Apptainer environments, and working-directory selection
  (`fileglancer/model.py`);
- file and directory parameters constrained to configured file shares and
  checked using the requesting user's permissions
  (`fileglancer/apps/command.py` and `fileglancer/filestore.py`);
- service variables `FG_HOSTNAME`, `FG_SERVICE_PORT`, and
  `FG_SERVICE_TOKEN`, plus automatic readiness detection and service URL
  publication (`fileglancer/apps/jobs.py`);
- an external “Open Service” link exposed only while the service job is
  running (`frontend/src/components/JobDetail.tsx`).

The companion `fg-interactive-apps` repository contains working service
manifests for Marimo, JupyterLab, OpenVSCode, TensorBoard, and a remote desktop.
Its Marimo example establishes the exact authentication handshake Palette can
reuse:

```text
FileGlancer mints FG_SERVICE_TOKEN
    -> Marimo receives --token-password="$FG_SERVICE_TOKEN"
    -> published URL includes ?access_token=${FG_SERVICE_TOKEN}
    -> FileGlancer shows Open Service after the port is ready
```

The source example is `fg-interactive-apps/marimo/runnables.yaml`. It launches
`marimo edit` in `ghcr.io/marimo-team/marimo:latest-sql`. That is appropriate
for a generic editable notebook environment but not for the fixed Palette
analytics application.

## System boundary

```text
FileGlancer
  authenticated user
  file-share picker and path validation
  service lifecycle and resource allocation
  app/code commit pinning
  FG_SERVICE_PORT + FG_SERVICE_TOKEN
             |
             v
Palette Marimo service
  fixed group_analytics_explorer.py application
  read-only selected export root
  optional read-only analytics registry snapshot
  reactive export_run_id selector
  capability-driven panels from available Parquet tables
             |
             v
Palette exported analytics
  immutable export manifest
  Parquet table partitions
  optional indexed report manifests
```

FileGlancer owns user identity, filesystem authorization, service lifecycle,
and app distribution. Palette owns analytics identity, scientific query and
aggregation semantics, visualization contracts, and report provenance.

The target notebook experience and its packaging acceptance criteria are
defined in
[Group Analytics Marimo Application Design](group_analytics_marimo_application_design.md).

## Dataset access model

FileGlancer does not need to understand Palette's scientific dataset schema to
give an app controlled access to datasets. Its app manifest declares typed
filesystem inputs, and FileGlancer validates and passes those paths to the
service.

| Dataset concern | FileGlancer mechanism | Palette interpretation |
| --- | --- | --- |
| Authorized dataset collection | Required `directory` parameter | One analytics export root |
| Optional catalog | Optional `file` parameter | Read-only SQLite registry or catalog snapshot |
| Individual dataset | Application choice inside the mounted root | Immutable `export_run_id` |
| Path authorization | Allowed-share containment and per-user readability checks | Upper bound on discoverable exports |
| Container visibility | Selected directory, or selected file's parent, is bind-mounted | Export manifests and Parquet tables become readable |
| Registry path references | Not inspected or mounted automatically | Must resolve beneath the explicit export root |
| Exact-dataset deep link | Future launch parameter or runnable | Preselect one verified `export_run_id` |
| Multiple roots | Multiple declared parameters or common parent required | Deferred from V1 |
| Writable output | Separate path parameter with explicit authorization | Disabled in read-only V1 |

The intended flow is:

```text
FileGlancer user selects /shared/palette_analytics
    -> FileGlancer verifies and mounts that directory
    -> Palette discovers valid export manifests beneath it
    -> Marimo displays friendly dataset choices
    -> user selects export_run_id=...
    -> Palette opens only tables verified beneath the mounted root
```

This separates two decisions that are easy to conflate:

- FileGlancer decides which collection of filesystem datasets the service may
  access.
- Marimo decides which scientific dataset within that collection is currently
  being viewed.

The registry is an index, not an authority to escape the mount boundary. If it
references `/nvme1`, `/groups`, or another root that was not explicitly
selected and mounted, the app must exclude or reject those rows.

## Palette service manifest direction

The production manifest should follow this shape:

```yaml
name: Palette Analytics Explorer
description: Explore registered Palette analytics exports in a read-only Marimo app.
requirements:
  - apptainer

runnables:
  - id: serve
    name: Explorer
    type: service
    auto_url: true
    service_url_suffix: "/?access_token=${FG_SERVICE_TOKEN}"
    container: docker://ghcr.io/<organization>/palette-analytics:<pinned-version-or-digest>
    command: >-
      marimo run
      --headless
      --host 0.0.0.0
      --port $FG_SERVICE_PORT
      --token-password="$FG_SERVICE_TOKEN"
      /opt/palette/apps/marimo/group_analytics_explorer.py
      --
    parameters:
      - flag: --export-root
        name: Analytics Export Root
        type: directory
        required: true
        exists: true
      - flag: --registry
        name: Analytics Registry Snapshot
        type: file
        required: false
        exists: true
    resources:
      cpus: 2
      memory: "16 GB"
      walltime: "08:00"
```

This is a target contract, not yet an implemented or published manifest. The
container location, version, shared paths, and FileGlancer executor remain
deployment decisions.

The Palette service differs intentionally from the generic Marimo app:

- use `marimo run`, not `marimo edit`;
- fix the application path rather than accepting an arbitrary notebook;
- use a pinned image containing Palette and its tested dependencies;
- expose only a trusted export root and optional catalog/registry input;
- select individual analytics exports inside Marimo;
- disable writes by default.

## Dataset selection contract

The FileGlancer launch form should select a shared analytics export root. An
optional registry input may provide friendly collection names, protocol
metadata, creation times, table inventories, report inventories, and immutable
export identities.

The Marimo application will query candidate active exports and present a
reactive selector. A selection resolves to one `export_run_id` and a verified
export manifest. Panels become available from the selected export's table
capabilities instead of from protocol-name conditionals alone.

The selection interface should show at least:

- collection or dataset name;
- `export_run_id`;
- protocol or cohort description when available;
- recording count;
- creation time;
- available table families;
- health or integrity status;
- available indexed reports.

Arbitrary browser-supplied filesystem paths are not dataset identities. The UI
selects immutable export identities obtained from the trusted root or registry.

## Path confinement

FileGlancer validates and mounts parameters explicitly declared as `file` or
`directory`. It does not inspect SQLite rows and therefore cannot automatically
mount or authorize paths merely referenced by the Palette registry.

Palette must independently enforce that:

1. the selected export manifest is beneath the FileGlancer-selected export
   root;
2. every resolved Parquet table and part file remains beneath that root after
   symlink-aware path resolution;
3. the manifest's `export_run_id` matches the selected registry row;
4. bound manifest hashes and collection identities match when present;
5. paths outside the selected root fail closed, even when the requesting user
   could otherwise read them.

If a registry file is provided, FileGlancer mounts its parent directory. The
export root must still be a separate directory parameter so all referenced
tables are available inside an Apptainer service.

## Storage and deployment

Current Palette defaults under `/nvme1` are workstation-local. A FileGlancer
service submitted to an LSF compute node cannot assume those paths exist.

The initial deployment must use one of these explicit models:

- place exported analytics and a read-only registry/catalog snapshot on shared
  storage mounted at stable paths on FileGlancer execution nodes; or
- use FileGlancer's local executor on the Palette data server where the current
  paths are valid.

A registry snapshot beside the exports is preferable when the authoritative
Palette registry should remain workstation-local. The snapshot needs only the
analytics collection, export, table, and report indexes required by the viewer.

The FileGlancer app should use a versioned container, preferably pinned by
immutable digest and built from a known Palette commit. The generic Marimo
image lacks `fisheye` and Palette's scientific/query dependencies. Runtime
package installation or reliance on every user's `palette-py311` environment
is not an acceptable shared-deployment contract.

## Network and authentication

FileGlancer's current automatic service URL is a direct
`http://<compute-host>:<allocated-port>` URL opened in a new browser tab. It is
not an iframe and is not reverse-proxied through FileGlancer in the inspected
repository.

Therefore deployment validation must establish:

- the user's browser can resolve and reach the execution host and allocated
  port;
- WebSocket traffic required by Marimo is allowed;
- the Marimo process enforces `FG_SERVICE_TOKEN` through `--token-password`;
- tokenized URLs are treated as bearer secrets;
- stopping the FileGlancer service job terminates Marimo cleanly.

If direct compute-node access is not supported, a later launcher must establish
an institutional WebSocket-capable proxy and write its externally usable URL
to `SERVICE_URL_PATH` instead of using `auto_url`.

## Report writing

The first web application is read-only. It may render interactive plots in
memory, query indexed report manifests, and display existing report artifacts.
It must not write to source Zarrs, analytics exports, or the Palette registry.

Future report creation should require all of the following:

- an explicit output-directory parameter or server-authorized output root;
- a deliberate user action distinct from dataset selection;
- canonical `export_run_id/report_id` placement;
- immutable report manifest and artifact writes;
- the existing report hash, source-backend, and registry-index contracts;
- collision refusal rather than overwrite.

## Rejected alternatives

### Register Palette as an OME-Zarr viewer

Rejected for group analytics. FileGlancer viewer capability matching consumes
OME-Zarr metadata and passes a single data URL to an external viewer. Palette's
analytics export is a table-first directory with multiple Parquet datasets and
cohort-level semantics.

This route may remain useful for a future per-recording Zarr viewer link, but it
does not replace the registry-backed group analytics application.

### Use the generic FileGlancer Marimo app unchanged

Rejected for publication. It exposes `marimo edit`, accepts an arbitrary
notebook/folder, uses a floating generic image, and lacks Palette's package and
scientific contracts. It remains useful as a development tool.

### Put the implementation in `fg-interactive-apps`

Rejected as the source of truth. Palette's application is coupled to Palette's
registry, export, visualization, and report contracts. Keeping the manifest
and app with Palette allows one immutable commit to identify both. A curated
pointer or catalog listing in FileGlancer infrastructure can be added later.

### Resolve derived-array references inside Palette

The FileGlancer recording explorer and editable recording workspace must use
Palette's logical readers for derived analyses instead of assuming every
coordinate array is copied into every run. In particular, run-schema-8 compact
swim-bout detector traces carry a versioned `frame_axis_contract` whose
archive-relative `authoritative_path` points to the exact source
track-kinematics `frame_indices` array. `fisheye.analysis.swim_bout_io` resolves
that path and retains schema-7 or explicitly embedded fallback compatibility,
so the FileGlancer runners require no additional mount: the source and derived
run are inside the same read-only Zarr root. The generic Zarr workspace may
display the reference metadata and source array separately, but scientific
renderers should use the logical resolver.

## Implementation sequence

1. Refactor `group_analytics_explorer.py` to accept `--export-root` and an
   optional `--registry`, then select `export_run_id` reactively in the UI.
2. Add a library-level read-only export catalog query and root-confinement
   validation; keep this logic out of Marimo cells.
3. Make panels capability-driven from available tables and provide explicit
   empty, unsupported, and unhealthy states.
4. Add a Palette-owned `runnables.yaml` using the established FileGlancer
   Marimo token and service protocol.
5. Validate the manifest against FileGlancer's `AppManifest` model and add a
   generated-command test.
6. Run a local-executor smoke test covering launch, one-click authentication,
   dataset switching, WebSockets, and clean service cancellation.
7. Build and pin a Palette analytics container.
8. Test shared-storage and LSF deployment only after network reachability and
   path visibility are confirmed.
9. Add explicit report creation later, after the read-only viewer is stable.

## Open deployment questions

- Which shared filesystem root will contain publishable analytics exports?
- Will the viewer consume the authoritative registry or a reduced read-only
  analytics registry snapshot?
- Can browsers reach FileGlancer compute-node service ports directly?
- Where will the Palette analytics container be built and hosted?
- Should the initial service use FileGlancer local execution or LSF?
- Which users or groups may view each export root?
- When report writing is enabled, which output root and registry writer owns
  publication?
