# Synthetic parent-intake canary

This workstation-only test package creates two tiny synthetic camera parents,
real H264 clips, and declared synthetic acquisition envelopes. It runs the
**actual pinned Citrus transfer CLI**, then Palette's unpatched import CLI,
receipt/registry readers, staging retirement, and byte-stable saved-plan replay.
It never installs software, contacts acquisition hardware, submits a cluster
job, or uses the live registry/poller. Do not run pytest on login nodes or LSF.

Prerequisites: an existing Palette `scripts/py` environment with the test
dependencies, `/usr/bin/ffmpeg` with libx264, `/usr/bin/ffprobe`, and an explicitly
supplied Citrus source directory. The four hashes in `generate_fixture.py`
must match Citrus commit `e881f5258be83231b62a00a6f9c4e5fcd69cd548` and the
recorded transfer schema. A source snapshot is sufficient; no Citrus Git
checkout or acquisition-agent connection is required. Do not replace the pins
with whatever is currently installed just to obtain a pass.

## Run a fresh canary

Run from the desired Palette checkout, outside the Codex sandbox. The importer
requires that checkout to be **clean** and verifies its exact full commit and
imported `fisheye` location. The generator must also run through that checkout's
own wrapper. It changes only fresh synthetic directories directly under `/tmp`.

```bash
scripts/py -m tests.canaries.parent_intake.generate_fixture \
  --producer-repo /absolute/path/to/pinned/citrus-source \
  --with-context --stimulus renderer-only
```

The final JSON prints the fresh `work_dir`. Read the clean code identity with
`git rev-parse HEAD`, then use those exact values:

```bash
scripts/py -m tests.canaries.parent_intake.run \
  --palette-repo /absolute/path/to/this/palette-checkout \
  --commit FULL_40_CHARACTER_COMMIT \
  --fixture /tmp/palette-citrus-encoded-transfer-PRINTED_SUFFIX \
  --full-stimulus
```

Neither `--recording-only` nor the metadata-only stimulus bypass is passed to
the real CLI in this mode. A fresh temporary registry is created by the harness;
there is deliberately no option to supply a live registry or destination.
Success retires **only the disposable delivery**, after complete hash/size
coverage and real import/registry gates. All synthetic acquisition originals
remain untouched. Replay must run no importer and preserve recording and
registry bytes. Both success and refusal reports are written in a separate
fresh `/tmp/palette-parent-intake-e2e-*` directory; failures retain their inputs.

For the malformed-H5 control, generate a **separate fresh fixture** using
`--with-context --stimulus missing-frame-metadata`, then invoke the runner with
`--full-stimulus --negative-stimulus`. Citrus transport must pass, but Palette
must refuse missing frame metadata, retain staging, produce no acknowledged
import receipt/admission, and leave the registry unchanged. Failed stimulus
candidates must remain selector-ineligible. Registry integrity is checked using
the SQLite library loaded by `scripts/py` in both positive and negative runs.

For recording-only coverage, generate with `--with-context` and omit
`--full-stimulus` from the runner. A **separate fresh** recording-only fixture
can exercise `--negative-corruption`, which deliberately appends bytes only to
its disposable delivery's opaque sidecar before invoking intake. Never reuse
the successful fixture (its delivery is empty) or a fixture already materialized
into recording parents (its files can now have shared hard links).

## Safety and preservation

The wrapper refuses broad/noncanonical locations, symlinks, special files,
shared hard links, missing/contradictory synthetic labels, and oversized input
trees before an importer or negative-control mutation. Inputs must have at
most 256 files and 64 MiB total. These are **test custody limits**, not changes
to Palette's production admission contract. `-O` and `PYTHONOPTIMIZE` are
rejected so the canary's preservation assertions cannot disappear.

The expected scientific values and existing validators are unchanged from the
original canary: two cameras, two full/crop clips per camera, three parent frame
rows, an explicit blank crop row, original timestamps/IDs, geometry and H5
custody. Full stimulus adds exact event/protocol/calibration readback and two
registered runs/four protocol-step rows. Fresh paths, H5 artifact mtimes,
publication times, producing commits, snapshot hashes and receipts may differ;
never rewrite old evidence to match a new execution.

The H5 fixture intentionally uses the existing test-owned semantic-v1 grammar
and renderer-only snapshot contract. It is not a finalized semantic-v2
execution-index, chaser-coordinate, sealed stimulus-to-acquisition mapping,
PTP/UTC qualification, authentic Orange encoder, or real scientific calibration
test. Canonical/selected bindings inside these isolated archives are fixture
checks, not production scientific acceptance. No production selection changes.

Focused portable package tests (no FFmpeg or Citrus source required):

```bash
scripts/py -m pytest tests/unit/fisheye/test_parent_intake_canary_package.py -q
```

The actual subprocess canary is explicitly invoked, not silently skipped in
ordinary CI when Citrus/FFmpeg are absent. Required CI still covers the portable
package tests and the rest of Palette. A local experimental canary before CI
finishes does not make that commit merge-ready.

Historical reports and original harness-source bytes are preserved in the
[evidence archive](../../../docs/diagnostics/parent_intake_synthetic_canary_2026-09-07/README.md).
