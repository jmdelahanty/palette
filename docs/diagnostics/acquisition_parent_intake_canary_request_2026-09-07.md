# Draft acquisition-agent request: parent-intake canary readiness

**Draft — not sent. Request readiness information and an artifact/capture plan,
not deployment, acquisition, transfer, source deletion or production activation.**
Any capture or real-data canary needs separate user authorization and an agreed
destination. Status remains `INGEST-001` in the
[owning queue](authority_consolidation_work_queue_2026-08-25.md).

## Repository, agent and service names

User clarification, September 7: Orange is the acquisition repository, Citrus
is the separate stimulation-library repository, and Palette is the separate
analysis/intake repository. The acquisition agent is our current contact for
both Orange and Citrus; those repository names do not imply separate AI
sessions, machines or communication endpoints.

Use **`cluster-login1-poller`** for the cluster login-1 polling/submission
method and service previously called the "Citrus login-1 poller". It is not the
Citrus stimulation library and is not itself an AI agent. This is the preferred
descriptive name, not a claim that an SSH alias or installed service has already
been renamed. The existing executable alias `login1-citrus-poller` remains a
compatibility reference pending an explicitly authorized migration.

For the proposed command center, distinguish repository, agent-session and
service identities. Ask the acquisition agent about Orange/Citrus; report
`cluster-login1-poller` as operational infrastructure through its confirmed
owner, without inventing a separate persistent agent. Actual Citrus source,
transfer-tool, schema and provenance references keep their original identities.
See the [naming and compatibility checkpoint](parent_recording_intake_handoff_2026-09-06.md#repository-and-cluster-service-naming--september-7).

## Message to forward when approved

Palette's synthetic rolling-clip intake now passes both recording-only and
renderer-only full-stimulus tests, including exact source-file preservation,
registry admission, empty staging after acceptance, safe refusal, and replay.
We want to plan one bounded test with authentic finalized acquisition outputs.
Please provide the following information without changing installed software,
scientific defaults, `cluster-login1-poller`, the workstation staging-marker
poller, source custody, or production selectors.

1. **Installed and candidate versions.** Report exact Orange, Citrus and shared
   contract commits/build identities; dirty state or packaged-source hashes;
   active transfer version/entry point; actual executable/script locations; and
   relevant required-check results. Distinguish installed versions from proposed
   ones. Our synthetic transfer used Citrus
   `e881f5258be83231b62a00a6f9c4e5fcd69cd548` and shared contracts
   `6cb56b4deae6a9a56bf4b3a3186168c15e91f542`; these are compatibility reference
   pins, not assertions that those revisions are installed or instructions to
   deploy them. Report differences before proposing a test.
2. **A representative recording or capture proposal.** Identify an existing
   short, finalized two-camera session with genuine rolling-clip rollover and
   full/crop outputs. Include camera serials, session UUID, effective recording
   recipe/settings, clip/frame counts and estimated bytes. Include a naturally
   occurring blank crop row if available; never manufacture one in real data.
   Prefer renderer-only stimulus for the first test, matching the bounded
   synthetic coverage. If no suitable recording exists, propose duration,
   resources and exact capture procedure for approval. Do not change normal
   rollover or scientific settings just to make the recording shorter.
3. **The complete finalized bundle as emitted.** List locations and sizes for
   `recording_session.json`, every full/crop clip and its metadata/finalization
   evidence, per-parent H5/protocol context, original recording snapshot,
   geometry contract and all referenced assets, PTP/synchronization records,
   and any other emitted sidecars/control files. Identify exact H5 ownership
   and selected active-camera calibration for each parent. Do not rename,
   duplicate or synthesize H5 evidence to force a per-camera association.
   Report the finalized snapshot/transfer marker and digests if an authorized
   delivery already exists; otherwise report them as not yet produced.
4. **Frame and timing semantics.** Describe the emitted domains/units and
   full/crop/parent correspondence for recording IDs, camera IDs, frame IDs,
   timestamps and stimulus events. Distinguish measured PTP/UTC qualification
   from unavailable or disabled synchronization. Include the actual protocol
   snapshot schema and execution-index/finalization state. Do not equate a
   triggering-camera ID with a zero-based acquisition index or invent a sealed
   mapping. A future chaser test must separately supply its genuine coordinate,
   row-identity and source/held-target acquisition-mapping contracts.
5. **Custody and proposed destination.** Identify the acquisition-source owner,
   exact source root, source retention conditions, any already-active transfer
   job, and the proposed isolated delivery location. Keep acquisition originals
   intact. Successful Palette intake retires its disposable staging delivery
   only after complete destination-byte coverage and import/registry acceptance;
   that is not an acquisition-machine release acknowledgment. Do not send real
   inputs to the synthetic harness or the live marker poller for this canary.

Please respond with exact observed values and explicit missing items. Unknown
information should remain unknown; do not repair historical artifacts or
generate substitute evidence. No extra human scientific-review gate is being
invented here: existing stage-specific validators and supplier contracts apply.

## Palette-side plan after an explicit go-ahead

Choose the exact CI-green Palette commit and authorized isolated destination,
run directory and registry. Record the producing code and input identities.
For a cluster canary, use the existing commit-pinned deployment helper and pass
its absolute `--palette-repo` path/full commit; do not fast-forward or switch the
shared `/groups` checkout or install/change a poller. Never run test suites on
login nodes or use LSF as a substitute for workstation pytest validation.

Before execution, independently validate the finalized transfer and recording
contracts. Preserve all genuine source bytes and reject failed/incomplete or
wrong-source evidence without admitting it. The real-data plan must use the
maintained intake workflow with explicit destinations and registry, not the
tiny synthetic test wrapper with its fixed three-frame expectations.

Check parent identity and all clip/crop/frame mappings, original clocks and
validity, H5/protocol/stimulus bindings, geometry asset loadback, exact recording
receipts, isolated registry readback, and Palette-runtime SQLite integrity.
Confirm all delivered files reach their declared recording destinations before
staging retirement, and that saved-plan replay is byte-stable. Failures retain
staging and remain visible; report computation/presentation/deliverable status
according to the selected workflow without manufacturing acceptance.

No production registry, cohort, selector, authority, publication or live poller
is activated by this plan. Evaluate the resulting evidence before separately
proposing rollout or acquisition-source release. The synthetic result does not
already establish authentic hardware timing, semantic-v2 execution-index or
chaser-coordinate acceptance.
