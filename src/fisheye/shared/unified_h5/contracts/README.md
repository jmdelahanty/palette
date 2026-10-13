# Pinned Citrus unified-H5 contracts

These runtime resources are copied byte-for-byte from the reviewed Citrus
producer evidence. `schema.py` verifies their SHA-256 before using them; changes
require an explicit producer/consumer contract review and compatibility decision.
They are not Palette scientific-acceptance or activation schemas.

| Resource | Producer commit | SHA-256 |
| --- | --- | --- |
| `experimental_h5_core_v1.json` | `74083cee35196b4b1fc58e8bea180455a0b924ad` | `febb7516ff96f359b6784bf137682abe387b7922270e9637ebc58b369e07f9c1` |
| `experimental_h5_geometry_v1.json` | `0a9e2d490ab0772dbaa7fdda3f837693806caad0` | `4c0c707192cf1fa6ceafb2f632672972f41dc7e4c7cc5cdb0c0e3b7920a1474b` |
| `experimental_h5_identity_claims_v1.json` | `0a9e2d490ab0772dbaa7fdda3f837693806caad0` | `83ca66b336f3e3a40ebaafc869d7febd020509e238243381dfe871cbeb89507e` |
| `object_appearance_replay_dependency_manifest_v1.json` | `74083cee35196b4b1fc58e8bea180455a0b924ad` | `eccfcfcb3c6e66549b2fdfcfcf88fdc4fa9def3d6ce4a375494d011b04f2607c` |
| `unified_h5_admission_v2.json` | agent-contracts PR 52 `86638d74a9cf9b0f883683c0c695e7310b403158` | `ad0aab702e6714131db857afe8d96323999df184e0a3b357705c1acf56a90c38` |
| `unified_h5_producer_policy_d544b081.json` (frozen producer policy named by capacity preflight) | agent-contracts `f3620fe1a05611db14497035d88f8c734449ac79` | `d544b0814006360b11b02cd9d171eefbddae7cc9cc41ad57470bbd240bc83cb9` |
| `experimental_h5_core_chaser_v2.json` (core v1 with chaser state table v2) | `e437fa4752679e5a614378d824e636104c4f095a` | `cb68ab6ecea614030b8f2a3d2096e9122b1efa5156b56ed7a495b3072c70316a` |
| `experimental_h5_correspondence_tables_v2.json` | `6bd8cbc443c3af699f9f13cbc2db037d8d99d82d` | `cb3ac574c76750c23de730129c4a32371328ffb34d792b1cfab164b7201ba1ea` |
| `experimental_h5_correspondence_input_v2.schema.json` | `6bd8cbc443c3af699f9f13cbc2db037d8d99d82d` | `a33e06819e966b7273015bfffe4aa0d01f24a04fb9ffac69abb29ed5c76329a3` |
| `experimental_h5_correspondence_receipt_v2.schema.json` | `6bd8cbc443c3af699f9f13cbc2db037d8d99d82d` | `8d2a414f46f6d6263f2609c55e15add9d7a2b61ad98349cf1ac3aef598b8d4e8` |
| `experimental_h5_capacity_preflight_v1.schema.json` | `6bd8cbc443c3af699f9f13cbc2db037d8d99d82d` | `d5d7ebf5249de35ce7234ddfec3c7cb51b57ac3b6fdcb25bf75724f25223d068` |
| `unified_h5_admission_v3.json` (admission revision 3: bounding_boxes v2, per-catalog policy bindings, declared-table rule) | agent-contracts PR 73, merged via PR 52 `1a9d8469d550c0274382c054986f3fb79e712fee` | `c32282b68dc5d0057abe3b4144df27a19afbc69f4a920a64aabd09db6b8647b0` |
| `unified_h5_producer_policy_b2af7764.json` (frozen producer policy bound to `bounding_boxes_v2`) | agent-contracts `1a9d8469` `citrus-palette/producer_policies/unified_h5_admission_b2af7764.json` | `b2af7764c7f27f6295007ba3049dcf8a35d0f84c5801f8a1ff56150c7f9e7fc1` |
| `experimental_h5_core_bounding_boxes_v2.json` (core with bounding_boxes v2, one row per frame per object) | Citrus `4d9d04d0d001cee9d46b2610a87089b4dd46a8ca` (PR 8) | `55f94f83e6f43824488161b7238846b2233933be871ac8044bef04d3fa6df4f8` |
| `experimental_h5_capacity_preflight_v1_ba8333a3.schema.json` (Citrus `docs/schemas/experimental_h5_capacity_preflight_v1.schema.json`, renamed; bound to `bounding_boxes_v2`) | Citrus `4d9d04d0d001cee9d46b2610a87089b4dd46a8ca` | `ba8333a35526dc1fc655734d938b14a816fd143d2c5eb395ad17065f79718a54` |
| `unified_h5_admission_v4.json` (admission revision 4: frame identity v1, table-version-map selection, `catalog_id`) | agent-contracts PR 88 `ccbefa9f426b769232e8f3656488ea8cab299644` (unchanged at head `32a3a709`), merged `9e373166a4d4bb11bf530911297d1aabfa93cb63` | `e702df25e5164548c72fa94a683922836567245ed7b7880316ea8ead2d33a0a4` |
| `unified_h5_producer_policy_3c9866ed.json` (agent-contracts `citrus-palette/producer_policies/unified_h5_admission_3c9866ed.json`, renamed; bound to `frame_identity_v1`) | agent-contracts PR 88 `ccbefa9f`, merged `9e373166` | `3c9866eda2f78c4aa37d1b68a9f4b6245aab8415b3ce8e3412cb3cbf5ab7eb78` |
| `experimental_h5_core_frame_identity_v1.json` (core catalog `frame_identity_v1`: nine tables re-versioned, no bare `*_frame_id`) | Citrus PR 24 `526c218a26ea711f8dbad4110401f5b8fbd81b86` (same bytes at `43f57c18`), merged `53b68a7` | `4365821df9cdde35d0417fec5b656de12607c211e24232b408e87344883a8264` |
| `experimental_h5_capacity_preflight_v2.schema.json` (adds `catalog_id`; bound to `frame_identity_v1`) | Citrus PR 24 `43f57c18097896fd1aef449f475982bead4be681` (the policy-pinning commit; the file differs at `526c218`), merged `53b68a7` | `3f78bff9fd51207d035ed9a2feb32660f5232b91d3674389882a4914fbf6bbc3` |
| `experimental_h5_correspondence_receipt_v3.schema.json` (embedded core table versions no longer enumerated) | Citrus PR 24 `526c218a` (same bytes at `43f57c18`), merged `53b68a7` | `7143f9cff038b4dda1c184e5f0ca7b1b924b541074d24b25b9b0d67dab9d34b4` |
| `unified_h5_admission_v5.json` (admission revision 5: catalog `angle_v1`, stimulus angle convention per binding) | agent-contracts PR 92 head `aa337ec66a2bfae05e2635567da8528351bdbbb4` (unchanged), merged `7a2e5d2517f690190fc3ae172bb7e36a4c9a4228` | `7b1448f83b68df898e1022388581ae833151c8a83d375bb51abb82c13c99ce71` |
| `unified_h5_producer_policy_9af921ca.json` (agent-contracts `citrus-palette/producer_policies/unified_h5_admission_9af921ca.json`, renamed; bound to `angle_v1`) | agent-contracts PR 92 head `aa337ec6`, merged `7a2e5d25` | `9af921cab2aedae8af66a00600f5eee6b076c5d8218b6f0e928f93b4df3fbedd` |
| `experimental_h5_core_angle_v1.json` (core catalog `angle_v1`: moving_grating 2 -> 3, independent_motion_grid 1 -> 2, angles counter-clockwise) | Citrus PR 27 `c9d52be6dddf07da8f13607f31ae3bcf2aca9d93` `docs/schemas/` (unchanged), merged `576481f3c58140dcc0fc6bc6fadbf1456341db48` | `0c054c7d5d3c1a4f7b39f800b63e272bb43bac30b5b640993a94974e2be24f07` |
| `experimental_h5_capacity_preflight_v3.schema.json` (`catalog_id` const `angle_v1`, policy const `9af921ca…`) | Citrus PR 27 `c9d52be6` `docs/schemas/`, merged `576481f3` | `00b201da455c8f1fc021506ceaf291092f0d61b1d2d5a2689f3ba386be5c60ea` |

The geometry and presentation adapters follow that producer's
`src/logging/experimental_h5_geometry.cpp`, `stimulus_geometry_contract.cpp`, and
`presentation_mapping_contract.cpp`. Recorded geometry references are interpreted
through the closed native source namespace; captured source JSON is not rewritten.

`../vendor/object_appearance_reference.py` is the unchanged portable evaluator
from the appearance review bundle (SHA-256
`e77ead403cee979db26d7b0a453e432f0c5f08f1c2e1aa3ea99b494e5c503f30`). Keeping it
unchanged preserves the float32 formula and RGBA8 quantization bridge; do not run
formatters over this vendor file. The corresponding golden vectors and exact
synthetic H5 evidence are under `tests/fixtures/unified_h5_v1`.

Sources were reviewed against agent-contracts PR 51 at
`408307c9a6b11258546b4465fa43323cb3668013`, with PR 50 at
`f775aa652e524a13fbdff9e832937eb74ba191bc`. See the repository's unified-H5
integration handoff for provenance, scope, and validation status.

Admission v2 is applied and is the only admitted revision: the resource
profile, the chaser state table at v2 (explicit camera-id validity), streamed
correspondence v2 (live identity tables checked against the catalog separately
from their closed-logical-value digests; closed input/receipt schemas),
grid/grating state rows validated as exact `stimulus_frame_num` joins to
`/correspondence/frames/sources` (the contract's `correspondence_scope`; no
per-component source tables), and whole-recording capacity-preflight evidence
naming the frozen producer policy. `experimental_h5_core_v1.json` stays pinned
byte-for-byte for provenance only; chaser v1 / correspondence v1 files are
refused. Optional pose v1 is not yet admitted.

Admission revision 4 (agent-contracts PR 88) sits next to revision 3. The
capacity preflight's `schema_version` chooses the revision: preflight v1 files
keep revision 3 and its `(chaser_state, bounding_boxes)` pair key; preflight v2
files are selected by the full `{core table path -> schema_version}` map over
their declared, present tables, the declared `catalog_id` must name the same
catalog (otherwise `catalog_id_mismatch`), and the declared policy must be the
one bound to it. A `(table, schema_version)` pair names one definition across
every pinned catalog (`schema.table_schema`), and every present core table must
carry the selected catalog's version. Correspondence receipts are validated by
their own `schema_version` (v2 or v3), which must be the one the file's
admission revision pins. The frame identity v1 witness H5s are under
`tests/fixtures/citrus_unified_h5_frame_identity_v1` (agent-contracts
`citrus-palette/fixtures/frame_identity_v1/`: complete and partial-adoption
from PR 88 `32a3a709`, pose-on and its README row from PR 89 `60856f3f`).

Admission revision 5 (agent-contracts PR 92, Citrus PR 27; merged `7a2e5d25`
and `576481f3`; every digest re-checked byte-for-byte against the merges) sits next to
revisions 3 and 4: preflight v3 files (`catalog_id angle_v1`) are selected by
its table-version map plus `catalog_id`, as revision 4. The selected binding
fixes the stimulus angle convention (agent-contracts PR 91
`citrus-palette/angles/stimulus_angle_convention_v1.json`), returned as
`admitted_binding(...)["stimulus_angle_convention"]` and carried on the
admission result: `angle_v1` -> `atan2_negative_y_x_degrees` (+90 = up),
every catalog of revisions 3-4 and the legacy pins of revision 5 ->
`canvas_y_down_clockwise_legacy` (+90 = down). It is never taken from the
file's date or from the session attribute alone. Under `angle_v1` the
`/metadata/session` attribute `stimulus_angle_convention` and the authored
protocol definition, trial index (v3) and semantic identity (policy v2) must
all declare `atan2_negative_y_x_degrees` (`stimulus_angle_convention_mismatch`);
under a legacy binding none may declare it (`stimulus_angle_convention_unexpected`).
The angle_v1 witness H5s are under `tests/fixtures/citrus_unified_h5_angle_v1`
(agent-contracts PR 92 `citrus-palette/fixtures/angle_v1/`, README verbatim).
