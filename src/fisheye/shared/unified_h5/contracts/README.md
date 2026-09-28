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
| `unified_h5_admission_v2.json` | agent-contracts PR 52 `13b48a1ed4a50a945796de995bc70b1a10684811` | `851ba36a7ab0757c90399e785ea350f285c5b1984f74409e125a79b6ba0b82b5` |
| `unified_h5_producer_policy_d544b081.json` (frozen producer policy named by capacity preflight) | agent-contracts `f3620fe1a05611db14497035d88f8c734449ac79` | `d544b0814006360b11b02cd9d171eefbddae7cc9cc41ad57470bbd240bc83cb9` |
| `experimental_h5_core_chaser_v2.json` (core v1 with chaser state table v2) | `e437fa4752679e5a614378d824e636104c4f095a` | `cb68ab6ecea614030b8f2a3d2096e9122b1efa5156b56ed7a495b3072c70316a` |
| `experimental_h5_correspondence_tables_v2.json` | `6bd8cbc443c3af699f9f13cbc2db037d8d99d82d` | `cb3ac574c76750c23de730129c4a32371328ffb34d792b1cfab164b7201ba1ea` |
| `experimental_h5_correspondence_input_v2.schema.json` | `6bd8cbc443c3af699f9f13cbc2db037d8d99d82d` | `a33e06819e966b7273015bfffe4aa0d01f24a04fb9ffac69abb29ed5c76329a3` |
| `experimental_h5_correspondence_receipt_v2.schema.json` | `6bd8cbc443c3af699f9f13cbc2db037d8d99d82d` | `8d2a414f46f6d6263f2609c55e15add9d7a2b61ad98349cf1ac3aef598b8d4e8` |
| `experimental_h5_capacity_preflight_v1.schema.json` | `6bd8cbc443c3af699f9f13cbc2db037d8d99d82d` | `d5d7ebf5249de35ce7234ddfec3c7cb51b57ac3b6fdcb25bf75724f25223d068` |

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

Admission v2 is applied: the resource profile, the chaser state table at v2
(explicit camera-id validity), streamed correspondence v2 (live identity
tables checked against the catalog separately from their closed-logical-value
digests; closed input/receipt schemas) and capacity-preflight evidence naming
the frozen producer policy. Each file's declared chaser/correspondence versions
select its rules, with no mixing in one file. The earlier revision (chaser v1 /
correspondence v1, catalog `experimental_h5_core_v1.json`) stays admissible
only until Citrus regenerates the consumer fixture corpus under v2; then it is
deleted. Optional pose v1 is not yet admitted.
