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
