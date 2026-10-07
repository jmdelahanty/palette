# Pinned producer contracts (shared)

Copied byte-for-byte from agent-contracts; the loader verifies each SHA-256
before use. Change only with an explicit producer/consumer contract review.

| Resource | Source | SHA-256 |
| --- | --- | --- |
| `orange_recording_subject_reference_v1.schema.json` | agent-contracts PR 53 `6098ca4710b52b9b67fc145d67bcae47946216a6` (Orange `b967dc9` + `9569a79`) | `3c4ba74f0f95f76dbb8b3d65aa8bd39fd409383388ee8debec3845ef26c1a930` |
| `orange_recording_subject_reference_v2.schema.json` | agent-contracts PR 53 `879ae3f68c74abaea2b7d1284b684e213434802d` | `d0f300fdbd71747f44219baf7bb5aea262ebbd275f85f877df9f1a00aa92e935` |
| `citrus_zebrobot_snapshot_v3.schema.json` | agent-contracts PR 52 `e16f62511b14a0f99a2236621ed7b03279bc41a3` (`subject_identity_v3_2026-10-05.md` `b6c8031333293c2e910fb40ec33d4cc0a39ad7e9216fdeeec7e116f0889caac9`) | `6d84dcfd97fca45cdf40363124b208f3547e6333c77bb8ed78fb5c0d2def6f26` |
| `recording_transfer_v2.schema.json` | Citrus `citrus-recording-transfer` 2.0.0 draft, branch `agent/citrus/sealer-2.0.0-marker-v3-20261007` @ `cfd477463719dc1ee519e168a381361d94da4ad6` (adds `marker_v3`; v2 marker and snapshot unchanged from 1.0.x `4cf61131…`); reliance in agent-contracts `citrus-recording-transfer-consumers` | `c12832b35657f21514392215d838f401be670e76ac22ae6dab28bf304391503c` |
| `orange_recording_output_v1.schema.json` | Orange `e11e841839f449876b11a2c7a80e816124235122` `docs/schemas/` (output descriptors in `recording_session.json` / `clip_manifest.json`); used by `acquisition_crop_stream_ledger` for crop descriptors that declare a size | `4db5c324cb041a2e31f3c9f96189c5d630fa94a7c352f447b5d9a62a7655d147` |
| `orange_recording_crop_output_v1.schema.json` | Orange `e11e841839f449876b11a2c7a80e816124235122` `docs/schemas/` (sealed `recording_snapshot_start.json` `crop_outputs.<serial>`) | `6a67811e9162116bb318694275eeca1e9d82ab09ff3b424dca957f67236c8894` |
