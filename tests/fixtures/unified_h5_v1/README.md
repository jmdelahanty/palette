# Frozen synthetic producer fixtures

The `.h5.gz.b64` files contain exact emitted synthetic H5 bytes, compressed with
gzip and represented as base64 text for repository transport. This packaging is
not a recording format or a receipt contract. `unified_h5_fixtures.py` decodes
only into a test's temporary directory and checks the inventory's original size
and SHA-256 before opening HDF5. Tests do not need a `/groups` mount or Citrus
checkout, and they never rewrite historical producer evidence.

`inventory.json` records source paths, producer commits, byte identities, and
scope. In particular, component-only witnesses are **not** globally finalized
recordings. They test scoped row/reference joins; the full importer separately
requires global completion, the external exact-file receipt, full protocol and
geometry checks, and numerical appearance replay.

`base.receipt.json` and `appearance.receipt.json` are the corresponding external
synthetic finalization receipts. Tests that mutate a fresh temporary copy may
build an explicitly test-only outer receipt to reach inner refusal checks. Such
a receipt is never written back to these fixtures or treated as historical
producer evidence.

The unchanged appearance golden-vector JSON has SHA-256
`ea1a9be5c3ed2b7d750a14bcd70b1eed6e26f9e9b9f6df88153c11d3be5774f4` and is
checked against the pinned portable evaluator as well as the emitted H5 witness.
