"""Emit sealer 3.3.0 receipt-v4 fixtures with the tag's own generator (fake media only)."""
import json
import sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(repo / "tests"), str(repo / "src")]
import recording_transfer_fixture as f  # noqa: E402
out = Path(sys.argv[2])
assert not out.exists()
# Sealed deliveries.
f.seal_existing(out / "bound_v4_video", lambda r: f.bound_fixture(r, version=4))
def dish_group(root):
    sessions = f.bound_fixture(root, version=4, video=False, timing=False, diagnostic=False)
    group = json.loads((repo / "tests/fixtures/subject_reference_v2/accepted/dish_group.json").read_text())
    assert group["subject_id"] is None
    sessions[0]["subject"] = group
    f.bind_citrus_sessions(root, sessions, version=4)
f.seal_existing(out / "bound_v4_dish_group", dish_group)
f.seal_existing(out / "bound_v4_chain_v4_v4",
                lambda r: f.bound_fixture(r, version=4, upgrade_to_v2=True, upgrade_version=4))
# Unsealed source trees the sealer refuses (no snapshot/marker).
refused = out / "refused_sources"
def rebind(name, version, edit, **kw):
    root = refused / name
    sessions = f.bound_fixture(root, version=version, **kw)
    edit(sessions)
    f.bind_citrus_sessions(root, sessions, version=version)
rebind("mixed_v3_v4_head", 4, lambda s: s[0].update(version=3))
rebind("v4_with_reference_v1", 4, lambda s: s[0].update(subject=f.subject_reference()))
rebind("v3_with_reference_v2", 3, lambda s: s[1].update(subject=f.subject_reference_v2()))
rebind("v4_without_subject_reference", 4,
       lambda s: s[0].update(contract_edit=lambda c: c.pop("subject_reference")))
rebind("v4_extra_contract_key", 4,
       lambda s: s[1].update(contract_edit=lambda c: c.update(subject_declaration={})))
f.bound_fixture(refused / "chain_v3_then_v4", version=3, upgrade_to_v2=True, upgrade_version=4)
f.bound_fixture(refused / "chain_v4_then_v3", version=4, upgrade_to_v2=True, upgrade_version=3)
f.bound_fixture(refused / "chain_v2_then_v4", version=2, upgrade_to_v2=True, upgrade_version=4)
f.bound_fixture(refused / "chain_v1_then_v4", version=1, upgrade_to_v2=True, upgrade_version=4,
                pre310_diagnostic=True)
sessions = f.bound_fixture(refused / "v4_unreadable_superseded", version=4, upgrade_to_v2=True,
                           upgrade_version=4)
(refused / "v4_unreadable_superseded" / f.BINDINGS /
 f"receipts/obsctx_{sessions[0]['hex']}.json").write_text("{ not json")
print(out)
