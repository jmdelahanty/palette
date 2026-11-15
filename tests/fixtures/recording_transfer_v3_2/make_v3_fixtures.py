"""Emit sealer 3.2.0 receipt-v3 fixtures with the tag's own generator (fake media only)."""
import sys
from pathlib import Path
tests = Path(sys.argv[1]).resolve() / "tests"
sys.path.insert(0, str(tests))
import recording_transfer_fixture as f  # noqa: E402
out = Path(sys.argv[2])
assert not out.exists()
# Sealed deliveries.
f.seal_existing(out / "bound_v3_video", lambda r: f.bound_fixture(r, version=3))
def null_objects(root):
    s = f.subject_reference(); s.update(dish=None, fish=None, zebrobot_version=None)
    sessions = f.bound_fixture(root, version=3, video=False, timing=False, diagnostic=False)
    sessions[0]["subject"] = s
    f.bind_citrus_sessions(root, sessions, version=3)
f.seal_existing(out / "bound_v3_null_subject_objects", null_objects)
# Unsealed source trees the sealer refuses (no snapshot/marker).
def rebind(name, version, edit, **kw):
    root = out / "refused_sources" / name
    sessions = f.bound_fixture(root, version=version, **kw)
    edit(sessions)
    f.bind_citrus_sessions(root, sessions, version=version)
rebind("mixed_v2_v3", 3, lambda s: s[0].update(version=2))
rebind("v3_without_subject_reference", 3,
       lambda s: s[0].update(contract_edit=lambda c: c.pop("subject_reference")))
rebind("v3_extra_contract_key", 3,
       lambda s: s[1].update(contract_edit=lambda c: c.update(subject_declaration={})))
rebind("v2_with_subject_reference", 2,
       lambda s: s[0].update(contract_edit=lambda c: c.update(subject_reference=f.subject_reference())))
print(out)
