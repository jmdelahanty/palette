# ⚠️ CRITICAL: Re-Import Required

## What Changed

The enum string storage format has been updated to use **2D uint8 arrays** instead of variable-length UTF-8 strings for TensorStore compatibility.

## Why Re-Import is Needed

Your current zarr file has:
```
analysis/enums/events/name/zarr.json:
  "data_type": "string"
  "codecs": [{"name": "vlen-utf8"}]  ← TensorStore can't read this in Zarr v3
```

After re-import, it will have:
```
analysis/enums/events/name/zarr.json:
  "data_type": "uint8"
  "shape": [56, 32]  ← 2D array: 56 strings, each up to 32 bytes
```

## How to Re-Import

```bash
python -m fisheye.analysis.import_stimulus_to_zarr \
    /nvme1/sesh3/2025-09-23T22-11-11Z_arena_4_chaser_arena4.h5 \
    /nvme1/sesh3/2025-09-23T22-11-11Z_arena_4_chaser_arena4.zarr/ \
    --overwrite
```

This will:
1. Re-read enums from the H5 file
2. Convert to columnar format with 2D uint8 arrays
3. Overwrite existing enum data

## Verification After Import

```bash
# Check the name array is now uint8
cat /nvme1/sesh3/2025-09-23T22-11-11Z_arena_4_chaser_arena4.zarr/analysis/enums/events/name/zarr.json

# Should show:
# "data_type": "uint8"  ← Not "string"!
# "shape": [56, <max_len>]  ← 2D, not 1D!
```

## What Your C++ Code Needs

After re-import, your C++ code should:

1. **Use `zarr3` driver** (not `zarr`)
2. **Read name array as 2D uint8**
3. **Decode each row to UTF-8 string**

See `ENUM_PATHS_QUICK_REFERENCE.md` for full C++ example.

## Timeline

- ✅ Python code updated to generate 2D uint8 arrays
- ⚠️ **Current zarr file still has old format (variable-length strings)**
- 🔄 **Re-import needed to get new format**
- 🔧 **Then update C++ code to read 2D uint8 arrays**

---

**Bottom line: Re-run the import command above, then your C++ code can read the enums with TensorStore!**
