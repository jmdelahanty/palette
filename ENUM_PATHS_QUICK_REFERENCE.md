# Enum Paths Quick Reference for C++ Code

## ✅ NEW Columnar Paths (TensorStore Compatible)

Use these paths for all enum loading:

### Event Types (56 entries: 0-55)
```
analysis/enums/events/id        → int32 array [56]
analysis/enums/events/name      → uint8 array [56, max_len] (UTF-8 encoded strings)
```

### Stimulus Modes (17 entries: -1, 2-16, 99)
```
analysis/enums/stimulus_modes/id        → int32 array [17]
analysis/enums/stimulus_modes/name      → uint8 array [17, max_len] (UTF-8 encoded strings)
```

### Chaser Trial States (3 entries: 0-2)
```
analysis/enums/chaser_trial_states/id        → int32 array [3]
analysis/enums/chaser_trial_states/name      → uint8 array [3, max_len] (UTF-8 encoded strings)
```

---

## 🔴 OLD Structured Paths (Don't Use - Compound Dtypes)

These may still exist in the file but should be avoided:

```
analysis/enums/events/events                    ❌ Compound dtype [('id', 'i4'), ('name', 'S128')]
analysis/enums/events/stimulus_modes            ❌ Compound dtype
analysis/enums/events/chaser_trial_states       ❌ Compound dtype
```

**Why avoid:** TensorStore v0.1.x has limited support for compound dtypes in Zarr v3.

---

## C++ Access Pattern

### Option 1: TensorStore (Recommended)

```cpp
#include "tensorstore/tensorstore.h"
#include "tensorstore/kvstore/kvstore.h"

auto load_enums(const std::string& zarr_path) {
    // Load IDs (int32 array)
    auto id_spec = tensorstore::Spec::FromJson({
        {"driver", "zarr3"},  // ← Use zarr3 for Zarr v3 format!
        {"kvstore", {
            {"driver", "file"},
            {"path", zarr_path}
        }},
        {"path", "analysis/enums/events/id"}
    }).value();

    auto id_store = tensorstore::Open(id_spec, tensorstore::OpenMode::open,
                                      tensorstore::ReadWriteMode::read).result().value();
    auto ids = tensorstore::Read(id_store).result().value();  // [56] int32

    // Load names (2D uint8 array)
    auto name_spec = tensorstore::Spec::FromJson({
        {"driver", "zarr3"},  // ← Use zarr3 for Zarr v3 format!
        {"kvstore", {
            {"driver", "file"},
            {"path", zarr_path}
        }},
        {"path", "analysis/enums/events/name"}
    }).value();

    auto name_store = tensorstore::Open(name_spec, tensorstore::OpenMode::open,
                                        tensorstore::ReadWriteMode::read).result().value();
    auto name_bytes = tensorstore::Read(name_store).result().value();  // [56, max_len] uint8

    // Decode uint8 array to strings
    std::vector<std::string> names;
    for (int i = 0; i < name_bytes.shape()[0]; ++i) {
        const uint8_t* row = &name_bytes.data()[i * name_bytes.shape()[1]];
        size_t len = 0;
        while (len < name_bytes.shape()[1] && row[len] != 0) ++len;
        names.emplace_back(reinterpret_cast<const char*>(row), len);
    }

    // Combine into map
    std::map<int, std::string> enum_map;
    for (size_t i = 0; i < ids.size(); ++i) {
        enum_map[ids.data()[i]] = names[i];
    }
    return enum_map;
}
```

### Option 2: Direct Zarr V3 JSON Reading

If you prefer to parse zarr metadata directly:

```cpp
// Check if path exists and is a group
bool is_columnar = std::filesystem::exists(zarr_path + "/analysis/enums/events/zarr.json");

if (is_columnar) {
    // Parse zarr.json to verify node_type == "group"
    // Load id array from: analysis/enums/events/id/c/
    // Load name array from: analysis/enums/events/name/c/
} else {
    // Fall back to legacy structured array
}
```

---

## Path Fallback Strategy

For maximum compatibility, try paths in this order:

```cpp
const std::vector<std::pair<std::string, std::string>> enum_path_candidates = {
    // Try new columnar format first
    {"analysis/enums/events/id", "analysis/enums/events/name"},

    // Try root-level columnar (alternative location)
    {"enums/events/id", "enums/events/name"},

    // Fall back to legacy structured (if new code fails)
    {"analysis/enums/events/events", ""},  // Empty second element = structured array
    {"enums/events", ""},
};

for (const auto& [id_path, name_path] : enum_path_candidates) {
    if (name_path.empty()) {
        // Legacy structured array - use compound dtype reader
        if (auto result = try_load_structured_enum(id_path)) {
            return *result;
        }
    } else {
        // Columnar format - load two separate arrays
        if (auto result = try_load_columnar_enum(id_path, name_path)) {
            return *result;
        }
    }
}
```

---

## Verification Commands

```bash
# Check the new structure exists
ls -la /path/to/file.zarr/analysis/enums/

# Should see three directories:
# - events/
# - stimulus_modes/
# - chaser_trial_states/

# Check events enum structure
ls -la /path/to/file.zarr/analysis/enums/events/

# Should see:
# - id/         (directory - the int32 array)
# - name/       (directory - the string array)
# - zarr.json   (metadata with storage_layout: "columnar")

# Inspect metadata
cat /path/to/file.zarr/analysis/enums/events/zarr.json

# Should show:
# {
#   "attributes": {
#     "storage_layout": "columnar",
#     "field_names": ["id", "name"]
#   },
#   "zarr_format": 3,
#   "node_type": "group"
# }
```

---

## Data Types

### ID Arrays
- **Dtype:** `int32` (4 bytes per entry)
- **Shape:** `(n,)` where n = number of enum entries
- **Range:**
  - Events: 0-55
  - Stimulus modes: -1, 2-16, 99
  - Chaser states: 0-2

### Name Arrays
- **Dtype:** Variable-length UTF-8 string
- **Shape:** `(n,)` matching ID array
- **Format:** Uppercase with underscores (e.g., `"PROTOCOL_START"`)
- **No null terminators** in modern zarr v3

---

## Common Issues

### Issue: "Path not found"
**Solution:** Ensure you imported/re-imported data with the updated Python code

### Issue: "Compound dtype not supported"
**Solution:** You're reading the old structured path. Switch to columnar paths.

### Issue: "Mismatched array lengths"
**Solution:** ID and name arrays should have identical length. If not, data is corrupted.

### Issue: "Can't decode strings"
**Solution:** Names are UTF-8. Use `std::string` or proper UTF-8 decoder.

---

## Test Data

Successfully tested with:
- **File:** `/nvme1/sesh3/2025-09-23T22-11-11Z_arena_4_chaser_arena4.zarr/`
- **Import date:** October 31, 2025
- **Zarr format:** v3
- **Structure verified:** ✅ Columnar format present

---

## Need Help?

See full documentation: `ENUM_COLUMNAR_FORMAT_CHANGES.md`
