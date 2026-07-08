# Sandbox Zarr Fallback

When running inside Codex sandbox, sync `zarr.open_group(...)` may hang.
If that happens, use direct `zarr.json` reads with `jq` for review-status checks.

## Single Run Check

```bash
ZARR=/nvme1/recordings/<session>/zarr/<session>.zarr
RUN=refined_keypoints_<timestamp>

RUN_JSON="$ZARR/refined_keypoints_runs/$RUN/zarr.json"
PARENT_JSON="$ZARR/refined_keypoints_runs/zarr.json"

echo "run zarr.json:"
jq '.attributes.keypoint_review_status // null' "$RUN_JSON"

echo "parent consolidated metadata:"
jq --arg run "$RUN" '.consolidated_metadata.metadata[$run].attributes.keypoint_review_status // null' "$PARENT_JSON"
```

## Recursive Strict-Style Check

This checks only run-group metadata files:
`.../refined_keypoints_runs/refined_keypoints_*/zarr.json`

```bash
root=/nvme1/recordings
ok=0
bad=0

while IFS= read -r f; do
  state=$(jq -r '.attributes.keypoint_review_status.state // empty' "$f")
  use=$(jq -r '.attributes.keypoint_review_status.intended_use // empty' "$f")
  if [[ "$state" == "approved" && "$use" == "training" ]]; then
    ((ok++))
  else
    ((bad++))
    printf "NONMATCH\t%s\tstate=%s\tintended_use=%s\n" "$f" "${state:--}" "${use:--}"
  fi
done < <(find "$root" -type f | rg '/refined_keypoints_runs/refined_keypoints_[^/]+/zarr\.json$')

printf "SUMMARY\tok=%d\tbad=%d\n" "$ok" "$bad"
```

## Notes

- This is a sandbox fallback only, not a replacement for normal Zarr tooling.
- Avoid scanning all `zarr.json` under run directories; that includes array metadata files and will produce false non-matches.
