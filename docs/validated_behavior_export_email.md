# Validated-behavior export availability email

The one-shot notifier announces an **exact, validated export run**. It reads
the selected manifest through `ValidatedBehaviorExportDataset.open` in receipt
mode before composing a message. The email has Dataset, Access, Tables, How to
read it, and Provenance sections in both plain text and formatted HTML. It does
not modify the export, grant access, or activate a production selector. Preview
is the default and sends nothing.

Publish a reviewed guide as a versioned companion **outside** the export's
sealed `.generations` tree. For the August sleepyfish guide:

```bash
scripts/py -m fisheye.utils.publish_validated_behavior_handoff \
  --publication-root /groups/johnson/johnsonlab/jeremy/exports/sleepyfish_core_behavior_bout_kinematics_v1_20260912_v001 \
  --export-run-id sleepyfish-core-fullrate-bout-kinematics-v1-canary-20260912-v001 \
  --source /home/delahantyj@hhmi.org/sleepyfish_august_2026_colleague_data_handoff.md \
  --version v001
```

This writes `handoffs/export_run_id=<run>/versions/v001.md` beside the export
and an explicit `handoff.json` selection record. The record binds the selected
guide's SHA-256 to the exact export run and manifest digest. The publisher
requires those identities in the guide. It leaves existing guide versions and
the immutable Parquet generation untouched; a revision uses a new version.
The companion record is schema v1 and independent of the scientific export
manifest. Older exports without one remain readable; the notifier can still
preview them, but delivery requires either a selected companion or an explicit
`--handoff` guide.

The notifier automatically validates and links the selected companion. Preview
the current sleepyfish message with a real recipient:

```bash
scripts/py -m fisheye.utils.notify_validated_behavior_export \
  --publication-root /groups/johnson/johnsonlab/jeremy/exports/sleepyfish_core_behavior_bout_kinematics_v1_20260912_v001 \
  --export-run-id sleepyfish-core-fullrate-bout-kinematics-v1-canary-20260912-v001 \
  --to colleague@example.org \
  --audience 'Johnson Lab collaborator' \
  --location /groups/johnson/johnsonlab/jeremy/exports/sleepyfish_core_behavior_bout_kinematics_v1_20260912_v001 \
  --access-note 'Request Johnson Lab group access from the dataset owner.'
```

Repeat `--to` for additional recipients. `--location` may describe another
shared path or URL, but the email still identifies the **validated publication
root and manifest digest** separately. Confirm that a different access location
points to the same export before sharing it. `--note` adds a one-line message.
An optional `--handoff` adds an additional guide when a companion is selected,
or supplies a guide for an export without a companion. Delivery refuses to
proceed without some reading guide. Confirm the selected path is accessible to
the recipient; an email does not grant storage permissions.

Add `--deliver` to queue the email. The default transport is the existing
Palette outbox; the notifier uses `~/.palette/export_email_outbox` by default,
or `PALETTE_EXPORT_NOTIFICATION_OUTBOX` if set:

```bash
scripts/py -m fisheye.utils.notify_validated_behavior_export \
  --publication-root /path/to/export-root \
  --export-run-id exact-export-run-id \
  --to colleague@example.org \
  --mode outbox --deliver
```

To send through the SMTP relay already used by Palette labeling and geometry
review, configure `PALETTE_LABELING_SMTP_HOST`, `PALETTE_LABELING_SMTP_PORT`,
and any required TLS/authentication environment variables as for those
services, then use `--mode smtp --deliver`. The optional
`PALETTE_EXPORT_NOTIFICATION_FROM` overrides the existing
`PALETTE_LABELING_NOTIFICATION_FROM` sender for this command. Keep recipient
addresses and credentials out of repository files. A successful SMTP command
reports `sent`; an outbox delivery reports `queued` and the `.eml` path.

This is an explicit announcement command, so rerunning `--deliver` creates
another email. Review the preview and use the delivery command once for each
intended announcement. An email is evidence of notification, not of access
control or scientific approval.
