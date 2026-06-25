# Web Labeling Production Decision Record

<!-- design-meta
status: template
last_updated: 2026-06-23
scope: required deployment decisions before production web-labeling use
-->

## Purpose

Use this record before enabling production browser labeling for non-developer
users.

The implementation enforces assignment ownership, signed links, trusted-header
auth opt-in, production launch checks, session locking, audit events, and
operator repair paths. This document captures the deployment decisions that
cannot be safely inferred by code.

Use `docs/web_labeling_deployment_examples.md` for example systemd and reverse
proxy shapes after filling out this decision record.

## Decision Summary

```text
decision_record_owner:
decision_date:
service_url:
production_ready: no
```

## Authentication Boundary

Choose and document the production identity boundary.

```text
auth_provider:
proxy_or_gateway:
trusted_user_header:
header_strip_rule_confirmed: no
header_rewrite_rule_confirmed: no
admin_users:
```

Required answers:

- Which system authenticates users before traffic reaches Palette?
- Which proxy strips inbound client-supplied copies of the trusted user header?
- Which exact header does the proxy set after authentication?
- Which Palette users are admins?
- How is proxy configuration reviewed before launch?

Production service flags should include:

```bash
--production --trust-auth-header --auth-header <trusted-user-header> --admin-user <admin-user>
```

## Host, Service Account, and Filesystem

Choose where the Palette-capable service runs.

```text
host:
service_account:
working_directory:
labeling_store_path:
palette_repo_path:
registry_path:
zarr_mounts:
backup_location:
```

Required answers:

- Which host has access to the required Palette environment and zarr storage?
- Which service account owns the process?
- Which paths must be readable?
- Which mutable zarr paths may be written?
- Where is the sidecar SQLite store located?
- Where are sidecar-store backups written?
- Where are mutable zarr backups written?

## Network and TLS Boundary

Document the network exposure model.

```text
service_bind_host: 127.0.0.1
service_bind_port:
external_url:
tls_termination:
allowed_clients_or_networks:
non_loopback_bind_required: no
```

Required answers:

- Is Palette bound only to loopback behind a TLS/auth proxy?
- If direct non-loopback bind is required, who approved `--allow-non-loopback`?
- Where is TLS terminated?
- Which clients or networks can reach the service?
- Where are access logs retained?

Preferred launch shape:

```bash
PALETTE_LABELING_LINK_SECRET='<secret-from-secret-store>' \
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  serve --production --trust-auth-header --auth-header X-Forwarded-User \
  --admin-user admin@example.org \
  --host 127.0.0.1 --port 8795 --access-log
```

## Operational Checks

Complete before sharing links with labelers.

```text
static_validation_passed: no
focused_unit_tests_passed: no
real_zarr_smoke_passed: no
sidecar_backup_tested: no
mutable_zarr_backup_tested: no
admin_preflight_clean: no
labeler_links_safe_to_share: no
safe_share_inspection_command: TODO
safe_share_inspection_report: TODO
preferred_labeler_entry_is_guarded_my_datasets: no
browser_writes_target_assigned_training_zarr: no
csv_handoff_artifacts_are_metadata_only: no
```

Commands:

```bash
scripts/check_labeling_web_readiness.sh
```

```bash
PALETTE_LABELING_LINK_SECRET='<secret-from-secret-store>' \
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  preflight --production --trust-auth-header --auth-header X-Forwarded-User \
  --admin-user admin@example.org \
  --host 127.0.0.1 --port 8795 --access-log
```

```bash
scripts/py scripts/check_labeling_production_decision_record.py
```

```bash
scripts/check_labeling_web_static.sh
```

```bash
scripts/check_labeling_web_unit.sh
```

```bash
PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC=/path/to/copied_web_labeling_real_zarr_smoke_spec.json \
PYTHONPYCACHEPREFIX=/tmp/palette-pycache \
scripts/py -m pytest -p no:cacheprovider \
  tests/integration/fisheye/test_labeling_web_real_zarr_smoke.py -q
```

## Sign-Off

```text
operator:
reviewer:
approved_for_labelers: no
notes:
```
