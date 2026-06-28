# Web labeling deployment notes

## Current expected use case

The near-term deployment is small and operator-controlled:

- Around five labelers.
- Labelers are campus users.
- Users are unlikely to label concurrently most of the time.
- Access is expected through the campus network or VPN.
- Labelers should not need a full Palette or Crimson installation.
- The web server should expose only assigned work for the resolved user.
- Labelers mutate assigned training Zarrs through server-owned routes, not by receiving direct Zarr write authority.

One IP address and one port are sufficient for this use case. Multiple users can access the same web service at the same time as long as the application keeps user identity, assignment filtering, task opening, sessions, and mutation authorization separated server-side.

## Current recommended operating model

For the current scale, prefer a simple, conservative setup:

```text
labeler browser
  -> VPN or SSH tunnel
  -> Palette labeling web server
  -> assignment store
  -> server-owned writes to assigned training Zarrs
```

The safest default is to keep the app bound to localhost on the workstation:

```bash
PALETTE_REGISTRY_PATH=/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
scripts/py -m fisheye.labeling.web \
  --store /home/delahantyj@hhmi.org/.palette/labeling_work.sqlite \
  serve \
  --host 127.0.0.1 \
  --port 8795 \
  --user delahantyj \
  --admin-user delahantyj
```

Remote access from home can then use an SSH local-forward:

```bash
ssh -N -L 8795:127.0.0.1:8795 <workstation-hostname>
```

The browser URL remains:

```text
http://127.0.0.1:8795/admin/datasets
```

This keeps the web process private to the workstation while allowing a remote operator to access it through authenticated SSH.

## Home-use tunnel helper

For now, keep the Palette web server bound to `127.0.0.1` on the workstation and access it from home through SSH. This avoids exposing the Python web process to campus network scanners or general VPN traffic.

On the workstation, start the web server:

```bash
scripts/start_labeling_web.sh
```

On the home machine, open a tunnel:

```bash
PALETTE_LABELING_REMOTE_HOST=<workstation-hostname> scripts/tunnel_labeling_web.sh
```

Then open:

```text
http://127.0.0.1:8795/admin/datasets
```

The helper is equivalent to:

```bash
ssh -N -L 8795:127.0.0.1:8795 <workstation-hostname>
```

Optional overrides:

```bash
PALETTE_LABELING_LOCAL_PORT=8875 \
PALETTE_LABELING_REMOTE_PORT=8795 \
PALETTE_LABELING_REMOTE_HOST=<workstation-hostname> \
scripts/tunnel_labeling_web.sh
```

Then open:

```text
http://127.0.0.1:8875/admin/datasets
```

Use this mode until IT confirms the preferred campus/VPN deployment pattern.

## VPN-only direct access option

If the workstation is reachable only from campus/VPN networks, the server can be bound to a non-loopback interface:

```bash
PALETTE_REGISTRY_PATH=/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
scripts/py -m fisheye.labeling.web \
  --store /home/delahantyj@hhmi.org/.palette/labeling_work.sqlite \
  serve \
  --host 0.0.0.0 \
  --port 8795 \
  --user delahantyj \
  --admin-user delahantyj
```

Then labelers on VPN could open:

```text
http://<workstation-hostname-or-vpn-ip>:8795/my-datasets
```

Use this only after confirming the port is not reachable from outside campus/VPN networks. The web app has assignment and mutation guards, but network exposure should still be minimized because assigned training Zarr mutation is possible through server-authorized routes.

## Identity and assignment assumptions

The current workflow relies on application-level authorization rather than full institutional SSO:

- The server resolves a browser user.
- `/my-datasets` filters work by resolved user.
- Task opening requires the task to be assigned to the resolved user.
- Mutation requires an active task session.
- Mutation requires the current assignment owner.
- Mutation target is resolved server-side from the active assignment/task.
- Invite links are entry hints, not write authority.
- One recording has one active owner.
- CSV or handoff artifacts are metadata/control-plane artifacts, not label mutation targets.

This is acceptable for the current small, VPN-restricted group. Full SSO can be deferred until the service is exposed more broadly or the number of users increases.

## Future domain-backed deployment

For a smoother lab-wide experience, the target deployment would be:

```text
https://palette-labeling.<campus-domain>
  -> campus/VPN DNS
  -> firewall-restricted host
  -> nginx or Caddy reverse proxy
  -> Palette labeling app bound to 127.0.0.1:8795
```

Benefits:

- Users get a normal domain name.
- No one needs to create an SSH tunnel manually.
- TLS/HTTPS is handled by the reverse proxy.
- The Palette app can remain bound to localhost.
- The reverse proxy can add request size limits, timeouts, logs, and security headers.
- Firewall rules can restrict access to campus/VPN networks.

This is the preferred future direction if more labelers start using the system regularly.

## Future hardening checklist

Before broader use, consider adding:

- A `systemd` user service or similar process supervisor for the web server.
- A stable internal DNS name.
- A reverse proxy such as Caddy or nginx.
- HTTPS, even on VPN.
- Campus/VPN-only firewall rules.
- Explicit operator restart instructions.
- Log rotation.
- Backup confirmation for mutable training Zarrs before labelers start.
- Periodic export of assignment/progress/audit summaries.
- Clear invite-token rotation policy.
- Optional institutional SSO if the service becomes broadly accessible.

## Practical recommendation

For the immediate five-user case:

- Keep the app simple.
- Keep the service VPN-only or SSH-tunnel-only.
- Use invite links and assignment-store authorization.
- Use the admin dataset page to monitor assignments and progress.
- Do not expose the port publicly.

For the next stage:

- Put a reverse proxy and DNS name in front of the same app.
- Keep the Palette app bound to localhost behind the proxy.
- Add SSO only if network restriction plus assignment authorization no longer feels sufficient.
