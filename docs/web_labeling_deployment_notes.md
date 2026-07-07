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


## Temporary per-user fixed-user servers

Until a trusted auth proxy is available, a practical short-term multi-labeler setup is one fixed-user server process per labeler, each on a different local port. All servers can point at the same labeling SQLite store, while the assignment store still enforces that each user only sees and mutates their assigned recordings.

Start one server per labeler on the workstation:

```bash
scripts/start_labeling_web_for_user.sh alice 8791
scripts/start_labeling_web_for_user.sh bob 8792
```

The helper delegates to `scripts/start_labeling_web.sh` and prints the matching tunnel command. It keeps the server bound to `127.0.0.1` by default and uses fixed-user auth for the requested labeler.

Important safety rules:

- Use a different port for each labeler.
- Give each labeler only their own tunnel/link for their assigned fixed-user server.
- Assign distinct recordings to each labeler; do not have two users mutate the same training zarr.
- Keep all servers pointed at the same `PALETTE_LABELING_STORE` if you want one unified admin/progress view.
- `PALETTE_LABELING_ADMIN_USER` defaults to `delahantyj` in this helper so a fixed labeler is not automatically made an admin.

Example with explicit store and remote host:

```bash
PALETTE_LABELING_STORE=/home/delahantyj@hhmi.org/.palette/labeling_work.sqlite \
PALETTE_LABELING_REMOTE_HOST=delahantyj-ws1 \
scripts/start_labeling_web_for_user.sh alice 8791
```

Stop a specific per-user server by port and pid file shown by the helper:

```bash
PALETTE_LABELING_PORT=8791 \
PALETTE_LABELING_PID=/tmp/palette-labeling-web-alice-8791.pid \
scripts/stop_labeling_web.sh
```

List currently running fixed-user labeling servers:

```bash
scripts/list_labeling_web_servers.sh
```

This helper is read-only. It reports matching PID files and live `fisheye.labeling.web` processes for the current workstation user.

For repeated multi-user sessions, keep the per-user port mapping in the local
operator roster instead of remembering ports by hand:

```bash
scripts/labeling_web_fixed_roster.sh set alice 8791 8791 alice@delahantyj-ws1.hhmi.org
scripts/labeling_web_fixed_roster.sh set bob 8792 8792 bob@delahantyj-ws1.hhmi.org
scripts/labeling_web_fixed_roster.sh list
scripts/labeling_web_fixed_roster.sh start --all
```

The roster defaults to `~/.palette/labeling_fixed_servers.tsv`. It is local
operator state and should not be committed. Each row stores:

```text
user_id<TAB>server_port<TAB>local_tunnel_port<TAB>ssh_target
```

Useful commands:

```bash
scripts/labeling_web_fixed_roster.sh start alice
scripts/labeling_web_fixed_roster.sh restart --all
scripts/labeling_web_fixed_roster.sh stop bob
scripts/labeling_web_fixed_roster.sh message --all
```

`message` prints the copy/paste SSH tunnel instructions using the stable port
from the roster. This is the preferred short-term workflow until a trusted auth
proxy replaces fixed-user per-port servers.

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
