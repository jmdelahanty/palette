# Recording Status Page Deployment

Purpose: document the recommended ways to expose the recording status page,
including when to use localhost only, when to use SSH port forwarding, and how
to put auth/TLS in front of the app with a reverse proxy.

Applies to:
- `scripts/py -m fisheye.utils.serve_recording_status_page`

## Short Recommendation

Preferred order:

1. local workstation only:
   - bind `127.0.0.1`
   - no auth/TLS needed
2. occasional remote access:
   - keep app on `127.0.0.1`
   - use SSH port forwarding
3. shared internal page:
   - keep app on `127.0.0.1`
   - put `caddy` or `nginx` in front of it
   - terminate TLS in the proxy
   - require auth in the proxy

Avoid exposing the Python server directly on `0.0.0.0` unless the network is
small, trusted, and additionally restricted by firewall rules.

## Why Auth/TLS Lives In The Proxy

The status page server is intentionally simple:
- read-only
- no session handling
- no user management
- no TLS termination

That is the right boundary for the app itself. Reverse proxies are better for:
- HTTPS/TLS
- password or SSO auth
- request logging
- network-facing hardening

## Local-Only Mode

Run:

```bash
scripts/py -m fisheye.utils.serve_recording_status_page \
  --registry /nvme1/palette_registry.sqlite \
  --host 127.0.0.1 \
  --port 8765
```

Open on the same machine:

```text
http://127.0.0.1:8765
```

Use this when:
- you are on the workstation itself
- no other machine needs access

## SSH Port Forwarding

Keep the app local-only on the server:

```bash
scripts/py -m fisheye.utils.serve_recording_status_page \
  --registry /nvme1/palette_registry.sqlite \
  --host 127.0.0.1 \
  --port 8765
```

From another machine:

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server-hostname>
```

Then open locally on the client machine:

```text
http://127.0.0.1:8765
```

This gives you:
- encrypted transport
- access control via SSH
- no need to expose the Python server on the LAN

## Reverse Proxy Pattern

Recommended architecture:

```text
browser --> https://status.example.org --> caddy/nginx --> http://127.0.0.1:8765
```

Run the app on localhost only:

```bash
scripts/py -m fisheye.utils.serve_recording_status_page \
  --registry /nvme1/palette_registry.sqlite \
  --host 127.0.0.1 \
  --port 8765
```

Then put a proxy in front of it:
- `caddy` example:
  [Caddyfile.example](/home/delahantyj@hhmi.org/gitrepos/palette/docs/examples/recording_status_page/Caddyfile.example)
- `nginx` example:
  [nginx.conf.example](/home/delahantyj@hhmi.org/gitrepos/palette/docs/examples/recording_status_page/nginx.conf.example)

## Caddy Notes

Use Caddy when you want the simplest secure setup.

Advantages:
- straightforward reverse proxy configuration
- easy HTTP basic auth
- automatic certificate handling when DNS is available

Two common modes:
- real hostname with public/internal DNS and normal TLS
- `tls internal` for lab-internal use with Caddy's internal CA

## Nginx Notes

Use nginx when:
- it is already standard on the machine
- you want tighter control over TLS and auth settings

You will usually provide:
- certificate and key paths
- an htpasswd file for basic auth, or external auth/SSO integration

## Firewall Guidance

If you expose the proxy to the LAN:
- allow inbound only on the proxy port you intend to use (`443` or `80/443`)
- keep the Python app bound to `127.0.0.1`
- do not open `8765` externally unless you deliberately want unauthenticated
  plain HTTP access

## Quick Safety Matrix

| Mode | Other machines can access | Encrypted | Authenticated | Recommended |
| --- | --- | --- | --- | --- |
| `127.0.0.1` only | no | local only | local only | yes |
| SSH tunnel | yes | yes | yes | yes |
| `0.0.0.0:8765` plain HTTP | yes | no | no | only for short-lived trusted LAN use |
| reverse proxy + TLS + auth | yes | yes | yes | yes |

## Minimal LAN Exposure Without Proxy

If you intentionally want quick LAN access without a proxy:

```bash
scripts/py -m fisheye.utils.serve_recording_status_page \
  --registry /nvme1/palette_registry.sqlite \
  --host 0.0.0.0 \
  --port 8765
```

Then:
- verify the machine firewall allows only the intended subnet
- accept that traffic is plain HTTP
- accept that anyone who can reach the port can read the page

This is operationally convenient, but not the preferred long-term setup.

## Troubleshooting

Get the machine LAN IP:

```bash
hostname -I | awk '{print $1}'
```

Check whether the app is listening:

```bash
ss -ltnp | rg 8765
```

Check whether the proxy can reach the app:

```bash
curl -sS http://127.0.0.1:8765/healthz
```

## Related Docs

- [recording_status_page_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/recording_status_page_design.md)
- [recording_status_page_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/recording_status_page_todo.md)
