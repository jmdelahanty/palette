# Web Labeling Deployment Examples

<!-- design-meta
status: template
last_updated: 2026-06-23
scope: example production wrappers for the assigned web-labeling service
-->

## Purpose

These examples show safe deployment shapes for the web-labeling service. They
are templates, not final site configuration.

Before using them, fill out:

```text
docs/web_labeling_production_decision_record.md
```

## Required Safety Properties

Production deployments should preserve these properties:

- Palette binds to loopback unless direct exposure is explicitly approved.
- Browser users authenticate before requests reach Palette.
- The proxy strips any inbound client-supplied copy of the trusted user header.
- The proxy sets exactly one trusted user header from the authenticated identity.
- Palette starts with `--production --trust-auth-header`.
- TLS termination and access logging happen at the proxy or an equivalent
  protected boundary.

## Example Palette Service Command

```bash
PALETTE_LABELING_LINK_SECRET='<secret-from-secret-store>' \
PALETTE_LABELING_LINK_NOT_BEFORE_UTC='' \
/home/delahantyj@hhmi.org/gitrepos/palette/scripts/py \
  -m fisheye.utils.labeling_work \
  --store /srv/palette-labeling/labeling_work.sqlite \
  serve \
  --production \
  --trust-auth-header \
  --auth-header X-Forwarded-User \
  --admin-user admin@example.org \
  --host 127.0.0.1 \
  --port 8795 \
  --access-log
```

Replace paths, admin users, and secrets with values from the production decision
record.

## Example systemd Unit

```ini
[Unit]
Description=Palette web labeling service
After=network.target

[Service]
Type=simple
User=palette-labeling
Group=palette-labeling
WorkingDirectory=/home/delahantyj@hhmi.org/gitrepos/palette
Environment=PYTHONUNBUFFERED=1
Environment=PALETTE_LABELING_LINK_SECRET=replace-with-secret-manager-value
ExecStart=/home/delahantyj@hhmi.org/gitrepos/palette/scripts/py -m fisheye.utils.labeling_work --store /srv/palette-labeling/labeling_work.sqlite serve --production --trust-auth-header --auth-header X-Forwarded-User --admin-user admin@example.org --host 127.0.0.1 --port 8795 --access-log
Restart=on-failure
RestartSec=5

# Keep filesystem access narrow for the chosen deployment host.
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Notes:

- Use a secret manager or systemd credential mechanism for the link secret.
- The service account must have write access only to approved mutable zarrs and
  the sidecar SQLite store.
- If registry or zarr paths are mounted separately, document them in the
  production decision record.

## Example Reverse Proxy Header Contract

This pseudocode is intentionally generic because the actual SSO mechanism may
be nginx, Apache, Caddy, an institutional gateway, or another proxy.

```text
1. Require authenticated browser identity.
2. Remove inbound X-Forwarded-User from the client request.
3. Set X-Forwarded-User to the authenticated username/email.
4. Set X-Forwarded-Host to the browser-visible host.
5. Forward to http://127.0.0.1:8795.
6. Log authenticated user, request path, status, and client IP.
```

Palette must not be the component that decides whether an unauthenticated
browser user is allowed through in production.

## Example nginx Location Shape

This is only a shape. Replace the authentication directives with the lab's real
SSO integration.

```nginx
server {
    listen 443 ssl http2;
    server_name labeling.example.org;

    # TLS config omitted: use the lab's managed certificate policy.

    # SSO/auth directives omitted: require authenticated user before proxy_pass.

    location / {
        # Strip any user identity header supplied by the browser/client.
        proxy_set_header X-Forwarded-User "";

        # Then set the trusted identity from the proxy-authenticated user.
        # Replace $remote_user with the variable populated by the chosen SSO.
        proxy_set_header X-Forwarded-User $remote_user;

        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        proxy_http_version 1.1;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        client_max_body_size 32m;

        proxy_pass http://127.0.0.1:8795;
    }
}
```

Important:

- Do not use this snippet without real authentication directives.
- Confirm the proxy does not forward a client-controlled `X-Forwarded-User`.
- Keep Palette's own `--auth-header` value synchronized with the proxy header.

## Example Local Operator Smoke

After deployment and before sharing with labelers:

```bash
scripts/check_labeling_web_static.sh
```

Then open `/admin` through the browser-visible URL and confirm:

- `Production mode` is `true`.
- `Trusted auth header` is `true`.
- `Same-origin POST guard` is `true`.
- The expected admin user is listed.
- Preflight warnings are expected and documented, or there are no warnings.
