# Vendored browser libraries

The labeling pages use native ES modules with no build step (decision
2026-09-27, `docs/design/2026-09-26-labeling-frontend-review/README.md`).
These files are served from `/static/<build>/vendor/` by `web_static.py` and
imported by relative path, for example
`import { h, render } from "../vendor/preact.module.js"`.

| File | Package | Upstream file | License |
|---|---|---|---|
| `preact.module.js` | preact 10.29.8 | `dist/preact.module.js` | MIT (`LICENSE-preact`) |
| `preact-hooks.module.js` | preact 10.29.8 | `hooks/dist/hooks.module.js` | MIT (`LICENSE-preact`) |
| `htm.module.js` | htm 3.1.1 | `dist/htm.module.js` | Apache-2.0 (`LICENSE-htm`) |

## Provenance

The npm registry tarballs were downloaded on 2026-09-28. Their sha512 was checked against the registry's `dist.integrity`:

- `https://registry.npmjs.org/preact/-/preact-10.29.8.tgz`
  `sha512-ej2aVZ+vZ8WO7tvlQWRM9N63A0KzF9q4mWJfDUHgYaIofWY9hu74QdnQrjoPMmZi2/nZ5gN0bJCQF49xQqx09Q==`
- `https://registry.npmjs.org/htm/-/htm-3.1.1.tgz`
  `sha512-983Vyg8NwUE7JkZ6NmOqpCZ+sh1bKv2iYTlUkzlWmA5JD2acKoxd4KVxbMmxX/85mtfdnDmTFoNKcg5DGAvxNQ==`

## Local modifications

The source-map comments were removed from both preact files, because the map files are not vendored. One import was rewritten in the hooks file: `from"preact"` became `from"./preact.module.js"`. Browsers cannot resolve the bare specifier without an import map. `htm.module.js` is byte-identical to upstream.

| File | Upstream sha256 | Vendored sha256 |
|---|---|---|
| `preact.module.js` | `c30e721ebfdc6e2ad4c18c14d2dfb82667829c8aec27de1207774e3fc16858a8` | `25a5df7e9f628a587743c4641a368737a3b218f28ef738ba0376a2d5bdfc948c` |
| `preact-hooks.module.js` | `a6ee626f2d01570592dd569a792e3f050154aa02890eead8c223fa3ed5aa3d5a` | `9e7eff58e0ae604583461eba1344da69a6894eab575eb591c635cc0d3f9ec57d` |
| `htm.module.js` | `ab33dd3f38059b9be4d5f5350128eefb2356639c4e0bbe9d9e8b3ba75847e9e4` | `ab33dd3f38059b9be4d5f5350128eefb2356639c4e0bbe9d9e8b3ba75847e9e4` |

`tests/unit/fisheye/test_labeling_static_vendor.py` pins the vendored digests.

## Updating

1. Download the new tarballs from the registry and check their integrity.
2. Apply the same modifications, and never edit the code in any other way.
3. Replace the files and license texts.
4. Update this table and the pinned digests in the same change.
5. Keep the two preact files at the same version, because hooks rely on preact's internal `options` object.
