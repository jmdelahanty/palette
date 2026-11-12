"""Workflow-runner support: the Palette side of ``workflows/*.smk``.

Snakemake (separate ``palette-flow`` conda env, ws1 only) owns the graph,
keep-going and the operator view; this package owns everything Palette-
specific the Snakefiles call (docs/design/2026-10-07-workflow-runner):

- :mod:`fisheye.flow.config` - one runner config, validated (state never
  inside the v1 ``.processing_state``, explicit destination root);
- :mod:`fisheye.flow.lsf` - submit one LSF job over SSH and wait on NFS
  evidence, with a shared ``bjobs`` cache refreshed at most once per
  ``bjobs_min_interval_s`` (the login-node budget);
- :mod:`fisheye.flow.intake` - the intake steps the Snakefile runs, built
  only on the ``python -m fisheye.intake`` CLI contract.

A sentinel ``<flow_root>/intake/<sha>/<step>.done.json`` is written only
from a ``fisheye.intake`` JSON result whose verdict is true: it caches
Palette's verdict and is never read back as evidence by Palette.
"""
