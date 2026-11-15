# Design documents

One directory per design, named `YYYY-MM-DD-<topic>/`, dated by when the design
was opened. The design itself is `README.md` in that directory; supporting
material (measurements, diagrams, question lists) sits beside it.

Each design README starts with a short header:

- **Status:** `draft`, `accepted`, `superseded`, or `withdrawn`.
- **Owner** and **last reviewed** date.
- **Builds on / supersedes / superseded by:** links, so readers can find the
  current design without guessing from file dates.

Change a design by editing its README and noting the change in its decision
log. A new direction that replaces a design gets a new dated directory, and the
old header is updated to point to it. Work status belongs in the owning queue
(`docs/diagnostics/authority_consolidation_work_queue_2026-08-25.md`), not here.
