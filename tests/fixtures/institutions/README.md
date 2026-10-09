# Institutions replay evidence

These compressed bundles are exported from independently archived source and prompt versions:

- Initial customer-support smoke: replay verified; reliability gate failed (21 rejected turns out of 40).
- Corrected customer-support smoke: replay verified; reliability gate passed (two games).
- Completed customer-support pilot: replay verified; 18 games, eight phases and 40 turns per game.
- Scripted customer-support evidence: a synthetic three-round game demonstrating an excluded real case, blocked peer closure, successful retry after a false claim, and rejected review with no applied updates or report. No model inference produced this fixture.

The timestamp directories are original source identifiers retained for provenance. Each manifest records the source snapshot, hashes, configuration, parent and adapter revisions, and separate replay and reliability results. Game event IDs are unchanged and scoped to a game. The initial archive did not provide an artifact hash list; its source hashes and complete deterministic replay were checked, and this limitation is recorded explicitly.

Regenerate real bundles with `experiments/support-integrity/export_replay.py` in realignment-benchmark, using `--snapshot <archived-results-directory> --split full|smoke --out <fresh-bundle-root>`. ChatLab's tests read these exported JSON records; they never import the experiment or load weights. The exporter’s `make_synthetic_fixture.py` reproduces the scripted fixture using the corrected archived source.
