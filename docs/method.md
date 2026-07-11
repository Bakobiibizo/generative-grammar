# Method and evaluation

The generator samples a family, builds a seeded motif library, sequences motifs into
bars and phrases, then applies logged tempo, swing, jitter, density, dropout, accent,
and timbre controls. A clip seed is derived from the run seed and sample index, so a
given version, seed, and configuration produces identical metadata and audition audio.

Parquet is the canonical corpus. `events_json` contains step-domain event time,
instrument, velocity, timbre, and instantaneous tempo. WAV files are deliberately
simple synthesized previews, not production drum rendering or training targets.

`manifest.json` records schema version, seed, file SHA-256 hashes, metrics, and pass/fail
results. The checked-in thresholds require 99% non-empty clips, mean event density of
0.04, mean instrument entropy of 1.4 bits, motif reuse of 0.12, and 75% family coverage.
Changing the algorithm or schema requires a version change and regenerated manifests.

Generated corpora are intentionally ignored by Git. Publish large corpora through a
release or dataset store with their manifest; never commit WAV/Parquet output.
