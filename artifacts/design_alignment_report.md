# Generative Grammar Design Alignment

Date: 2026-02-20
Project: `projects/generative-grammar`

## What was implemented
- Replaced symbolic text grammar with structured rhythm clips.
- Added 4 controllable families:
  - `straight_rock`
  - `syncopated_funk`
  - `minimal_techno`
  - `jazz_swing`
- Added 2-level hierarchy in generation:
  - motif (1 beat, 4 step)
  - bar (4 motifs)
  - phrase (multiple bars)
- Added structured noise/diversity knobs:
  - swing
  - microtiming jitter
  - push/pull
  - fill probability
  - ghost probability
  - dropout
  - tempo ramp
- Added metadata logging per clip in parquet:
  - family, motif sequence, beat patterns, knob values, timbre profile, accent profile, per-event JSON
- Return metrics for harness thresholds:
  - `h_generate_valid_rate`
  - `h_generate_event_density`
  - `h_generate_event_entropy`
  - `h_generate_motif_reuse`
  - `h_generate_family_coverage`

## Ground-truth vs. model separation
- Latent knobs and family IDs are generated and stored as metadata.
- No training loop or model conditioning logic exists in this project.
- The generator only emits sample data + metadata for downstream experiments.

## Next step
- Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run harness run --project projects/generative-grammar --seed 42
```
