# Generative Grammar

Lightweight rhythm-corpus generator project for test-data synthesis.

## Design

Each generated clip is controlled by a small latent control set that is logged and
held constant per clip, but never provided as training inputs. The generator combines:

- Family-level templates (`straight_rock`, `syncopated_funk`, `minimal_techno`, `jazz_swing`)
- Motif library per family
- Motif sequencing rules (hierarchical structure: motifs -> bars -> phrases)
- Microtiming knobs (swing, jitter, push/pull)
- Accent and density knobs
- Timbre metadata profiles

## Latent knobs per clip

- family
- motif sequence
- tempo start/end
- swing
- microtiming jitter
- push/pull
- density
- fill probability
- instrumental dropout
- accent profile
- timbre profile

## Outputs

For each seed, the run writes:

- `rhythm_samples_seed_<seed>.parquet`

Columns include:
- `seed`, `sample_idx`, `family`, `family_id`
- timing knobs (`tempo_start_bpm`, `tempo_end_bpm`, `swing`, ...)
- `motif_ids`, `beat_patterns`, `accent_profile`, `timbre_profile`
- `events_json` with per-event timing and velocity metadata
- `event_count`, `valid`

## Run

From repo root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run harness run --project projects/generative-grammar --seed 42
```

## License

MIT. See `LICENSE`.
