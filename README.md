# Generative Grammar

A reproducible, grammar-controlled rhythm corpus generator. It creates hierarchical
motif, bar, and phrase metadata plus deterministic WAV previews for test-data synthesis,
benchmarking, and controlled music-model experiments.

## Install and run

Python 3.10 or newer is required.

```bash
pip install .
generative-grammar --seed 42 --config configs/default.yaml --output-dir corpus/seed-42
```

The command writes a versioned Parquet corpus, one mono PCM WAV preview per clip, and
`manifest.json` with the seed, evaluation results, and SHA-256 hashes. The default run
produces 160 four-bar clips. Repeating a run with the same release, configuration, and
seed produces byte-identical artifacts.

The existing research-harness contract remains available:

```python
from experiments.generators import run
metrics = run(seed=42, output_dir=output_dir, config=flattened_config)
```

## Controls and data contract

Controls include rhythm family, motif sequence, tempo/ramp, swing, microtiming jitter,
push/pull, density, fills, dropout, accents, and timbre labels. Every sampled value is
recorded in the Parquet row; latent controls are never used as model inputs here.

`events_json` is an ordered list of events with step-domain time, instrument, velocity,
timbre, and instantaneous tempo. Parquet schema metadata contains `schema_version=1.0`.
The WAV files are audition previews; Parquet is the canonical dataset. See
[`docs/method.md`](docs/method.md) for generation and evaluation details.

## Development

```bash
uv run --extra test pytest -q
```

Generated corpora are ignored because a default run is about 62 MB. Distribute corpora
through a release or dataset store together with their manifest, not in Git history.

MIT licensed.
