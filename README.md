# Generative Grammar

Test-data generator project for the research harness.

This project contains synthetic sequence generators driven by simple formal grammars.
The goal is to produce reproducible datasets for downstream experiments.

## Structure

- `project.yaml`: Harness project manifest
- `configs/default.yaml`: Default experiment configuration
- `experiments/generators.py`: Generator implementations and harness entrypoint
- `thresholds/generators.yaml`: Optional threshold checks
- `artifacts/`: Folder for generated local results

## Run

From repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run harness run \
  --project projects/generative-grammar \
  --seed 42
```

## License

MIT License (`LICENSE`).
