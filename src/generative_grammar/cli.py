from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from experiments.generators import flatten_config, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic rhythm corpus")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    metrics = run(args.seed, args.output_dir, flatten_config(raw))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
