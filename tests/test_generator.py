import hashlib
import json
import wave

import pyarrow.parquet as pq
import pytest

from experiments.generators import _parse_config, flatten_config, run


SMALL = {
    "grammar.sample_count": "8",
    "grammar.phrase_bars": "1",
    "grammar.motifs_per_family": "4",
    "grammar.render_audio": "true",
}


def test_reproducible_corpus_and_valid_audio(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    metrics = run(17, first, SMALL)
    run(17, second, SMALL)
    parquet = "rhythm_samples_seed_17.parquet"
    assert hashlib.sha256((first / parquet).read_bytes()).digest() == hashlib.sha256((second / parquet).read_bytes()).digest()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["seed"] == 17 and manifest["schema_version"] == "1.0"
    assert metrics["thresholds_passed"] is True
    table = pq.read_table(first / parquet)
    assert table.schema.metadata[b"schema_version"] == b"1.0"
    for relative in manifest["audio_files"]:
        with wave.open(str(first / relative)) as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 22050
            assert wav.getnframes() > 0


def test_nested_config_flattening():
    assert flatten_config({"grammar": {"sample_count": 3}}) == {"grammar.sample_count": "3"}


def test_invalid_controls_fail_fast():
    with pytest.raises(ValueError, match="sample_count"):
        _parse_config({"grammar.sample_count": "0"})
    with pytest.raises(ValueError, match="tempo"):
        _parse_config({"grammar.tempo_min": "140", "grammar.tempo_max": "80"})
