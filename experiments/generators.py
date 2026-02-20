"""Hierarchical rhythm generators for synthetic test data.

The generator combines: family-level rhythmic templates, motif sequencing,
microtiming controls, and composition-level knobs. All knobs are logged in
metadata and stored with each clip, but they are not fed back into any model
inside this project.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


INSTRUMENTS = ("kick", "snare", "hat", "clap")
STEP_PER_BEAT = 4
BEATS_PER_BAR = 4


@dataclass(frozen=True)
class _ClipConfig:
    seed_list: list[int]
    families: int
    motifs_per_family: int
    sample_count: int
    phrase_bars: int
    tempo_min: float
    tempo_max: float
    tempo_ramp_max: float
    swing_min: float
    swing_max: float
    jitter_ms_min: float
    jitter_ms_max: float
    push_pull_min: float
    push_pull_max: float
    density_min: float
    density_max: float
    ghost_probability: float
    fill_probability_min: float
    fill_probability_max: float
    dropout_min: float
    dropout_max: float
    instrument_timbre_variants: int


@dataclass(frozen=True)
class _FamilySpec:
    name: str
    tempo_preference: tuple[float, float]
    motif_templates: list[dict[str, list[int]]]
    motif_transition_strength: list[float]


_FAMILY_DEFINITIONS: dict[str, _FamilySpec] = {
    "straight_rock": _FamilySpec(
        name="straight_rock",
        tempo_preference=(86.0, 112.0),
        motif_templates=[
            {"kick": [0], "snare": [2], "hat": [0, 1, 2, 3], "clap": []},
            {"kick": [0, 2], "snare": [2], "hat": [1, 3], "clap": []},
            {"kick": [0], "snare": [2], "hat": [0, 2], "clap": [1]},
            {"kick": [1], "snare": [2], "hat": [0, 1, 2, 3], "clap": []},
            {"kick": [0, 3], "snare": [1, 3], "hat": [0, 2], "clap": []},
        ],
        motif_transition_strength=[1.2, 1.0, 0.9, 1.1],
    ),
    "syncopated_funk": _FamilySpec(
        name="syncopated_funk",
        tempo_preference=(96.0, 126.0),
        motif_templates=[
            {"kick": [1], "snare": [2], "hat": [0, 2], "clap": [2]},
            {"kick": [1, 2], "snare": [2], "hat": [1, 3], "clap": []},
            {"kick": [0], "snare": [1, 2], "hat": [0, 1, 3], "clap": [3]},
            {"kick": [2], "snare": [2], "hat": [1, 2, 3], "clap": []},
            {"kick": [1, 3], "snare": [0, 2], "hat": [0, 2], "clap": [1]},
        ],
        motif_transition_strength=[1.0, 1.2, 1.1, 1.0],
    ),
    "minimal_techno": _FamilySpec(
        name="minimal_techno",
        tempo_preference=(110.0, 136.0),
        motif_templates=[
            {"kick": [0], "snare": [0], "hat": [0, 1, 2, 3], "clap": []},
            {"kick": [0], "snare": [1], "hat": [0, 1, 2, 3], "clap": []},
            {"kick": [0], "snare": [2], "hat": [1, 3], "clap": []},
            {"kick": [0, 2], "snare": [3], "hat": [0, 2], "clap": []},
            {"kick": [0], "snare": [], "hat": [0, 1, 3], "clap": [2]},
        ],
        motif_transition_strength=[1.3, 1.0, 1.0, 0.9],
    ),
    "jazz_swing": _FamilySpec(
        name="jazz_swing",
        tempo_preference=(84.0, 110.0),
        motif_templates=[
            {"kick": [1], "snare": [1, 3], "hat": [0, 2], "clap": []},
            {"kick": [0, 2], "snare": [2], "hat": [0, 3], "clap": [1]},
            {"kick": [0], "snare": [1], "hat": [1, 2], "clap": []},
            {"kick": [2], "snare": [3], "hat": [0, 2], "clap": [2]},
            {"kick": [0, 3], "snare": [1, 2], "hat": [1, 3], "clap": []},
        ],
        motif_transition_strength=[1.1, 1.0, 1.2, 0.9],
    ),
}



def _coerce_scalar(value: object, caster: type, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, caster):
        return value
    try:
        if caster is bool:
            return str(value).lower() in {"1", "true", "yes", "on"}
        if caster in (int, float):
            return caster(value)
        if caster is str:
            return str(value)
    except Exception:
        return default
    return default


def _parse_config(config: dict[str, str] | None) -> _ClipConfig:
    cfg = config or {}
    get = cfg.get

    seed_list = [
        int(v)
        for v in str(get("experiment.seed_list", "42,123,321,777,999")).split(",")
        if str(v).strip()
    ]

    return _ClipConfig(
        seed_list=seed_list,
        families=_coerce_scalar(get("grammar.families"), int, 4),
        motifs_per_family=_coerce_scalar(get("grammar.motifs_per_family"), int, 10),
        sample_count=_coerce_scalar(get("grammar.sample_count"), int, 160),
        phrase_bars=_coerce_scalar(get("grammar.phrase_bars"), int, 4),
        tempo_min=_coerce_scalar(get("grammar.tempo_min"), float, 82.0),
        tempo_max=_coerce_scalar(get("grammar.tempo_max"), float, 138.0),
        tempo_ramp_max=_coerce_scalar(get("grammar.tempo_ramp_max"), float, 0.08),
        swing_min=_coerce_scalar(get("grammar.swing_min"), float, 0.0),
        swing_max=_coerce_scalar(get("grammar.swing_max"), float, 0.3),
        jitter_ms_min=_coerce_scalar(get("grammar.jitter_ms_min"), float, 1.0),
        jitter_ms_max=_coerce_scalar(get("grammar.jitter_ms_max"), float, 12.0),
        push_pull_min=_coerce_scalar(get("grammar.push_pull_min"), float, -0.03),
        push_pull_max=_coerce_scalar(get("grammar.push_pull_max"), float, 0.03),
        density_min=_coerce_scalar(get("grammar.density_min"), float, 0.25),
        density_max=_coerce_scalar(get("grammar.density_max"), float, 0.75),
        ghost_probability=_coerce_scalar(get("grammar.ghost_probability"), float, 0.35),
        fill_probability_min=_coerce_scalar(get("grammar.fill_probability_min"), float, 0.05),
        fill_probability_max=_coerce_scalar(get("grammar.fill_probability_max"), float, 0.28),
        dropout_min=_coerce_scalar(get("grammar.dropout_min"), float, 0.0),
        dropout_max=_coerce_scalar(get("grammar.dropout_max"), float, 0.2),
        instrument_timbre_variants=_coerce_scalar(
            get("grammar.instrument_timbre_variants"), int, 3
        ),
    )


def _normalize_motif(motif: dict[str, list[int]]) -> dict[str, list[int]]:
    return {
        instrument: [1 if step in set(motif.get(instrument, [])) else 0 for step in range(STEP_PER_BEAT)]
        for instrument in INSTRUMENTS
    }


def _build_family_motifs(
    rng: random.Random, family: _FamilySpec, count: int
) -> list[dict[str, list[int]]]:
    motifs: list[dict[str, list[int]]] = []

    while len(motifs) < count:
        template = rng.choice(family.motif_templates)
        candidate = {
            instrument: list(template.get(instrument, [])) for instrument in INSTRUMENTS
        }

        for instrument in INSTRUMENTS:
            current = set(candidate.get(instrument, []))
            for step in range(STEP_PER_BEAT):
                if rng.random() < 0.18:
                    if step in current:
                        current.remove(step)
                    else:
                        current.add(step)
            candidate[instrument] = sorted(current)

        normalized = _normalize_motif(candidate)
        if any(any(v) for v in normalized.values()):
            motifs.append(normalized)

    # Deduplicate.
    dedup: list[dict[str, list[int]]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for m in motifs:
        key = tuple(tuple(row) for row in (m[instrument] for instrument in INSTRUMENTS))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(m)
    while len(dedup) < count:
        dedup.append(dedup[-1])

    return dedup[:count]


def _pick_transition_probs(motif_count: int, family: _FamilySpec, prev: int | None) -> list[float]:
    if prev is None or motif_count == 0:
        return [1.0 for _ in range(motif_count)]

    base = family.motif_transition_strength
    probs = []
    for idx in range(motif_count):
        base_weight = base[idx % len(base)]
        if idx == prev:
            base_weight *= 1.35
        else:
            base_weight *= 1.0
        distance = abs(idx - prev)
        proximity = 1.0 / (1.0 + 0.15 * distance)
        probs.append(base_weight * max(0.22, proximity))

    total = sum(probs)
    if total <= 0:
        return [1.0 for _ in probs]
    return [p / total for p in probs]


def _select_motif(rng: random.Random, family: _FamilySpec, motifs: list[dict[str, list[int]]], prev: int | None) -> int:
    probs = _pick_transition_probs(len(motifs), family, prev)
    cut = rng.random()
    acc = 0.0
    for idx, p in enumerate(probs):
        acc += p
        if cut <= acc:
            return idx
    return len(motifs) - 1


def _timing_with_swing_and_jitter(
    step: int,
    global_step: int,
    swing: float,
    jitter_ms: float,
    push_pull: float,
    rng: random.Random,
) -> float:
    base = float(global_step)
    if step % 2 == 1:
        base += swing * 0.5
    jitter = rng.uniform(-jitter_ms, jitter_ms) / 1000.0
    return base + push_pull + jitter


def _make_clip(
    rng: random.Random,
    family: _FamilySpec,
    cfg: _ClipConfig,
    motif_library: list[dict[str, list[int]]],
    sample_index: int,
) -> dict[str, Any]:
    # Tempo controls.
    tempo_min, tempo_max = family.tempo_preference
    tempo_start = rng.uniform(max(cfg.tempo_min, tempo_min), min(cfg.tempo_max, tempo_max))
    tempo_end = tempo_start * (1.0 + rng.uniform(-cfg.tempo_ramp_max, cfg.tempo_ramp_max))
    tempo_ramp = (tempo_end - tempo_start) / max(1, cfg.phrase_bars - 1)

    # Control knobs.
    swing = rng.uniform(cfg.swing_min, cfg.swing_max)
    jitter_ms = rng.uniform(cfg.jitter_ms_min, cfg.jitter_ms_max)
    push_pull = rng.uniform(cfg.push_pull_min, cfg.push_pull_max)
    density = rng.uniform(cfg.density_min, cfg.density_max)
    fill_probability = rng.uniform(cfg.fill_probability_min, cfg.fill_probability_max)
    dropout = rng.uniform(cfg.dropout_min, cfg.dropout_max)
    ghost = rng.uniform(0.0, 1.0) < cfg.ghost_probability

    accent_profile = {
        "kick": [1.0, 0.92, 1.04, 0.88],
        "snare": [0.9, 1.03, 0.96, 1.06],
        "hat": [1.0, 1.02, 0.97, 1.01],
        "clap": [0.94, 1.07, 0.98, 1.00],
    }
    for key in accent_profile:
        accent_profile[key] = [val * rng.uniform(0.85, 1.15) for val in accent_profile[key]]

    timbre_profile = {
        inst: f"{inst}_{rng.randrange(cfg.instrument_timbre_variants)}" for inst in INSTRUMENTS
    }

    events: list[dict[str, Any]] = []
    motif_ids: list[int] = []
    beat_patterns: list[list[int]] = []
    prev_motif: int | None = None

    bars = max(1, cfg.phrase_bars)
    total_beats = bars * BEATS_PER_BAR

    for beat in range(total_beats):
        motif_idx = _select_motif(rng, family, motif_library, prev_motif)
        prev_motif = motif_idx
        motif_ids.append(motif_idx)

        if beat % BEATS_PER_BAR == 0:
            beat_patterns.append([])
        beat_patterns[-1].append(motif_idx)

        motif = motif_library[motif_idx]

        for step in range(STEP_PER_BEAT):
            global_step = beat * STEP_PER_BEAT + step
            time_step = _timing_with_swing_and_jitter(
                step,
                global_step,
                swing,
                jitter_ms,
                push_pull,
                rng,
            )
            for instrument in INSTRUMENTS:
                if rng.random() < dropout:
                    continue

                hit = motif[instrument][step] == 1 and rng.random() < density
                if instrument == "snare" and ghost and rng.random() < 0.12:
                    hit = True
                if instrument in {"hat", "clap"} and rng.random() < fill_probability:
                    hit = hit or (rng.random() < 0.15)

                if not hit:
                    continue

                # Keep surface stats overlapping across families.
                accent = accent_profile[instrument][beat % BEATS_PER_BAR]
                velocity = max(
                    0.05,
                    min(
                        1.0,
                        rng.normalvariate(0.78, 0.08) * accent,
                    ),
                )
                tempo_now = tempo_start + (tempo_end - tempo_start) * (global_step / max(1, total_beats * STEP_PER_BEAT - 1))

                events.append(
                    {
                        "beat": float(global_step),
                        "time": float(time_step),
                        "instrument": instrument,
                        "velocity": float(velocity),
                        "timbre": timbre_profile[instrument],
                        "tempo_bpm": float(tempo_now),
                    }
                )

    # Encourage motif reuse as a compositional signal.
    motif_reuse = 1.0 - (len(set(motif_ids)) / max(1, len(motif_ids)))

    events.sort(key=lambda e: e["time"])

    return {
        "motif_ids": motif_ids,
        "beat_patterns": beat_patterns,
        "events": events,
        "motif_reuse": float(motif_reuse),
        "tempo_start": float(tempo_start),
        "tempo_end": float(tempo_end),
        "tempo_ramp_per_bar": float(tempo_ramp),
        "swing": float(swing),
        "microtiming_jitter_ms": float(jitter_ms),
        "push_pull": float(push_pull),
        "density": float(density),
        "fill_probability": float(fill_probability),
        "accent_profile": accent_profile,
        "timbre_profile": timbre_profile,
    }


def _entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    probs = np.array([count / total for count in counts.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def run(seed: int, output_dir: Path, config: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = _parse_config(config)
    rng = random.Random(seed)

    family_names = list(_FAMILY_DEFINITIONS.keys())[: max(1, min(cfg.families, len(_FAMILY_DEFINITIONS)))]
    families = [_FAMILY_DEFINITIONS[name] for name in family_names]

    motif_libraries: dict[str, list[dict[str, list[int]]]] = {}
    for idx, family in enumerate(families):
        family_rng = random.Random(seed * 9973 + idx * 31 + len(family.name))
        motif_libraries[family.name] = _build_family_motifs(
            family_rng, family, cfg.motifs_per_family
        )

    rows: list[dict[str, Any]] = []
    valid_rates: list[float] = []
    densities: list[float] = []
    entropies: list[float] = []
    motif_reuses: list[float] = []
    family_hits = Counter()

    for sample_idx in range(cfg.sample_count):
        family = rng.choice(families)
        clip_rng = random.Random(seed * 19 + sample_idx)
        family_hits[family.name] += 1

        clip = _make_clip(clip_rng, family, cfg, motif_libraries[family.name], sample_idx)
        events = clip["events"]
        event_count = len(events)
        max_events = (cfg.phrase_bars * BEATS_PER_BAR * STEP_PER_BEAT) * len(INSTRUMENTS)

        density = event_count / max(1.0, max_events)
        valid = event_count > 0
        entropy = _entropy([event["instrument"] for event in events])

        valid_rates.append(1.0 if valid else 0.0)
        densities.append(density)
        entropies.append(entropy)
        motif_reuses.append(clip["motif_reuse"])

        rows.append(
            {
                "seed": seed,
                "sample_idx": sample_idx,
                "family": family.name,
                "family_id": family_names.index(family.name),
                "tempo_start_bpm": clip["tempo_start"],
                "tempo_end_bpm": clip["tempo_end"],
                "tempo_ramp_per_bar": clip["tempo_ramp_per_bar"],
                "swing": clip["swing"],
                "microtiming_jitter_ms": clip["microtiming_jitter_ms"],
                "push_pull": clip["push_pull"],
                "density": clip["density"],
                "fill_probability": clip["fill_probability"],
                "motif_reuse": clip["motif_reuse"],
                "motif_ids": json.dumps(clip["motif_ids"]),
                "beat_patterns": json.dumps(clip["beat_patterns"]),
                "accent_profile": json.dumps(clip["accent_profile"]),
                "timbre_profile": json.dumps(clip["timbre_profile"]),
                "events_json": json.dumps(events),
                "event_count": event_count,
                "valid": valid,
            }
        )

    if not rows:
        return {
            "seed": seed,
            "h_generate_valid_rate": 0.0,
            "h_generate_event_density": 0.0,
            "h_generate_event_entropy": 0.0,
            "h_generate_motif_reuse": 0.0,
            "h_generate_family_coverage": 0.0,
            "h_generate_sample_count": 0,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_dir / f"rhythm_samples_seed_{seed}.parquet")

    valid_rate = float(np.mean(valid_rates))
    density = float(np.mean(densities))
    entropy = float(np.mean(entropies))
    motif_reuse = float(np.mean(motif_reuses))
    family_coverage = len(family_hits) / max(1.0, len(families))

    return {
        "seed": seed,
        "h_generate_valid_rate": valid_rate,
        "h_generate_event_density": density,
        "h_generate_event_entropy": entropy,
        "h_generate_motif_reuse": motif_reuse,
        "h_generate_family_coverage": family_coverage,
        "h_generate_sample_count": cfg.sample_count,
    }
