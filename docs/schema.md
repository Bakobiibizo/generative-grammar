# Corpus schema 1.0

Each Parquet row describes one clip. Scalar columns record `seed`, `sample_idx`, family
identity, tempo, timing controls, density/fill controls, motif reuse, event count, and
validity. JSON string columns contain motif IDs, bar-grouped motif patterns, instrument
accent and timbre maps, and ordered events. JSON is used deliberately so downstream
Arrow readers receive a stable flat table across languages.

Required event fields are `beat`, `time`, `instrument`, `velocity`, `timbre`, and
`tempo_bpm`. `instrument` is one of `kick`, `snare`, `hat`, or `clap`; velocity is in
`[0.05, 1.0]`. `time` is measured in sixteenth-note steps after microtiming controls.

Consumers must reject unsupported major schema versions and may accept new columns in
minor versions. A clip is valid when it contains at least one event.
