# Storage audit — 2026-11-10

This is a dated measurement, not the storage contract. See `STORAGE.md` for
the authoritative architecture.

## Baseline

- Commit: `38e011e`
- Working tree: clean before implementation
- Tests: 254 passed in 66.19 seconds
- Working project folders: 11
- Parse-history events: 33
- Fragment folders: 4

## Measured storage

- Git-tracked files: about 13.0 MiB
- Parse-history events: about 11.3 MiB
- Fragment folders: about 119 KiB
- Git object database: about 2.16 MiB before repacking
- Machine-local `.local` caches: about 24.8 MiB
- Byte-identical redundancy inside parse history: about 1.21 MiB
- Byte-identical redundancy inside fragment folders: none measured

The measurements exclude `.venv`, Python bytecode, and the external model
checkpoint. Git already stores byte-identical committed blobs once even
though each checked-out event folder has its own file.

## Findings

1. Local-versus-published divergence is a larger risk than repository size.
2. Existing parse events are small enough that complete, independently
   restorable snapshots remain the safer event model.
3. First-class MIDI projects need `source.mid`, full tempo and meter maps,
   staff assignments, and score provenance in future published snapshots.
4. Fragment artifacts are intentional portable copies. Their SQLite rows,
   generated pages, and preview audio are disposable projections.
5. Recovery copies are durable user data. They are not included in
   reclaimable-cache totals and must never be removed automatically.
6. `pedal.json` appears redundant with pedals in `notes.json`, but removal is
   deferred until every writer, reader, restore, export, test, UI, and
   external workflow has been audited and replacement behavior approved.

## Decision

Revision safety is implemented before storage optimization. Event schema v2
adds lineage, fingerprints, and restore metadata while score-producing events
continue to store complete snapshots. Existing events, fragments, and Git
history are not rewritten. Event-to-event artifact references remain a
separately approved future optimization.

## Revision-safety gate — 2026-11-10

The gate ran on Windows 10 with Python 3.12.10:

- The complete suite passed: 281 tests in 73.33 seconds.
- Focused revision, fragment, and MIDI-import suites passed after the final
  compatibility changes.
- Byte-compilation, `git diff --check`, and configured IDE lints passed.
- A live Streamlit smoke test rendered the ten-project library and the
  legacy unavailable-project state without an exception.
- Git reported no changes under `data/parses/`, `data/fragments/`, or
  `transcriptions/`. Existing event and fragment bytes were not rewritten.

Representative new metadata is deliberately additive. A score event keeps
its complete files beside an `event.json` revision block:

```json
{
  "version": 2,
  "revision": {
    "lineage_id": "...",
    "parent_event_id": "... or null",
    "complete_snapshot": true,
    "snapshot": {
      "sha256": "...",
      "project_state_sha256": "...",
      "artifacts": {"notes.json": "canonical component hash ..."},
      "files": {"notes.json": "raw file SHA-256 ..."}
    }
  }
}
```

The corresponding project manifest records only its causal base in
`revision` (`lineage_id`, `base_event_id`, and `base_snapshot_sha256`) and
the hashes from its last successful output generation in `generation`.
Current state, published heads, and conflict status are derived when read.

A post-gate read-only audit produced additional adversarial cases, all
covered before closure:

- event/project/artifact names are plain and resolved paths must remain
  inside their storage roots;
- successful score actions cannot publish or advance a non-restorable event;
- semantic MusicXML fingerprints ignore volatile IDs/dates while independent
  raw hashes still detect any byte change;
- attached video does not replace local-audio or MIDI source identity, and
  local audio is retained only when source hashes match;
- analysis overlays require an unchanged causal score; v1 overlays and
  edited-v1 status remain compatible;
- independent same-name lineages conflict, and an interrupted revision-marker
  write is recoverable through exact adoption;
- imported MIDI staff survives literal fragments, and staged fragments are
  read back before publication; and
- verified recovery rollback is explicit, retains the selected recovery, and
  safeguards the state it replaces.

The remaining risks are explicit user choice around destructive updates,
external/manual reliance on `pedal.json`, and growth of complete snapshots.
Updates therefore require preview plus confirmation and leave a permanent
verified recovery. `pedal.json` remains part of every relevant path.
Snapshot deduplication remains deferred because the measured saving is only
about 1.21 MiB and independently restorable events are the safer contract.

## Post-gate optimization audit (proposal only)

No candidate below was removed, relocated, or cleaned by the revision-safety
implementation.

### `pedal.json` dependency inventory

- Writers: initial transcription and requantization in `pipeline.py`; MIDI
  import in `midi_import.py`; shared shape in `schema.write_pedal_file`.
- Musical authority: the same pedal intervals also remain in `notes.json`;
  `raw.mid` carries them as CC 64. No internal musical calculation currently
  reads `pedal.json` back.
- Storage/portability consumers: event publication and format metadata in
  `parse_store.py`; canonical inventories in `revision.py`; complete-snapshot
  validation, materialization, generation hashes, recovery, and restore in
  `project_restore.py` and `pipeline.py`.
- User/export consumers: the Files-tab download in `ui.py`; project ZIP
  export includes it; event folders expose it as an ordinary standalone file.
- Tests: parse-store, CLI/tempo, and revision-safety fixtures assert its
  writing, publication, complete-event presence, and restoration.
- Documentation/external boundary: `README.md`, `STORAGE.md`, and `HANDOFF.md`
  promise the standalone file. External scripts may consume it even though
  this repository does not read it musically.
- Measured tracked footprint on 2026-11-10: 23 files, 90,621 bytes.

Conclusion: stopping future writes is **not approved**. The saving is small,
and internal plus documented export/restore contracts currently require the
file. A future proposal would need on-demand synthesis/export parity,
complete-event compatibility for both schemas, external-workflow approval,
and a forward-only policy that never deletes existing copies.

### Other candidate classifications

- Manifest counts, analysis summaries, fixed artifact names, and
  `constant_tempo` are small projections. Removing them needs consumer-by-
  consumer equivalence tests and old-manifest support.
- Project playback HTML/OSMD, notation diagnostics, diagnostics, comparisons,
  MIDI upload staging, previews, pages, thumbnails, and temporary folders are
  generated candidates. Some are useful user outputs, so there is no default
  deletion scope.
- `.local/` fragment pages/audio and the SQLite fragment index, plus
  `data/thumbnails/`, are rebuildable caches. They remain the strongest
  ordinary-cleanup candidates.
- `notes.json`, `annotations.json`, source audio, `source.mid`, project
  manifests, complete parse events, every self-contained fragment artifact,
  and all recovery copies are authority/portable data and are ineligible.
- Diagnostics/comparisons may only be removed by an explicit selected scope.
  Recovery removal remains its own itemized command with exact confirmation.

The recommended future approval package is conservative cache cleanup plus
optional page relocation. It excludes `pedal.json`, event artifact
deduplication, diagnostics/comparisons by default, and every authority path.
