# Storage: what lives where

This program does not keep its work in a database. Almost every durable object is a file in a folder. One SQLite file exists, and it is a **search cache** of fragment metadata. It can be deleted. The fragment folders cannot.

If you are looking at `.local/library.sqlite` and wondering why notes, MIDI, and scores are not rows in it: they never were, and they are not supposed to be. That file does not own data. It indexes fields copied out of `fragment.json` so the library page can filter without opening every folder.

The rest of this document is the map of the tree, the JSON contracts, the SQLite schema, and the reasons the system is files-first.

## Design rule

There are four classes of storage. Mixing them up is what makes the tree look like “stray files.”

| Class | Where | Owned by | Survives deleting `.local/` | Travels with `git pull` |
| --- | --- | --- | --- | --- |
| Working project | `transcriptions/<name>/` | this machine | yes | no |
| Published run | `data/parses/events/<id>/` | git | yes | yes |
| Fragment | `data/fragments/<uuid>/` | git | yes | yes |
| Machine cache | `.local/`, `data/thumbnails/`, the model checkpoint | this process / this machine | no (that is the point) | no |

A **working project** is the folder you edit, requantize, and listen to. A **published run** is an immutable snapshot of one command (`transcribe`, `quantize`, `analyze`, …) copied into parse history so another machine can restore it. A **fragment** is a self-contained excerpt of printed measures; it does not need the parent project to open. A **cache** is anything derived that this computer can rebuild.

SQLite sits only in the last class, and only for fragments.

There is no SQLite (or any other database) for transcriptions, parse history, notes, harmony, MIDI, audio, or the home-page library. Those are JSON, MIDI, MusicXML, and WAV on disk. The home page builds its cards by reading `data/parses/events/*/event.json` and the folders under `transcriptions/` (`library.py`).

## Why this is not “everything in a database”

The repository is the sync mechanism between computers. Each clone has its own `transcriptions/`, its own `.venv`, and its own model weights. What must travel is committed as ordinary files so `git pull` / `git push` is the whole publish step.

Putting the library in SQLite would fight that:

1. **Git merges text, not database files.** Two machines adding fragments or parse-history runs produce new folders. Git merges those as new paths. Two machines writing the same `.sqlite` file produce an unmergeable binary blob.
2. **Other programs consume files.** MuseScore opens `score.musicxml`. A DAW imports `raw.mid` / `performance.mid`. `notes.json` is meant to be edited by hand and re-quantized. Those are file formats, not table rows.
3. **Audio is large and private.** `source.wav` / `working.wav` stay on the machine that downloaded them. History records that they existed; it does not copy them. A database of the library would still need those files beside it.
4. **History is append-only on purpose.** `event.json` is never rewritten. A later harmony edit is a new `analyze` event. That avoids Git conflicts on a shared record.
5. **The index is disposable.** Search fields are a projection of `fragment.json`. If the SQLite file is missing, wrong-version, or corrupt, the program deletes it and rebuilds from the folders. That is only safe because the folders are the library.

SQLite is used for fragment search because opening every `fragment.json` on every keystroke does not stay cheap, and SQL `LIKE` / `EXISTS` is a convenient way to combine text, tags, key, source, and a chord-sequence needle. That is an index over files, not a store of files.

## Repository tree

Paths are relative to the clone root (the directory that contains `pyproject.toml`).

```text
<repo>/
  src/baroque_transcribe/     program (committed)
  tests/                      tests (committed)
  data/
    parses/events/<event-id>/ committed snapshots of each run
    fragments/<uuid>/         committed fragment folders  ← the fragment library
    thumbnails/               ignored JPEG cache of YouTube stills
  transcriptions/<name>/      ignored working projects
  .local/
    library.sqlite            ignored fragment search index
    fragment_pages/           ignored generated HTML for fragments
  .venv/                      ignored
```

Outside the clone:

```text
%USERPROFILE%\.cache\baroque_transcribe\
  note_F1=0.9677_pedal_F1=0.9186.pth    ByteDance checkpoint (~165 MB)
```

`.gitignore` ignores audio globally (`*.wav`, `*.mp3`, `*.flac`), MIDI, and MusicXML, then **re-includes** the score copies that are supposed to travel:

- `!data/parses/**/*.mid` and `!data/parses/**/*.musicxml`
- `!data/fragments/**/*.mid` and `!data/fragments/**/*.musicxml`

Staging folders `data/fragments/.tmp-*/` are ignored. `.local/` is ignored in full.

## 1. Working project — `transcriptions/<name>/`

This is the live workspace of one transcription. It is gitignored, including its `manifest.json`. It is created by `prepare` / `transcribe`, rebuilt by `quantize` / `analyze`, and restored onto a second machine by `history restore`.

```text
transcriptions/<name>/
  manifest.json              project metadata (tempo, excerpt, sync, warnings)
  source.wav                 downloaded or copied original (may be a section)
  working.wav                mono 16 kHz peak-normalized excerpt; MIDI time 0
  notes.json                 SOURCE OF TRUTH for pitches and times after the model
  pedal.json                 sustain-pedal intervals (also CC 64 in raw.mid)
  raw.mid                    unquantized performance MIDI (120 BPM as a clock)
  quantized.mid              metronomic MIDI on the chosen grid
  score.musicxml             piano MusicXML (two linked staves)
  score.voices.musicxml      readable style only; Color voices sibling
  notation_map.json          readable style only
  preservation.json          readable style only
  annotations.json           harmonic draft (chords, sequences, analyzed key)
  diagnostics/               plots and comparison.wav
  comparison/                side-by-side notation experiments (does not replace the score)
```

### `notes.json` (schema version 1)

Defined in `src/baroque_transcribe/schema.py`. After the model runs, this file is the editable source of truth. Quantization, MusicXML, diagnostics, and harmonic analysis all read it. They do not go back to `raw.mid`. Hand-edit this file and run `quantize` to correct a transcription.

```json
{
  "version": 1,
  "model": "bytedance",
  "source": "https://www.youtube.com/watch?v=…",
  "notes": [{ "pitch": 67, "onset": 1.204, "offset": 1.551, "velocity": 84 }],
  "pedals": [{ "onset": 0.0, "offset": 4.2 }]
}
```

Times are seconds on the **working-audio** clock (the start of `working.wav`). If `--start` was used, that is the excerpt, not the start of the YouTube video.

### `manifest.json` (project)

Written by `pipeline.py`. This is not the fragment manifest. Typical keys:

| Key | Role |
| --- | --- |
| `origin`, `title`, `youtube_url` | where the audio came from; YouTube URL can be attached later |
| `source_kind` | `"midi"` for MIDI imports; omitted for transcribed audio |
| `audio.*` | `source.wav` / `working.wav`, sample rate, excerpt start/end, durations; MIDI imports set duration from the file end time |
| `model`, `device`, `thresholds` | how notes were produced (`midi-import` for MIDI) |
| `counts` | note and pedal counts |
| `tempo` | bpm, downbeat, subdivision, time signature, grid mode, estimate; MIDI imports also store full `tempo_map` and `time_signature_map` |
| `staff_split`, `staff_assignment`, `notation` | staff split MIDI pitch; `tracks` vs `pitch_split`; style and settings used to write the score |
| `warnings` | assumptions (e.g. 4/4 placeholder) and failures that did not abort the run |
| `sync_offset_seconds` | ear correction on the synced page; survives a later `quantize` |
| `analysis` | summary pointing at `annotations.json` when harmony exists |

Nothing in this file is a path into SQLite. `sync_offset_seconds` is a **linear** offset when mapping the canonical note timeline to a YouTube player. It does not provide nonlinear alignment with an expressive performance; that remains out of scope.

### MIDI import projects

Importing a `.mid` / `.midi` file creates a normal project under `transcriptions/` with `notes.json` times taken from pretty_midi absolute seconds (tempo map applied). Multi-track piano files keep track→staff layout in `score.musicxml` (and optional per-note `staff` in `notes.json`). Playback without YouTube uses a synthesized preview WAV under `.local/project_audio/`, or a silent scrubber if synthesis fails. Attaching a YouTube URL later switches the shared player to the video clock; only `sync_offset_seconds` aligns the two.

### `annotations.json` (schema version 1)

Defined in `src/baroque_transcribe/analysis.py`. A draft harmonic reading of the **quantized grid**, not of the counterpoint. One global key. Chord labels (Roman numeral + lead-sheet symbol) and sequence spans. User edits in the Harmony tab are stored here (`edited`, `edited_fields`) and preserved on a normal rebuild.

MusicXML does not embed these chords. The browser score pages draw them from this file.

### What a project is not

- It is not the fragment library. Saving a passage copies a slice into `data/fragments/`; the project is only read.
- It is not the published history. History *copies* `notes.json` and the score files out of it into a new event folder.
- It is not searchable via `.local/library.sqlite`.

## 2. Parse history — `data/parses/events/<event-id>/`

Defined in `src/baroque_transcribe/parse_store.py`. One new folder per successful or failed `prepare` / `transcribe` / `quantize` / `analyze` (and later `pdf`). The folder name is the event id. `event.json` is written once and never updated.

This is how a second computer learns that a transcription exists without receiving `transcriptions/` or the WAV files.

```text
data/parses/events/<event-id>/
  event.json                 immutable record of the run
  notes.json                 copy, when the run produced notes
  pedal.json
  raw.mid
  quantized.mid
  score.musicxml
  annotations.json           copy, when the run produced or edited harmony
```

`score.pdf` is reserved (`"present": false`, `"status": "planned"`). Audio is **described** in `event.json` (`stored: false`) and left in `transcriptions/`.

`event.json` (schema version 1) includes: `id`, `recorded_at`, `machine`, `action`, `status`, `project` (the working-folder name), `source` (`kind` youtube/file/url, url, video_id, title), `excerpt`, `audio` (file names and durations, not bytes), `formats` (which score files were copied), `error`.

The home-page library (`library.py`) merges these events with whatever project folders exist on this computer. A run named in history but missing as a folder is shown greyed out (“not on this computer”). `history restore` copies the stored score files into a new `transcriptions/<project>/` and rebuilds a `manifest.json` from the event. It does not run the model. Audio is still missing unless you pass `--with-audio`.

There is no index database for parse history. Listing history is “read every `event.json`.”

## 3. Fragment — `data/fragments/<uuid>/`

Defined in `src/baroque_transcribe/fragments.py`. A fragment is a run of **printed measures** taken from a project and stored as its own folder. The parent is never modified. The folder is named after a new UUID. Extraction writes into `data/fragments/.tmp-<uuid>/` and then `os.replace`s that directory into place, so a reader never sees a half-written fragment. A folder is a fragment if and only if it contains `fragment.json`.

```text
data/fragments/<uuid>/
  fragment.json              AUTHORITATIVE manifest (metadata, clocks, pointers)
  notes.json                 rebased performance notes + pedal (working clock → fragment clock)
  grid.json                  sliced parent grid (steps from 0; not a second quantization)
  performance.mid            as-played MIDI on the fragment clock
  quantized.mid              metronomic MIDI on the sliced grid
  score.musicxml             MusicXML of the selected measures
  harmony.json               optional sliced HarmonicAnalysis + provenance
```

No WAV in the fragment folder. Audio is never copied into a fragment. The library **Preview** control synthesizes a short tone WAV from `notes.json` or the grid into `.local/fragment_audio/` (derived, deletable). A YouTube fragment can still open on its own page with the video at the stored timestamp. A local-file fragment has the score and downloads only.

### `fragment.json` (schema version 1)

Built by `build_manifest()`. Validated by `validate_manifest()`. This is the document Git tracks for title, tags, clocks, and sync. Edits from the fragment details form and a sync nudge on the fragment page write **this file**, then the index is refreshed from it.

User-editable after extraction: `title`, `description`, `practice_notes`, `tags`, `composer`, `performer`, `sync.offset_seconds`. Everything else is written once.

| Block | What it records |
| --- | --- |
| `schema_version`, `id` | version 1; UUID, also the folder name |
| `title`, `description`, `practice_notes`, `tags`, `composer`, `performer` | library metadata |
| `source` | parent’s source: `kind`, `title`, `url`, `video_id` (or local `name`) |
| `parent` | project folder name, latest parse-history event id, notes fingerprint, analysis hash, measure/step span |
| `time.working_audio` | window on the parent’s working-audio clock (`t0` … `t1`) |
| `time.source_video` | same window on the source-video clock, plus the parent’s excerpt start `E` |
| `time.duration_seconds` | `t1 - t0`; not changed by a later sync nudge |
| `sync.offset_seconds` | fragment’s own ear correction (copied from the parent at extraction, then independent) |
| `sync.parent_offset_at_extraction` | snapshot of the parent’s offset; not updated later |
| `music` | analyzed key (tonic/mode/source/confidence), written key signature, meter, bpm, subdivision, staff split, measure count, parent grid mode, notation style |
| `harmony` | Roman numerals and chord symbols as parallel lists (searchable projection of `harmony.json`) |
| `boundary` | owned / carried-in / cut-off / unprinted-overlap counts; pedal already down |
| `artifacts` | **relative file names only** (`notes.json`, `grid.json`, …). No absolute paths, no machine name |
| `warnings`, `created_at`, `modified_at` | extraction notes and UTC timestamps |

`artifacts` is how the program finds sibling files. It is a map of roles to names in the same folder, not a database foreign key.

### `notes.json` and `grid.json` inside a fragment

Same `notes.json` shape as a project, with onsets/offsets in **fragment seconds** (zero at the playback-window start). `grid.json` (schema version 1) is the notation record: bpm, meter, subdivision, barline times, sliced `GridNote`s (integer steps minus the first selected step), and measure anchors. The notes are never quantized a second time.

### Clocks (why four numbers exist)

`V` source-video seconds, `W` parent working-audio seconds, `F` fragment seconds, `G` grid steps. With `E` the parent excerpt start and `t0` the window start in `W`:

```text
F = W - t0
V = F + (E + t0) + sync.offset_seconds
```

`time.source_video.start_seconds` is `E + t0`. It does **not** include the sync offset. Nudging sync writes only `sync.offset_seconds` and must not change duration or the stored interval.

### What a fragment is not

- It is not a row in SQLite. The row is a copy of some of these fields.
- It is not a pointer into `transcriptions/`. The parent name is recorded so “Open source transcription” can find a local project when one exists; opening the fragment does not need that folder.
- It does not hold generated HTML. Those pages are rebuilt under `.local/fragment_pages/`.

## 4. Machine cache — `.local/`, thumbnails, checkpoint

### `.local/library.sqlite`

The only database. Module: `src/baroque_transcribe/fragment_index.py`. Schema version: `PRAGMA user_version = 1` (`INDEX_SCHEMA_VERSION`). Gitignored. Not shared between machines.

**Invariant:** every column on a fragment row is derived from that fragment’s `fragment.json`. Nothing is stored here that is not stored there, except:

- `rel_dir` — which folder the id was read from (usually the UUID)
- `manifest_sha256` — hash of the canonicalized manifest, so a later pass can see that content changed
- the `folders` table — filesystem snapshot (mtime, size) plus last error, so an incremental reconcile can skip unchanged files and the UI can show problems without scanning again

Those three are bookkeeping for the index, not musical data.

If the file is missing, the schema version does not match, or SQLite reports it unreadable, the program creates a new empty file and fills it from the folders. `fragments reindex --rebuild` does the same on purpose.

### SQLite schema (live)

```sql
CREATE TABLE fragments (
    id TEXT PRIMARY KEY,                 -- fragment.json "id"
    rel_dir TEXT NOT NULL UNIQUE,        -- folder name under data/fragments/
    manifest_sha256 TEXT NOT NULL,       -- index bookkeeping
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    practice_notes TEXT NOT NULL,
    composer TEXT NOT NULL,
    performer TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    video_id TEXT NOT NULL,
    parent_project TEXT NOT NULL,
    parent_event_id TEXT NOT NULL,
    video_start REAL NOT NULL,           -- time.source_video.start_seconds
    video_end REAL NOT NULL,
    working_start REAL NOT NULL,         -- time.working_audio.start_seconds
    working_end REAL NOT NULL,
    duration REAL NOT NULL,
    sync_offset REAL NOT NULL,           -- sync.offset_seconds
    analyzed_key_tonic TEXT NOT NULL,    -- music.analyzed_key.tonic
    analyzed_key_mode TEXT NOT NULL,
    time_signature TEXT NOT NULL,
    bpm REAL NOT NULL,
    measures INTEGER NOT NULL,
    measure_start INTEGER NOT NULL,      -- parent.measure_start
    measure_end INTEGER NOT NULL,
    note_count INTEGER NOT NULL,         -- boundary.notes
    harmony_roman TEXT NOT NULL,         -- " iv6 V7 i " (collapsed repeats)
    harmony_symbols TEXT NOT NULL,
    created_at TEXT NOT NULL,
    modified_at TEXT NOT NULL
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE
);

CREATE TABLE fragment_tags (
    fragment_id TEXT NOT NULL REFERENCES fragments(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    name TEXT NOT NULL,                  -- spelling from the winning fragment
    position INTEGER NOT NULL,
    PRIMARY KEY (fragment_id, tag_id)
);

CREATE TABLE folders (
    rel_dir TEXT PRIMARY KEY,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    fragment_id TEXT,                    -- NULL if this folder was not indexed
    invalid TEXT,
    problem TEXT
);
```

Search (`search_fragments`) is SQL over this projection: word `LIKE` on title/description/practice notes/composer/performer/source title/tags; required tags; analyzed key; a contiguous chord-sequence needle in `harmony_roman` or `harmony_symbols`; source as video id or parent project name.

What is **not** in SQLite, by design:

| Lives in the fragment folder | Why it is not a column |
| --- | --- |
| `notes.json` (every pitch, onset, offset, velocity) | large; opened when you play or export, not when you search |
| `grid.json` (every grid note, anchors) | same |
| MIDI and MusicXML bytes | binary files for other programs |
| `harmony.json` (per-chord inversion, confidence, provenance) | the index keeps only the label lists for sequence search |
| `boundary` detail beyond `note_count` | shown on the extract form from a live `plan_fragment` / the manifest |
| `parent.notes_fingerprint`, `step_start` / `step_end` | provenance; not a search filter in v1 |
| `written_key_signature` | separate from analyzed key; filter uses analyzed key |
| `artifacts` map | the files are found by well-known names in the same folder |
| `warnings` | displayed from the manifest on the details page |

Opening a fragment always reads the folder (`read_manifest`, `read_grid`, `read_harmony`). The SQLite row is not used as the document.

### How the index is kept in agreement

Reconcile looks at the folders as they are **now**, never at what the index used to mean:

- A folder whose `fragment.json` cannot be read or is invalid is **not** indexed. Its stale row is deleted. The problem is reported (`IndexProblem`).
- Duplicate ids: the folder named after the id wins, else the first by name. The others are reported.
- Every other valid folder is indexed.

Three modes apply the same rule:

| Mode | When | What it reads |
| --- | --- | --- |
| `incremental` | library page, throttled to once per 10 s per process | a manifest only if mtime or size changed |
| `full` | UI process start (`prepare_index`), “Check for changes”, `fragments reindex` | every manifest, hashed |
| `rebuild` | “Rebuild library index”, `fragments reindex --rebuild` | delete the file, then full |

After save or metadata/sync edit, `refresh_folder` updates that one row. Incremental mode will not notice a file whose content changed but mtime and size did not; full / rebuild / `refresh_folder` will.

Tag spelling is canonicalized from the oldest fragment that uses the tag (`created_at`, then id).

### `.local/fragment_pages/`

Generated `*.playback.html` and `*.notation.html`, plus one copy of the OSMD script. Rebuilt whenever the fragment is opened. Not committed. Deleting the directory only forces a rewrite.

Project playback pages are also generated on demand by `PlaybackServer` (a localhost `ThreadingHTTPServer` on a port chosen at launch). Tokens in `/p/<token>/` and `/f/<token>/` live **in memory for that process**. They are not in SQLite and they are not stable URLs.

### `.local/fragment_audio/`

Tone renderings for the fragment library **Preview** control. Named `<id>-<performance|quantized>-<hash>.wav` from `notes.json` or `grid.json`. Not committed. Deleting the directory only forces a rewrite the next time Preview is used.

### Other caches (not SQLite)

| Location | What |
| --- | --- |
| `data/thumbnails/` | YouTube `hqdefault.jpg` copies for the home-page grid; re-fetched if missing |
| `%USERPROFILE%\.cache\baroque_transcribe\` | neural-net checkpoint |
| Browser `localStorage` | UI toggles only: `baroque-transcribe.color-voices`, note-highlight preference. Not musical data |

## What to open for a given job

| Job | Read | Write |
| --- | --- | --- |
| Correct pitches or timing | `transcriptions/<name>/notes.json` | same, then `quantize` |
| Change tempo / meter / style | project `manifest.json` + `notes.json` | `quantize` rewrites MIDI, MusicXML, manifest tempo block |
| Edit harmony labels | `annotations.json` | Harmony tab or `analyze`; new parse-history event |
| Nudge video sync on a project | project `manifest.json` → `sync_offset_seconds` | same file |
| Nudge video sync on a fragment | `data/fragments/<id>/fragment.json` → `sync.offset_seconds` | same file; index row refreshed |
| Save a passage | parent project (read-only) | new folder under `data/fragments/` |
| Search the fragment library | `.local/library.sqlite` | nothing (reconcile may rewrite the db) |
| Open / play / download a fragment | the fragment folder | generated pages under `.local/fragment_pages/` |
| Publish a transcription to the other computer | copy score files into a new `data/parses/events/` folder (the pipeline already does this) | `git add data/parses` |
| Publish fragments | the folders already under `data/fragments/` | `git add data/fragments` |
| Rebuild search after `git pull` | fragment folders | `fragments reindex` or open the UI (full reconcile at start) |

## Two computers, compressed

```text
Machine A transcribes  →  transcriptions/piece/     (local WAV + notes)
                       →  data/parses/events/<id>/  (committed snapshot, no audio)
Machine A extracts     →  data/fragments/<uuid>/    (committed excerpt, no audio)
Machine A searches     →  .local/library.sqlite     (local cache)

git push / pull moves data/parses and data/fragments only.

Machine B after pull   →  history shows the video greyed out
                       →  history restore copies notes/MIDI/MusicXML into transcriptions/
                       →  fragments are already openable (no restore step)
                       →  first UI start rebuilds .local/library.sqlite from those folders
```

Fragments do not go through restore because they already are the portable object. Projects do, because the working folder is gitignored and the audio may still be missing.

## Source code that defines each contract

| Contract | File |
| --- | --- |
| `notes.json` / `pedal.json` | `src/baroque_transcribe/schema.py` |
| Project `manifest.json` | `src/baroque_transcribe/pipeline.py` |
| `annotations.json` | `src/baroque_transcribe/analysis.py` |
| Parse-history `event.json` | `src/baroque_transcribe/parse_store.py` |
| Home-page cards (no database) | `src/baroque_transcribe/library.py` |
| `fragment.json`, `grid.json`, extraction | `src/baroque_transcribe/fragments.py` |
| SQLite index | `src/baroque_transcribe/fragment_index.py` (`_SCHEMA`, `_manifest_row`) |
| Generated fragment pages | `src/baroque_transcribe/fragment_pages.py` |
| Filenames | `src/baroque_transcribe/defaults.py` |
| Git ignore / re-include rules | `.gitignore` |

Operator-facing summary of fragments remains in `README.md` (Fragment library, Using two computers). This file is the storage map those sections assume.
