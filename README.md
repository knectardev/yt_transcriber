# Piano performance transcription

A local Python program that transcribes a solo piano recording into MIDI and MusicXML. It is meant for learning a performance, including Baroque-style improvisation: the notes should be the notes that were played, not a new arrangement and not a stylistic reconstruction.

Two results are written every time:

- **Raw transcription.** Unquantized MIDI. Onsets, durations, velocities, and overlaps stay where the model heard them.
- **Notation transcription.** The same pitches snapped to a grid you choose, as MIDI and as MusicXML for MuseScore.

Those are different objects. The raw file is the one that should line up with the audio. The score is a rhythmic reading, and under rubato it will not.

## Model

The default model is [piano-transcription-inference](https://github.com/qiuqiangkong/piano_transcription_inference) 0.0.6, the inference code for ByteDance's high-resolution piano transcription (Kong, Li, Wang, Wang). It was trained for solo piano. It predicts onset, offset, frame activation, velocity, and sustain pedal together. On the authors' own test conditions the published note F1 is about 0.97 and the pedal F1 is about 0.92. Those numbers describe clean acoustic-piano recordings in the MAESTRO style. A YouTube recording, room noise, or a different keyboard will do worse. Nothing in this repository re-measures that paper, and a run of this program is not a claim that the score is note-for-note.

[Basic Pitch](https://github.com/spotify/basic-pitch) 0.4 is included as a lighter, instrument-agnostic baseline (`--model basic-pitch`). It has no pedal head. Its "Melodia" post-process, which pulls a texture toward a single line, is turned off here because that fights counterpoint. On Python 3.12 the published package tries to install a TensorFlow build that does not support 3.12, so the baseline is run through the ONNX file instead. Install steps are below.

Heavier multi-instrument systems such as MT3 were set aside. They need a separate JAX stack and are not better suited to solo piano than the ByteDance model.

The upstream package says Windows was not part of its test matrix, and it downloads the checkpoint with `wget`. This project downloads that file in Python and then calls the same model. The code path is the one to use on Windows. It was run here on Linux with Python 3.12 and a CPU build of PyTorch. `baroque-transcribe doctor` reports what your machine can actually see. `--device auto` uses CUDA when that PyTorch install can see a GPU, and otherwise uses the CPU.

## What it will not do

- It does not add harmony that the model did not detect.
- It does not quantize `raw.mid`.
- It does not separate real contrapuntal voices, find imitation, or mark cadences. `src/baroque_transcribe/analysis.py` is the place those passes will plug in. They are not implemented.
- The treble/bass split is a pitch threshold (middle C by default), not a decision about which hand played.

## Install

Use Python 3.11 or 3.12 (check with `py -3.12 --version`; a machine that only has Python 3.13 needs 3.12 installed first). FFmpeg must be on `PATH` (`ffmpeg` and `ffprobe`). On Windows, `winget install Gyan.FFmpeg`, then open a new terminal.

YouTube downloads also need a JavaScript runtime for yt-dlp. On Windows, `winget install OpenJS.NodeJS.LTS`, then open a new terminal. `baroque-transcribe doctor` reports whether Node or Deno was found; without one, YouTube extraction fails even though everything else works.

To work from the repository rather than a zip, install Git, sign in to GitHub on that machine (for example with Git Credential Manager or `gh auth login`), and clone instead of unzipping:

```text
git clone https://github.com/knectardev/yt_transcriber.git
cd yt_transcriber
```

Then create the environment in the cloned folder:

```text
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install torch
python -m pip install -r requirements.txt
python -m pip install -e .
baroque-transcribe doctor
```

Do not copy another machine's `.venv` or clone into a folder that already has one. The virtual environment records the folder it was created in; a copied or stale one can import the program from the wrong checkout. If `python -c "import baroque_transcribe.audio; print(baroque_transcribe.audio.__file__)"` does not print a path inside the clone you are working in, delete `.venv` and recreate it.

On macOS or Linux the virtual environment is activated with `source .venv/bin/activate`.

Install PyTorch **before** the requirements file, so you keep the CPU or CUDA build you chose. A CUDA example, which you should match to the current command on [pytorch.org](https://pytorch.org/get-started/locally/):

```text
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`audioread` is listed on purpose. Current Librosa no longer installs it, and the ByteDance package still imports it.

The first ByteDance transcription downloads about 165 MB from Zenodo into:

- Windows: `%USERPROFILE%\.cache\baroque_transcribe\`
- macOS and Linux: `~/.cache/baroque_transcribe/`

Override that path with `--checkpoint` or the environment variable `BAROQUE_TRANSCRIBE_CHECKPOINT`.

Optional local review window:

```text
python -m pip install -r requirements-ui.txt
```

Optional Basic Pitch baseline:

```text
python -m pip install onnxruntime scikit-learn mir_eval resampy
python -m pip install "basic-pitch==0.4.0" --no-deps
```

Tests, which do not download the checkpoint and do not run the network:

```text
python -m pip install -r requirements-dev.txt
python -m pytest
```

A separate check, after the checkpoint is downloaded, is to transcribe a short acoustic-piano excerpt and listen to `diagnostics/comparison.wav`. On a CPU run here, the 40-second Liszt excerpt distributed with the ByteDance repository (`cut_liszt.mp3`) produced 523 notes and 25 pedal events, with pitches from MIDI 27 through 94, and a two-staff MusicXML file. That shows the pipeline runs. It is not a note-for-note audit, and it is not your recording. YouTube blocked the download of `https://youtu.be/IuZCq1cMq1A` from this machine (it asked for a sign-in), so that performance still has to be transcribed locally.

## Transcribe the performance

Listen once and cut out talk, applause, and demonstrations. The model will try to transcribe speech as pitches. Times below are only an illustration.

```text
baroque-transcribe transcribe "https://youtu.be/IuZCq1cMq1A" ^
  -o transcriptions\baroque_improvisation ^
  --start 0:00 --end 3:00 ^
  --time-signature 4/4 ^
  --tempo auto ^
  --subdivision 16
```

On macOS or Linux, continue the command with `\` instead of `^`.

A file already on disk:

```text
baroque-transcribe transcribe "C:\Recordings\recital.flac" -o transcriptions\recital --time-signature 3/4 --tempo 72 --subdivision 32 --downbeat 0.35
```

`--downbeat` is the time, in the trimmed audio, of the first strong beat. Notes before it are written in the previous bar. In MuseScore you can turn that opening bar into a pickup if you want an anacrusis.

YouTube often answers a download from a fresh or remote machine with a sign-in check. The command then looks for a signed-in Chrome, Edge, Firefox, Brave, or Chromium profile on the same computer. A login in a browser on a different computer is not visible to it. Close the browser if the cookie database is locked, or name it yourself:

```text
baroque-transcribe transcribe "https://youtu.be/IuZCq1cMq1A" -o transcriptions\baroque_improvisation --cookies-from-browser chrome
```

`edge` and `firefox` work the same way. A `cookies.txt` export can be passed with `--cookies` instead. Recent yt-dlp also wants Node or Deno installed for YouTube extraction. `baroque-transcribe doctor` reports whether either one is on `PATH`.

Prepare the audio without running the model:

```text
baroque-transcribe prepare recital.mp3 -o transcriptions\recital --start 1:05 --end 6:40
```

Accepted local files are WAV, MP3, and FLAC. The original file is not modified. `source.wav` is a decoded copy at the file's own rate and channel count. `working.wav` is the excerpt actually transcribed: mono, 16 kHz, peak-normalized to 0.95. MIDI time zero is the start of `working.wav`, not the start of the video, when you pass `--start`.

For a YouTube source, passing `--start` or `--end` downloads only that section of the video (yt-dlp `--download-sections`), so a short excerpt of a long video does not download the whole recording. In that case `source.wav` holds just the downloaded section, at the excerpt's offset recorded in `manifest.json`; choosing a different excerpt later means downloading again. If a section download fails, the whole audio is downloaded instead, as before. With no excerpt times the whole video is downloaded and `source.wav` is the complete recording, so a local re-trim needs no new download. Local files are always copied in full.

## Output

```text
transcriptions/baroque_improvisation/
  source.wav
  source.mid          # MIDI imports only: original source bytes
  working.wav
  notes.json
  pedal.json
  raw.mid
  quantized.mid
  score.musicxml
  score.voices.musicxml   # readable style: optional Color voices display
  annotations.json
  manifest.json
  diagnostics/
    overview.png
    waveform.png
    spectrogram.png
    piano_roll.png
    resynthesis.wav
    comparison.wav
```

`raw.mid` stores a tempo of 120 only so that MIDI ticks line up with seconds. Leave that tempo alone if you are comparing it with the recording. Musical tempo belongs on `quantized.mid`.

`notes.json` is the file to correct. Each note is pitch, onset, offset, and velocity, in seconds from the start of `working.wav`. Pedal down/up times are in the same file and, separately, in `pedal.json`. `raw.mid` also carries them as controller 64. The quantized score does not, so a noisy pedal track is not baked into the notation.

After an edit:

```text
baroque-transcribe quantize transcriptions\baroque_improvisation --tempo 84 --time-signature 4/4 --subdivision 16 --downbeat 0.2
baroque-transcribe quantize transcriptions\baroque_improvisation --grid follow --notation readable
baroque-transcribe compare transcriptions\baroque_improvisation --variants literal_fixed,literal_follow,readable_follow,readable_follow_voices
```

With `--notation readable`, the project also writes `score.voices.musicxml`. The playback and score-only pages then offer a **Color voices** checkbox (default off, remembered in the browser). Colours mark interpreted contrapuntal voices from the same interpretation as the black-and-white score; toggling does not re-transcribe or move the video. Continuity prefers reactivating same-staff lines and discourages a third simultaneous voice, but treat the colours as the algorithm's current reading, not ground-truth voice identity. Defaults for new runs remain `fixed` + `literal`. Known follow-ups: staff assignment after continuity (instead of the middle-C threshold during assignment), and Tip No. 1's follow-grid rest regression.

`--subdivision` is the smallest note value: `8` eighths, `16` sixteenths, `32` thirty-seconds. Notes that share a start and an end on one staff become a chord. Notes with different durations stay in separate voices. A note that crosses a barline is tied.

If you omit `--time-signature`, the score uses 4/4 and `manifest.json` records that this was an assumption. `--tempo auto` estimates one tempo from onset spacing and lists half-tempo and double-tempo candidates, because that estimator often locks onto the wrong metrical level. One tempo cannot represent rubato.

## Check the result

```text
baroque-transcribe ui
```

That opens a local site at <http://127.0.0.1:8765>. Links inside the site open in the tab you are in. **Open on YouTube** is the one that opens a new tab, because it leaves the site.

The home page is the **Fragment library** (searchable practice excerpts). Primary navigation also covers the **Bach catalog**, **Projects**, and **Transcribe / import**. Manual YouTube, local file, upload, and MIDI import live on `/transcribe`. The project grid (YouTube thumbnails, MIDI tiles, opened Bach works) lives on `/projects`. **Bach catalog** is the full keyboard list; **Opened Bach** on Projects is only works you have opened from that list. A run named in the parse history but missing from this computer is shown greyed out.

A transcription's page is the transport with the score under it. For YouTube projects that is the video; for MIDI imports (and other projects without a video) it is synthesized preview audio, or a silent scrubber if synthesis fails. The header gives the note and pedal counts, excerpt length, tempo, meter, model, and the project folder. The highlighted measure follows the transport; optional **Highlight notes** places a small attack dot above the staff (pitch names stay in the measure label). Scrolling the score turns off **Follow score** so playback stops forcing the viewport. Clicking a measure jumps there; with note highlight on, clicking the attack dot seeks to that onset. **Select passage** marks a run of measures (click the first, then the last; the − and + buttons move either end). **Loop** replays that run at whatever speed the player is set to. **Extract fragment** sends the range to a form that saves it. **Score only** shows the static two-staff score, and **Play with score** on that page comes back. The page is written again each time it is opened, so it shows the notes and harmony as they are now. It is served on a second local port that is chosen at each launch, so its address is not one to bookmark. Start from the home page.

MIDI imports keep `source.mid` as the original source authority and pretty_midi absolute note times in `notes.json` as the editable app timeline. Full tempo/meter maps, source SHA-256, track/pitch-split staff assignment, optional per-note `staff`, and score origin survive publication, restore, and requantization. A YouTube URL can be attached later; only a fixed sync offset aligns video time to that timeline (nonlinear performance alignment is out of scope). Importing an already-published source warns and requires the explicit duplicate override in the UI.

## Bach catalog

The checked-in manifest at `data/catalog/bach-keyboard.json` lists 194 keyboard works (BWV 772–943) from Tobis Notenarchiv, with source URLs and attribution. MIDI bytes are **not** committed. Download them into the local mirror `catalog/midi/<entry-id>/source.mid` (gitignored):

```text
baroque-transcribe catalog list --text invention
baroque-transcribe catalog sync --all --overnight --keep-awake --confirm
baroque-transcribe catalog preprocess --all --confirm --keep-awake
baroque-transcribe catalog preprocess --all --confirm --force --keep-awake
baroque-transcribe catalog open bwv-0870
baroque-transcribe catalog movements-draft --write
```

`--overnight` waits a fresh uniform-random **30–200 seconds** between requests, one file at a time, and resumes verified files on restart. `--keep-awake` asks Windows not to sleep for the duration of that process. The sync process is independent of the Streamlit UI and of git commits; leave both alone while it runs.

**Preprocess** imports each downloaded MIDI into a Library project and engraves it with **`--notation readable`**, writing `score.voices.musicxml` and `notation_map.json` so Color voices and dual-instrument preview work. It skips entries that already have those voice artifacts; **`--force`** re-quantizes every selected entry in place (needed after engraver changes or to upgrade older literal catalog scores). Progress prints `[n/N] id: opened|requantized|skipped|failed`.

Opening a catalog entry (UI or CLI) downloads that one file if needed, then imports it through the same MIDI-project path as a manual upload (catalog opens default to readable), including playback, measure selection, and fragment extraction. **Open score** navigates this tab straight to the playback page (no second button, no new tab). The playback **← Bach catalog** link returns to `/catalog`. Downloaded rows show a disabled **Downloaded** button. Catalog provenance is stored on the project and copied into new fragments. The YouTube thumbnail grid lives on **Projects** (`/projects`) and again under **Transcribe** (defaults to YouTube).

**Edit & files** on a project page opens `/project?project=<folder>`, which holds the rest. Its tabs are the working audio and the mix of that audio with a simple tone at each detected pitch (a check of timing, not a piano sound) with the plots, the editable note table with the tempo, meter, subdivision, staff split, and optional MIDI staff used to rebuild MIDI and MusicXML, the Harmony tab described below, the downloadable files with the run's warnings, and a **Fragments** tab that lists passages saved from this transcription and can open the extract form from measure numbers (the way to save a passage when there is no video). Revision indicators warn about unpublished, newer, divergent, or conflicting history. A published update is previewed and selected by event, then requires confirmation and creates a permanent recovery copy before replacement. The Files tab can write a validated ZIP backup, optionally with audio, to a user-chosen local path. A project transcribed from a local file can have a YouTube address attached there so the score can follow the video. **Back to video and score** returns to the transcription's page. A transcription that has just finished opens here, and so does a project that has no score yet.

`baroque-transcribe ui --project transcriptions\piece` opens straight to that project's Edit & files page.

## Fragment library

A fragment is a run of printed measures taken from a transcription and stored as its own folder under `data/fragments/<id>/`. It keeps its own `notes.json`, `grid.json`, `performance.mid`, `quantized.mid`, `score.musicxml`, and (when the parent had one) `harmony.json`. The parent transcription is only read. The quantized score is a slice of the parent's grid, not a second quantization, so the selected measures match the parent except at the two outer barlines.

Membership is decided on the grid. A note the parent prints in the selection is kept even if it was played early or late. A note already sounding at the first barline is struck again at the start of the fragment, at its original loudness — an intentional approximation, because the real attack came earlier. A note that rings past the last barline is cut there. Those counts are shown on the extract form before you save.

The fragment has its own clocks. Fragment time zero is the start of the playback window. The source-video interval is written once. A later sync nudge on the fragment's page writes only `sync.offset_seconds` in `fragment.json`; it moves where the player looks and does not change the duration or the stored interval.

The home page is the Fragment library (`/` and `/fragments`). Search by text, analyzed key, tags, source, catalog entry, or a chord sequence (`V7 i` matches those labels next to each other; `V` and `v` are different chords). The list is a dense row per fragment: **Play**, title, a piano-roll strip, and a few facts, with a synthetic piano or electric-piano tone rendering (as-played or quantized) cached under `.local/fragment_audio/`. Audio is never copied into a fragment. A fragment of a YouTube transcription can still open on its own page with the video cued and looping on the passage. A fragment of a local recording has the score and the downloads, and no source-audio player. **Open source transcription** goes to the parent at that place when the parent is on this computer; otherwise **Open on YouTube** uses the stored timestamp. New fragment manifests record the parent lineage, base event, base snapshot, and extraction snapshot; the details page reports whether the parent is current, changed locally, changed in published history, or unavailable. Parent restore/update never mutates a fragment.

```text
baroque-transcribe fragments extract transcriptions\piece --measures 12-15 --title "Descending fifths"
baroque-transcribe fragments extract transcriptions\piece --measures 12-15 --dry-run
baroque-transcribe fragments list --text cadence --tag sequence --key "G minor"
baroque-transcribe fragments reindex
baroque-transcribe fragments reindex --rebuild
```

The SQLite index at `.local/library.sqlite` is a local cache. JSON files under `data/fragments/` are authoritative. Deleting the index and running `fragments reindex --rebuild` restores search. **Check for changes** and **Rebuild library index** on the library page do the same. The index is not committed.

The full storage map — working projects, parse history, fragment folders, the SQLite schema, and why the library is files rather than a database — is [STORAGE.md](STORAGE.md).

Analyzed key (from the parent's `annotations.json`, used for Roman numerals and the key filter) is stored separately from the written MusicXML key signature. The MusicXML key is the analyzed key so spelling matches the parent; the pitches are not transposed.

The same plots are written by the `transcribe` command. `baroque-transcribe diagnostics transcriptions\baroque_improvisation --position 12.5` redraws them with a cursor.

Import `working.wav` and `raw.mid` into a DAW if you want to solo the transcription against the recording. Use the raw file for that. The quantized file is metronomic on purpose.

## Harmony draft

A transcription also gets a first-pass harmonic reading, stored in `annotations.json` beside the score: one chord label per measure (split at a beat when the harmony clearly changes mid-measure), each with both a Roman numeral and a lead-sheet chord symbol, plus any repeated root-motion spans the detector finds — descending or ascending fifths, stepwise root motion, and repeated one- or two-measure modules. The key is estimated from the notes (Krumhansl profiles) unless you set it. A fresh transcription writes this file automatically; a failure to analyze never fails the transcription.

```text
baroque-transcribe analyze transcriptions\baroque_improvisation --key Bb --mode major
```

`--key`/`--mode` pin the tonic; omit them to keep the estimate (or the key you set before). The **Harmony** tab on the Edit & files page does the same thing and lets you edit every label and span in place. Edits you save there are marked as yours: a normal rebuild keeps them and only refreshes the rest, while "Replace my edited labels" (or `--force`) discards them. If the notes or tempo changed since the analysis, the tab says so.

On both score pages the labels sit under each measure, Roman numerals by default. The **Harmony** checkbox hides the layer and **Lead-sheet chord symbols** swaps every label to its chord symbol (the hover text shows the other form and the detector's confidence; paler labels are the ones it was unsure about). Sequence spans are underlined under the measures they cover. Exported MusicXML does not carry the chords yet. `annotations.json` does travel: a transcribe or quantize copies it into that run's history folder, and a later edit or `analyze` adds a new history event that holds the file. Restore copies it back with the score. A project folder that already exists keeps its notes, and gains `annotations.json` only when that file is missing locally.

Treat the whole thing as a draft. It reads the quantized grid, not the counterpoint: Baroque figuration, arpeggiation, passing tones, and pedal blur will produce wrong or over-specific labels (sus and six-chords are common false positives), the key is one global estimate with no modulation tracking, and inversions follow the lowest attack near the slice, which ornamentation can fool. Correct it in the Harmony tab rather than trusting it.

## When the transcription is wrong

Raise `--onset-threshold` (ByteDance default 0.3) when the sustain pedal is leaving extra notes, especially octaves and fifths that were not played. Lower it when soft notes or ornaments disappear. `--frame-threshold` (default 0.1) and `--offset-threshold` (default 0.3) change how long a note is held. Try a 10–20 second excerpt before a long file.

Common failures, even when the model is the right tool:

- Trills, mordents, and fast repeated notes merge into one note or shatter into several.
- Pedal resonance creates notes that were not struck, and it smears offsets.
- Speech, coughing, and page turns become pitches if they are inside the excerpt.
- YouTube compression and a distant microphone hide inner voices.
- The model was trained on acoustic piano. Harpsichord, organ, and heavily processed electric piano are outside that training.

Do not treat `score.musicxml` as a finished edition until you have checked it against the recording.

## Parse history

Every `prepare`, `transcribe`, `quantize`, MIDI `import`, and `analyze` adds a folder under `data/parses/events/`. Each folder is one run. `event.json` is written once and left as it was, so history added on two machines merges as new folders. Existing schema-v1 events remain readable. New schema-v2 events carry causal lineage, canonical component hashes, separate raw file-integrity hashes, a portable settings projection, and a complete independently restorable artifact snapshot for score-producing runs. Analyze-only events are causal annotation overlays and are published only while their score still matches the causal base; no event reads artifact bytes through another event.

The folder also holds the score files from that run, when the run produced them: `notes.json`, `pedal.json`, `raw.mid`, `quantized.mid`, `score.musicxml`, and `annotations.json`; complete MIDI events also hold `source.mid`. An `analyze` event, including one written when harmony edits are saved, stores `annotations.json` and leaves the earlier score files where they are. `source.wav`, `working.wav`, and the diagnostic wavs stay in `transcriptions/` and are only described in `event.json`. Cookie files and absolute home-directory paths are not stored. The machine name in the event is the computer that wrote it.

`score.pdf` is listed in every event with `"present": false` and `"status": "planned"`. A later `baroque-transcribe pdf <project>` command will write that PDF next to `score.musicxml` by asking music21 for a MusicXML PDF, which needs MuseScore on `PATH`. The PDF will be copied into the event folder as a new file, with a sibling `pdf.json` saying when and on which machine. `event.json` stays as first written. The review page can then offer the PDF beside MusicXML.

```text
baroque-transcribe history
baroque-transcribe history import transcriptions
baroque-transcribe history status
baroque-transcribe history preview <event-id>
baroque-transcribe history update <event-id> --confirm <event-id>
```

Import reads each `manifest.json` once. Running it again skips a project already imported with the same video (or file name), excerpt, and folder name.

Commit `data/parses/` and pull it on the other machine. The history and the score files come with the pull. The audio does not, so a YouTube download still has to be run on a computer where YouTube is signed in.

Status is ancestry-based: synchronized, newer published, unpublished local edits, diverged, multiple published heads, independent, not restored, or a legacy relationship. A timestamp never resolves incomparable branches. `history restore` remains non-destructive for missing projects. `history update` is event-targeted, validates before writing, makes a verified recovery under `transcriptions/.recovery/`, stages the replacement, and rolls back on failure. It preserves local annotations unless `--published-annotations` is given and retains local audio only when source/excerpt identity matches.

## Using two computers

The repository is the sync mechanism. Each machine has its own clone, its own `.venv`, its own model checkpoint, and its own `transcriptions/` folder. The parse history under `data/parses/` and the fragment library under `data/fragments/` travel between them.

On a laptop that already has an editable install, stop the running UI, run `git pull`, and start `baroque-transcribe ui` again. This revision adds no packages and no migration. Use `history restore <project>` for a pulled run whose project folder is absent; leave an existing project folder in place and inspect it with `history status`. The colored `score.voices.musicxml` sibling is local, so a restored readable project needs one local `quantize --notation readable` before its Color voices checkbox appears. The full sequence is in [HANDOFF.md](HANDOFF.md).

**What travels, and what does not**

| Comes with `git pull` | Stays on the machine that made it |
| --- | --- |
| The program, the tests, and these documents | `transcriptions/` (every project folder, including `manifest.json`) |
| `data/parses/events/`: `event.json`, complete score artifacts, annotations, and `source.mid` for new MIDI score events | All audio (`source.wav`, `working.wav`, diagnostics) |
| `data/fragments/<id>/`: `fragment.json`, `notes.json`, `grid.json`, `harmony.json`, `performance.mid`, `quantized.mid`, `score.musicxml` | `.local/` (the fragment search index and generated fragment pages; rebuild with `fragments reindex`) |
| | The 165 MB model checkpoint (`%USERPROFILE%\.cache\baroque_transcribe\`) |
| | The thumbnail cache (`data/thumbnails\`); it is re-fetched on first view |
| | `.venv\`, browser cookies, and YouTube sign-ins |

**Routine on the machine that transcribes**

```text
git pull
baroque-transcribe transcribe "https://youtu.be/…" -o transcriptions\piece --cookies-from-browser chrome
git add data/parses
git commit -m "Transcribe piece"
git push
```

Pull first, push after. Event folders are new every time and are never edited, so two machines adding runs do not conflict in the files themselves; a push is only rejected when the other machine pushed first, and `git pull` then merges the two histories as new folders. Never edit an `event.json` by hand, and do not copy another machine's `transcriptions/` folder into the clone as part of the routine: `git add data/parses` is the whole publish step. YouTube sign-in is per machine, so the download still has to run on a computer whose browser is signed in (close the browser first if its cookie database is locked).

**What you get on the other machine after `git pull`**

The library shows the new video and its runs greyed out, marked "not on this computer". That is expected: the history and the score files have arrived, but no project folder has. Restore the project from the history:

```text
baroque-transcribe history restore
```

That rebuilds every missing project (or one project, with `baroque-transcribe history restore <project>`) from a validated complete score event plus its causal analysis overlay. **Nothing is downloaded and no model is run**: a restored project is the pulled transcription itself, not a new one, and restoring records no new history. A project that already exists on this machine keeps its notes and score. If `annotations.json` is missing there, restore fills that file in and leaves an analysis that is already present. The same restore is a button on each greyed-out library card in the review window.

Two limits of a restored project, both because audio does not travel in the repo:

- There is no `working.wav`, so the Listen tab has nothing to play and there are no diagnostic plots. The score page and the synced video play without it, because the video is streamed from YouTube and the measure timing is re-derived from the restored tempo and notes.
- `baroque-transcribe history restore --with-audio` additionally downloads the excerpt audio -- for a YouTube source, only the requested section, using the same section download as a transcription -- so the Listen tab works. It still never runs the model. The diagnostics can then be redrawn with `baroque-transcribe diagnostics transcriptions/<project>`.

The alternatives are still there: open `data/parses/events/<run>/score.musicxml` directly in MuseScore for just the score, or re-transcribe on this machine, which downloads and runs the model again and records a *new* history run rather than reusing the pulled one. Do **not** run `baroque-transcribe history import transcriptions` for a folder whose runs are already in the pulled history; import is for projects that predate the history, and it would record those runs a second time.

The same warning in reverse: a `git pull` and a restore never update a project folder you already have. If a newer event arrives, inspect it with `history status` and `history preview <event-id>`, then use the explicit confirmed `history update` command or the project-page action. Do not delete the local folder to update it: the safe path preserves it as a verified recovery and rolls back if application fails.

Recovery copies are durable and never expire automatically:

```text
baroque-transcribe storage report
baroque-transcribe storage recoveries
baroque-transcribe storage export <project> <backup.zip> --with-audio
baroque-transcribe storage rollback-recovery <project>/<recovery-id> --confirm "RESTORE <project>/<recovery-id>"
baroque-transcribe storage delete-recovery <project>/<recovery-id> --confirm "DELETE <project>/<recovery-id>"
```

The storage report lists recovery bytes separately from rebuildable caches. Recovery listing shows the path, prior/target events, and exact confirmation strings. A rollback verifies and retains the selected recovery and first preserves the current project as another recovery. Deletion requires the exact selected recovery id and confirmation. Git does not back up unpublished working projects; use `storage export` for a validated local ZIP.

This safety revision does not remove or deduplicate score artifacts. `pedal.json` is still written, published, restored, downloaded, and tested. Generated files and cache cleanup remain a separately approved storage-optimization decision; existing events, fragments, and Git history are never rewritten.

## Layout

See [STORAGE.md](STORAGE.md) for the on-disk layout, JSON contracts, and SQLite index. The table below is the Python modules.

| Piece | Role |
| --- | --- |
| `audio.py` | YouTube or local file, trim, normalize |
| `transcribe.py` | ByteDance and Basic Pitch |
| `midi_export.py` | Raw and quantized MIDI |
| `quantize.py` | Grid, tempo, meter, downbeat (`fixed` or `follow`) |
| `notation_settings.py` | Notation thresholds and style defaults |
| `engrave.py` | Readable voice/duration interpretation and note-preservation map |
| `notation.py` | Piano MusicXML (linked staves; literal or readable) |
| `compare.py` | Side-by-side score variants under `comparison/` |
| `metrics.py` | Comparison metrics |
| `diagnostics.py` | Plots and the comparison mix |
| `analysis.py` | Harmonic draft: key estimate, chord labels, sequence spans in `annotations.json`; voice/imitation/cadence hooks still reserved |
| `parse_store.py` | JSON history of each run, with the score files and a reserved PDF slot; also restores project folders from that history on another machine |
| `revision.py` | Canonical fingerprints, lineage heads/ancestry, and derived project revision states |
| `project_manifest.py` | Shared versioned manifest IO, portable settings projection, revision and generation blocks |
| `project_restore.py` | Event validation/materialization, preview, recovery, rollback, and explicit update |
| `storage_management.py` | Category disk report and validated project ZIP export |
| `library.py` | Cards for the home page: the parse history merged with the folders under `transcriptions/` |
| `playback.py` | YouTube id parsing and the localhost server that writes and serves each transcription's page (video with score) and its score-only page, plus `/f/<token>/` for a saved fragment |
| `fragments.py` | Plan and extract a measure range; fragment manifests, clocks, and metadata edits. Does not write the parent. |
| `fragment_index.py` | Rebuildable SQLite search index over `data/fragments/` |
| `fragment_pages.py` | Playback and score pages for a saved fragment, written under `.local/fragment_pages/` |
| `fragment_ui.py` | Streamlit `/fragments`, `/extract`, `/fragment` |
| `ui.py` | The local site: home page with the transcribe controls and the library, an Edit & files page per project, and the fragment library |
| `cli.py` | `prepare`, `transcribe`, `quantize`, `compare`, `analyze`, `diagnostics`, `doctor`, `ui`, `history`, `fragments`, `storage` |

`notes.json` is the stable intermediate those later analyzers should read. They should write annotations beside the transcription and leave `raw.mid` alone. The harmonic analyzer does exactly that: it reads `notes.json` and writes `annotations.json`.

## License of the models

The ByteDance inference package and Basic Pitch are Apache-2.0. The piano checkpoint is downloaded from Zenodo record 4034264 and is not stored in this repository. Basic Pitch's ONNX file ships inside its own package.
