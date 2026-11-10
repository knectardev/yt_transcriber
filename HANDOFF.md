# Handoff

This file is for continuing the project in Cursor on a Windows PC. It describes the program as it stands. It does not add behavior.

## What this is

A local Python command-line program, plus an optional browser review page, that transcribes a solo piano recording into:

- an unquantized MIDI file that keeps the performance timing
- a separate quantized MIDI file and a MusicXML score for MuseScore

The musical goal is a decoding of the notes that were played, for studying Baroque-style improvisation. It is not an arrangement and it does not fill in harmony.

## Architecture

`notes.json` is the editable source of truth after the model runs. Raw MIDI, quantized MIDI, MusicXML, pedal data, and the plots are all derived from it. Later analysis should read that file and write annotations beside it, without rewriting `raw.mid`.

Where each object lives on disk, what Git tracks, and why the only SQLite file is a disposable search cache of fragment metadata: [STORAGE.md](STORAGE.md).

| Module | Role |
| --- | --- |
| `src/baroque_transcribe/cli.py` | Commands: `prepare`, `transcribe`, `quantize`, `analyze`, `diagnostics`, `doctor`, `ui`, `history`, `fragments` |
| `audio.py` | Local WAV, MP3, or FLAC, or a YouTube URL. Writes `source.wav` and a trimmed, mono, 16 kHz, peak-normalized `working.wav`. Does not modify the original file. |
| `checkpoint.py` | Downloads the Kong et al. checkpoint with Python, so Windows does not need `wget`. Accepts both the old pickle checkpoint and newer zip checkpoints. |
| `transcribe.py` | ByteDance piano model by default. Basic Pitch ONNX as a baseline. CUDA when PyTorch can see a GPU, otherwise CPU. |
| `midi_export.py` | `raw.mid` uses 120 BPM only as a clock so note times stay in seconds. `quantized.mid` uses the musical tempo and splits treble and bass by pitch. |
| `quantize.py` | Grid from tempo, time signature, subdivision (8, 16, or 32), and a downbeat time. Modes: `fixed` (default) or tempo-`follow`. Pitches and velocities are copied. |
| `notation_settings.py` | Grid/style thresholds (`cluster_ms`, gap fill, rest keep, soft extra-voice cost). Missing manifest keys mean fixed + literal. |
| `engrave.py` | Readable interpretation: continuity voices (soft two-voice preference), guarded chords, gap fill, selective rests, `notation_map.json`, preservation check. |
| `notation.py` | Piano MusicXML as one part with two linked staves. Literal packing or readable interpretation; key-aware spelling; optional coloured sibling `score.voices.musicxml`. |
| `voice_metrics.py` | Continuity metrics and staff-split assessment for interpreted voices. |
| `compare.py` | Side-by-side variants under `project/comparison/` without touching the saved score. |
| `metrics.py` | Rest/tie/voice/on-beat metrics for comparison reports. |
| `diagnostics.py` | Waveform, spectrogram, piano roll, a tone rendering, and a mix of that rendering with the working audio. |
| `ui.py` | Local Streamlit site. Home page: transcribe controls plus a library grid of transcribed videos, whose cards link to the transcription's page on the playback server. Edit & files page (`/project?project=<folder>`): audio and plots, the note table, the harmony labels, the files, and a Fragments tab. Fragment library (`/fragments`), extract form (`/extract`), fragment details (`/fragment?id=`). Links are `st.html` anchors, because `st.image(link=)`, Markdown links, and `st.link_button` always open a new tab. |
| `library.py` | Builds the library cards by merging the parse history with the project folders on disk, resolves the `project` address parameter to a folder, and lists the facts shown in a transcription's header (`run_facts`). |
| `playback.py` | YouTube id parsing, plus a localhost server for each transcription's page (`/p/<token>/`, the video with the score following it) and its score-only page, and for each saved fragment (`/f/<token>/`). Passage selection, loop, and `?w=` / `?sel=` on the playback template. A project or fragment with no video is redirected to its score. |
| `fragments.py` | `plan_fragment` / `extract_fragment`: grid membership, playback window, clipping, sliced `Quantization`, clocks, harmony slice, metadata and sync edits. Does not write the parent. |
| `fragment_index.py` | Rebuildable SQLite index over `data/fragments/`. Incremental (stat-gated), full, and rebuild. Search by text, tags, key, progression, source. |
| `fragment_pages.py` | Playback context and generated pages for a saved fragment. Pages go under `.local/fragment_pages/`. |
| `fragment_ui.py` | Streamlit handlers for the fragment library, extract form, fragment details, and the project Fragments tab. |
| `analysis.py` | Harmonic draft implemented: key estimate/override, measure/beat chord labels (Roman numerals + lead-sheet symbols), sequence spans, edit preservation, `annotations.json` IO. Voice separation, imitation, and cadences remain `NotImplementedError` hooks. |
| `pipeline.py` | Runs the stages and writes `manifest.json`. |

Output of one run:

```text
transcriptions/<name>/
  source.wav
  working.wav
  notes.json
  pedal.json
  raw.mid
  quantized.mid
  score.musicxml
  score.voices.musicxml   # readable style only; optional Color voices toggle
  notation_map.json       # readable style only
  preservation.json       # readable style only
  manifest.json
  diagnostics/
```

MIDI time zero is the start of `working.wav`. If you pass `--start`, that is the excerpt, not the start of the original video.

## What was implemented

- Audio prepare, including a start and end time, for WAV, MP3, FLAC, and a YouTube URL via `yt-dlp` and FFmpeg.
- YouTube sign-in retry against Chrome, Edge, Firefox, Brave, and Chromium on the same computer, plus `--cookies-from-browser` and `--cookies`.
- ByteDance transcription with onset, offset, frame, and pedal thresholds. The official checkpoint is loaded with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because it predates PyTorch 2.6's safe unpickler.
- Basic Pitch through its ONNX file, with the melody post-process turned off and no pedal output.
- Unquantized MIDI with velocities and sustain-pedal controller 64. Pedal times are also stored in `pedal.json`.
- Quantized MIDI and MusicXML. Meter and tempo can be set by hand. Tempo can be estimated. The default meter, if you omit it, is 4/4 and is recorded as an assumption.
- Diagnostic plots and a comparison mix.
- A local site: a Streamlit home page with the transcribe controls and a thumbnail library, one page per transcription with the video and score in sync, and a Streamlit Edit & files page per project with the audio and plots, the note table, the harmony labels, and the files. Links between them open in the same tab.
- Musical fragment extraction: a measure range becomes a self-contained folder under `data/fragments/<id>/` (manifest JSON, notes, sliced grid, both MIDI files, MusicXML, optional harmony). The parent is never modified. Search index in `.local/library.sqlite` (not committed). CLI `fragments extract|list|reindex`. Playback page: Select passage, Loop, Extract fragment. Streamlit `/fragments`, `/extract`, `/fragment`.
- `baroque-transcribe doctor` for FFmpeg, PyTorch, the checkpoint, and optional packages.

## What was tested

On Linux, Python 3.12, CPU PyTorch 2.15:

- `python -m pytest`: 19 tests passed. They cover timestamps, trimming without modifying the source file, the quantization grid, MIDI round-trip, MusicXML pitch content, tempo estimation on a click train, the command-line prepare and quantize commands, and the checkpoint header check. They do not download the network or run the neural net.
- The ByteDance model on the 40-second `cut_liszt.mp3` excerpt from the upstream repository: 523 notes, MIDI pitches 27 through 94, 25 pedal events, controller 64 in `raw.mid`, and a two-staff MusicXML score of 20 measures at the tempo that was passed in (120). The automatic tempo estimate on that clip was about 57 BPM, with about 114 as the double-time candidate.
- Basic Pitch ONNX on the first 8 seconds of that same audio: 137 notes, no pedal events.
- The Streamlit script loaded that 523-note project and presented the note table without an exception.

Windows was not executed in this environment. The upstream piano package also says Windows was not part of its own test matrix. The code path avoids `wget` and uses normal Python and FFmpeg.

YouTube was not transcribed. Both `https://youtu.be/IuZCq1cMq1A` and `https://www.youtube.com/watch?v=Anx1SFCbGow` (Baroque Improvisation Tip No. 1, the video open in the browser screenshot) returned YouTube's "Sign in to confirm you're not a bot" response. This machine has no browser cookie database, so a login in a browser on another computer could not be used.

## Install and run on Windows

1. Install Python 3.11 or 3.12 from python.org and check "Add python to PATH".
2. Install FFmpeg and open a new terminal: `winget install Gyan.FFmpeg`. Confirm `ffmpeg -version` and `ffprobe -version`. Also install Node.js LTS (`winget install OpenJS.NodeJS.LTS`) and open a new terminal; current yt-dlp needs Node or Deno on `PATH` to extract YouTube audio.
3. Clone the repository (`git clone https://github.com/knectardev/yt_transcriber.git`) — or unzip this project — then in that folder:

```text
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install torch
python -m pip install -r requirements.txt
python -m pip install -e .
baroque-transcribe doctor
```

Install PyTorch before `requirements.txt`. For an NVIDIA GPU, use the CUDA command currently shown at https://pytorch.org/get-started/locally/ instead of the plain `pip install torch`. One example is:

```text
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

The first ByteDance run downloads about 165 MB from Zenodo into `%USERPROFILE%\.cache\baroque_transcribe\`.

Optional review page:

```text
python -m pip install -r requirements-ui.txt
baroque-transcribe ui
```

That opens http://127.0.0.1:8765 on the PC where the command is running. The home page starts transcriptions and lists what has been transcribed; clicking a thumbnail opens that transcription's page, the video with the score, in the same tab. Edit & files on that page opens the project's tools. **Fragment library** lists saved passages. `--project <folder>` opens a project's Edit & files page directly. `baroque-transcribe` exists only after `.venv\Scripts\activate`. In PowerShell or Command Prompt that is the activation line above. In Git Bash it is `source .venv/Scripts/activate`.

Optional Basic Pitch baseline, because its TensorFlow pin does not support Python 3.12:

```text
python -m pip install onnxruntime scikit-learn mir_eval resampy
python -m pip install "basic-pitch==0.4.0" --no-deps
```

Tests, after `python -m pip install -r requirements-dev.txt`:

```text
python -m pytest
```

Transcribe the performance from the PC where YouTube is signed in. Close the browser if the cookie database is locked. Use the browser you are actually signed in with (`chrome`, `edge`, or `firefox`).

```text
baroque-transcribe transcribe "https://www.youtube.com/watch?v=Anx1SFCbGow" -o transcriptions\baroque_improvisation --cookies-from-browser chrome --time-signature 4/4 --subdivision 16
baroque-transcribe ui --project transcriptions\baroque_improvisation
```

Set `--start` and `--end` after listening, so speech stays outside the excerpt. Set `--tempo` when you know the beat. `--downbeat` is the time of the first strong beat inside the excerpt. Node or Deno on `PATH` helps current `yt-dlp` read YouTube pages. `doctor` reports whether one is present.

In PowerShell, continue a long command with a backtick, not `^`. In Git Bash, use forward slashes in paths (`transcriptions/baroque_improvisation`). A backslash inside double quotes is not a folder separator there.

After editing `notes.json`:

```text
baroque-transcribe quantize transcriptions\baroque_improvisation --tempo 84 --time-signature 4/4 --subdivision 16
```

Open `score.musicxml` in MuseScore. Compare `working.wav` with `raw.mid` when checking timing. Leave the 120 BPM mark in `raw.mid` alone if you want it to stay aligned with the audio. Musical tempo belongs on `quantized.mid`.

## Known limitations

- The published model scores are not a measurement of a YouTube recording. Ornamentation, repeated notes, and notes hidden by the pedal are the usual errors.
- The program does not invent notes, but a wrong detection still adds or drops notes. Treat the score as unchecked until you have listened.
- One global tempo cannot follow rubato. The quantized score is a reading aid. The raw MIDI is the timing document.
- Default grid/style for new runs remain `fixed` + `literal`. Pass `--grid follow` and/or `--notation readable` (or use `baroque-transcribe compare`) to opt in.
- **Voice fragmentation:** Readable continuity prefers reactivating same-staff lines and steeply discourages a 3rd+ simultaneous voice, but dense Tip No. 4 passages can still open extra IDs. Soft two-voice preference is not a finished editorial reading.
- **Pitch-threshold staff assignment:** Staff is chosen during voice assignment from middle C (default) with ±5 sticky continuation. Near-split notes bias toward the treble. A later improvement is staff-per-segment from each voice's median pitch after continuity IDs are set.
- **Tip No. 1 follow-grid regression:** Tempo-follow can worsen rest counts on Tip No. 1; investigation is deferred. Keep fixed + literal as the default.
- **Color voices** is a presentation toggle on the playback and score-only pages when `score.voices.musicxml` exists. Preference is stored in `localStorage` (`baroque-transcribe.color-voices`). Default off. Toggling reloads OSMD only and must not seek or reset YouTube playback. Colours visualize the current interpreter's voice IDs — pedagogical, not authoritative analysis. Features such as voice soloing need stronger cross-phrase continuity first.
- **Highlight notes** on the synced playback page is optional (checkbox, remembered in `localStorage`). Measure highlight stays as before. With the option on, a small attack dot sits above the staff at the current onset (chord pitches listed in the measure label); click the dot to seek. Manual score scrolling turns **Follow score** off so the viewport is free.
- **Fragments.** Membership is grid-only; the playback window is derived from members; clipping applies only at the outer barlines. A carried-in note is re-struck at time zero in MIDI (intentional approximation). Readable-style fragments keep the parent's voice assignment and omit a trimmed-overlap note from the printed score (`printed=False`) while leaving it in notes/grid/MIDI. A gap-filled tail that the parent prints in the first selected measure but whose parent note is not a member is omitted (documented boundary difference). `write_musicxml(..., end_step=)` is used only for fragments; the parent path is unchanged. Readable interpretation can leave overlapping events in one voice, so music21 may write an overfull measure; `_cut_at_barline` runs only on the `end_step` path so a fragment does not print an extra bar. Incremental index reconcile that sees the same mtime and size will not reread a file whose content changed in place; `full` / `refresh_folder` / Rebuild catch that. Local-file fragments have no source-audio playback in v1. Analyzed key and written key signature are separate manifest fields.
- The harmonic layer is a draft over the quantized grid: one global key, no modulation, figuration and passing tones produce wrong or over-specific labels. It is drawn on the browser score pages and stored in `annotations.json`, which is copied into the parse history with the score. It is not embedded in MusicXML.
- Pedal is kept in `raw.mid` and `pedal.json`. It is not written into the quantized score.
- If you omit `--time-signature`, the score uses 4/4 and the manifest says that was an assumption.
- Tempo estimation often locks onto half time or double time. The manifest lists those candidates.
- The model was trained on acoustic piano. Harpsichord, organ, and heavily processed electric piano will do worse.
- Speech inside the excerpt is transcribed as pitches.
- YouTube downloads depend on `yt-dlp`, a JavaScript runtime, and cookies from a signed-in browser on the same computer.
- This environment did not run the program on Windows.

## Recommended next steps

1. Unzip on the Windows PC, create the virtual environment, and run `baroque-transcribe doctor`.
2. Download and transcribe a short excerpt of the Think Baroque video with `--cookies-from-browser`, then listen to `diagnostics/comparison.wav` before transcribing a long span.
3. Set the real tempo, meter, and downbeat, and fix notes in the review page or in `notes.json`.
4. Open the MusicXML in MuseScore and make the pickup, beaming, and hand assignment editorial.
5. Only after that, extend the analysis in `analysis.py`. The harmonic layer is done: `analyze_harmony` writes chord labels (Roman + chord symbol) and sequence spans to `annotations.json`, generated automatically after notation, rebuilt with `baroque-transcribe analyze` or the Harmony tab of the Edit & files page, and drawn under the measures on both score pages with a lead-sheet toggle. Still reserved: voice separation, imitation, and cadences (`CounterpointAnalyzer` raises `NotImplementedError`). Those should annotate `notes.json`. They should not change `raw.mid`.
