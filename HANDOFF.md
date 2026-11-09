# Handoff

This file is for continuing the project in Cursor on a Windows PC. It describes the program as it stands. It does not add behavior.

## What this is

A local Python command-line program, plus an optional browser review page, that transcribes a solo piano recording into:

- an unquantized MIDI file that keeps the performance timing
- a separate quantized MIDI file and a MusicXML score for MuseScore

The musical goal is a decoding of the notes that were played, for studying Baroque-style improvisation. It is not an arrangement and it does not fill in harmony.

## Architecture

`notes.json` is the editable source of truth after the model runs. Raw MIDI, quantized MIDI, MusicXML, pedal data, and the plots are all derived from it. Later analysis should read that file and write annotations beside it, without rewriting `raw.mid`.

| Module | Role |
| --- | --- |
| `src/baroque_transcribe/cli.py` | Commands: `prepare`, `transcribe`, `quantize`, `diagnostics`, `doctor`, `ui` |
| `audio.py` | Local WAV, MP3, or FLAC, or a YouTube URL. Writes `source.wav` and a trimmed, mono, 16 kHz, peak-normalized `working.wav`. Does not modify the original file. |
| `checkpoint.py` | Downloads the Kong et al. checkpoint with Python, so Windows does not need `wget`. Accepts both the old pickle checkpoint and newer zip checkpoints. |
| `transcribe.py` | ByteDance piano model by default. Basic Pitch ONNX as a baseline. CUDA when PyTorch can see a GPU, otherwise CPU. |
| `midi_export.py` | `raw.mid` uses 120 BPM only as a clock so note times stay in seconds. `quantized.mid` uses the musical tempo and splits treble and bass by pitch. |
| `quantize.py` | Grid from tempo, time signature, subdivision (8, 16, or 32), and a downbeat time. Pitches and velocities are copied. |
| `notation.py` | Two-staff MusicXML. Notes that share a start and an end become a chord. Independent durations stay in separate voices. Barline crossings are tied. |
| `diagnostics.py` | Waveform, spectrogram, piano roll, a tone rendering, and a mix of that rendering with the working audio. |
| `ui.py` | Local Streamlit site. Home page: transcribe controls plus a library grid of transcribed videos. Project page (`/project?project=<folder>`): synced video and score, audio and plots, the note table, and the files. |
| `library.py` | Builds the library cards by merging the parse history with the project folders on disk, and resolves the `project` address parameter to a folder. |
| `playback.py` | YouTube id parsing, plus a localhost server that serves each project's synced playback page and static score page. The project page frames those with `?embed=1`. |
| `analysis.py` | Empty hooks for voice separation, imitation, sequences, harmony, and cadences. Every method raises `NotImplementedError`. |
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
- A Streamlit site: a home page with the transcribe controls and a thumbnail library, and one page per project with the video and score in sync, the audio and plots, the note table, and the files.
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

That opens http://127.0.0.1:8765 on the PC where the command is running. The home page starts transcriptions and lists what has been transcribed; clicking a thumbnail opens that project's page. `--project <folder>` opens a project page directly. `baroque-transcribe` exists only after `.venv\Scripts\activate`. In PowerShell or Command Prompt that is the activation line above. In Git Bash it is `source .venv/Scripts/activate`.

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
- The staff split is a pitch threshold, middle C by default. It is not a hand assignment and not a contrapuntal voice analysis.
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
5. Only after that, implement the hooks in `analysis.py`: real voice separation, imitation, sequences, harmonic progressions, and cadences. Those should annotate `notes.json`. They should not change `raw.mid`.
