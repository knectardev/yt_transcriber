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

Use Python 3.11 or 3.12. FFmpeg must be on `PATH` (`ffmpeg` and `ffprobe`). On Windows, `winget install Gyan.FFmpeg`, then open a new terminal.

```text
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install torch
python -m pip install -r requirements.txt
python -m pip install -e .
baroque-transcribe doctor
```

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
  working.wav
  notes.json
  pedal.json
  raw.mid
  quantized.mid
  score.musicxml
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
```

`--subdivision` is the smallest note value: `8` eighths, `16` sixteenths, `32` thirty-seconds. Notes that share a start and an end on one staff become a chord. Notes with different durations stay in separate voices. A note that crosses a barline is tied.

If you omit `--time-signature`, the score uses 4/4 and `manifest.json` records that this was an assumption. `--tempo auto` estimates one tempo from onset spacing and lists half-tempo and double-tempo candidates, because that estimator often locks onto the wrong metrical level. One tempo cannot represent rubato.

## Check the result

```text
baroque-transcribe ui
```

That opens a local site at <http://127.0.0.1:8765> with two kinds of page.

The home page has the controls that start a transcription (YouTube URL, local file, upload, or a MIDI file to draw without running the model; excerpt and notation settings are under an expander) and, below them, the library: a grid of the videos and recordings that already have a transcription. Each YouTube video appears once with its thumbnail; several runs of the same video are listed as chips on its card. A recording without a video gets a plain tile. Clicking a card opens that transcription's page. A run named in the parse history but missing from this computer is shown greyed out.

The project page, `/project?project=<folder>`, is one transcription. The header gives the note and pedal counts, excerpt length, tempo, meter, and model. **Play with score** opens the video with the measures following it in a new tab, and **Score only** opens the static two-staff score; the same synced view is also framed in the first tab of the page. The other tabs hold the working audio and the mix of that audio with a simple tone at each detected pitch (a check of timing, not a piano sound), the plots, the editable note table with the tempo, meter, subdivision, and staff split used to rebuild MIDI and MusicXML, and the downloadable files with the run's warnings. A project transcribed from a local file can have a YouTube address attached on that page so the score can follow the video.

`baroque-transcribe ui --project transcriptions\piece` opens straight to that project's page.

The same plots are written by the `transcribe` command. `baroque-transcribe diagnostics transcriptions\baroque_improvisation --position 12.5` redraws them with a cursor.

Import `working.wav` and `raw.mid` into a DAW if you want to solo the transcription against the recording. Use the raw file for that. The quantized file is metronomic on purpose.

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

Every `prepare`, `transcribe`, and `quantize` adds a folder under `data/parses/events/`. Each folder is one run. `event.json` is written once and left as it was, so history added on two machines merges as new folders.

The folder also holds the score files from that run, when the run produced them: `notes.json`, `pedal.json`, `raw.mid`, `quantized.mid`, and `score.musicxml`. `source.wav`, `working.wav`, and the diagnostic wavs stay in `transcriptions/` and are only described in `event.json`. Cookie files and absolute home-directory paths are not stored. The machine name in the event is the computer that wrote it.

`score.pdf` is listed in every event with `"present": false` and `"status": "planned"`. A later `baroque-transcribe pdf <project>` command will write that PDF next to `score.musicxml` by asking music21 for a MusicXML PDF, which needs MuseScore on `PATH`. The PDF will be copied into the event folder as a new file, with a sibling `pdf.json` saying when and on which machine. `event.json` stays as first written. The review page can then offer the PDF beside MusicXML.

```text
baroque-transcribe history
baroque-transcribe history import transcriptions
```

Import reads each `manifest.json` once. Running it again skips a project already imported with the same video (or file name), excerpt, and folder name.

Commit `data/parses/` and pull it on the other machine. The history and the score files come with the pull. The audio does not, so a YouTube download still has to be run on a computer where YouTube is signed in.

## Layout

| Piece | Role |
| --- | --- |
| `audio.py` | YouTube or local file, trim, normalize |
| `transcribe.py` | ByteDance and Basic Pitch |
| `midi_export.py` | Raw and quantized MIDI |
| `quantize.py` | Grid, tempo, meter, downbeat |
| `notation.py` | Two-staff MusicXML |
| `diagnostics.py` | Plots and the comparison mix |
| `analysis.py` | Later voice, imitation, sequence, harmony, and cadence hooks |
| `parse_store.py` | JSON history of each run, with the score files and a reserved PDF slot |
| `library.py` | Cards for the home page: the parse history merged with the folders under `transcriptions/` |
| `playback.py` | YouTube id parsing and the localhost server that serves the synced playback page and the score page |
| `ui.py` | The local site: home page with the transcribe controls and the library, and one page per project |
| `cli.py` | `prepare`, `transcribe`, `quantize`, `diagnostics`, `doctor`, `ui`, `history` |

`notes.json` is the stable intermediate those later analyzers should read. They should write annotations beside the transcription and leave `raw.mid` alone.

## License of the models

The ByteDance inference package and Basic Pitch are Apache-2.0. The piano checkpoint is downloaded from Zenodo record 4034264 and is not stored in this repository. Basic Pitch's ONNX file ships inside its own package.
