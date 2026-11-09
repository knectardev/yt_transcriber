"""Filenames and musical defaults shared by the command line and the UI."""

MODEL_SAMPLE_RATE = 16_000
WORKING_PEAK = 0.95
RAW_MIDI_TEMPO = 120.0
RAW_MIDI_RESOLUTION = 480

DEFAULT_SUBDIVISION = 16
DEFAULT_TIME_SIGNATURE = (4, 4)
DEFAULT_STAFF_SPLIT = 60
DEFAULT_UI_PORT = 8765

# Paper defaults from Kong et al., high-resolution piano transcription.
BYTEDANCE_ONSET_THRESHOLD = 0.3
BYTEDANCE_OFFSET_THRESHOLD = 0.3
BYTEDANCE_FRAME_THRESHOLD = 0.1
BYTEDANCE_PEDAL_OFFSET_THRESHOLD = 0.2

# Basic Pitch defaults, with a shorter minimum note so ornaments are less
# likely to be discarded. The Melodia post-process is off: it suppresses
# polyphony, which is the opposite of what a contrapuntal texture needs.
BASIC_PITCH_ONSET_THRESHOLD = 0.5
BASIC_PITCH_FRAME_THRESHOLD = 0.3
BASIC_PITCH_MIN_NOTE_MS = 50.0

SOURCE_WAV = "source.wav"
WORKING_WAV = "working.wav"
NOTES_JSON = "notes.json"
PEDAL_JSON = "pedal.json"
RAW_MID = "raw.mid"
QUANTIZED_MID = "quantized.mid"
SCORE_XML = "score.musicxml"
MANIFEST = "manifest.json"
DIAGNOSTICS_DIR = "diagnostics"

CHECKPOINT_FILENAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_URL = (
    "https://zenodo.org/records/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
# The upstream loader treats anything under 160 MB as a failed download.
CHECKPOINT_MIN_BYTES = 160_000_000
