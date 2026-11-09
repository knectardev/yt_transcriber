from pathlib import Path

import numpy as np

from baroque_transcribe.analysis import CounterpointAnalyzer
from baroque_transcribe.cli import main
from baroque_transcribe.schema import NoteEvent, Transcription
from baroque_transcribe.tempo import estimate_tempo


def test_periodic_bursts_estimate_near_120_bpm():
    sample_rate = 16000
    audio = np.zeros(sample_rate * 8, dtype=np.float64)
    for index in range(16):
        start = int(index * 0.5 * sample_rate)
        audio[start : start + 500] = 0.9
    estimate = estimate_tempo(audio, sample_rate)
    assert estimate.bpm is not None
    nearby = [estimate.bpm, *estimate.candidates]
    assert any(abs(candidate - 120) < 6 or abs(candidate - 60) < 4 for candidate in nearby)


def test_prepare_command_writes_the_working_copy(tmp_path):
    sample_rate = 22050
    tone = (0.2 * np.sin(2 * np.pi * 330 * np.linspace(0, 1.0, sample_rate, endpoint=False))).astype(np.float32)
    source = tmp_path / "phrase.wav"
    import soundfile as sf

    sf.write(source, tone, sample_rate)
    output = tmp_path / "project"
    code = main(["prepare", str(source), "-o", str(output), "--start", "0.1", "--end", "0.6"])
    assert code == 0
    assert (output / "source.wav").exists()
    assert (output / "working.wav").exists()
    assert (output / "manifest.json").exists()


def test_quantize_command_rebuilds_a_score(tmp_path):
    project = tmp_path / "edited"
    project.mkdir()
    Transcription(
        notes=[
            NoteEvent(67, 0.0, 0.5, 80),
            NoteEvent(43, 0.0, 1.0, 50),
        ],
        model="edited",
        source="hands",
    ).write(project / "notes.json")
    code = main(
        [
            "quantize",
            str(project),
            "--tempo",
            "96",
            "--time-signature",
            "3/4",
            "--subdivision",
            "8",
            "--no-diagnostics",
        ]
    )
    assert code == 0
    assert (project / "raw.mid").exists()
    assert (project / "quantized.mid").exists()
    assert (project / "score.musicxml").exists()
    assert (project / "pedal.json").exists()


def test_diagnostics_write_plots_and_a_comparison_mix(tmp_path):
    import numpy as np
    import soundfile as sf

    from baroque_transcribe.diagnostics import render_diagnostics

    sample_rate = 16000
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sample_rate, endpoint=False))).astype(np.float32)
    working = tmp_path / "working.wav"
    sf.write(working, tone, sample_rate)
    written = render_diagnostics(
        working,
        [NoteEvent(69, 0.1, 0.4, 80)],
        tmp_path / "diagnostics",
        position=0.25,
    )
    for path in written.values():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_analysis_hooks_are_not_implemented():
    analyzer = CounterpointAnalyzer()
    try:
        analyzer.split_voices(Transcription())
    except NotImplementedError:
        return
    raise AssertionError("voice splitting should stay unimplemented")
