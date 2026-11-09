from datetime import datetime
from pathlib import Path

from baroque_transcribe.pipeline import holds_saved_work, unique_output_dir, unique_path


def test_empty_directory_is_used_as_given(tmp_path: Path):
    target = tmp_path / "piece"
    assert unique_output_dir(target) == target
    assert holds_saved_work(target) is False


def test_existing_quantized_midi_is_left_in_place(tmp_path: Path):
    target = tmp_path / "piece"
    target.mkdir()
    (target / "quantized.mid").write_bytes(b"MThd")
    (target / "notes.json").write_text("{}", encoding="utf-8")
    claimed = unique_output_dir(target, now=datetime(2026, 11, 7, 16, 45, 0))
    assert claimed == tmp_path / "piece-20261107-164500"
    assert (target / "quantized.mid").read_bytes() == b"MThd"
    assert not claimed.exists()


def test_timestamp_collision_adds_a_number(tmp_path: Path):
    target = tmp_path / "piece"
    target.mkdir()
    (target / "raw.mid").write_bytes(b"MThd")
    (tmp_path / "piece-20261107-164500").mkdir()
    claimed = unique_output_dir(target, now=datetime(2026, 11, 7, 16, 45, 0))
    assert claimed == tmp_path / "piece-20261107-164500-2"


def test_unique_filename_keeps_an_existing_file(tmp_path: Path):
    existing = tmp_path / "quantized.mid"
    existing.write_bytes(b"old")
    claimed = unique_path(existing, now=datetime(2026, 11, 7, 16, 45, 0))
    assert claimed == tmp_path / "quantized-20261107-164500.mid"
    assert existing.read_bytes() == b"old"
