from baroque_transcribe.schema import (
    NoteEvent,
    PedalEvent,
    Transcription,
    clean_notes,
    transcription_from_dict,
)


def test_velocity_is_clamped_into_midi_range():
    notes = clean_notes(
        [
            NoteEvent(pitch=60, onset=0.0, offset=0.2, velocity=128),
            NoteEvent(pitch=61, onset=0.0, offset=0.2, velocity=0),
            NoteEvent(pitch=10, onset=0.0, offset=0.2, velocity=40),
        ]
    )
    assert [(note.pitch, note.velocity) for note in notes] == [(60, 127), (61, 1)]


def test_times_are_rounded_for_editing():
    notes = clean_notes([NoteEvent(pitch=60, onset=0.6499999761581421, offset=1.0, velocity=40)])
    assert notes[0].onset == 0.65


def test_checkpoint_header_accepts_the_zenodo_pickle(tmp_path):
    from baroque_transcribe.checkpoint import is_checkpoint_payload

    path = tmp_path / "model.pth"
    path.write_bytes(b"\x80\x02rest-of-an-old-checkpoint")
    assert is_checkpoint_payload(path)
    path.write_bytes(b"PK\x03\x04zip-checkpoint")
    assert is_checkpoint_payload(path)
    path.write_bytes(b"<html>error</html>")
    assert not is_checkpoint_payload(path)


def test_notes_json_roundtrip(tmp_path):
    original = Transcription(
        notes=[NoteEvent(64, 0.25, 0.5, 70)],
        pedals=[PedalEvent(0.0, 1.0)],
        model="bytedance-note-pedal",
        source="recital.wav",
    )
    path = tmp_path / "notes.json"
    original.write(path)
    loaded = transcription_from_dict(__import__("json").loads(path.read_text()))
    assert loaded.notes[0].pitch == 64
    assert loaded.pedals[0].offset == 1.0
    assert loaded.model == "bytedance-note-pedal"
