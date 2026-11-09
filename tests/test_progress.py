from baroque_transcribe.progress import (
    eta_label,
    fraction_for,
    load_seconds_per_window,
    pace_preview,
    remaining_seconds,
    save_pace,
    window_count,
    window_label,
)


def test_window_count_matches_the_ten_second_hop():
    assert window_count(0) == 0
    assert window_count(16_000 * 10) == 1
    assert window_count(16_000 * 20) == 3
    # 5:18 of audio pads to 320 seconds and becomes 63 windows, the count
    # printed as "Segment i / 63" by the ByteDance loop.
    assert window_count(16_000 * 318) == 63


def test_remaining_time_uses_the_finished_windows():
    assert remaining_seconds(0, 63, 0.0) is None
    assert remaining_seconds(10, 20, 100.0) == 100.0
    assert remaining_seconds(63, 63, 100.0) == 0.0


def test_eta_label_waits_for_the_first_window_then_speaks_the_rest():
    assert eta_label(0, 63, 0.0) == "Timing the first window…"
    assert eta_label(1, 2, 30.0) == "about 30 seconds left"
    elapsed = 540 * 22 / 41
    assert eta_label(22, 63, elapsed) == "about 9 minutes left"
    assert window_label(22, 63, elapsed) == "Window 22 of 63. about 9 minutes left"
    assert eta_label(63, 63, 100.0) == "Finishing the last window…"


def test_transcription_occupies_most_of_the_bar():
    start = fraction_for("transcribe", 0.0)
    end = fraction_for("transcribe", 1.0)
    assert abs(start - 0.18) < 1e-9
    assert abs(end - 0.92) < 1e-9
    assert abs(fraction_for("write", 1.0) - 1.0) < 1e-9


def test_saved_pace_previews_the_next_run(tmp_path):
    path = tmp_path / "pace.json"
    save_pace("cpu", 15.0, path=path)
    assert load_seconds_per_window("cpu", path=path) == 15.0
    assert load_seconds_per_window("cuda", path=path) is None
    assert pace_preview("cpu", 4, path=path) == (
        "about 1 minute on cpu, from the last run on this PC"
    )
