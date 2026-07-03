import numpy as np

from src.preprocessing.ubi_fights import (
    label_segments,
    load_frame_labels,
    sample_neutral_segments,
)


def test_load_frame_labels_lines_and_commas(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("0\n0\n1\n1\n0\n")
    assert load_frame_labels(p).tolist() == [0, 0, 1, 1, 0]
    p.write_text("0,0,1,1,0")
    assert load_frame_labels(p).tolist() == [0, 0, 1, 1, 0]


def test_label_segments_contiguous_spans():
    labels = np.array([0] * 100 + [1] * 80 + [0] * 60, dtype=np.int8)
    segs = label_segments(labels, min_frames=48)
    assert segs == [(0, 100, False), (100, 180, True), (180, 240, False)]


def test_label_segments_drops_short_spans():
    labels = np.array([0] * 100 + [1] * 10 + [0] * 100, dtype=np.int8)
    segs = label_segments(labels, min_frames=48)
    # the 10-frame fight blip is too short to score with a 64-frame window
    assert all(not is_fight for _, _, is_fight in segs)


def test_sample_neutral_segments_caps_volume():
    segments = [(0, 3000, False), (3000, 3100, True)]
    chunks = sample_neutral_segments(segments, max_per_video=4, segment_frames=150)
    assert len(chunks) == 4
    assert all(end - start <= 150 for start, end in chunks)
    # spread across the span, not clustered at the start
    assert chunks[-1][0] > 1500
