from src.preprocessing.labeled_folder import find_labeled_videos, folder_label


def test_folder_label_negative_beats_fight_substring():
    assert folder_label("noFight") == 0
    assert folder_label("no_fight") == 0
    assert folder_label("NonViolence") == 0
    assert folder_label("normal") == 0
    assert folder_label("fight") == 1
    assert folder_label("Violence") == 1
    assert folder_label("Fights") == 1
    assert folder_label("random_folder") is None


def test_find_labeled_videos_by_subfolder(tmp_path):
    (tmp_path / "fight").mkdir()
    (tmp_path / "noFight").mkdir()
    (tmp_path / "misc").mkdir()
    (tmp_path / "fight" / "fi001.mp4").write_bytes(b"x")
    (tmp_path / "fight" / "fi002.avi").write_bytes(b"x")
    (tmp_path / "noFight" / "nf001.mp4").write_bytes(b"x")
    (tmp_path / "misc" / "unlabeled.mp4").write_bytes(b"x")  # no class folder -> skipped
    (tmp_path / "fight" / "notes.txt").write_bytes(b"x")  # non-video -> skipped

    found = find_labeled_videos(tmp_path)
    labels = {p.name: lbl for p, lbl in found}
    assert labels == {"fi001.mp4": 1, "fi002.avi": 1, "nf001.mp4": 0}
