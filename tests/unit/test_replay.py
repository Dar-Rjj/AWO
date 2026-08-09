from pathlib import Path

from awo.benchmarks.replay import detect_header_row, replay_result_csv


def test_replay_accepts_extra_metadata_row(tmp_path: Path) -> None:
    path = tmp_path / "result.csv"
    path.write_text(
        "0.50000_20240926_112034\n"
        "question,prediction,expected_output,score,cost\n"
        'q,"The answer",answer,1.0,0.1\n'
        'q2,wrong,right,0.0,0.2\n',
        encoding="utf-8",
    )

    assert detect_header_row(path) == 1
    replay = replay_result_csv("hotpotqa", path)
    assert replay["records"] == 2
    assert replay["mismatches"] == 0
    assert replay["stored_mean"] == replay["replayed_mean"] == 0.5
