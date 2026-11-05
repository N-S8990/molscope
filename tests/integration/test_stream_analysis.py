import json

import numpy as np
import pandas as pd
import pytest

from molscope.stream_analysis import Reservoir, RunningMoments, StreamAnalysisConfig, analyze_stream


def make_store(path, rows=83):
    path.mkdir()
    rng = np.random.default_rng(44)
    values = rng.normal(size=(rows, 3))
    chunks = []
    for i, start in enumerate(range(0, rows, 17)):
        table = pd.DataFrame(values[start:start + 17], columns=["a", "b", "c"])
        table.insert(0, "frame_index", np.arange(start, start + len(table)))
        name = f"chunk_{i}.csv"
        table.to_csv(path / name, index=False)
        chunks.append({"file": name, "rows": len(table)})
    (path / "manifest.json").write_text(json.dumps({"complete": True, "feature_columns": ["a", "b", "c"], "chunks": chunks, "total_rows": rows}))
    return values


def test_running_moments_matches_batch():
    values = np.random.default_rng(23).normal(size=(137, 4)) + 10000
    running = RunningMoments(4)
    for batch in np.array_split(values, 13):
        running.update(batch)
    assert running.count == len(values)
    np.testing.assert_allclose(running.mean, values.mean(axis=0))
    np.testing.assert_allclose(running.m2 / (running.count - 1), values.var(axis=0, ddof=1))
    np.testing.assert_array_equal(running.minimum, values.min(axis=0))
    np.testing.assert_array_equal(running.maximum, values.max(axis=0))


def test_reservoir_bounded_and_batch_independent():
    values = np.arange(3000).reshape(1000, 3)
    a, b = Reservoir(19, 3, 42), Reservoir(19, 3, 42)
    a.update(values)
    for batch in np.array_split(values, 37):
        b.update(batch)
    assert a.values.shape == (19, 3)
    assert a.seen == b.seen == 1000
    np.testing.assert_array_equal(a.sample, b.sample)
    assert len(np.unique(a.sample[:, 0])) == 19


def test_stream_analysis_all_frames_and_sample_scope(tmp_path):
    values = make_store(tmp_path / "source")
    output = tmp_path / "output"
    report = analyze_stream(tmp_path / "source", output, StreamAnalysisConfig(batch_rows=7, sample_rows=21))
    assert report["frames_analyzed"] == 83
    assert report["training_sample_rows"] == 21
    assert "sample-trained" in report["scope"]
    assert any("no replica stability or SAM-MFP" in line for line in report["limitations"])
    summary = pd.read_csv(output / "feature_summary.csv")
    np.testing.assert_allclose(summary.Mean, values.mean(axis=0))
    np.testing.assert_allclose(summary.Std, values.std(axis=0, ddof=1))
    assignments = pd.concat([pd.read_csv(output / name) for name in report["assignment_files"]])
    np.testing.assert_array_equal(assignments.frame_index, np.arange(83))
    assert len(assignments) == 83
    assert (output / "report.html").is_file()
    assert pd.read_csv(output / "state_populations.csv").Frame_Count.sum() == 83
    with pytest.raises(FileExistsError):
        analyze_stream(tmp_path / "source", output)


def test_stream_rejects_nonfinite_and_budget(tmp_path):
    make_store(tmp_path / "source")
    with pytest.raises(ValueError, match="memory budget"):
        analyze_stream(tmp_path / "source", tmp_path / "output", StreamAnalysisConfig(sample_rows=100000, memory_budget_mb=1))
    table = pd.read_csv(tmp_path / "source/chunk_0.csv")
    table.loc[0, "a"] = np.nan
    table.to_csv(tmp_path / "source/chunk_0.csv", index=False)
    with pytest.raises(ValueError, match="Nonfinite"):
        analyze_stream(tmp_path / "source", tmp_path / "output")


def test_stream_rejects_incomplete_and_row_mismatch(tmp_path):
    make_store(tmp_path / "source")
    path = tmp_path / "source/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["complete"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed"):
        analyze_stream(tmp_path / "source", tmp_path / "output")
    manifest["complete"] = True
    manifest["total_rows"] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="total_rows"):
        analyze_stream(tmp_path / "source", tmp_path / "output")
