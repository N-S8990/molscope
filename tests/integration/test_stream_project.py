from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from molscope.project_spec import AnalysisSpec, ProjectSpec
from molscope.stream_project import run_stream_project, streaming_config
from molscope.streaming import CA, GEOMETRY


def test_stream_project_parameter_mapping_and_rejection():
    project = ProjectSpec("study", (), analysis=AnalysisSpec(
        generic_feature_parameters={"max_frames": 10, "frame_stride": 2, "use_pbc": False},
        plugins=(CA,), plugin_parameters={CA: {"selection": "resid 1:20", "max_pairs": 30}}))
    config = streaming_config(project)
    assert config.frame_stride == 2 and config.max_frames == 10
    assert config.plugins == (GEOMETRY, CA)
    assert config.ca_selection == "resid 1:20" and config.max_pairs == 30
    with pytest.raises(ValueError, match="unsupported"):
        streaming_config(replace(project, analysis=AnalysisSpec(generic_feature_parameters={"unknown": 1})))
    with pytest.raises(ValueError, match="use_pbc"):
        streaming_config(replace(project, analysis=AnalysisSpec(generic_feature_parameters={"use_pbc": True})))


def test_project_preserves_identity_and_independent_analysis(tmp_path, monkeypatch):
    import molscope.stream_project as module
    inputs = [SimpleNamespace(topology="top", trajectory="traj", trajectory_id=f"s:e:r{i}",
                              system_id="s", ensemble_id="s:e", replica_id=f"r{i}",
                              metadata={"label_condition": "declared"}) for i in range(2)]
    project = SimpleNamespace(project_id="study", analysis=AnalysisSpec(), labels=(), prediction_tasks=(),
                              validate_paths=lambda: [], trajectory_inputs=lambda: inputs)
    monkeypatch.setattr(module, "load_project_spec", lambda path: project)
    monkeypatch.setattr(module, "profile_system", lambda *args: SimpleNamespace(to_dict=lambda: {"atom_count": 2}))
    def extract(top, traj, directory, **kwargs):
        directory.mkdir()
        return directory / "manifest.json"
    calls = []
    def analyze(source, directory, config):
        calls.append((source, directory))
        directory.mkdir()
        return {"frames_analyzed": 10}
    monkeypatch.setattr(module, "extract_streaming", extract)
    monkeypatch.setattr(module, "analyze_stream", analyze)
    report = run_stream_project("project.json", tmp_path / "run")
    assert len(calls) == 2 and calls[0][0] != calls[1][0]
    assert report["status"] == "completed"
    assert "not comparable" in report["scope"]
    identity = json.loads((tmp_path / "run/trajectory_00001/identity.json").read_text())
    assert identity["trajectory_id"] == "s:e:r1"
    assert identity["metadata"]["label_condition"] == "declared"
    with pytest.raises(FileExistsError):
        run_stream_project("project.json", tmp_path / "run")
