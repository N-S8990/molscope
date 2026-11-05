import json

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest

from molscope import streaming
from molscope.plugins import ProteinCADistanceContactPlugin


@pytest.fixture
def inputs(tmp_path):
    universe = mda.Universe.empty(4, n_residues=4, atom_resindex=np.arange(4), trajectory=True)
    for name, values in {"name": ["CA"] * 4, "type": ["C"] * 4,
                         "resname": ["ALA"] * 4, "resid": [1, 2, 5, 9],
                         "segid": ["A"], "chainID": ["A"] * 4}.items():
        universe.add_TopologyAttr(name, values)
    topology, trajectory = tmp_path / "input.pdb", tmp_path / "input.dcd"
    universe.atoms.positions = np.arange(12).reshape(4, 3)
    universe.atoms.write(str(topology))
    with mda.Writer(str(trajectory), n_atoms=4) as writer:
        for frame in range(9):
            universe.atoms.positions = np.arange(12).reshape(4, 3) * (1 + frame / 20)
            writer.write(universe.atoms)
    return topology, trajectory


def table(path):
    manifest = json.loads(path.read_text())
    return manifest, pd.concat([pd.read_csv(path.parent / chunk["file"]) for chunk in manifest["chunks"]], ignore_index=True)


def test_stream_matches_standard_plugin_and_resume_is_idempotent(inputs, tmp_path):
    config = streaming.StreamingConfig(chunk_size=2, frame_stride=2, max_frames=4)
    output = tmp_path / "out"
    path = streaming.extract_streaming(*inputs, output, config=config)
    manifest, actual = table(path)
    expected = ProteinCADistanceContactPlugin().extract(*inputs, max_frames=4, frame_stride=2).frame_features
    np.testing.assert_allclose(actual[expected.columns], expected, atol=1e-10)
    assert actual.frame_index.tolist() == [0, 2, 4, 6]
    assert manifest["complete"] and len(manifest["chunks"]) == 2
    assert streaming.extract_streaming(*inputs, output, config=config, resume=True) == path
    pd.testing.assert_frame_equal(table(path)[1], actual)


def test_resume_after_failed_manifest_commit_rewrites_uncommitted_chunk(inputs, tmp_path, monkeypatch):
    original = streaming._atomic_json
    def fail_second_chunk(path, payload):
        if payload["total_rows"] == 4:
            raise OSError("simulated interruption")
        original(path, payload)
    monkeypatch.setattr(streaming, "_atomic_json", fail_second_chunk)
    config = streaming.StreamingConfig(chunk_size=2)
    output = tmp_path / "out"
    with pytest.raises(OSError, match="interruption"):
        streaming.extract_streaming(*inputs, output, config=config)
    assert json.loads((output / "manifest.json").read_text())["total_rows"] == 2
    monkeypatch.setattr(streaming, "_atomic_json", original)
    path = streaming.extract_streaming(*inputs, output, config=config, resume=True)
    manifest, actual = table(path)
    assert manifest["total_rows"] == 9
    assert actual.frame_index.tolist() == list(range(9))


def test_resume_rejects_configuration_source_and_chunk_mutations(inputs, tmp_path):
    output = tmp_path / "out"
    path = streaming.extract_streaming(*inputs, output)
    with pytest.raises(ValueError, match="configuration mismatch"):
        streaming.extract_streaming(*inputs, output, resume=True, config=streaming.StreamingConfig(frame_stride=2))
    manifest = json.loads(path.read_text())
    chunk = output / manifest["chunks"][0]["file"]
    chunk.write_text(chunk.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        streaming.extract_streaming(*inputs, output, resume=True)
    inputs[0].write_text(inputs[0].read_text() + "REMARK changed source\n")
    with pytest.raises(ValueError, match="sources"):
        streaming.extract_streaming(*inputs, output, resume=True)


def test_limits_and_unavailable_features_are_explicit(inputs, tmp_path):
    with pytest.raises(ValueError, match="max_pairs"):
        streaming.extract_streaming(*inputs, tmp_path / "pairs", config=streaming.StreamingConfig(max_pairs=1))
    with pytest.raises(ValueError, match="memory_budget"):
        streaming.extract_streaming(*inputs, tmp_path / "memory", config=streaming.StreamingConfig(memory_budget_mb=0.0001))
    path = streaming.extract_streaming(*inputs, tmp_path / "unsupported",
                                      config=streaming.StreamingConfig(plugins=(streaming.GEOMETRY, "protein.secondary_structure_dssp")))
    manifest, actual = table(path)
    assert manifest["assessments"][1]["status"] == "unsupported"
    assert len(manifest["feature_columns"]) == 3
    assert actual.shape == (9, 6)
    with pytest.raises(ValueError, match="imaging"):
        streaming.StreamingConfig(imaging_policy="minimum_image")
