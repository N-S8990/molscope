"""Audit evidence must survive cache reuse and session restoration."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from molscope.generic_features import GenericFeatureConfig
from molscope.manifest import TrajectoryInput
from molscope.pipeline import MDContactMLPipeline
from molscope.storage import extraction_cache_key


def test_unavailable_assessment_and_summaries_survive_cache_and_session():
    path = Path(__file__).parents[1] / "fixtures" / "nucleic_base_centers.pdb"
    inputs = [TrajectoryInput(path, path, "t1", "r1", "e1")]
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache"
        first = MDContactMLPipeline().load_trajectories(
            inputs, GenericFeatureConfig(max_frames=1),
            plugins=["protein.backbone_torsions"], cache_dir=cache,
        )
        evidence = first.extraction_evidence[0]
        assert evidence["input_integrity"]["trajectory"]["complete_scan"]
        assert evidence["trajectory_features"]
        assert evidence["provenance"]
        assert any(a["feature_id"] == "protein.backbone_torsions" and a["status"] == "not_applicable" for a in evidence["assessments"])
        with patch("molscope.pipeline.GenericFeatureExtractor.extract", side_effect=AssertionError("cache missed")):
            second = MDContactMLPipeline().load_trajectories(
                inputs, GenericFeatureConfig(max_frames=1),
                plugins=["protein.backbone_torsions"], cache_dir=cache,
            )
        assert first.extraction_evidence == second.extraction_evidence
        session = Path(directory) / "session"
        second.save_session(session)
        restored = MDContactMLPipeline.load_session(session)
        assert restored.extraction_evidence == second.extraction_evidence


def test_invalid_input_preflight_prevents_extraction():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.pdb"
        path.write_text("invalid molecular input\n", encoding="utf-8")
        with patch("molscope.pipeline.GenericFeatureExtractor.extract") as extract:
            with pytest.raises(ValueError, match="Input preflight failed"):
                MDContactMLPipeline().load_trajectories([TrajectoryInput(path, path, "t1", "r1", "e1")])
            extract.assert_not_called()


def test_plugin_parameter_changes_invalidate_cache_identity():
    path = Path(__file__).parents[1] / "fixtures" / "nucleic_base_centers.pdb"
    first = extraction_cache_key(path, path, generic_config=GenericFeatureConfig(), plugins={"protein.ca_distances_contacts": {"version": "1", "parameters": {"contact_cutoff_angstrom": 8.0}}})
    second = extraction_cache_key(path, path, generic_config=GenericFeatureConfig(), plugins={"protein.ca_distances_contacts": {"version": "1", "parameters": {"contact_cutoff_angstrom": 6.0}}})
    assert first != second
