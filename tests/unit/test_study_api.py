"""Unit tests for the fluent Study API."""

from pathlib import Path

import pandas as pd
import pytest

from molscope.api import Study

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"


def test_study_initialization_and_config():
    study = Study(project_id="test_api_study")
    study.add_system(
        system_id="test_sys",
        topology=FIXTURE_PDB,
        trajectories=[FIXTURE_PDB],
        condition_label="baseline",
    )
    study.configure_imaging(unwrap=True, center_selection="protein")
    study.configure_features(["protein.ca_distances_contacts"], plugin_parameters={"protein.ca_distances_contacts": {"max_pairs": 50}})

    spec = study.to_project_spec()
    assert spec.project_id == "test_api_study"
    assert len(spec.systems) == 1
    assert spec.analysis.imaging["unwrap"] is True
    assert spec.analysis.imaging["center_selection"] == "protein"
    assert "protein.ca_distances_contacts" in spec.analysis.plugins


def test_study_extract_only(tmp_path):
    study = Study(project_id="extract_test", out_dir=tmp_path / "study_out")
    study.add_system(
        system_id="sys1",
        topology=FIXTURE_PDB,
        trajectories=[FIXTURE_PDB],
    )
    study.configure_features(
        ["protein.ca_distances_contacts"],
        plugin_parameters={"protein.ca_distances_contacts": {"selection": "resid 1:5", "max_pairs": 100}},
    )
    study.run(extract_only=True)

    features = study.get_features()
    manifest = study.get_manifest()

    assert not features.empty
    assert not manifest.empty
    assert "trajectory_id" in manifest.columns
    assert (
        (tmp_path / "study_out" / "frame_manifest.parquet").is_file()
        or (tmp_path / "study_out" / "frame_manifest.csv").is_file()
    )
