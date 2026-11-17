"""Unit tests for JSON schema validation and versioning of MolScope project specifications."""

import json
from pathlib import Path
import tempfile
import pytest

from molscope.core.project import load_project_spec, SCHEMA_VERSION

SCHEMAS_DIR = Path(__file__).parents[2] / "schemas"
BENCHMARK_PROJECT = Path(__file__).parents[2] / "studies" / "2rjy_benchmark" / "project.json"


def test_project_schema_file_exists_and_parses():
    schema_path = SCHEMAS_DIR / "project.schema.json"
    assert schema_path.is_file(), "schemas/project.schema.json must exist."
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["title"] == "MolScope Project Specification"
    assert "properties" in schema
    assert "schema_version" in schema["properties"]


def test_benchmark_project_matches_schema_version():
    if BENCHMARK_PROJECT.is_file():
        spec = json.loads(BENCHMARK_PROJECT.read_text(encoding="utf-8"))
        assert spec.get("schema_version", "1") == SCHEMA_VERSION


def test_unsupported_schema_version_raises_clear_error(tmp_path):
    invalid_project = tmp_path / "future_project.json"
    payload = {
        "schema_version": "99",
        "project_id": "future_experiment",
        "systems": []
    }
    invalid_project.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported project manifest schema_version"):
        load_project_spec(invalid_project)


def test_schema_migration_path():
    """Verify that projects without explicit schema_version default to SCHEMA_VERSION gracefully."""
    raw = {
        "project_id": "legacy_v0",
        "systems": [{
            "system_id": "sys1",
            "topology": "tests/fixtures/1A22.pdb",
            "ensembles": [{
                "ensemble_id": "ens1",
                "replicas": [{"replica_id": "rep1", "trajectory": "tests/fixtures/1A22.pdb"}]
            }]
        }]
    }
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        f.write(json.dumps(raw))
        f.flush()
        spec = load_project_spec(f.name)
        assert spec.project_id == "legacy_v0"
        assert len(spec.systems) == 1
