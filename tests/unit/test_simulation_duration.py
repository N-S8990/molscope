import pytest
from molscope.project_spec import _openmm_protocol


def test_physical_duration_and_save_interval_convert_to_steps():
    protocol = _openmm_protocol({"production_duration_ns": 10, "save_interval_ps": 10,
                                "timestep_femtoseconds": 2}, "simulation")
    assert protocol.production_steps == 5_000_000
    assert protocol.report_interval_steps == 5_000


def test_ambiguous_duration_rejected():
    with pytest.raises(ValueError, match="not both"):
        _openmm_protocol({"production_duration_ns": 1, "production_steps": 10}, "simulation")


def test_fractional_timestep_interval_rejected():
    with pytest.raises(ValueError, match="integer number"):
        _openmm_protocol({"save_interval_ps": .001, "timestep_femtoseconds": 2}, "simulation")
