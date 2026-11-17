"""Unit tests for Free Energy Landscape (FES) and Potential of Mean Force (PMF)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molscope.api import Study
from molscope.reporting.free_energy import (
    BOLTZMANN_KCAL,
    BOLTZMANN_KJ,
    FreeEnergyLandscapeResult,
    compute_free_energy_landscape,
    plot_landscape_interactive,
    plot_landscape_static,
)

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"


def _generate_synthetic_2well(n_samples: int = 6000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D coordinates sampled from a 2-well potential.

    Well 1 is centered at (-2.0, 0.0) with 60% population.
    Well 2 is centered at (+2.0, 0.0) with 40% population.
    """
    rng = np.random.default_rng(seed)
    n1 = int(n_samples * 0.6)
    n2 = n_samples - n1
    w1 = rng.normal(loc=[-2.0, 0.0], scale=[0.35, 0.35], size=(n1, 2))
    w2 = rng.normal(loc=[2.0, 0.0], scale=[0.35, 0.35], size=(n2, 2))
    data = np.vstack([w1, w2])
    return data[:, 0], data[:, 1]


def _generate_synthetic_3well(n_samples: int = 7000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D coordinates sampled from a 3-well potential.

    Well 1: (-2.0, -1.0) with 50% population (deepest minimum).
    Well 2: (+2.0, -1.0) with 30% population (intermediate minimum).
    Well 3: (0.0, +2.0) with 20% population (shallowest minimum).
    """
    rng = np.random.default_rng(seed)
    n1 = int(n_samples * 0.5)
    n2 = int(n_samples * 0.3)
    n3 = n_samples - n1 - n2
    w1 = rng.normal(loc=[-2.0, -1.0], scale=[0.35, 0.35], size=(n1, 2))
    w2 = rng.normal(loc=[2.0, -1.0], scale=[0.35, 0.35], size=(n2, 2))
    w3 = rng.normal(loc=[0.0, 2.0], scale=[0.35, 0.35], size=(n3, 2))
    data = np.vstack([w1, w2, w3])
    return data[:, 0], data[:, 1]


class TestFreeEnergyCalculation:
    def test_free_energy_calculation_2well(self):
        """Test PMF calculation and basin detection on a synthetic 2-well potential."""
        x, y = _generate_synthetic_2well(n_samples=6000, seed=42)
        result = compute_free_energy_landscape(
            x=x,
            y=y,
            temperature_k=300.0,
            bins=50,
            units="kcal/mol",
            smooth=1.0,
        )

        assert isinstance(result, FreeEnergyLandscapeResult)
        assert result.units == "kcal/mol"
        assert result.temperature_k == 300.0
        assert result.delta_g.shape == (50, 50)
        assert result.density.shape == (50, 50)

        # Global minimum must be exactly 0.0 kcal/mol
        assert result.min_free_energy == pytest.approx(0.0, abs=1e-7)
        assert np.nanmin(result.delta_g) == pytest.approx(0.0, abs=1e-7)

        # Sampling region has finite non-negative free energy
        finite_vals = result.delta_g[np.isfinite(result.delta_g)]
        assert len(finite_vals) > 0
        assert np.all(finite_vals >= 0.0)

        # Basin detection: exactly 2 basins should be identified
        basins = result.basins
        assert isinstance(basins, pd.DataFrame)
        assert len(basins) == 2

        # Basin 1 is the global minimum near (-2, 0)
        b1 = basins.iloc[0]
        assert b1["basin_id"] == 1
        assert b1["free_energy"] == pytest.approx(0.0, abs=1e-6)
        assert b1["x"] == pytest.approx(-2.0, abs=0.35)
        assert b1["y"] == pytest.approx(0.0, abs=0.35)

        # Basin 2 is near (+2, 0) with Delta G > 0
        b2 = basins.iloc[1]
        assert b2["basin_id"] == 2
        assert b2["x"] == pytest.approx(2.0, abs=0.35)
        assert b2["y"] == pytest.approx(0.0, abs=0.35)

        # Theoretical Delta Delta G for 60% vs 40% population at 300 K:
        # Delta Delta G = -k_B * T * ln(0.4 / 0.6) = 0.001987204 * 300 * ln(1.5) ~ 0.2417 kcal/mol
        expected_ddg = -BOLTZMANN_KCAL * 300.0 * np.log(0.4 / 0.6)
        assert b2["free_energy"] == pytest.approx(expected_ddg, abs=0.1)

    def test_free_energy_calculation_3well(self):
        """Test PMF calculation and basin detection on a synthetic 3-well potential."""
        x, y = _generate_synthetic_3well(n_samples=7000, seed=42)
        result = compute_free_energy_landscape(
            x=x,
            y=y,
            temperature_k=300.0,
            bins=50,
            units="kcal/mol",
            smooth=1.0,
        )

        assert isinstance(result, FreeEnergyLandscapeResult)
        basins = result.basins
        assert len(basins) == 3

        # Basins must be sorted in strictly ascending free energy order
        assert basins["free_energy"].is_monotonic_increasing
        assert basins.iloc[0]["free_energy"] == pytest.approx(0.0, abs=1e-6)

        # Check basin coordinates match the 3 clusters
        # Well 1: (-2, -1) -> deepest
        assert basins.iloc[0]["x"] == pytest.approx(-2.0, abs=0.35)
        assert basins.iloc[0]["y"] == pytest.approx(-1.0, abs=0.35)

        # Well 2: (+2, -1) -> second deepest
        assert basins.iloc[1]["x"] == pytest.approx(2.0, abs=0.35)
        assert basins.iloc[1]["y"] == pytest.approx(-1.0, abs=0.35)

        # Well 3: (0, +2) -> third
        assert basins.iloc[2]["x"] == pytest.approx(0.0, abs=0.35)
        assert basins.iloc[2]["y"] == pytest.approx(2.0, abs=0.35)

        # Check theoretical Delta Delta G rankings:
        # DDG(1->2) = -k_B*T*ln(0.3 / 0.5) ~ 0.304 kcal/mol
        # DDG(1->3) = -k_B*T*ln(0.2 / 0.5) ~ 0.546 kcal/mol
        expected_ddg_2 = -BOLTZMANN_KCAL * 300.0 * np.log(0.3 / 0.5)
        expected_ddg_3 = -BOLTZMANN_KCAL * 300.0 * np.log(0.2 / 0.5)
        assert basins.iloc[1]["free_energy"] == pytest.approx(expected_ddg_2, abs=0.12)
        assert basins.iloc[2]["free_energy"] == pytest.approx(expected_ddg_3, abs=0.15)


class TestUnitsConversion:
    def test_units_conversion_kcal_vs_kj(self):
        """Test energy unit conversion between kcal/mol and kJ/mol."""
        x, y = _generate_synthetic_2well(n_samples=3000, seed=123)

        res_kcal = compute_free_energy_landscape(x, y, temperature_k=300.0, units="kcal/mol")
        res_kj = compute_free_energy_landscape(x, y, temperature_k=300.0, units="kJ/mol")

        assert res_kcal.units == "kcal/mol"
        assert res_kj.units == "kJ/mol"

        conversion_ratio = BOLTZMANN_KJ / BOLTZMANN_KCAL
        assert conversion_ratio == pytest.approx(4.1840002, rel=1e-6)

        # Compare positive finite free energy grid points
        valid = np.isfinite(res_kcal.delta_g) & np.isfinite(res_kj.delta_g) & (res_kcal.delta_g > 1e-4)
        assert np.sum(valid) > 50

        ratios = res_kj.delta_g[valid] / res_kcal.delta_g[valid]
        np.testing.assert_allclose(ratios, conversion_ratio, rtol=1e-5)

        # Compare basin free energy values
        assert len(res_kcal.basins) == len(res_kj.basins)
        for b_kcal, b_kj in zip(res_kcal.basins.itertuples(), res_kj.basins.itertuples()):
            assert b_kj.free_energy == pytest.approx(b_kcal.free_energy * conversion_ratio, rel=1e-4)

    def test_case_insensitive_units(self):
        x, y = _generate_synthetic_2well(n_samples=500, seed=1)
        r1 = compute_free_energy_landscape(x, y, units="KCAL/MOL")
        assert r1.units == "kcal/mol"
        r2 = compute_free_energy_landscape(x, y, units="kJ/mol")
        assert r2.units == "kJ/mol"
        r3 = compute_free_energy_landscape(x, y, units="kj")
        assert r3.units == "kJ/mol"

    def test_unsupported_units_raises(self):
        x, y = _generate_synthetic_2well(n_samples=500, seed=1)
        with pytest.raises(ValueError, match="Unsupported energy units"):
            compute_free_energy_landscape(x, y, units="electron_volts")


class TestBasinDetection:
    def test_basin_dataframe_schema_and_properties(self):
        x, y = _generate_synthetic_2well(n_samples=2000, seed=99)
        result = compute_free_energy_landscape(x, y, temperature_k=310.0, units="kcal/mol")

        basins = result.basins
        expected_cols = ["basin_id", "x", "y", "free_energy", "probability"]
        assert list(basins.columns) == expected_cols
        assert len(basins) >= 1
        assert basins["basin_id"].iloc[0] == 1
        assert basins["free_energy"].iloc[0] == pytest.approx(0.0, abs=1e-6)

        # Dict access and property access on FreeEnergyLandscapeResult
        assert result["delta_g"] is result.delta_g
        assert result["basins"] is result.basins
        assert result.get("units") == "kcal/mol"
        assert result.get("nonexistent", 42) == 42
        assert result.min_free_energy == 0.0
        assert result.max_free_energy > 0.0

    def test_basin_max_energy_cutoff(self):
        x, y = _generate_synthetic_3well(n_samples=4000, seed=42)
        # With cutoff at 0.1 kcal/mol, only the global minimum (Basin 1) is retained
        result = compute_free_energy_landscape(x, y, max_energy_cutoff=0.1)
        assert len(result.basins) == 1
        assert result.basins.iloc[0]["basin_id"] == 1


class TestVisualizations:
    def test_plot_landscape_static(self, tmp_path):
        x, y = _generate_synthetic_2well(n_samples=2000, seed=42)
        result = compute_free_energy_landscape(x, y)

        out_png = tmp_path / "fes_static.png"
        saved_path = plot_landscape_static(
            result=result,
            output_path=out_png,
            xlabel="Collective Variable 1",
            ylabel="Collective Variable 2",
            title="Synthetic 2-Well Potential",
            cmap="viridis_r",
            levels=20,
        )

        assert saved_path.is_file()
        assert saved_path == out_png.resolve()
        assert out_png.stat().st_size > 5000  # Non-trivial image file

    def test_plot_landscape_interactive(self, tmp_path):
        x, y = _generate_synthetic_2well(n_samples=2000, seed=42)
        result = compute_free_energy_landscape(x, y)

        out_html = tmp_path / "fes_interactive.html"
        saved_path = plot_landscape_interactive(
            result=result,
            output_path=out_html,
            xlabel="TIC 1",
            ylabel="TIC 2",
            title="Interactive Kinetic FES",
            colorscale="Viridis",
        )

        assert saved_path.is_file()
        assert saved_path == out_html.resolve()
        assert out_html.stat().st_size > 10000

        content = out_html.read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "Interactive Kinetic FES" in content

    def test_result_convenience_plot_methods(self, tmp_path):
        x, y = _generate_synthetic_2well(n_samples=1000, seed=7)
        result = compute_free_energy_landscape(x, y)

        png_path = result.plot_static(tmp_path / "method_static.png")
        html_path = result.plot_interactive(tmp_path / "method_interactive.html")

        assert png_path.is_file()
        assert html_path.is_file()


class TestKDEMethod:
    def test_free_energy_calculation_with_kde(self):
        x, y = _generate_synthetic_2well(n_samples=1500, seed=42)
        result = compute_free_energy_landscape(
            x=x,
            y=y,
            method="kde",
            temperature_k=300.0,
            units="kcal/mol",
        )

        assert result.method == "kde"
        assert result.delta_g.shape == (50, 50)
        assert result.min_free_energy == pytest.approx(0.0, abs=1e-7)
        assert len(result.basins) == 2
        assert result.basins.iloc[0]["x"] == pytest.approx(-2.0, abs=0.35)
        assert result.basins.iloc[1]["x"] == pytest.approx(2.0, abs=0.35)


class TestValidationAndEdgeCases:
    def test_mismatched_coordinates_raises(self):
        with pytest.raises(ValueError, match="identical lengths"):
            compute_free_energy_landscape(x=[1.0, 2.0], y=[1.0])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            compute_free_energy_landscape(x=[1.0, 2.0], y=[1.0, 2.0])

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="Temperature must be strictly positive"):
            compute_free_energy_landscape(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0], temperature_k=-10.0)

    def test_invalid_bins_raises(self):
        with pytest.raises(ValueError, match="bins must be at least 2"):
            compute_free_energy_landscape(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0], bins=1)

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported density estimation method"):
            compute_free_energy_landscape(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0], method="spline")

    def test_filters_non_finite_coordinates(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        y = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0])
        result = compute_free_energy_landscape(x, y, bins=10)
        assert isinstance(result, FreeEnergyLandscapeResult)


def _create_multi_frame_trajectory(tmp_path: Path, n_frames: int = 25) -> tuple[Path, Path]:
    import MDAnalysis as mda
    fixture = Path(__file__).parents[1] / "fixtures" / "solvated_protein.pdb"
    u = mda.Universe(str(fixture))
    traj_path = tmp_path / "multi_traj.pdb"
    rng = np.random.default_rng(42)
    with mda.Writer(str(traj_path), len(u.atoms)) as w:
        for _ in range(n_frames):
            u.atoms.positions += rng.normal(0, 0.08, size=u.atoms.positions.shape)
            w.write(u)
    return fixture, traj_path


class TestStudyAPIIntegration:
    def test_study_plot_free_energy_landscape(self, tmp_path):
        """Test study.plot_free_energy_landscape() on an active Study session."""
        fixture, multi_pdb = _create_multi_frame_trajectory(tmp_path, n_frames=20)
        study = Study(project_id="test_fes_study", out_dir=tmp_path / "study_out")
        study.add_system(
            system_id="sys1",
            topology=fixture,
            trajectories=[multi_pdb],
        )
        study.configure_features(
            ["protein.ca_distances_contacts"],
            plugin_parameters={"protein.ca_distances_contacts": {"selection": "resid 1:10", "max_pairs": 50}},
        )

        out_dir = tmp_path / "fes_output"
        fes_dict = study.plot_free_energy_landscape(
            dim1="PC1",
            dim2="PC2",
            out_dir=out_dir,
            temperature_k=300.0,
            bins=25,
            units="kcal/mol",
        )

        assert "result" in fes_dict
        assert "static_plot" in fes_dict
        assert "interactive_plot" in fes_dict
        assert "basins_file" in fes_dict
        assert "basins" in fes_dict

        result = fes_dict["result"]
        assert isinstance(result, FreeEnergyLandscapeResult)
        assert result is study.get_free_energy_result()

        static_path = fes_dict["static_plot"]
        interactive_path = fes_dict["interactive_plot"]
        basins_path = fes_dict["basins_file"]

        assert static_path.is_file()
        assert interactive_path.is_file()
        assert basins_path.is_file()

        # Check basins CSV file
        df_basins = pd.read_csv(basins_path)
        assert "basin_id" in df_basins.columns
        assert "free_energy" in df_basins.columns

    def test_study_plot_free_energy_landscape_invalid_dimension_raises(self, tmp_path):
        fixture, multi_pdb = _create_multi_frame_trajectory(tmp_path, n_frames=20)
        study = Study(project_id="test_invalid_dim", out_dir=tmp_path / "study_out")
        study.add_system(
            system_id="sys1",
            topology=fixture,
            trajectories=[multi_pdb],
        )
        study.configure_features(
            ["protein.ca_distances_contacts"],
            plugin_parameters={"protein.ca_distances_contacts": {"selection": "resid 1:10", "max_pairs": 20}},
        )
        study.run(extract_only=True)

        with pytest.raises(ValueError, match="could not be resolved"):
            study.plot_free_energy_landscape(dim1="TOTALLY_NONEXISTENT_DIM", dim2="PC2")
