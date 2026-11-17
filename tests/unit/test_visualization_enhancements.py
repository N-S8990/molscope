"""Unit tests for enhanced interpretability and visualization modules."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molscope.reporting.free_energy import (
    compute_pca_free_energy_landscape,
    plot_pca_free_energy,
)
from molscope.visualization.pdb_mapping import (
    extract_residues_from_feature_name,
    extract_residue_scores,
    map_feature_importance_to_pdb,
    generate_pymol_script,
    generate_chimerax_script,
    generate_3d_visualization,
)
from molscope.visualization.timeline import (
    compute_transition_matrix,
    plot_cluster_timeline,
    plot_state_transitions,
    generate_cluster_timeline_plots,
)
from molscope.visualization.rules import (
    plot_feature_importance,
    plot_decision_tree_rules,
    render_decision_tree_html,
)
from molscope.analysis.discovery.decision_tree import DecisionTreeExplainer

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"


class TestPDBMappingAnd3DVisualizers:
    """Tests for PDB B-factor importance mapping and 3D PyMOL / ChimeraX scripts."""

    def test_extract_residues_from_various_feature_patterns(self):
        # Pair patterns
        assert extract_residues_from_feature_name("chainA__ARG_37__chainA__LYS_42") == [37, 42]
        assert extract_residues_from_feature_name("CA_DIST__A__PHE_16__A__ALA_25") == [16, 25]

        # Sidechain dihedrals
        assert extract_residues_from_feature_name("SIDECHAIN_CHI1__ARG_37") == [37]
        assert extract_residues_from_feature_name("SIDECHAIN_chi2__TRP_45") == [45]

        # Chain residue pattern
        assert extract_residues_from_feature_name("__chain_A__ARG_37") == [37]
        assert extract_residues_from_feature_name("__X__VAL_12") == [12]

        # Standard residue or resSeq patterns
        assert extract_residues_from_feature_name("ARG_37") == [37]
        assert extract_residues_from_feature_name("residue_14") == [14]
        assert extract_residues_from_feature_name("res 99") == [99]

        # Global feature without residue
        assert extract_residues_from_feature_name("GENERIC_rg") == []

    def test_extract_residue_scores_aggregations(self):
        df = pd.DataFrame([
            {"Feature": "CA_DIST__A__ARG_10__A__LYS_20", "Gini_Importance": 0.4},
            {"Feature": "SIDECHAIN_CHI1__ARG_10", "Gini_Importance": 0.8},
            {"Feature": "GENERIC_sasa", "Gini_Importance": 0.1},
        ])

        # Default max aggregation
        max_scores = extract_residue_scores(df, aggregation="max")
        assert max_scores[10] == pytest.approx(0.8)
        assert max_scores[20] == pytest.approx(0.4)

        # Sum aggregation
        sum_scores = extract_residue_scores(df, aggregation="sum")
        assert sum_scores[10] == pytest.approx(1.2)
        assert sum_scores[20] == pytest.approx(0.4)

        # Mean aggregation
        mean_scores = extract_residue_scores(df, aggregation="mean")
        assert mean_scores[10] == pytest.approx(0.6)

    def test_map_feature_importance_to_pdb(self, tmp_path):
        out_pdb = tmp_path / "annotated.pdb"
        scores = {1: 10.0, 16: 50.0, 25: 100.0}

        mapped_path = map_feature_importance_to_pdb(
            pdb_input_path=FIXTURE_PDB,
            importance_df=scores,
            pdb_output_path=out_pdb,
            normalize=True,
            max_scale=100.0,
        )
        assert mapped_path.is_file()

        lines = mapped_path.read_text(encoding="utf-8").splitlines()
        atom_lines = [l for l in lines if l.startswith("ATOM  ")]
        assert len(atom_lines) > 0

        # Check that residue 25 has 100.00 in B-factor cols (61-66)
        res25_atoms = [l for l in atom_lines if int(l[22:26].strip()) == 25]
        assert len(res25_atoms) > 0
        bfactor_25 = float(res25_atoms[0][60:66].strip())
        assert bfactor_25 == pytest.approx(100.0, abs=0.1)

        # Residue 16 has 50.00
        res16_atoms = [l for l in atom_lines if int(l[22:26].strip()) == 16]
        assert len(res16_atoms) > 0
        bfactor_16 = float(res16_atoms[0][60:66].strip())
        assert bfactor_16 == pytest.approx(50.0, abs=0.1)

    def test_generate_3d_visualization_suite(self, tmp_path):
        df = pd.DataFrame([
            {"Feature": "CA_DIST__A__PHE_16__A__ALA_25", "Gini_Importance": 0.95},
            {"Feature": "SIDECHAIN_CHI1__ARG_10", "Gini_Importance": 0.45},
        ])

        outputs = generate_3d_visualization(
            topology_path=FIXTURE_PDB,
            importance_df=df,
            output_dir=tmp_path / "3d_viz",
            basename="test_protein",
            top_label_count=5,
        )

        assert outputs["pdb"].is_file()
        assert outputs["pymol_script"].is_file()
        assert outputs["chimerax_script"].is_file()
        assert outputs["residue_scores_csv"].is_file()
        assert 16 in outputs["top_residues"]
        assert 25 in outputs["top_residues"]

        pml_content = outputs["pymol_script"].read_text(encoding="utf-8")
        assert "cartoon putty" in pml_content
        assert "spectrum b, blue_white_red" in pml_content
        assert "select top_discriminators" in pml_content

        cxc_content = outputs["chimerax_script"].read_text(encoding="utf-8")
        assert "color bfactor palette" in cxc_content


class TestClusterTimelineAndTransitions:
    """Tests for cluster timeline and Markovian state transition plots."""

    @pytest.fixture
    def mock_trajectory_assignments(self):
        # 2 trajectories, alternating between states 0, 1, 2
        data = []
        # Trajectory 1: spends time in 0 then transitions to 1
        for t in range(50):
            data.append({"trajectory_id": "traj_1", "time_ps": float(t * 10), "State": 0})
        for t in range(50, 100):
            data.append({"trajectory_id": "traj_1", "time_ps": float(t * 10), "State": 1})

        # Trajectory 2: transitions between 1 and 2
        for t in range(50):
            data.append({"trajectory_id": "traj_2", "time_ps": float(t * 10), "State": 1})
        for t in range(50, 100):
            data.append({"trajectory_id": "traj_2", "time_ps": float(t * 10), "State": 2})

        return pd.DataFrame(data)

    def test_compute_transition_matrix_probabilities(self, mock_trajectory_assignments):
        matrix = compute_transition_matrix(mock_trajectory_assignments, normalize=True)
        assert matrix.shape == (3, 3)
        # Each row should sum to 1.0 (or 0 for absorbing/unvisited states)
        row_sums = matrix.sum(axis=1).to_numpy()
        assert np.allclose(row_sums, 1.0)
        # High self-transition probability (state persistence)
        assert matrix.iloc[0, 0] > 0.9

    def test_plot_cluster_timeline(self, mock_trajectory_assignments, tmp_path):
        out_png = tmp_path / "timeline.png"
        res = plot_cluster_timeline(mock_trajectory_assignments, out_png)
        assert res.is_file()
        assert res.stat().st_size > 1000

    def test_plot_state_transitions(self, mock_trajectory_assignments, tmp_path):
        out_png = tmp_path / "transitions.png"
        res = plot_state_transitions(mock_trajectory_assignments, out_png)
        assert res.is_file()
        assert res.stat().st_size > 1000

    def test_generate_cluster_timeline_plots(self, mock_trajectory_assignments, tmp_path):
        out_dir = tmp_path / "timeline_suite"
        outputs = generate_cluster_timeline_plots(mock_trajectory_assignments, out_dir)
        assert outputs["timeline"].is_file()
        assert outputs["transitions"].is_file()
        assert outputs["transition_matrix_csv"].is_file()


class TestFeatureImportanceAndDecisionTreeRules:
    """Tests for feature importance bar plots and decision tree rule visualization."""

    @pytest.fixture
    def mock_features_and_states(self):
        np.random.seed(42)
        n = 100
        # Synthetic features separating State A and State B
        x1 = np.concatenate([np.random.normal(3.0, 0.5, n // 2), np.random.normal(8.0, 0.5, n // 2)])
        x2 = np.concatenate([np.random.normal(50.0, 10.0, n // 2), np.random.normal(-30.0, 10.0, n // 2)])
        x3 = np.random.normal(0.0, 1.0, n)  # noise

        X = pd.DataFrame({
            "DIST_PHE16_ALA25": x1,
            "CHI1_ARG37": x2,
            "NOISE_VAR": x3,
        })
        y = pd.Series(["State_A"] * (n // 2) + ["State_B"] * (n // 2))
        return X, y

    def test_plot_feature_importance(self, tmp_path):
        df_imp = pd.DataFrame([
            {"Feature": "DIST_PHE16_ALA25", "Gini_Importance": 0.75},
            {"Feature": "CHI1_ARG37", "Gini_Importance": 0.20},
            {"Feature": "NOISE_VAR", "Gini_Importance": 0.05},
        ])
        out_png = tmp_path / "feat_imp.png"
        res = plot_feature_importance(df_imp, out_png)
        assert res.is_file()
        assert res.stat().st_size > 1000

    def test_plot_decision_tree_rules_and_html(self, mock_features_and_states, tmp_path):
        X, y = mock_features_and_states
        explainer = DecisionTreeExplainer(max_depth=2, random_state=42)
        explainer.fit(X, y)

        # Plot diagram
        out_png = tmp_path / "dt_rules.png"
        res = plot_decision_tree_rules(explainer, out_png)
        assert res.is_file()
        assert res.stat().st_size > 1000

        # Render HTML
        html_str = render_decision_tree_html(explainer)
        assert "dt-rule-card" in html_str
        assert "State_A" in html_str
        assert "Precision" in html_str
        assert "Recall" in html_str


class TestPCAFreeEnergyLandscape:
    """Tests for 2D Free Energy Landscape computation and plotting on leading PCA components."""

    def test_compute_and_plot_pca_free_energy(self, tmp_path):
        np.random.seed(42)
        # Generate 2 conformational clusters on PC1 and PC2
        cluster1 = np.random.multivariate_normal([-2.0, -1.0], [[0.3, 0.0], [0.0, 0.3]], 500)
        cluster2 = np.random.multivariate_normal([2.5, 1.5], [[0.4, 0.0], [0.0, 0.4]], 300)
        coords = np.vstack([cluster1, cluster2])

        df_coords = pd.DataFrame(coords, columns=["PC1", "PC2"])

        fes_res = compute_pca_free_energy_landscape(
            df_coords,
            pc_x="PC1",
            pc_y="PC2",
            temperature_k=300.0,
            units="kcal/mol",
        )
        assert fes_res.min_free_energy == pytest.approx(0.0, abs=1e-5)
        assert fes_res.max_free_energy > 2.0
        assert len(fes_res.basins) >= 2

        # Plot
        outputs = plot_pca_free_energy(
            df_coords,
            output_dir=tmp_path / "fes",
            prefix="pca_fes",
            interactive=True,
        )
        assert outputs["static"].is_file()
        assert outputs["interactive"].is_file()
        assert outputs["basins_csv"].is_file()
