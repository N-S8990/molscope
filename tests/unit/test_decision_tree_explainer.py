"""Unit tests for DecisionTreeExplainer and Study.explain_clusters."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless test runner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from molscope.analysis.discovery.decision_tree import (
    DecisionRule,
    DecisionTreeExplainer,
    _format_threshold,
    _infer_feature_unit,
)
from molscope.api import Study

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"


def _generate_synthetic_biophysical_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic conformational data with known ground-truth boundaries."""
    rng = np.random.default_rng(seed)
    n_per_cluster = n_samples // 2

    # Cluster 1: Closed state (short Phe47-Trp23 distance, Tyr60 chi1 > -60 deg)
    dist_c1 = rng.uniform(2.5, 5.5, size=n_per_cluster)
    chi1_c1 = rng.uniform(-55.0, 10.0, size=n_per_cluster)
    noise_c1 = rng.normal(0, 1.0, size=n_per_cluster)

    # Cluster 2: Open state (longer Phe47-Trp23 distance, Tyr60 chi1 <= -65 deg)
    dist_c2 = rng.uniform(6.5, 9.5, size=n_per_cluster)
    chi1_c2 = rng.uniform(-120.0, -70.0, size=n_per_cluster)
    noise_c2 = rng.normal(0, 1.0, size=n_per_cluster)

    features = pd.DataFrame({
        "dist(Phe47, Trp23)": np.concatenate([dist_c1, dist_c2]),
        "chi1(Tyr60)": np.concatenate([chi1_c1, chi1_c2]),
        "noise_coord": np.concatenate([noise_c1, noise_c2]),
    })

    # Integer labels 1 and 2
    labels = pd.Series([1] * n_per_cluster + [2] * n_per_cluster, name="cluster")
    return features, labels


class TestDecisionTreeExplainerHelpers:
    """Test unit inference and threshold formatting."""

    def test_format_threshold(self):
        assert _format_threshold(5.80001, precision=2) == "5.8"
        assert _format_threshold(-60.0001, precision=2) == "-60.0"
        assert _format_threshold(5.85, precision=2) == "5.85"
        assert _format_threshold(0.0, precision=2) == "0.0"
        assert _format_threshold(12.0, precision=2) == "12.0"

    def test_infer_feature_unit_automatic(self):
        assert _infer_feature_unit("dist(Phe47, Trp23)") == " Å"
        assert _infer_feature_unit("CA_dist_12_45") == " Å"
        assert _infer_feature_unit("chi1(Tyr60)") == "°"
        assert _infer_feature_unit("phi_angle") == "°"
        assert _infer_feature_unit("psi(Ser12)") == "°"
        assert _infer_feature_unit("noise_coordinate") == ""

    def test_infer_feature_unit_explicit(self):
        custom_units = {"rmsd_core": "nm", "angle_deg": "°"}
        assert _infer_feature_unit("rmsd_core", custom_units) == " nm"
        assert _infer_feature_unit("angle_deg", custom_units) == "°"


class TestDecisionTreeExplainerInitialization:
    """Test constructor parameters and input validation."""

    def test_default_init(self):
        explainer = DecisionTreeExplainer()
        assert explainer.max_depth == 3
        assert explainer.min_samples_leaf == 5
        assert explainer.criterion == "gini"
        assert explainer.random_state == 42

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="max_depth must be an integer >= 1"):
            DecisionTreeExplainer(max_depth=0)
        with pytest.raises(ValueError, match="min_samples_leaf must be an integer >= 1"):
            DecisionTreeExplainer(min_samples_leaf=0)
        with pytest.raises(ValueError, match="criterion must be"):
            DecisionTreeExplainer(criterion="invalid_crit")

    def test_fit_invalid_inputs_raise(self):
        explainer = DecisionTreeExplainer()
        with pytest.raises(ValueError, match="X must contain at least one sample"):
            explainer.fit(pd.DataFrame(), pd.Series())

        with pytest.raises(ValueError, match="Length mismatch"):
            explainer.fit(pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([1, 2, 3]))

        with pytest.raises(ValueError, match="At least 2 unique clusters"):
            explainer.fit(pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([1, 1]))

        with pytest.raises(ValueError, match="non-numeric, NaN, or infinite"):
            explainer.fit(pd.DataFrame({"a": [1.0, np.nan]}), pd.Series([1, 2]))


class TestDecisionTreeRuleExtraction:
    """Test rule extraction on synthetic biophysical clusters."""

    def test_extracts_rules_with_precision_recall_and_units(self):
        X, y = _generate_synthetic_biophysical_data(n_samples=200, seed=42)
        explainer = DecisionTreeExplainer(max_depth=3, min_samples_leaf=5, random_state=42)
        explainer.fit(X, y)

        assert 1 in explainer.rules
        assert 2 in explainer.rules

        rules_c2 = explainer.get_rules(2)
        assert len(rules_c2) >= 1

        top_rule_c2 = explainer.get_top_rule(2)
        assert isinstance(top_rule_c2, DecisionRule)
        assert top_rule_c2.cluster == 2
        assert top_rule_c2.precision >= 0.90
        assert top_rule_c2.recall >= 0.90
        assert top_rule_c2.f1 >= 0.90

        rule_str = top_rule_c2.rule_string
        assert rule_str.startswith("IF ")
        assert "THEN Cluster 2" in rule_str
        assert "[precision=" in rule_str
        assert "recall=" in rule_str

        # Check that biophysical units were inferred and attached
        assert "Å" in rule_str or "°" in rule_str

    def test_rule_matches_and_apply(self):
        X, y = _generate_synthetic_biophysical_data(n_samples=100, seed=42)
        explainer = DecisionTreeExplainer(max_depth=2, min_samples_leaf=5)
        explainer.fit(X, y)

        rule = explainer.get_top_rule(1)
        mask = rule.matches(X)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(X)
        assert mask.sum() > 0

        subset = rule.apply(X)
        assert len(subset) == mask.sum()
        assert not subset.empty

    def test_rules_dataframe_table(self):
        X, y = _generate_synthetic_biophysical_data(n_samples=100, seed=42)
        explainer = DecisionTreeExplainer(max_depth=2, min_samples_leaf=5)
        explainer.fit(X, y)

        df = explainer.rules_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "Cluster" in df.columns
        assert "Rule" in df.columns
        assert "Precision" in df.columns
        assert "Recall" in df.columns
        assert "F1" in df.columns
        assert "Support" in df.columns
        assert len(df) >= 2


class TestDecisionTreeFeatureImportance:
    """Test Gini feature importance and normalized scoring."""

    def test_gini_importance_ranks_key_features_over_noise(self):
        X, y = _generate_synthetic_biophysical_data(n_samples=200, seed=42)
        explainer = DecisionTreeExplainer(max_depth=3, min_samples_leaf=5, random_state=42)
        explainer.fit(X, y)

        imp_df = explainer.get_feature_importances(cluster=1)
        assert isinstance(imp_df, pd.DataFrame)
        assert list(imp_df.columns[:4]) == ["Feature", "Gini_Importance", "Normalized_Score", "Relative_Importance"]

        # Top feature should have Normalized_Score == 1.0
        assert pytest.approx(imp_df.iloc[0]["Normalized_Score"], 1e-4) == 1.0

        # Discriminating feature (dist or chi1) should have much higher importance than noise
        noise_row = imp_df[imp_df["Feature"] == "noise_coord"]
        top_row = imp_df.iloc[0]
        if not noise_row.empty:
            assert top_row["Gini_Importance"] > noise_row.iloc[0]["Gini_Importance"]

    def test_get_feature_importances_combined(self):
        X, y = _generate_synthetic_biophysical_data(n_samples=100, seed=42)
        explainer = DecisionTreeExplainer(max_depth=2, min_samples_leaf=5)
        explainer.fit(X, y)

        all_imp = explainer.get_feature_importances()
        assert isinstance(all_imp, pd.DataFrame)
        assert "Cluster" in all_imp.columns
        assert len(all_imp["Cluster"].unique()) == 2


class TestDecisionTreePairwiseComparison:
    """Test State A vs State B pairwise explainability."""

    def test_pairwise_fit_extracts_rules_for_both_states(self):
        X, y_int = _generate_synthetic_biophysical_data(n_samples=150, seed=42)
        # String state names
        y = y_int.map({1: "Active_State", 2: "Inactive_State"})

        explainer = DecisionTreeExplainer(max_depth=2, min_samples_leaf=5, random_state=42)
        explainer.fit_pairwise(X, y, state_a="Active_State", state_b="Inactive_State")

        assert explainer.mode == "pairwise"
        assert "Active_State" in explainer.rules
        assert "Inactive_State" in explainer.rules

        rule_act = explainer.get_top_rule("Active_State")
        assert "THEN Active_State" in rule_act.rule_string
        assert rule_act.precision >= 0.85

        rule_inact = explainer.get_top_rule("Inactive_State")
        assert "THEN Inactive_State" in rule_inact.rule_string
        assert rule_inact.precision >= 0.85

        # Feature importances for pairwise comparison
        imp_df = explainer.get_feature_importances("Active_State")
        assert not imp_df.empty


class TestDecisionTreeTextSummaryAndDiagram:
    """Test text summary, ASCII export, and matplotlib diagram generation."""

    def test_export_text_and_get_text_summary(self):
        X, y = _generate_synthetic_biophysical_data(n_samples=100, seed=42)
        explainer = DecisionTreeExplainer(max_depth=2, min_samples_leaf=5)
        explainer.fit(X, y)

        ascii_tree = explainer.export_text()
        assert "|---" in ascii_tree

        summary = explainer.get_text_summary()
        assert "MolScope Decision Tree Explainer Summary" in summary
        assert "Extracted Boolean Rules:" in summary
        assert "Top Gini Explanatory Features:" in summary
        assert "[precision=" in summary

    def test_plot_tree_and_generate_diagram(self, tmp_path):
        X, y = _generate_synthetic_biophysical_data(n_samples=100, seed=42)
        explainer = DecisionTreeExplainer(max_depth=2, min_samples_leaf=5)
        explainer.fit(X, y)

        # Matplotlib plot
        fig = explainer.plot_tree(cluster=1)
        assert fig is not None

        # Save diagram
        out_png = tmp_path / "tree_diag.png"
        fig_saved = explainer.generate_diagram(cluster=1, output_path=out_png)
        assert out_png.is_file()
        assert out_png.stat().st_size > 0
        plt.close("all")


class TestStudyExplainClustersIntegration:
    """Test integration of explain_clusters into the MolScope Study API."""

    def test_study_explain_clusters_with_mock_state_assignments(self):
        X, y = _generate_synthetic_biophysical_data(n_samples=100, seed=42)

        # Create a stateful study and mock features + state_assignments
        study = Study(project_id="test_dt_study")

        from molscope.execution.pipeline import MDContactMLPipeline
        pipeline = MDContactMLPipeline()

        manifest = pd.DataFrame({
            "trajectory_id": [f"t{i // 50}" for i in range(100)],
            "replica_id": [f"r{i // 50}" for i in range(100)],
            "ensemble_id": ["ens_default"] * 100,
            "frame_index": list(range(100)),
            "frame_order": list(range(100)),
            "time_ps": [float(i) for i in range(100)],
            "condition": ["control" if i < 50 else "treated" for i in range(100)],
        })
        pipeline.load_feature_matrix(X, manifest)

        # Attach mock state outputs
        pipeline.outputs = {
            "state_assignments": pd.DataFrame({
                "State": [0 if val == 1 else 1 for val in y],
                "State_Label": [f"state_{0 if val == 1 else 1}" for val in y],
            })
        }
        study._pipeline = pipeline

        # Explain conformational states
        explainer = study.explain_clusters(clustering_name="states", max_depth=3)
        assert isinstance(explainer, DecisionTreeExplainer)
        assert explainer is study.get_tree_explainer()

        rules = explainer.get_rules()
        assert len(rules) >= 2
        assert any("state_0" in r.rule_string for r in rules)
        assert any("state_1" in r.rule_string for r in rules)

    def test_study_explain_clusters_by_manifest_column(self):
        X, _ = _generate_synthetic_biophysical_data(n_samples=100, seed=42)

        study = Study(project_id="test_condition_study")
        from molscope.execution.pipeline import MDContactMLPipeline
        pipeline = MDContactMLPipeline()

        manifest = pd.DataFrame({
            "trajectory_id": ["t0"] * 100,
            "replica_id": ["r0"] * 100,
            "ensemble_id": ["ens_default"] * 100,
            "frame_index": list(range(100)),
            "frame_order": list(range(100)),
            "time_ps": [float(i) for i in range(100)],
            "condition": ["wildtype" if i < 50 else "mutant" for i in range(100)],
        })
        pipeline.load_feature_matrix(X, manifest)
        pipeline.outputs = {}
        study._pipeline = pipeline

        # Explain manifest condition labels
        explainer = study.explain_clusters(clustering_name="condition", max_depth=2)
        assert isinstance(explainer, DecisionTreeExplainer)
        assert "wildtype" in explainer.rules
        assert "mutant" in explainer.rules

    def test_study_explain_clusters_pairwise(self):
        X, y = _generate_synthetic_biophysical_data(n_samples=100, seed=42)

        study = Study(project_id="test_pairwise_study")
        from molscope.execution.pipeline import MDContactMLPipeline
        pipeline = MDContactMLPipeline()

        manifest = pd.DataFrame({
            "trajectory_id": ["t0"] * 100,
            "replica_id": ["r0"] * 100,
            "ensemble_id": ["ens_default"] * 100,
            "frame_index": list(range(100)),
            "frame_order": list(range(100)),
            "time_ps": [float(i) for i in range(100)],
            "State_Label": [f"state_{val}" for val in y],
        })
        pipeline.load_feature_matrix(X, manifest)
        pipeline.outputs = {"state_assignments": manifest[["State_Label"]].copy()}
        study._pipeline = pipeline

        # Explain pairwise comparison
        explainer = study.explain_clusters(
            clustering_name="states",
            mode="pairwise",
            pair=("state_1", "state_2"),
            max_depth=2,
        )
        assert explainer.mode == "pairwise"
        assert "state_1" in explainer.rules
        assert "state_2" in explainer.rules

    def test_study_explain_clusters_errors_when_unrun_without_systems(self):
        study = Study(project_id="empty_study")
        with pytest.raises(RuntimeError, match="Run the study"):
            study.explain_clusters(clustering_name="states")

    def test_study_explain_clusters_errors_on_unknown_clustering(self):
        X, _ = _generate_synthetic_biophysical_data(n_samples=50, seed=42)
        study = Study(project_id="test_err_study")
        from molscope.execution.pipeline import MDContactMLPipeline
        pipeline = MDContactMLPipeline()
        manifest = pd.DataFrame({
            "trajectory_id": ["t0"] * 50,
            "replica_id": ["r0"] * 50,
            "ensemble_id": ["ens0"] * 50,
            "frame_index": list(range(50)),
            "frame_order": list(range(50)),
            "time_ps": list(range(50)),
        })
        pipeline.load_feature_matrix(X, manifest)
        pipeline.outputs = {}
        study._pipeline = pipeline

        with pytest.raises(ValueError, match="Clustering 'nonexistent' not found"):
            study.explain_clusters(clustering_name="nonexistent")
