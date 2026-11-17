"""High-level interactive Python API for MolScope.

Enables fluent, in-notebook exploratory analysis, featurization,
Pareto prioritization (SAM-MFP), predictive modeling, and 3D visualization.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from molscope.analysis.discovery.contact_kpca import ContactKPCAConfig, ContactKPCAResult, fit_contact_kpca
from molscope.analysis.discovery.decision_tree import DecisionTreeExplainer
from molscope.analysis.discovery.density_peaks import DPAConfig, DPAResult, cluster_dpa
from molscope.analysis.discovery.tica import TICAConfig, TICAResult, fit_tica
from molscope.core.identity import TrajectoryInput
from molscope.core.project import (
    AnalysisSpec,
    EnsembleSpec,
    LabelSpec,
    PredictionTask,
    ProjectSpec,
    ReplicaSpec,
    SystemSpec,
)
from molscope.execution.discovery import DiscoveryPipelineConfig
from molscope.execution.pipeline import MDContactMLPipeline
from molscope.features.geometry.generic import GenericFeatureConfig
from molscope.io.transformations import ImagingPolicy
from molscope.profiling.system import SystemProfile, profile_system
from molscope.reporting.free_energy import (
    FreeEnergyLandscapeResult,
    compute_free_energy_landscape,
    plot_landscape_interactive,
    plot_landscape_static,
)
from molscope.reporting.pymol import export_3d_visualization, extract_residue_scores


class Study:
    """Stateful, interactive MolScope study for notebooks and Python scripts.

    Example:
        >>> study = Study(project_id="gpr6_exploration")
        >>> study.add_system("receptor", "inputs/topology.pdb", ["inputs/rep1.xtc", "inputs/rep2.xtc"])
        >>> study.configure_features(["protein.ca_distances_contacts", "protein.backbone_torsions"])
        >>> study.configure_imaging(unwrap=True, center_selection="protein")
        >>> results = study.run()
        >>> pareto_front = study.get_pareto_front()
        >>> study.export_3d()
    """

    def __init__(
        self,
        project_id: str = "molscope_study",
        out_dir: str | Path | None = None,
    ) -> None:
        self.project_id = project_id
        self.out_dir = Path(out_dir) if out_dir is not None else Path(f"./results_{project_id}")
        self._systems: list[SystemSpec] = []
        self._labels: list[LabelSpec] = []
        self._prediction_tasks: list[PredictionTask] = []
        self._plugins: list[str] = []
        self._plugin_parameters: dict[str, dict[str, Any]] = {}
        self._generic_parameters: dict[str, Any] = {}
        self._imaging_policy: ImagingPolicy = ImagingPolicy()
        self._pipeline: MDContactMLPipeline | None = None
        self._tica_result: TICAResult | None = None
        self._contact_kpca_result: ContactKPCAResult | None = None
        self._dpa_result: DPAResult | None = None
        self._tree_explainer: DecisionTreeExplainer | None = None
        self._fes_result: FreeEnergyLandscapeResult | None = None

    @classmethod
    def from_project(cls, project_spec_path: str | Path, out_dir: str | Path | None = None) -> Study:
        """Create a Study initialized from a project specification JSON or dict."""
        from molscope.core.project import load_project_spec
        spec = load_project_spec(project_spec_path)
        study = cls(project_id=spec.project_id, out_dir=out_dir)
        study._systems = list(spec.systems)
        study._labels = list(spec.labels)
        study._prediction_tasks = list(spec.prediction_tasks)
        if spec.analysis:
            study._plugins = list(spec.analysis.plugins)
            study._plugin_parameters = dict(spec.analysis.plugin_parameters)
            study._generic_parameters = dict(spec.analysis.generic_feature_parameters)
            study._imaging_policy = spec.analysis.imaging
        return study

    def add_system(
        self,
        system_id: str,
        topology: str | Path,
        trajectories: str | Path | Sequence[str | Path],
        ensemble_id: str = "default_ensemble",
        condition_label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Study":
        """Add a molecular system and its associated trajectory replica(s)."""
        top_path = Path(topology).resolve()
        traj_list: list[Path] = []
        if isinstance(trajectories, (str, Path)):
            traj_list = [Path(trajectories).resolve()]
        else:
            traj_list = [Path(t).resolve() for t in trajectories]

        replicas: list[ReplicaSpec] = []
        for i, traj in enumerate(traj_list, start=1):
            replica_id = f"rep_{i:02d}"
            rep_meta = dict(metadata or {})
            if condition_label is not None:
                rep_meta["condition"] = condition_label
            replicas.append(ReplicaSpec(replica_id=replica_id, trajectory=traj, metadata=rep_meta))

        ensemble = EnsembleSpec(ensemble_id=ensemble_id, replicas=tuple(replicas))
        system = SystemSpec(
            system_id=system_id,
            topology=top_path,
            ensembles=(ensemble,),
            metadata=dict(metadata or {}),
        )
        self._systems.append(system)
        return self

    def add_label(
        self,
        label_id: str,
        assignments: Mapping[str, str | int | float | bool],
        label_type: str = "categorical",
        description: str = "",
    ) -> "Study":
        """Add trajectory-level categorical or numerical labels for supervised learning."""
        label = LabelSpec(
            label_id=label_id,
            label_type=label_type,
            level="trajectory",
            source="user_study",
            assignments=dict(assignments),
            description=description,
        )
        self._labels.append(label)
        return self

    def configure_features(
        self,
        plugins: Sequence[str] | None = None,
        plugin_parameters: Mapping[str, Mapping[str, Any]] | None = None,
        generic_parameters: Mapping[str, Any] | None = None,
    ) -> "Study":
        """Select feature plugins and their numerical options."""
        if plugins is not None:
            self._plugins = list(plugins)
        if plugin_parameters is not None:
            self._plugin_parameters = {k: dict(v) for k, v in plugin_parameters.items()}
        if generic_parameters is not None:
            self._generic_parameters = dict(generic_parameters)
        return self

    def configure_imaging(
        self,
        unwrap: bool = True,
        center_selection: str | None = "protein",
        center_to: str = "box",
        wrap: bool = False,
        fit_selection: str | None = None,
        require_pbc: bool = False,
    ) -> "Study":
        """Configure coordinate unwrapping, centering, and PBC handling."""
        self._imaging_policy = ImagingPolicy(
            unwrap=unwrap,
            center_selection=center_selection,
            center_to=center_to,
            wrap=wrap,
            fit_selection=fit_selection,
            require_pbc=require_pbc,
        )
        return self

    def to_project_spec(self) -> ProjectSpec:
        """Export current study state into a validated ProjectSpec."""
        analysis = AnalysisSpec(
            generic_feature_parameters=self._generic_parameters,
            plugins=tuple(self._plugins),
            plugin_parameters=self._plugin_parameters,
            imaging=self._imaging_policy.to_dict() if hasattr(self._imaging_policy, "to_dict") else (self._imaging_policy if isinstance(self._imaging_policy, dict) else {}),
        )
        return ProjectSpec(
            project_id=self.project_id,
            systems=tuple(self._systems),
            labels=tuple(self._labels),
            prediction_tasks=tuple(self._prediction_tasks),
            analysis=analysis,
        )

    def profile_systems(self) -> list[dict[str, Any]]:
        """Run system profiling across all registered topology/trajectory inputs."""
        profiles = []
        for system in self._systems:
            first_traj = system.ensembles[0].replicas[0].trajectory if system.ensembles and system.ensembles[0].replicas else None
            p = profile_system(system.topology, first_traj)
            profiles.append(p.to_dict())
        return profiles

    def run(
        self,
        components: int = 5,
        states: int = 2,
        supervised_folds: int = 3,
        target_column: str | None = None,
        feature_costs: pd.DataFrame | None = None,
        extract_only: bool = False,
    ) -> "Study":
        """Run extraction, quality checks, SAM-MFP Pareto discovery, and predictive ML."""
        project = self.to_project_spec()
        config = DiscoveryPipelineConfig(
            pca_components=components,
            state_count=states,
            supervised_splits=supervised_folds,
        )
        pipeline = MDContactMLPipeline(config)
        pipeline.load_project(project)

        if extract_only:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            pipeline.save_session(self.out_dir)
            self._pipeline = pipeline
            return self

        # Automatically resolve target if labels are defined and target_column is None
        if target_column is None and self._labels:
            cat_labels = [l for l in self._labels if l.label_type == "categorical"]
            if len(cat_labels) == 1:
                target_column = f"metadata_label_{cat_labels[0].label_id}"

        pipeline.run(target_column=target_column, feature_costs=feature_costs)
        pipeline.write_results(self.out_dir)
        self._pipeline = pipeline
        return self

    def get_features(self) -> pd.DataFrame:
        """Return the extracted frame feature table."""
        if self._pipeline is None or self._pipeline.features is None:
            raise RuntimeError("Run the study before requesting features.")
        return self._pipeline.features.copy()

    def get_manifest(self) -> pd.DataFrame:
        """Return the trajectory frame manifest with replica/condition identities."""
        if self._pipeline is None or self._pipeline.manifest is None:
            raise RuntimeError("Run the study before requesting manifest.")
        return self._pipeline.manifest.copy()

    def get_pareto_front(self) -> pd.DataFrame:
        """Return the SAM-MFP Pareto-optimal candidates."""
        if self._pipeline is None:
            raise RuntimeError("Run the study before requesting Pareto candidates.")
        table = self._pipeline.table("sam_mfp_candidates")
        return table[table["SAM_MFP_Pareto_Front"] == True].copy()

    def get_quality_metrics(self) -> pd.DataFrame:
        """Return the feature quality metrics (ESS, replica consistency, entropy)."""
        if self._pipeline is None:
            raise RuntimeError("Run the study before requesting quality metrics.")
        return self._pipeline.table("feature_quality_metrics")

    def run_tica(
        self,
        lag_time: int = 1,
        components: int = 2,
        shrinkage: float = 1e-4,
        dt: float | None = None,
    ) -> TICAResult:
        """Fit Time-lagged Independent Component Analysis (TICA) on extracted features.

        Parameters
        ----------
        lag_time : int, default=1
            Lag time tau (in frames) for time-lagged covariance within continuous trajectories.
        components : int, default=2
            Number of slowest kinetic dimensions (TICs) to compute and retain.
        shrinkage : float, default=1e-4
            Regularization shrinkage added to instantaneous covariance C_0 for stability.
        dt : float | None, default=None
            Time step per frame (in ps). If None, inferred from manifest `time_ps` if available,
            or defaults to 1.0.

        Returns
        -------
        TICAResult
            Fitted TICA result containing coordinates, eigenvalues, implied timescales,
            and feature loadings (kinetic salience).
        """
        if self._pipeline is None or self._pipeline.features is None:
            if self._systems:
                self.run(extract_only=True)
            else:
                raise RuntimeError("Run the study or configure systems before running TICA.")

        features = self.get_features()
        manifest = self.get_manifest()

        if dt is None:
            if "time_ps" in manifest.columns:
                diffs = manifest.groupby("trajectory_id", sort=False)["time_ps"].diff().dropna()
                positive_diffs = diffs[diffs > 0]
                dt = float(positive_diffs.median()) if not positive_diffs.empty else 1.0
            else:
                dt = 1.0

        config = TICAConfig(
            lag_time=lag_time,
            n_components=components,
            shrinkage=shrinkage,
            dt=dt,
        )
        result = fit_tica(features, manifest, config)
        self._tica_result = result
        return result

    def get_tica_result(self) -> TICAResult:
        """Return the fitted TICA result, or raise RuntimeError if not yet run."""
        if getattr(self, "_tica_result", None) is None:
            raise RuntimeError("Run TICA with study.run_tica(...) before requesting TICA results.")
        return self._tica_result

    def run_contact_kpca(
        self,
        n_components: int = 4,
        gamma: float | str | None = None,
        feature_columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ContactKPCAResult:
        """Run Contact Kernel PCA on extracted molecular contact features.

        Parameters
        ----------
        n_components : int, default=4
            Number of non-linear contact coordinates (CKPCA_1, CKPCA_2, ...) to retain.
        gamma : float | str | None, default=None
            RBF kernel parameter. If None or 'median', computed via median heuristic.
        feature_columns : Sequence[str] | None, default=None
            Specific contact feature columns to use. If None, automatically selects contact/distance
            columns, or all numeric features if no contact-specific columns exist.
        **kwargs : Any
            Additional options passed to ContactKPCAConfig (e.g., kernel='rbf', drop_constant=True).

        Returns
        -------
        ContactKPCAResult
            Fitted Contact Kernel PCA result containing non-linear embeddings and feature salience.
        """
        if self._pipeline is None or self._pipeline.features is None:
            if self._systems:
                self.run(extract_only=True)
            else:
                raise RuntimeError("Run the study or configure systems before running Contact Kernel PCA.")

        features = self.get_features()
        manifest = self.get_manifest()

        if feature_columns is not None:
            selected_features = features[list(feature_columns)]
        else:
            contact_cols = [
                c for c in features.columns
                if any(k in c.lower() for k in ("contact", "distance", "dist", "pair", "ca_"))
            ]
            if contact_cols:
                selected_features = features[contact_cols]
            else:
                selected_features = features

        config = ContactKPCAConfig(n_components=n_components, gamma=gamma, **kwargs)
        result = fit_contact_kpca(selected_features, manifest=manifest, config=config)
        self._contact_kpca_result = result
        return result

    def get_contact_kpca_result(self) -> ContactKPCAResult:
        """Return the fitted Contact Kernel PCA result, or raise RuntimeError if not yet run."""
        if getattr(self, "_contact_kpca_result", None) is None:
            raise RuntimeError("Run Contact Kernel PCA with study.run_contact_kpca(...) before requesting results.")
        return self._contact_kpca_result

    def cluster_dpa(
        self,
        coordinate_name: str | pd.DataFrame = "PCA",
        z_score: float = 2.0,
        cutoff_distance: float | None = None,
        percentile: float = 2.0,
        n_clusters: int | None = None,
        **kwargs: Any,
    ) -> DPAResult:
        """Cluster conformational coordinates using Density Peak Advanced (DPA) clustering.

        Parameters
        ----------
        coordinate_name : str | pd.DataFrame, default="PCA"
            Coordinate representation to cluster:
            - "PCA": Linear PCA representation coordinates (PC1, PC2, ...)
            - "CKPCA": Non-linear Contact Kernel PCA embeddings (CKPCA_1, CKPCA_2, ...)
            - "TICA": Slow kinetic coordinates (TIC_1, TIC_2, ...)
            - Or a custom DataFrame containing conformational coordinates.
        z_score : float, default=2.0
            Z-score threshold on decision metric gamma = rho * delta for identifying cluster centers.
        cutoff_distance : float | None, default=None
            Cutoff distance d_c for Gaussian kernel local density estimation.
        percentile : float, default=2.0
            Percentile of pairwise distances used to determine d_c if cutoff_distance is None.
        n_clusters : int | None, default=None
            Explicit number of cluster centers to retain (overrides z_score if specified).
        **kwargs : Any
            Additional options passed to DPAConfig (e.g., metric='euclidean', min_clusters=1).

        Returns
        -------
        DPAResult
            Fitted DPA clustering result with state assignments, decision graph, and transitions.
        """
        if isinstance(coordinate_name, pd.DataFrame):
            coords_df = coordinate_name
            coordinate_prefix = None
        elif isinstance(coordinate_name, str):
            coord_upper = coordinate_name.strip().upper()
            if coord_upper in ("PCA", "PC"):
                coordinate_prefix = "PCA"
                if self._pipeline is not None and "representation_coordinates" in (self._pipeline.outputs or {}):
                    coords_df = self._pipeline.table("representation_coordinates")
                else:
                    if self._pipeline is None or self._pipeline.features is None:
                        if self._systems:
                            self.run(extract_only=True)
                        else:
                            raise RuntimeError("Run the study or configure systems before clustering coordinates.")
                    try:
                        from molscope.analysis.discovery.representation import discover_representation_features
                        rep = discover_representation_features(self.get_features(), self.get_manifest())
                        coords_df = rep["representation_coordinates"]
                    except Exception:
                        from sklearn.decomposition import PCA
                        from sklearn.preprocessing import StandardScaler
                        feat = self.get_features().apply(pd.to_numeric, errors="coerce")
                        active = feat.columns[feat.std(axis=0) > 1e-12]
                        if not active.empty:
                            val = feat[active].to_numpy(dtype=float)
                        else:
                            val = feat.to_numpy(dtype=float)
                        scaled = StandardScaler().fit_transform(val)
                        n_comp = min(5, scaled.shape[0], scaled.shape[1])
                        pca = PCA(n_components=n_comp)
                        pcs = pca.fit_transform(scaled)
                        coords_df = self.get_manifest().copy().reset_index(drop=True)
                        for idx in range(n_comp):
                            coords_df[f"PC{idx + 1}"] = pcs[:, idx]
            elif coord_upper in ("CKPCA", "CONTACT_KPCA", "K_PCA"):
                coordinate_prefix = "CKPCA"
                if getattr(self, "_contact_kpca_result", None) is None:
                    self.run_contact_kpca()
                coords_df = self._contact_kpca_result.coordinates
            elif coord_upper in ("TICA", "TIC"):
                coordinate_prefix = "TICA"
                if getattr(self, "_tica_result", None) is None:
                    self.run_tica()
                coords_df = self._tica_result.coordinates
            elif self._pipeline is not None and coordinate_name in (self._pipeline.outputs or {}):
                coords_df = self._pipeline.table(coordinate_name)
                coordinate_prefix = None
            else:
                raise ValueError(
                    f"Unknown coordinate representation {coordinate_name!r}; "
                    f"expected 'PCA', 'CKPCA', 'TICA', or a custom DataFrame."
                )
        else:
            raise TypeError(f"coordinate_name must be str or DataFrame, got {type(coordinate_name).__name__}.")

        manifest = None
        if self._pipeline is not None and self._pipeline.manifest is not None:
            manifest = self.get_manifest()

        config = DPAConfig(
            cutoff_distance=cutoff_distance,
            percentile=percentile,
            z_score=z_score,
            n_clusters=n_clusters,
            **kwargs,
        )
        result = cluster_dpa(
            coords_df,
            manifest=manifest,
            config=config,
            coordinate_prefix=coordinate_prefix,
        )
        self._dpa_result = result
        return result

    def get_dpa_result(self) -> DPAResult:
        """Return the fitted DPA clustering result, or raise RuntimeError if not yet run."""
        if getattr(self, "_dpa_result", None) is None:
            raise RuntimeError("Run DPA clustering with study.cluster_dpa(...) before requesting DPA results.")
        return self._dpa_result

    def explain_clusters(
        self,
        clustering_name: str = "states",
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        mode: str = "ovr",
        pair: tuple[Any, Any] | None = None,
        target_cluster: Any | None = None,
        random_state: int = 42,
        feature_units: Mapping[str, str] | None = None,
        float_precision: int = 2,
    ) -> DecisionTreeExplainer:
        """Explain conformational states or trajectory clusters using shallow decision trees.

        Extracts interpretable, human-readable boolean if/then rules, computes Gini
        feature importance, and provides text and diagrammatic explanations.

        Parameters
        ----------
        clustering_name : str, default="states"
            Name of clustering to explain: "states", "state_assignments", "dpa",
            or any column in the trajectory frame manifest (e.g. metadata condition).
        max_depth : int, default=3
            Maximum tree depth for shallow, human-interpretable rules.
        min_samples_leaf : int, default=5
            Minimum samples required at a leaf node to avoid overfitting.
        mode : str, default="ovr"
            "ovr" for One-vs-Rest explanations of each cluster, or "pairwise" for pairwise comparisons.
        pair : tuple[Any, Any] | None, default=None
            Specific pair of clusters to compare when mode="pairwise" (e.g. ("state_0", "state_1")).
        target_cluster : Any | None, default=None
            Specific cluster to explain in OvR mode. If None, explains all clusters.
        random_state : int, default=42
            Random seed for reproducible tree splits.
        feature_units : Mapping[str, str] | None, default=None
            Optional dictionary mapping feature names to physical units (e.g. {"dist(A, B)": "Å"}).
        float_precision : int, default=2
            Decimal precision for threshold formatting.

        Returns
        -------
        DecisionTreeExplainer
            Fitted explainer with extracted rules, feature importances, and plotting methods.
        """
        if self._pipeline is None or self._pipeline.features is None:
            if self._systems:
                self.run(extract_only=True)
            else:
                raise RuntimeError("Run the study or configure systems before explaining clusters.")

        features = self.get_features()
        manifest = self.get_manifest()

        # Resolve cluster labels
        labels: pd.Series | None = None

        if clustering_name in ("states", "state_assignments", "conformational_states"):
            if self._pipeline.outputs is not None and "state_assignments" in self._pipeline.outputs:
                st_tbl = self._pipeline.table("state_assignments")
                labels = st_tbl["State_Label"] if "State_Label" in st_tbl.columns else st_tbl["State"]
            elif self._pipeline.outputs is not None and "representation_coordinates" in self._pipeline.outputs:
                from molscope.analysis.discovery.states import StateDiscoveryConfig, discover_conformational_states
                st_res = discover_conformational_states(
                    self._pipeline.table("representation_coordinates"),
                    manifest,
                )
                self._pipeline.outputs.update(st_res)
                st_tbl = st_res["state_assignments"]
                labels = st_tbl["State_Label"] if "State_Label" in st_tbl.columns else st_tbl["State"]
            else:
                raise RuntimeError("Run study.run() before explaining conformational states.")
        elif clustering_name in ("dpa", "density_peaks"):
            if getattr(self, "_dpa_result", None) is not None:
                dpa_tbl = self._dpa_result.assignments
                labels = dpa_tbl["Cluster_Label"] if "Cluster_Label" in dpa_tbl.columns else dpa_tbl["Cluster"]
            elif self._pipeline.outputs is not None and "dpa_clusters" in self._pipeline.outputs:
                dpa_tbl = self._pipeline.table("dpa_clusters")
                labels = dpa_tbl["Cluster_Label"] if "Cluster_Label" in dpa_tbl.columns else dpa_tbl["Cluster"]
            else:
                raise RuntimeError("Run study.cluster_dpa(...) before explaining DPA clusters.")
        elif clustering_name in manifest.columns:
            labels = manifest[clustering_name]
        elif f"metadata_{clustering_name}" in manifest.columns:
            labels = manifest[f"metadata_{clustering_name}"]
        elif self._pipeline.outputs is not None and clustering_name in self._pipeline.outputs:
            tbl = self._pipeline.table(clustering_name)
            for cand_col in ("State_Label", "State", "Cluster_Label", "Cluster"):
                if cand_col in tbl.columns:
                    labels = tbl[cand_col]
                    break
            if labels is None:
                labels = tbl.iloc[:, 0]
        else:
            manifest_cols = [c for c in manifest.columns if c not in ("frame_index", "frame_order", "time_ps")]
            raise ValueError(
                f"Clustering '{clustering_name}' not found. Available options: 'states', 'dpa', "
                f"or manifest columns: {manifest_cols}"
            )

        explainer = DecisionTreeExplainer(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            feature_units=feature_units,
            float_precision=float_precision,
        )

        if pair is not None:
            explainer.fit_pairwise(features, labels, state_a=pair[0], state_b=pair[1])
        elif mode == "pairwise":
            explainer.fit(features, labels, mode="pairwise")
        else:
            explainer.fit(features, labels, mode="ovr", target_cluster=target_cluster)

        self._tree_explainer = explainer
        return explainer

    def get_tree_explainer(self) -> DecisionTreeExplainer:
        """Return the fitted DecisionTreeExplainer, or raise RuntimeError if not yet run."""
        if getattr(self, "_tree_explainer", None) is None:
            raise RuntimeError("Run study.explain_clusters(...) before requesting tree explainer.")
        return self._tree_explainer

    def export_3d(
        self,
        score_column: str = "PCA_Salience",
        out_dir: str | Path | None = None,
        top_labels: int = 8,
    ) -> dict[str, Path]:
        """Export PyMOL (.pml), ChimeraX (.cxc), and B-factor scored PDB files."""
        if self._pipeline is None:
            raise RuntimeError("Run the study before exporting 3D scripts.")
        if not self._systems:
            raise ValueError("No system topology available for 3D export.")

        target_dir = Path(out_dir) if out_dir is not None else self.out_dir / "3d_visualization"
        topology = self._systems[0].topology

        # Prefer TICA loadings if requested and available
        if score_column in ("Kinetic_Salience", "TICA_Salience") and getattr(self, "_tica_result", None) is not None:
            score_table = self._tica_result.loadings
        # Prefer CKPCA salience if requested and available
        elif score_column in ("CKPCA_Salience", "Contact_Salience") and getattr(self, "_contact_kpca_result", None) is not None:
            score_table = self._contact_kpca_result.salience
        # Prefer SAM-MFP candidates or predictive rankings
        elif "sam_mfp_candidates" in (self._pipeline.outputs or {}):
            score_table = self._pipeline.table("sam_mfp_candidates")
        elif "supervised_predictive_feature_ranking" in (self._pipeline.outputs or {}):
            score_table = self._pipeline.table("supervised_predictive_feature_ranking")
            score_column = "Mean_LOFO_Importance"
        else:
            score_table = self._pipeline.table("representation_feature_salience")

        residue_scores = extract_residue_scores(score_table, feature_column="Feature", score_column=score_column)
        return export_3d_visualization(
            topology=topology,
            output_dir=target_dir,
            residue_scores=residue_scores,
            basename=f"{self.project_id}_importance",
            top_label_count=top_labels,
        )

    def _resolve_coordinate_series(self, dim: str) -> np.ndarray | None:
        """Resolve a coordinate name into a 1D numpy array from available representations."""
        # 1. Check TICA coordinates if available
        if getattr(self, "_tica_result", None) is not None:
            tica_df = self._tica_result.coordinates
            if dim in tica_df.columns:
                return np.asarray(tica_df[dim], dtype=float)
            if dim.upper().startswith("TIC"):
                alt = dim.replace("_", "") if "_" in dim else f"TIC_{dim[3:]}"
                if alt in tica_df.columns:
                    return np.asarray(tica_df[alt], dtype=float)

        # 2. Check Contact KPCA coordinates if available
        if getattr(self, "_contact_kpca_result", None) is not None:
            ckpca_df = self._contact_kpca_result.coordinates
            if dim in ckpca_df.columns:
                return np.asarray(ckpca_df[dim], dtype=float)
            if dim.upper().startswith("CKPCA"):
                alt = dim.replace("_", "") if "_" in dim else f"CKPCA_{dim[5:]}"
                if alt in ckpca_df.columns:
                    return np.asarray(ckpca_df[alt], dtype=float)

        # 3. Check PCA representation coordinates in pipeline outputs
        if self._pipeline is not None and "representation_coordinates" in (self._pipeline.outputs or {}):
            rep_df = self._pipeline.table("representation_coordinates")
            if dim in rep_df.columns:
                return np.asarray(rep_df[dim], dtype=float)
            if dim.upper().startswith("PC"):
                alt = dim.replace("_", "") if "_" in dim else f"PC_{dim[2:]}"
                if alt in rep_df.columns:
                    return np.asarray(rep_df[alt], dtype=float)

        # 4. Check extracted features in pipeline
        if self._pipeline is not None and self._pipeline.features is not None:
            feat_df = self._pipeline.features
            if dim in feat_df.columns:
                return np.asarray(feat_df[dim], dtype=float)

        return None

    def _resolve_coordinates_pair(self, dim1: str, dim2: str) -> tuple[np.ndarray, np.ndarray]:
        """Resolve a pair of coordinates, triggering extraction or dimensionality reduction if needed."""
        x = self._resolve_coordinate_series(dim1)
        y = self._resolve_coordinate_series(dim2)

        if x is not None and y is not None:
            return x, y

        # If missing and involves TIC, compute TICA
        if (dim1.upper().startswith("TIC") or dim2.upper().startswith("TIC")) and getattr(self, "_tica_result", None) is None:
            self.run_tica()
            x = self._resolve_coordinate_series(dim1)
            y = self._resolve_coordinate_series(dim2)
            if x is not None and y is not None:
                return x, y

        # If missing and involves CKPCA, compute Contact KPCA
        if (dim1.upper().startswith("CKPCA") or dim2.upper().startswith("CKPCA")) and getattr(self, "_contact_kpca_result", None) is None:
            self.run_contact_kpca()
            x = self._resolve_coordinate_series(dim1)
            y = self._resolve_coordinate_series(dim2)
            if x is not None and y is not None:
                return x, y

        # If missing and pipeline not run, extract features
        if self._pipeline is None or self._pipeline.features is None:
            if self._systems:
                self.run(extract_only=True)
            else:
                raise RuntimeError("Run the study or configure systems before plotting free energy landscape.")
            x = self._resolve_coordinate_series(dim1)
            y = self._resolve_coordinate_series(dim2)
            if x is not None and y is not None:
                return x, y

        # If representation coordinates missing (e.g. run with extract_only=True)
        if (dim1.upper().startswith("PC") or dim2.upper().startswith("PC")) and "representation_coordinates" not in (self._pipeline.outputs or {}):
            if self._pipeline.outputs is None:
                self._pipeline.outputs = {}
            features = self.get_features()
            manifest = self.get_manifest()
            trajectory_ids = pd.unique(manifest["trajectory_id"])
            if len(trajectory_ids) >= 2:
                from molscope.analysis.discovery.representation import discover_representation_features
                rep = discover_representation_features(features, manifest)
                self._pipeline.outputs["representation_coordinates"] = rep["representation_coordinates"]
            else:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                numeric = features.apply(pd.to_numeric, errors="coerce")
                active = numeric.columns[numeric.std(axis=0) > 1e-12]
                if len(active) > 0:
                    scaled = StandardScaler().fit_transform(numeric[active])
                    k = min(5, scaled.shape[0], scaled.shape[1])
                    pca = PCA(n_components=k)
                    coords = pca.fit_transform(scaled)
                    coord_table = manifest.copy()
                    for i in range(coords.shape[1]):
                        coord_table[f"PC{i + 1}"] = coords[:, i]
                    self._pipeline.outputs["representation_coordinates"] = coord_table

            x = self._resolve_coordinate_series(dim1)
            y = self._resolve_coordinate_series(dim2)
            if x is not None and y is not None:
                return x, y

        missing = []
        if x is None:
            missing.append(dim1)
        if y is None:
            missing.append(dim2)
        raise ValueError(
            f"Coordinate(s) {missing} could not be resolved from PCA, TICA, Contact KPCA, or feature tables."
        )

    def plot_free_energy_landscape(
        self,
        dim1: str = "PC1",
        dim2: str = "PC2",
        out_dir: str | Path | None = None,
        temperature_k: float = 300.0,
        bins: int = 50,
        units: str = "kcal/mol",
        method: str = "histogram",
        smooth: float | None = 1.0,
        title: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Compute and plot the 2D Free Energy Landscape (FES) / Potential of Mean Force.

        Parameters
        ----------
        dim1 : str, default="PC1"
            Coordinate for horizontal axis (e.g. "PC1", "TIC_1", "CKPCA_1", or feature name).
        dim2 : str, default="PC2"
            Coordinate for vertical axis (e.g. "PC2", "TIC_2", "CKPCA_2", or feature name).
        out_dir : str or Path or None, default=None
            Target directory for output files. Defaults to `self.out_dir / "free_energy"`.
        temperature_k : float, default=300.0
            Simulation temperature in Kelvin.
        bins : int, default=50
            Number of bins per coordinate dimension.
        units : str, default="kcal/mol"
            Thermodynamic units: "kcal/mol" or "kJ/mol".
        method : str, default="histogram"
            Density estimation method: "histogram" or "kde".
        smooth : float | None, default=1.0
            Gaussian smoothing kernel standard deviation (in bins) applied to histogram density.
        title : str | None, default=None
            Custom title for the static and interactive plots.
        **kwargs : Any
            Additional options forwarded to `plot_landscape_static`.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "result": FreeEnergyLandscapeResult
            - "static_plot": Path to saved PNG file
            - "interactive_plot": Path to saved HTML file
            - "basins_file": Path to saved basins CSV file
            - "basins": pd.DataFrame of detected conformational basins
        """
        x, y = self._resolve_coordinates_pair(dim1, dim2)

        result = compute_free_energy_landscape(
            x=x,
            y=y,
            temperature_k=temperature_k,
            bins=bins,
            units=units,
            method=method,
            smooth=smooth,
        )
        self._fes_result = result

        target_dir = Path(out_dir) if out_dir is not None else self.out_dir / "free_energy"
        target_dir.mkdir(parents=True, exist_ok=True)

        clean_dim1 = dim1.replace("/", "_").replace(" ", "_")
        clean_dim2 = dim2.replace("/", "_").replace(" ", "_")
        static_path = target_dir / f"fes_{clean_dim1}_{clean_dim2}.png"
        interactive_path = target_dir / f"fes_{clean_dim1}_{clean_dim2}.html"
        basins_path = target_dir / f"basins_{clean_dim1}_{clean_dim2}.csv"

        plot_title = title or f"Free Energy Landscape: {dim1} vs {dim2} ({self.project_id})"
        plot_landscape_static(
            result,
            static_path,
            xlabel=dim1,
            ylabel=dim2,
            title=plot_title,
            **kwargs,
        )
        plot_landscape_interactive(
            result,
            interactive_path,
            xlabel=dim1,
            ylabel=dim2,
            title=plot_title,
        )

        result.basins.to_csv(basins_path, index=False)

        return {
            "result": result,
            "static_plot": static_path,
            "interactive_plot": interactive_path,
            "basins_file": basins_path,
            "basins": result.basins,
        }

    def get_free_energy_result(self) -> FreeEnergyLandscapeResult:
        """Return the most recently computed Free Energy Landscape result."""
        if getattr(self, "_fes_result", None) is None:
            raise RuntimeError("Run study.plot_free_energy_landscape(...) before requesting FES results.")
        return self._fes_result

