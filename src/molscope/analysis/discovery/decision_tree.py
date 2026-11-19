"""Interpretable decision tree rule extraction and molecular dynamics state explainability."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


def _format_threshold(val: float, precision: int = 2) -> str:
    """Format numerical threshold cleanly without excessive floating point digits."""
    formatted = f"{val:.{precision}f}"
    if "." in formatted:
        parts = formatted.split(".")
        decimals = parts[1].rstrip("0")
        if not decimals:
            decimals = "0"
        return f"{parts[0]}.{decimals}"
    return formatted


def _infer_feature_unit(feature_name: str, explicit_units: Mapping[str, str] | None = None) -> str:
    """Infer physical units for molecular features (e.g. Å for distances, ° for dihedrals)."""
    if explicit_units and feature_name in explicit_units:
        unit = explicit_units[feature_name]
        if not unit:
            return ""
        return unit if (unit.startswith("°") or unit.startswith(" ")) else f" {unit}"

    name_lower = feature_name.lower()
    # Torsions and dihedrals
    if any(k in name_lower for k in ("chi1", "chi2", "chi3", "chi4", "phi", "psi", "omega", "dihedral", "torsion", "angle")):
        return "°"
    # Interatomic and contact distances
    if any(k in name_lower for k in ("dist", "distance", "ca_dist", "cb_dist", "mindist", "contact_dist")):
        return " Å"
    return ""


@dataclass
class DecisionRule:
    """Human-readable if/then boolean decision rule explaining a cluster or state.

    Attributes
    ----------
    cluster : Any
        Cluster or state label explained by this rule.
    rule_string : str
        Human-readable boolean rule string:
        e.g., "IF dist(Phe47, Trp23) <= 5.8 Å AND chi1(Tyr60) > -60.0° THEN Cluster 2 [precision=0.96, recall=0.92]"
    conditions : list[str]
        Individual human-readable condition clauses (e.g. "dist(Phe47, Trp23) <= 5.8 Å").
    precision : float
        Precision of the rule (positive predictive value) on evaluated data.
    recall : float
        Recall of the rule (sensitivity / true positive rate) on evaluated data.
    f1 : float
        Harmonic mean of precision and recall.
    support : int
        Number of positive samples in this cluster matching all conditions.
    total_samples : int
        Total number of samples matching all conditions.
    raw_conditions : list[tuple[str, str, float]]
        Tuple of (feature_name, operator, threshold) for programmatic evaluation.
    """

    cluster: Any
    rule_string: str
    conditions: list[str]
    precision: float
    recall: float
    f1: float
    support: int
    total_samples: int
    raw_conditions: list[tuple[str, str, float]] = field(default_factory=list)

    def __str__(self) -> str:
        return self.rule_string

    def __repr__(self) -> str:
        return f"<DecisionRule: {self.rule_string}>"

    def matches(self, X: pd.DataFrame | np.ndarray, feature_names: Sequence[str] | None = None) -> np.ndarray:
        """Evaluate rule conditions against a dataset and return boolean mask."""
        if isinstance(X, pd.DataFrame):
            n_samples = len(X)
            mask = np.ones(n_samples, dtype=bool)
            for feat, op, thresh in self.raw_conditions:
                if feat not in X.columns:
                    raise KeyError(f"Feature '{feat}' not found in X columns.")
                col = X[feat].to_numpy(dtype=float)
                if op == "<=":
                    mask &= (col <= thresh)
                elif op == ">":
                    mask &= (col > thresh)
            return mask
        else:
            arr = np.asarray(X, dtype=float)
            if feature_names is None:
                raise ValueError("feature_names required when X is a numpy ndarray.")
            feat_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            mask = np.ones(len(arr), dtype=bool)
            for feat, op, thresh in self.raw_conditions:
                if feat not in feat_to_idx:
                    raise KeyError(f"Feature '{feat}' not found in feature_names.")
                col = arr[:, feat_to_idx[feat]]
                if op == "<=":
                    mask &= (col <= thresh)
                elif op == ">":
                    mask &= (col > thresh)
            return mask

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the subset of DataFrame X that satisfies all rule conditions."""
        mask = self.matches(X)
        return X.loc[mask].copy()

    @property
    def cluster_id(self) -> Any:
        if isinstance(self.cluster, str) and self.cluster.startswith("cluster_"):
            try:
                return int(self.cluster.split("_")[1])
            except ValueError:
                pass
        return self.cluster

    @property
    def cluster_name(self) -> str:
        return f"Macrostate {self.cluster_id}"

    @property
    def purity(self) -> float:
        return self.precision

    @property
    def support_fraction(self) -> float:
        return float(self.support / self.total_samples) if self.total_samples > 0 else 0.0

    def simplified_rule_text(self) -> str:
        if self.conditions:
            return " AND ".join(self.conditions)
        return self.rule_string


class RuleStore(list):
    """Container holding DecisionRule objects with dual list and dict-by-cluster access."""

    def __init__(self, rules_by_cluster: dict[Any, list[DecisionRule]] | None = None):
        rules_by_cluster = rules_by_cluster or {}
        all_rules: list[DecisionRule] = []
        for rules_list in rules_by_cluster.values():
            all_rules.extend(rules_list)
        super().__init__(all_rules)
        self._by_cluster: dict[Any, list[DecisionRule]] = dict(rules_by_cluster)

    def __contains__(self, key: Any) -> bool:
        return key in self._by_cluster or super().__contains__(key)

    def __getitem__(self, key: Any) -> Any:
        if key in self._by_cluster:
            return self._by_cluster[key]
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        return self._by_cluster.get(key, [])

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, int):
            super().__setitem__(key, value)
        else:
            self._by_cluster[key] = value

    def items(self):
        return self._by_cluster.items()

    def values(self):
        return self._by_cluster.values()

    def keys(self):
        return self._by_cluster.keys()

    def get(self, key: Any, default: Any = None) -> Any:
        return self._by_cluster.get(key, default)

    def clear(self) -> None:
        super().clear()
        self._by_cluster.clear()


class DecisionTreeExplainer:
    """Interpretable shallow Decision Tree Explainer for conformational clusters and states.

    Fits shallow decision trees (One-vs-Rest or pairwise comparisons) to extract
    sparse, human-readable boolean if/then rules, Gini feature importances, and
    visual tree diagrams.

    Parameters
    ----------
    max_depth : int, default=3
        Maximum tree depth for shallow, human-interpretable rules.
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node.
    criterion : str, default="gini"
        Function to measure split quality: "gini", "entropy", or "log_loss".
    class_weight : str | dict | None, default=None
        Weights associated with classes (e.g. "balanced" for imbalanced clusters).
    random_state : int, default=42
        Seed for reproducible tree construction.
    feature_units : Mapping[str, str] | None, default=None
        Optional dictionary mapping feature names to physical units (e.g. {"dist(A, B)": "Å"}).
    float_precision : int, default=2
        Number of decimal digits for displaying thresholds in rule strings.
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        criterion: str = "gini",
        class_weight: str | dict | None = None,
        random_state: int = 42,
        feature_units: Mapping[str, str] | None = None,
        float_precision: int = 2,
    ) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be an integer >= 1, got {max_depth}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be an integer >= 1, got {min_samples_leaf}")
        if criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError(f"criterion must be 'gini', 'entropy', or 'log_loss', got {criterion!r}")

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.random_state = random_state
        self.feature_units = dict(feature_units) if feature_units is not None else {}
        self.float_precision = float_precision

        # Model storage
        self.models: dict[Any, DecisionTreeClassifier] = {}
        self.rules: RuleStore = RuleStore({})
        self.feature_importances: dict[Any, pd.DataFrame] = {}
        self.feature_names: list[str] = []
        self.cluster_labels: list[Any] = []
        self.mode: str = "ovr"
        self._fitted: bool = False

    def _validate_inputs(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | Sequence[Any],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Validate feature matrix X and target labels y."""
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}")
            cols = [f"feature_{i}" for i in range(arr.shape[1])]
            X_df = pd.DataFrame(arr, columns=cols)

        if X_df.empty:
            raise ValueError("X must contain at least one sample and one feature.")

        # Ensure numeric
        numeric_X = X_df.apply(pd.to_numeric, errors="coerce")
        nan_cols = numeric_X.columns[numeric_X.isna().any()].tolist()
        if nan_cols:
            raise ValueError(f"X contains non-numeric, NaN, or infinite values in columns: {nan_cols[:5]}")

        y_series = pd.Series(y).reset_index(drop=True)
        if len(y_series) != len(X_df):
            raise ValueError(f"Length mismatch: X has {len(X_df)} rows, y has {len(y_series)} elements.")

        # Drop NaN labels if any
        if y_series.isna().any():
            valid_mask = ~y_series.isna()
            X_df = X_df.loc[valid_mask].reset_index(drop=True)
            y_series = y_series.loc[valid_mask].reset_index(drop=True)

        if y_series.nunique() < 2:
            raise ValueError("At least 2 unique clusters/states are required.")

        return numeric_X, y_series

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | Sequence[Any],
        mode: str = "ovr",
        target_cluster: Any | None = None,
        pair: tuple[Any, Any] | None = None,
    ) -> "DecisionTreeExplainer":
        """Fit decision tree models to explain clusters or states.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Frame features table or numerical array.
        y : pd.Series | np.ndarray | Sequence[Any]
            Cluster or conformational state assignments.
        mode : str, default="ovr"
            "ovr" for One-vs-Rest cluster explanations, or "pairwise" for pairwise comparisons.
        target_cluster : Any | None, default=None
            Specific cluster to explain in OvR mode. If None, all clusters are explained.
        pair : tuple[Any, Any] | None, default=None
            Specific pair of clusters to compare when mode="pairwise".

        Returns
        -------
        DecisionTreeExplainer
            Self instance with fitted models, rules, and feature importances.
        """
        X_df, y_series = self._validate_inputs(X, y)
        self.feature_names = list(X_df.columns)
        self.mode = mode

        if mode == "pairwise":
            if pair is not None:
                return self.fit_pairwise(X_df, y_series, state_a=pair[0], state_b=pair[1])
            unique_vals = list(y_series.unique())
            if len(unique_vals) == 2:
                return self.fit_pairwise(X_df, y_series, state_a=unique_vals[0], state_b=unique_vals[1])
            else:
                raise ValueError(
                    "For mode='pairwise' with > 2 clusters, specify 'pair=(state_a, state_b)'."
                )

        # One-vs-Rest mode
        unique_clusters = sorted(y_series.unique(), key=lambda v: str(v))
        self.cluster_labels = unique_clusters
        clusters_to_fit = [target_cluster] if target_cluster is not None else unique_clusters

        self.models.clear()
        self.rules.clear()
        self.feature_importances.clear()

        for cluster in clusters_to_fit:
            y_bin = (y_series == cluster).astype(int).to_numpy()
            clf = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            clf.fit(X_df, y_bin)
            self.models[cluster] = clf

            # Extract rules for target cluster (class 1)
            cluster_rules = self._extract_rules_for_target(
                clf=clf,
                X=X_df,
                y_binary=y_bin,
                class_label=cluster,
                target_class_idx=1,
            )
            self.rules[cluster] = cluster_rules

            # Compute Gini feature importance & normalized scores
            self.feature_importances[cluster] = self._compute_feature_importance_df(clf, cluster)

        self.rules = RuleStore(self.rules._by_cluster if isinstance(self.rules, RuleStore) else self.rules)
        self._fitted = True
        return self

    def fit_pairwise(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | Sequence[Any],
        state_a: Any,
        state_b: Any,
    ) -> "DecisionTreeExplainer":
        """Fit a decision tree comparing two specific conformational states (State A vs State B).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Features table.
        y : pd.Series | np.ndarray | Sequence[Any]
            State assignments.
        state_a : Any
            First state to compare.
        state_b : Any
            Second state to compare.

        Returns
        -------
        DecisionTreeExplainer
            Self instance.
        """
        X_df, y_series = self._validate_inputs(X, y)
        self.feature_names = list(X_df.columns)
        self.mode = "pairwise"

        mask = y_series.isin([state_a, state_b])
        if not mask.any():
            raise ValueError(f"Neither {state_a!r} nor {state_b!r} found in labels.")

        X_sub = X_df.loc[mask].reset_index(drop=True)
        y_sub = y_series.loc[mask].reset_index(drop=True)

        if y_sub.nunique() < 2:
            raise ValueError(f"Pairwise comparison requires both states present in data, found only {y_sub.unique()}.")

        # Binary encoding: 1 for state_a, 0 for state_b
        y_bin = (y_sub == state_a).astype(int).to_numpy()

        clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        clf.fit(X_sub, y_bin)

        pair_key = (state_a, state_b)
        pair_str = f"{state_a}_vs_{state_b}"
        self.models[pair_key] = clf
        self.models[pair_str] = clf
        self.models[state_a] = clf
        self.models[state_b] = clf
        self.cluster_labels = [state_a, state_b]

        # Rules predicting state_a (class 1)
        rules_a = self._extract_rules_for_target(
            clf=clf,
            X=X_sub,
            y_binary=y_bin,
            class_label=state_a,
            target_class_idx=1,
        )
        # Rules predicting state_b (class 0)
        rules_b = self._extract_rules_for_target(
            clf=clf,
            X=X_sub,
            y_binary=(1 - y_bin),
            class_label=state_b,
            target_class_idx=0,
        )
        self.rules[state_a] = rules_a
        self.rules[state_b] = rules_b
        self.rules[pair_key] = rules_a + rules_b
        self.rules = RuleStore(self.rules._by_cluster if isinstance(self.rules, RuleStore) else self.rules)

        imp_df = self._compute_feature_importance_df(clf, pair_str)
        self.feature_importances[pair_key] = imp_df
        self.feature_importances[pair_str] = imp_df
        self.feature_importances[state_a] = imp_df
        self.feature_importances[state_b] = imp_df

        self._fitted = True
        return self

    def _compute_feature_importance_df(self, clf: DecisionTreeClassifier, cluster_label: Any) -> pd.DataFrame:
        """Compute Gini feature importance and max-normalized feature scores."""
        raw_importances = clf.feature_importances_
        max_imp = float(np.max(raw_importances)) if len(raw_importances) > 0 and np.max(raw_importances) > 0 else 1.0
        sum_imp = float(np.sum(raw_importances)) if len(raw_importances) > 0 and np.sum(raw_importances) > 0 else 1.0

        normalized_scores = raw_importances / max_imp
        relative_importance = raw_importances / sum_imp

        df = pd.DataFrame({
            "Feature": self.feature_names,
            "Gini_Importance": raw_importances,
            "Normalized_Score": normalized_scores,
            "Relative_Importance": relative_importance,
            "Cluster": str(cluster_label),
        })
        df = df.sort_values("Gini_Importance", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        return df

    def _extract_rules_for_target(
        self,
        clf: DecisionTreeClassifier,
        X: pd.DataFrame,
        y_binary: np.ndarray,
        class_label: Any,
        target_class_idx: int = 1,
    ) -> list[DecisionRule]:
        """Traverse tree paths to leaves and build simplified human-readable boolean rules."""
        tree = clf.tree_
        feature_names = self.feature_names

        # Traversal to find all paths from root to leaves
        paths: list[list[tuple[int, str, float]]] = []
        leaf_nodes: list[int] = []

        def recurse(node_id: int, current_path: list[tuple[int, str, float]]) -> None:
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            if left_child == -1 and right_child == -1:
                # Leaf node
                paths.append(list(current_path))
                leaf_nodes.append(node_id)
                return

            feat_idx = int(tree.feature[node_id])
            thresh = float(tree.threshold[node_id])

            # Left branch: feature <= thresh
            recurse(left_child, current_path + [(feat_idx, "<=", thresh)])
            # Right branch: feature > thresh
            recurse(right_child, current_path + [(feat_idx, ">", thresh)])

        recurse(0, [])

        # Candidate leaves predicting target_class_idx
        candidate_rules: list[DecisionRule] = []
        all_leaf_scores: list[tuple[float, int, list[tuple[int, str, float]]]] = []

        formatted_label = f"Cluster {class_label}" if isinstance(class_label, (int, np.integer)) else str(class_label)
        total_positives = int(np.sum(y_binary == 1))

        for path, leaf_id in zip(paths, leaf_nodes):
            counts = tree.value[leaf_id][0]
            pred_class = int(np.argmax(counts))
            purity = float(counts[target_class_idx] / counts.sum()) if counts.sum() > 0 else 0.0
            all_leaf_scores.append((purity, leaf_id, path))

            # Only consider leaves that predict target_class_idx
            if pred_class == target_class_idx:
                rule = self._build_rule(
                    path=path,
                    formatted_label=formatted_label,
                    class_label=class_label,
                    X=X,
                    y_binary=y_binary,
                    total_positives=total_positives,
                )
                candidate_rules.append(rule)

        # Fallback if no leaf has majority target class (e.g. rare cluster)
        if not candidate_rules and all_leaf_scores:
            all_leaf_scores.sort(key=lambda item: item[0], reverse=True)
            best_path = all_leaf_scores[0][2]
            rule = self._build_rule(
                path=best_path,
                formatted_label=formatted_label,
                class_label=class_label,
                X=X,
                y_binary=y_binary,
                total_positives=total_positives,
            )
            candidate_rules.append(rule)

        # Sort candidate rules by F1 score descending
        candidate_rules.sort(key=lambda r: (r.f1, r.precision, r.recall, r.support), reverse=True)
        return candidate_rules

    def _build_rule(
        self,
        path: list[tuple[int, str, float]],
        formatted_label: str,
        class_label: Any,
        X: pd.DataFrame,
        y_binary: np.ndarray,
        total_positives: int,
    ) -> DecisionRule:
        """Consolidate redundant conditions and compute precision and recall metrics."""
        # Consolidate bounds per feature while preserving root-to-leaf appearance order
        feature_order: list[int] = []
        lower_bounds: dict[int, float] = {}
        upper_bounds: dict[int, float] = {}

        for feat_idx, op, thresh in path:
            if feat_idx not in feature_order:
                feature_order.append(feat_idx)
            if op == "<=":
                upper_bounds[feat_idx] = (
                    min(upper_bounds[feat_idx], thresh) if feat_idx in upper_bounds else thresh
                )
            elif op == ">":
                lower_bounds[feat_idx] = (
                    max(lower_bounds[feat_idx], thresh) if feat_idx in lower_bounds else thresh
                )

        raw_conditions: list[tuple[str, str, float]] = []
        formatted_conditions: list[str] = []

        for feat_idx in feature_order:
            feat_name = self.feature_names[feat_idx]
            unit = _infer_feature_unit(feat_name, self.feature_units)

            # Order: if upper exists and lower exists, maintain natural logic
            if feat_idx in upper_bounds and feat_idx in lower_bounds:
                # E.g. dist <= 5.8 AND dist > 2.0
                u_val = upper_bounds[feat_idx]
                l_val = lower_bounds[feat_idx]
                raw_conditions.append((feat_name, "<=", u_val))
                formatted_conditions.append(f"{feat_name} <= {_format_threshold(u_val, self.float_precision)}{unit}")
                raw_conditions.append((feat_name, ">", l_val))
                formatted_conditions.append(f"{feat_name} > {_format_threshold(l_val, self.float_precision)}{unit}")
            elif feat_idx in upper_bounds:
                u_val = upper_bounds[feat_idx]
                raw_conditions.append((feat_name, "<=", u_val))
                formatted_conditions.append(f"{feat_name} <= {_format_threshold(u_val, self.float_precision)}{unit}")
            elif feat_idx in lower_bounds:
                l_val = lower_bounds[feat_idx]
                raw_conditions.append((feat_name, ">", l_val))
                formatted_conditions.append(f"{feat_name} > {_format_threshold(l_val, self.float_precision)}{unit}")

        # Compute empirical precision and recall across X
        mask = np.ones(len(X), dtype=bool)
        for feat_name, op, thresh in raw_conditions:
            col = X[feat_name].to_numpy(dtype=float)
            if op == "<=":
                mask &= (col <= thresh)
            elif op == ">":
                mask &= (col > thresh)

        total_matched = int(np.sum(mask))
        true_positives = int(np.sum(mask & (y_binary == 1)))

        precision = float(true_positives / total_matched) if total_matched > 0 else 0.0
        recall = float(true_positives / total_positives) if total_positives > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        cond_str = " AND ".join(formatted_conditions) if formatted_conditions else "TRUE"
        rule_string = f"IF {cond_str} THEN {formatted_label} [precision={precision:.2f}, recall={recall:.2f}]"

        return DecisionRule(
            cluster=class_label,
            rule_string=rule_string,
            conditions=formatted_conditions,
            precision=precision,
            recall=recall,
            f1=f1,
            support=true_positives,
            total_samples=total_matched,
            raw_conditions=raw_conditions,
        )

    def get_rules(self, cluster: Any | None = None) -> list[DecisionRule]:
        """Return extracted decision rules for a specific cluster or all clusters."""
        if not self._fitted:
            raise RuntimeError("Fit the explainer with .fit(...) before requesting rules.")
        if cluster is not None:
            if cluster in self.rules:
                return list(self.rules[cluster])
            raise KeyError(f"Cluster {cluster!r} not found in extracted rules.")
        # Return all rules across all clusters
        all_rules: list[DecisionRule] = []
        for rules_list in self.rules.values():
            all_rules.extend(rules_list)
        return all_rules

    def get_top_rule(self, cluster: Any) -> DecisionRule | None:
        """Return the highest-F1 rule for a given cluster."""
        rules = self.get_rules(cluster)
        return rules[0] if rules else None

    def rules_dataframe(self) -> pd.DataFrame:
        """Return a structured DataFrame of all extracted rules with evaluation metrics."""
        if not self._fitted:
            raise RuntimeError("Fit the explainer with .fit(...) before requesting rules table.")
        records = []
        for cluster, rules in self.rules.items():
            for i, r in enumerate(rules, start=1):
                records.append({
                    "Cluster": str(cluster),
                    "Rule_Index": i,
                    "Rule": r.rule_string,
                    "Precision": r.precision,
                    "Recall": r.recall,
                    "F1": r.f1,
                    "Support": r.support,
                    "Total_Samples": r.total_samples,
                    "Condition_Count": len(r.conditions),
                })
        return pd.DataFrame.from_records(records)

    def get_feature_importances(self, cluster: Any | None = None, top_n: int | None = None) -> pd.DataFrame:
        """Return Gini feature importances and normalized feature scores."""
        if not self._fitted:
            raise RuntimeError("Fit the explainer before requesting feature importances.")
        if cluster is not None:
            if cluster in self.feature_importances:
                df = self.feature_importances[cluster]
                return df.head(top_n) if top_n is not None else df.copy()
            raise KeyError(f"Cluster {cluster!r} not found in feature importances.")
        # Combine all clusters
        dfs = [df for df in self.feature_importances.values()]
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        return combined

    def export_text(self, cluster: Any | None = None) -> str:
        """Export scikit-learn ASCII decision tree representation.

        Parameters
        ----------
        cluster : Any | None, default=None
            Specific cluster model to export, or None for all fitted models.
        """
        if not self._fitted:
            raise RuntimeError("Fit the explainer before exporting text trees.")
        if cluster is not None:
            if cluster in self.models:
                return export_text(self.models[cluster], feature_names=self.feature_names)
            raise KeyError(f"Cluster {cluster!r} not found in fitted models.")

        out = []
        for c, model in self.models.items():
            out.append(f"=== Decision Tree for {c} ===")
            out.append(export_text(model, feature_names=self.feature_names))
        return "\n".join(out)

    def get_text_summary(self, cluster: Any | None = None) -> str:
        """Generate a comprehensive human-readable summary of rules and feature importances."""
        if not self._fitted:
            raise RuntimeError("Fit the explainer before requesting summary.")

        clusters = [cluster] if cluster is not None else list(self.rules.keys())
        lines = [
            "=" * 80,
            "                   MolScope Decision Tree Explainer Summary",
            "=" * 80,
            f"Mode: {self.mode.upper()} | Max Depth: {self.max_depth} | Min Samples Leaf: {self.min_samples_leaf}",
            f"Clusters Analyzed: {[str(c) for c in clusters]}",
            "",
        ]

        for c in clusters:
            lines.append("-" * 80)
            lines.append(f"Cluster / Comparison: {c}")
            lines.append("-" * 80)

            rules = self.rules.get(c, [])
            lines.append("Extracted Boolean Rules:")
            if rules:
                for idx, rule in enumerate(rules, start=1):
                    lines.append(f"  [{idx}] {rule.rule_string}")
                    lines.append(f"      (Support: {rule.support}/{rule.total_samples} samples, F1-score: {rule.f1:.2f})")
            else:
                lines.append("  (No rules extracted)")

            lines.append("")
            lines.append("Top Gini Explanatory Features:")
            imp_df = self.feature_importances.get(c)
            if imp_df is not None and not imp_df.empty:
                for _, row in imp_df.head(5).iterrows():
                    feat = row["Feature"]
                    gini = row["Gini_Importance"]
                    norm = row["Normalized_Score"]
                    lines.append(f"  - {feat:35s} Gini = {gini:.3f} (Normalized = {norm:.3f})")
            lines.append("")

            # Short tree excerpt
            if c in self.models:
                lines.append("Decision Tree ASCII Representation:")
                tree_text = export_text(self.models[c], feature_names=self.feature_names, max_depth=3)
                indented = "\n".join("  " + l for l in tree_text.strip().split("\n"))
                lines.append(indented)
            lines.append("")

        return "\n".join(lines)

    def plot_tree(
        self,
        cluster: Any | None = None,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] | None = None,
        ax: Any | None = None,
        filled: bool = True,
        rounded: bool = True,
        max_depth: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot a visual decision tree diagram using matplotlib and scikit-learn.

        Parameters
        ----------
        cluster : Any | None, default=None
            Specific cluster tree to plot. If None, plots all clusters in subplots or single plot.
        output_path : str | Path | None, default=None
            File path to save the generated diagram (e.g. .png or .pdf).
        figsize : tuple[float, float] | None, default=None
            Figure size in inches.
        ax : matplotlib.axes.Axes | None, default=None
            Target matplotlib axis.
        filled : bool, default=True
            Whether to paint nodes with colors representing class purity.
        rounded : bool, default=True
            Whether to draw node boxes with rounded corners.
        max_depth : int | None, default=None
            Depth limit for visualization.

        Returns
        -------
        matplotlib.figure.Figure
            The rendered figure object.
        """
        if not self._fitted:
            raise RuntimeError("Fit the explainer before plotting.")

        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree as sk_plot_tree

        target_cluster = cluster if cluster is not None else list(self.models.keys())[0]
        if target_cluster not in self.models:
            raise KeyError(f"Cluster {target_cluster!r} not found in models.")

        clf = self.models[target_cluster]

        if ax is None:
            default_figsize = figsize if figsize is not None else (12, 6)
            fig, ax = plt.subplots(figsize=default_figsize)
        else:
            fig = ax.figure

        class_names = ["Rest", str(target_cluster)]
        sk_plot_tree(
            clf,
            feature_names=self.feature_names,
            class_names=class_names,
            filled=filled,
            rounded=rounded,
            max_depth=max_depth or self.max_depth,
            ax=ax,
            **kwargs,
        )
        ax.set_title(f"MolScope Decision Tree: {target_cluster}", fontsize=12, fontweight="bold")

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")

        return fig

    def generate_diagram(
        self,
        cluster: Any | None = None,
        output_path: str | Path | None = None,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Alias for plot_tree to generate a visual decision tree diagram."""
        return self.plot_tree(cluster=cluster, output_path=output_path, figsize=figsize, **kwargs)
