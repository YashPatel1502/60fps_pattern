"""
Write per-pattern descriptive statistics (mean/std of segment features) to CSV.

Files (in ``out_dir``):
  - ``segment_measures.csv`` — one row per segment: measured features + pattern assignment
  - ``cluster_means_and_labels.csv`` — one row per **cluster id** (``pattern``): mean features used for
    post-hoc naming + assigned ``pattern_label`` / ``pattern_group`` + ``n_segments``
  - ``cluster_naming_order.csv`` — one row per **naming step** (priority list): which cluster got which
    label, which feature decided it, and that cluster’s mean of that feature (matches notebook ``assign_best``)
  - ``pattern_feature_means.csv`` — one row per (pattern, pattern_label, pattern_group)
  - ``pattern_feature_means_by_split.csv`` — if ``age_group`` or ``sex`` is present, one row
    per pattern × those columns for stratified means.
  - ``pca_specification.csv`` — optional: ``StandardScaler`` means/scales, sklearn ``PCA`` ``mean_`` and
    ``components_`` (PC1/PC2 loadings) + explained variance (see ``export_pca_specification_csv``).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd


def export_pca_specification_csv(
    scaler: Any,
    pca: Any,
    feature_cols: List[str],
    out_dir: Path,
    *,
    analysis_id: str,
    segment_length_mm: float | int,
    n_segments: int,
    k_patterns: int,
    filename: str = "pca_specification.csv",
    experiment_id: Optional[str] = None,
    detection_run_id: Optional[int] = None,
    sklearn_random_state: int = 42,
) -> Path:
    """
    Write a single CSV documenting PC1/PC2 as fitted by sklearn on ``StandardScaler``-transformed
    ``feature_cols``: comment header + table with per-feature scaler mean/scale, PCA input mean
    (``pca.mean_`` on scaled data), and loadings (rows of ``pca.components_`` for PC1 and PC2).

    Reconstructing scores: ``PCA`` centers scaled inputs with ``pca.mean_`` then projects with
    ``components_`` (see scikit-learn ``PCA`` implementation).

    Comment lines are written with :mod:`csv` as single-field rows so spreadsheet tools (Excel)
    do not split formulas on commas; use ``components_[k-1][j]`` in text instead of ``[k-1, j]``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    n_feat = len(feature_cols)
    sm = np.asarray(scaler.mean_, dtype=float)
    ss = np.asarray(scaler.scale_, dtype=float)
    pm = np.asarray(pca.mean_, dtype=float)
    comp = np.asarray(pca.components_, dtype=float)
    if sm.shape[0] != n_feat or ss.shape[0] != n_feat:
        raise ValueError("scaler mean_/scale_ length does not match feature_cols")
    if comp.ndim != 2 or comp.shape[1] != n_feat:
        raise ValueError(f"PCA components_ must be 2D with {n_feat} columns, got {comp.shape}")
    if comp.shape[0] < 2:
        raise ValueError("PCA must have at least n_components=2 for PC1/PC2 export")
    comp = comp[:2, :]
    if pm.shape[0] != n_feat:
        raise ValueError("pca.mean_ length does not match feature_cols")

    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cum2 = float(np.sum(evr[:2])) if len(evr) >= 2 else float("nan")

    header_lines = [
        "# PCA specification (sklearn StandardScaler + PCA on scaled features)",
        f"# analysis_id={analysis_id}",
        f"# segment_length_mm={segment_length_mm}",
        f"# n_segments={n_segments}",
        f"# k_patterns={k_patterns}",
        f"# explained_variance_ratio_PC1={evr[0]:.16g}",
        f"# explained_variance_ratio_PC2={evr[1]:.16g}" if len(evr) >= 2 else "# explained_variance_ratio_PC2=nan",
        f"# cumulative_explained_variance_PC1_PC2={cum2:.16g}",
        f"# sklearn_PCA_n_components=2_random_state={sklearn_random_state}",
    ]
    if experiment_id is not None:
        header_lines.append(f"# experiment_id={experiment_id}")
    if detection_run_id is not None:
        header_lines.append(f"# detection_run_id={detection_run_id}")
    header_lines.extend(
        [
            "#",
            "# z_j = (x_j - scaler_mean_j) / scaler_scale_j  for raw feature x_j.",
            "# PCk_i = sum_j components_[k-1][j] * ( z_ij - pca_mean_on_scaled_input_j ).",
            "",
        ]
    )

    rows = []
    for j, feat in enumerate(feature_cols):
        rows.append(
            {
                "feature": feat,
                "scaler_mean": sm[j],
                "scaler_scale": ss[j],
                "pca_mean_on_scaled_input": pm[j],
                "loading_PC1": comp[0, j],
                "loading_PC2": comp[1, j],
            }
        )
    df = pd.DataFrame(rows)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        for line in header_lines:
            if line == "":
                f.write("\n")
            else:
                w.writerow([line])
        df.to_csv(f, index=False, lineterminator="\n")

    return path

# Must match ``label_specs`` + ``assign_best`` in ``pattern_distance.ipynb`` / ``pattern_dehydration_step0.py``.
NAMING_LABEL_SPECS: List[Tuple[str, str, str]] = [
    ("zig-zag", "n_turns", "max"),
    ("straight", "straightness", "max"),
    ("meandering", "path_per_40", "max"),
    ("curved", "mean_abs_turn_deg", "max"),
    ("winding", "tortuosity", "max"),
    ("direct", "straightness", "max"),
    ("exploratory", "path_per_40", "max"),
]

# Extra segment-level columns copied when present (IDs, grouping, geometry summaries).
_SEGMENT_META_ORDER = (
    "experiment_id",
    "detection_run_id",
    "category",
    "age_group",
    "sex",
    "seg_40",
    "n_rows",
    "t_start_s",
    "t_end_s",
    "seg_path_mm",
    "seg_dur_s",
    "speed_mean_mm_s",
)
_PATTERN_COLS = ("pattern", "pattern_label", "pattern_group")


def export_segment_measures_csv(
    seg_clean: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
    *,
    segment_length_mm: float | int,
    step_name: str,
    experiment_id: Optional[str] = None,
    detection_run_id: Optional[int] = None,
    filename: str = "segment_measures.csv",
) -> Path:
    """
    One row per segment: kinematic measurements (``feature_cols``), optional metadata columns,
    and cluster labels. Does not include PCA columns or plot-only fields.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    miss = [c for c in feature_cols if c not in seg_clean.columns]
    if miss:
        raise ValueError(f"seg_clean missing feature columns: {miss}")

    meta = [c for c in _SEGMENT_META_ORDER if c in seg_clean.columns]
    pat = [c for c in _PATTERN_COLS if c in seg_clean.columns]

    n = len(seg_clean)
    pieces: dict = {
        "step_name": [step_name] * n,
        "segment_length_mm": [segment_length_mm] * n,
    }
    if experiment_id is not None and "experiment_id" not in seg_clean.columns:
        pieces["experiment_id"] = [experiment_id] * n
    if detection_run_id is not None and "detection_run_id" not in seg_clean.columns:
        pieces["detection_run_id"] = [detection_run_id] * n
    for c in meta:
        pieces[c] = seg_clean[c].values
    for c in feature_cols:
        pieces[c] = seg_clean[c].values
    for c in pat:
        pieces[c] = seg_clean[c].values

    order = ["step_name", "segment_length_mm"]
    if experiment_id is not None and "experiment_id" not in seg_clean.columns:
        order.append("experiment_id")
    if detection_run_id is not None and "detection_run_id" not in seg_clean.columns:
        order.append("detection_run_id")
    order.extend(meta + feature_cols + pat)

    df_out = pd.DataFrame(pieces)[order]

    path = out_dir / filename
    df_out.to_csv(path, index=False)
    return path


def export_cluster_means_and_labels_csv(
    seg_clean: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
    *,
    segment_length_mm: float | int,
    step_name: str,
    experiment_id: Optional[str] = None,
    detection_run_id: Optional[int] = None,
    filename: str = "cluster_means_and_labels.csv",
) -> Path:
    """
    One row per KMeans cluster id (``pattern``): cluster-level mean of each feature (same table as
    ``means = seg_clean.groupby('pattern')[FEATURE_COLS].mean()`` in the notebook), segment count,
    and the post-hoc ``pattern_label`` / ``pattern_group`` assigned to that cluster.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    miss = [c for c in feature_cols if c not in seg_clean.columns]
    if miss:
        raise ValueError(f"seg_clean missing feature columns: {miss}")
    for c in ("pattern", "pattern_label", "pattern_group"):
        if c not in seg_clean.columns:
            raise ValueError(f"seg_clean missing column: {c}")

    g = seg_clean.groupby("pattern", dropna=False)
    means = g[feature_cols].mean()
    means.columns = [f"mean_{fc}" for fc in feature_cols]
    nseg = g.size().rename("n_segments")
    lab = g["pattern_label"].first()
    grp = g["pattern_group"].first()
    df = means.join(nseg).join(lab.rename("pattern_label")).join(grp.rename("pattern_group"))
    if "experiment_id" in seg_clean.columns:
        df = df.join(g["experiment_id"].first().rename("experiment_id"))
    if "detection_run_id" in seg_clean.columns:
        df = df.join(g["detection_run_id"].first().rename("detection_run_id"))
    df = df.reset_index()

    df.insert(0, "step_name", step_name)
    df.insert(1, "segment_length_mm", segment_length_mm)
    idx = 2
    if experiment_id is not None and "experiment_id" not in df.columns:
        df.insert(idx, "experiment_id", experiment_id)
        idx += 1
    if detection_run_id is not None and "detection_run_id" not in df.columns:
        df.insert(idx, "detection_run_id", detection_run_id)

    order = ["step_name", "segment_length_mm"]
    if "experiment_id" in df.columns:
        order.append("experiment_id")
    if "detection_run_id" in df.columns:
        order.append("detection_run_id")
    order.extend(["pattern", "n_segments"])
    order.extend([f"mean_{fc}" for fc in feature_cols])
    order.extend(["pattern_label", "pattern_group"])
    df = df[order]

    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def export_cluster_naming_order_csv(
    seg_clean: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
    *,
    segment_length_mm: float | int,
    step_name: str,
    experiment_id: Optional[str] = None,
    detection_run_id: Optional[int] = None,
    filename: str = "cluster_naming_order.csv",
    label_specs: Optional[List[Tuple[str, str, str]]] = None,
) -> Path:
    """
    One row per post-hoc naming step: replicates ``assign_best`` on
    ``means = seg_clean.groupby('pattern')[features].mean()`` — which cluster (pattern id) receives
    which ``pattern_label`` at each priority, the decision feature, and that cluster’s mean for that feature.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = label_specs if label_specs is not None else NAMING_LABEL_SPECS
    miss = [c for c in feature_cols if c not in seg_clean.columns]
    if miss:
        raise ValueError(f"seg_clean missing feature columns: {miss}")

    means = seg_clean.groupby("pattern", dropna=False)[feature_cols].mean()
    patterns = sorted(means.index.tolist())
    n_clusters = len(patterns)
    assigned = set()
    rows: list[dict] = []

    for naming_step, (lab, feat, ord_) in enumerate(specs[:n_clusters], start=1):
        remaining = [p for p in patterns if p not in assigned]
        if not remaining:
            break
        if feat not in means.columns:
            raise ValueError(f"label_specs references unknown feature: {feat}")
        sub = means.loc[remaining, feat]
        pat = sub.idxmax() if ord_ == "max" else sub.idxmin()
        value = float(means.loc[pat, feat])
        assigned.add(pat)
        row = {
            "naming_step": naming_step,
            "pattern_label_assigned": lab,
            "decision_feature": feat,
            "decision_op": ord_,
            "pattern_cluster_id": int(pat) if not pd.isna(pat) else pat,
            "cluster_mean_of_decision_feature": value,
        }
        for fc in feature_cols:
            row[f"cluster_mean_{fc}"] = float(means.loc[pat, fc])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "step_name", step_name)
    df.insert(1, "segment_length_mm", segment_length_mm)
    idx = 2
    if experiment_id is not None:
        df.insert(idx, "experiment_id", experiment_id)
        idx += 1
    if detection_run_id is not None:
        df.insert(idx, "detection_run_id", detection_run_id)

    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def export_pattern_feature_stats_csv(
    seg_clean: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
    *,
    segment_length_mm: float | int,
    step_name: str,
    experiment_id: Optional[str] = None,
    detection_run_id: Optional[int] = None,
    write_pattern_aggregate_csv: bool = True,
    write_segment_csv: bool = True,
    write_cluster_csv: bool = True,
    write_naming_order_csv: bool = True,
    only_cluster_naming_order_csv: bool = False,
) -> List[Path]:
    """
    Export CSV(s) with mean and std of each feature in ``feature_cols`` for each named pattern.

    ``only_cluster_naming_order_csv=True`` skips **all** exports except ``cluster_naming_order.csv``
    (small file). Use this to avoid huge ``segment_measures.csv`` writes and extra aggregates.

    **Note:** Most runtime is still KMeans / silhouette / segment building in the notebook — turning
    off CSVs does not speed up clustering itself.

    When ``write_pattern_aggregate_csv`` is True, writes ``pattern_feature_means.csv`` (and by-split
    if applicable). When ``write_segment_csv`` is True, writes ``segment_measures.csv`` (can be very
    slow and large). When ``write_cluster_csv`` is True, writes ``cluster_means_and_labels.csv``.
    When ``write_naming_order_csv`` is True, writes ``cluster_naming_order.csv``.
    """
    if only_cluster_naming_order_csv:
        write_pattern_aggregate_csv = False
        write_segment_csv = False
        write_cluster_csv = False
        write_naming_order_csv = True

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    base_cols = [c for c in ("pattern", "pattern_label", "pattern_group") if c in seg_clean.columns]
    if len(base_cols) < 3 or "pattern_label" not in seg_clean.columns:
        return written

    miss = [c for c in feature_cols if c not in seg_clean.columns]
    if miss:
        raise ValueError(f"seg_clean missing feature columns: {miss}")

    def _build(group_cols: List[str]) -> pd.DataFrame:
        g = seg_clean.groupby(group_cols, dropna=False)
        stats = g[feature_cols].agg(["mean", "std"])
        stats.columns = [f"{stat}_{fc}" for fc, stat in stats.columns]
        df = stats.reset_index()
        sizes = g.size().reset_index(name="n_segments")
        df = df.merge(sizes, on=group_cols, how="left")
        df.insert(0, "step_name", step_name)
        idx = 1
        df.insert(idx, "segment_length_mm", segment_length_mm)
        idx = 2
        if experiment_id is not None:
            df.insert(idx, "experiment_id", experiment_id)
            idx += 1
        if detection_run_id is not None:
            df.insert(idx, "detection_run_id", detection_run_id)
        return df

    if write_pattern_aggregate_csv:
        df_main = _build(base_cols)
        p_main = out_dir / "pattern_feature_means.csv"
        df_main.to_csv(p_main, index=False)
        written.append(p_main)

        extra_dims = [
            c
            for c in ("age_group", "sex")
            if c in seg_clean.columns and seg_clean[c].notna().any()
        ]
        if extra_dims:
            gcols2 = base_cols + extra_dims
            df_split = _build(gcols2)
            p_split = out_dir / "pattern_feature_means_by_split.csv"
            df_split.to_csv(p_split, index=False)
            written.append(p_split)

    if write_cluster_csv:
        written.append(
            export_cluster_means_and_labels_csv(
                seg_clean,
                feature_cols,
                out_dir,
                segment_length_mm=segment_length_mm,
                step_name=step_name,
                experiment_id=experiment_id,
                detection_run_id=detection_run_id,
            )
        )

    if write_naming_order_csv:
        written.append(
            export_cluster_naming_order_csv(
                seg_clean,
                feature_cols,
                out_dir,
                segment_length_mm=segment_length_mm,
                step_name=step_name,
                experiment_id=experiment_id,
                detection_run_id=detection_run_id,
            )
        )

    if write_segment_csv:
        written.append(
            export_segment_measures_csv(
                seg_clean,
                feature_cols,
                out_dir,
                segment_length_mm=segment_length_mm,
                step_name=step_name,
                experiment_id=experiment_id,
                detection_run_id=detection_run_id,
            )
        )

    return written
