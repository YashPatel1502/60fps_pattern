# Fly trajectory pattern analysis (60 fps)

This repository holds two Jupyter pipelines that turn **2D fly trajectories** from automated detection into **discrete movement patterns** (clusters) and compare those patterns across biological conditions. Both use the same **segmentation logic**, **kinematic features**, **standardization**, **PCA for visualization**, and **K-means** machinery; they differ in **which experiments are pooled** and **how videos are labeled**.

---

## Shared methodology (both experiments)

### 1. From trajectory to segments

For each video, the code walks along the path in **image coordinates**, converts steps to millimetres using per-run or default **mm per pixel** scales (defaults: **0.04 mm/px** on x, **0.05 mm/px** on y unless overridden in `MM_PER_PX_BY_RUN` or similar), and builds **cumulative distance** along the track. The recording is then split into consecutive **segments of fixed path length**. By default the same six lengths are used everywhere: **5, 10, 15, 20, 30, and 40 mm**. Each segment is one row in the clustering table (one statistical unit).

### 2. Segment-level features

For every segment, five **shape / locomotion descriptors** are computed (`FEATURE_COLS` in the notebooks):

| Column | Role |
|--------|------|
| `straightness` | How direct the path is within the segment (end-to-end vs path length). |
| `tortuosity` | How winding or convoluted the path is. |
| `path_per_40` | Path length in the segment expressed relative to a **40 mm** reference scale (helps compare across segment lengths). |
| `mean_abs_turn_deg` | Mean absolute **heading change** between successive steps, in degrees (from `arctan2` of displacements). |
| `n_turns` | Count of heading changes above **`TURN_THRESHOLD_DEG`** (default **15°**); sharp turns are treated as discrete events. |

Together these capture whether a fly moved in a roughly straight line, curved gently, zig-zagged, meandered, etc., over a fixed distance window.

### 3. Clustering and choice of *k*

- Features are **z-scored** with `sklearn.preprocessing.StandardScaler`.
- **PCA** is fit on the scaled features for **2D visualisation** and interpretability; clustering is still done in the scaled feature space (PCA is not the sole driver of assignments unless the notebook path specifies otherwise for a given step).
- **K-means** is run for candidate cluster counts **k = 2 … 10**.
- The implementation uses the **silhouette score** to pick the **k** that best separates segments into coherent clusters at each segment length.
- Clusters receive **text labels** (e.g. straight, curved, meandering) from their mean feature profiles so figures are readable in a paper or talk.

### 4. Outputs (CSVs and figures)

Under each run’s `pattern_distance_output/` tree you typically find, **per segment length** (e.g. `10mm/`):

- **`segment_measures.csv`** — one row per segment with features, cluster assignment, and metadata keys (experiment, video id, etc.).
- **`cluster_means_and_labels.csv`** — cluster centroids and human-readable pattern names.
- **`cluster_naming_order.csv`** — stable ordering / naming for legends across runs.
- **`pattern_feature_means.csv`** and **`pattern_feature_means_by_split.csv`** — pattern summaries overall and split by the comparison hue (age, sex, dehydration group, …).
- **`pca_specification.csv`** — exports from `pattern_export_stats.py` so PCA/scaler choices can be audited or reproduced.
- **PNG figures** — silhouette vs *k*, pattern counts, PCA scatter coloured by cluster, movement statistics per pattern, and (where implemented) **trajectory snippets** overlaid for representative segments.

The notebooks expose toggles such as **`SKIP_PATTERN_FIGURES`** (recompute clusters but skip slow PNG regeneration) and **`ONLY_CLUSTER_NAMING_ORDER_CSV`** for lighter exports.

---

## Experiment 1 — `Exp1/pattern_distance.ipynb`

### Biological question

How do **locomotor movement patterns** differ between **young vs old** *Drosophila*, and between **males vs females**, for **WIG** (and related) genotypes, when pooling multiple recording days from several `sd_*` sessions?

### Data you need

1. **Raw detector CSVs** live under `Exp1/sd_*` (one folder per recording session, e.g. `sd_09_12`, `sd_09_17`, …).
2. **`merge_detection_csvs.py`** defines **`EXPERIMENT_CONFIG`**: which `detection_output_<N>.csv` files belong to which session, and how fly indices map to **sex** and **age** metadata.
3. From **`Exp1/`**, run:

   ```bash
   python merge_detection_csvs.py
   ```

   That produces **`merged_detection_all.csv`** — one long table with aligned columns plus keys such as `experiment_id`, `detection_run_id`, `category`, `age_group`, and (when configured) **`sex`**. This file can be **very large**; it is listed in `.gitignore` so you regenerate it locally rather than pushing it to GitHub.

### Notebook workflow (run top to bottom)

1. **Imports / environment** — The first code cell checks that you are using the project **`.venv`** kernel so `numpy`, `pandas`, `seaborn`, `matplotlib`, and `sklearn` resolve consistently.

2. **Load / setup** — Reads `merged_detection_all.csv`, normalises column names, and recomputes timing and distances: `dt_s`, `step_dist_mm`, `cum_dist_mm`, `total_dist_mm`, speeds, and per-video totals. Optional **`USER_VIDEO_METADATA_CSV`** can override or append columns keyed by `(experiment_id, detection_run_id)` (e.g. manual **sex** fixes).

3. **Step 0 — global pattern discovery** — Pools **all** trajectories in the merged table. For each segment length it chooses **k**, assigns **pattern labels**, and writes:

   - Summary: `pattern_distance_output/0_global/step0_pattern_summary.csv`
   - **Eight figures per distance folder** under `0_global/<N>mm/` (silhouette sweep, counts, PCA, movement breakdowns, etc.).

   When the merged CSV includes **`sex`** and **`age_group`**, cohort figures use **YM / YF / OM / OF** (young male, young female, old male, old female). Missing sex is shown in neutral styling and may be omitted from four-way breakdowns; if there is no `sex` column, plots fall back to **young vs old** only.

4. **Step 1 — per fly / per CSV** — Each `(experiment_id, detection_run_id)` is analysed like a miniature Step 0. Outputs go under `pattern_distance_output/1_entire_data/<exp>_run_<id>/` with the **same eight figure types** per `<N>mm/` plus per-fly summaries.

5. **Step 2 — males vs females (young and old pooled within sex)** — Pooled comparisons with outputs under `2_males/<N>mm/` and `2_females/<N>mm/`.

6. **Step 3 — young vs old (male vs female within each age)** — Outputs under `3_young/<N>mm/` and `3_old/<N>mm/`.

7. **Cohort experiments (single cell)** — Runs several predefined comparisons at once (e.g. young vs old with sex splits, male vs female with age splits) and writes under `pattern_distance_output/cohort_experiments/<experiment_key>/`.

**Practical note:** Run **Step 0 first** so the global *k* and pattern vocabulary are established before interpreting Steps 1–3.

---

## Experiment 2 — `Exp2/pattern_distance_main.ipynb`

### Biological question

For **w1118** flies, how do **movement patterns** compare **control animals on media** vs **dehydrated** animals when pooling two recording dates (`sd_09_08` and `sd_09_09`)?

### Data you need

- **No merged master CSV is required** for the main dehydration workflow. The notebook reads **`detection_output_*_filtered.csv`** (or the naming convention set in the notebook) directly from **`Exp2/sd_09_08/`** and **`Exp2/sd_09_09/`**.
- **`detection_run_id`** corresponds to the video index **`N`** in the filename.
- Condition assignments follow **`Exp2/Sign-up list (1).pdf`** (summarised in the notebook):

  | Session folder | Experiment id | Control (on media) | Dehydrated |
  |----------------|---------------|--------------------|------------|
  | `sd_09_09` | `sd_09_09` | Videos **1, 3, 5** | Videos **2, 4, 6** |
  | `sd_09_08` | `sd_09_08` | Videos **1–3** | Videos **4–7** |

### Notebook workflow

1. **Imports / setup** — Same libraries and feature definitions as Exp1; path and output root point at `Exp2/pattern_distance_output/`.

2. **Pooled dehydration analysis** — The notebook inlines **`run_dehydration_step0_style`**, which mirrors **Exp1 Step 0** logic but restricts the comparison to **two groups** (control w1118 vs dehydrated w1118). Internally the routine uses the same segmenting, scaling, PCA, K-means, and silhouette machinery; **hue** in plots is mapped so **control** and **dehydrated** are visually distinct (e.g. blue vs coral, analogous to young/old styling in Exp1).

3. **Outputs** — Written to:

   `Exp2/pattern_distance_output/w1118_dehydration_step0/<N>mm/`

   for each segment length **N**, including the usual Step-0-style PNG set, plus a **`step0_pattern_summary.csv`** (and related CSVs) at the run level as implemented in the notebook.

`Exp2/merge_detection_csvs.py` is kept for **parity with Exp1** if you ever want a single merged table from the Exp2 folders; the **primary** dehydration paper-style comparison is self-contained in **`pattern_distance_main.ipynb`**.

---

## Repository layout (current)

```text
60fps_pattern/
├── README.md
├── requirements.txt
├── .gitignore
├── Exp1/
│   ├── pattern_distance.ipynb      # WIG age / sex / cohort analysis
│   ├── merge_detection_csvs.py
│   ├── pattern_export_stats.py
│   ├── sd_*/                       # raw detection CSVs per session (tracked)
│   └── pattern_distance_output/    # generated (gitignored by default)
└── Exp2/
    ├── pattern_distance_main.ipynb # w1118 control vs dehydrated
    ├── Sign-up list (1).pdf        # video ↔ condition key
    ├── merge_detection_csvs.py
    ├── pattern_export_stats.py
    ├── sd_09_08/, sd_09_09/
    └── pattern_distance_output/
```

---

## Environment

Python **3.10+** is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=60fps-pattern --display-name='Python (.venv 60fps pattern)'
```

Select that kernel in Jupyter, VS Code, or Cursor. The notebooks optionally **raise a clear error** if the active interpreter is not the project `.venv` when one exists nearby—this avoids confusing `ModuleNotFoundError` for `sklearn` or `seaborn`.

---

## GitHub and large files

- **`merged_detection_all.csv`** and **`pattern_distance_output/`** are **gitignored** so pushes stay within normal GitHub size limits.
- Raw `sd_*` CSVs are smaller and can be tracked for reproducibility; if the total grows beyond GitHub’s limit, use **Git LFS**, **Zenodo**, or an internal file share and document download steps here.

### Push updates to the existing remote

If `origin` already points at your GitHub repo:

```bash
cd /path/to/60fps_pattern
git status
git add README.md requirements.txt .gitignore
git add Exp1/pattern_distance.ipynb Exp2/pattern_distance_main.ipynb Exp1/*.py Exp2/*.py
# Stage removal of renamed/removed notebooks if applicable:
# git add -u Exp2/
git commit -m "Document Exp1 and Exp2 pattern pipelines; restore project metadata files"
git push origin main
```

Replace `main` with your default branch name if different.

---

## License

Add a **LICENSE** file if you make the repository public.
