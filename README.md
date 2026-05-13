# Fly trajectory pattern analysis (60 fps)

This repository holds two Jupyter pipelines that turn **2D fly trajectories** from automated detection into **discrete movement patterns** (clusters) and compare those patterns across biological conditions. Both use the same **segmentation logic**, **kinematic features**, **standardization**, **PCA (for 2D plots)**, and **K-means on scaled features**; they differ in **which experiments are pooled** and **how videos are labeled**.

---

## Shared methodology (both experiments)

### 1. From trajectory to segments

For each video, the code walks along the path in **image coordinates**, converts steps to millimetres using per-run or default **mm per pixel** scales (defaults: **0.04 mm/px** on x, **0.05 mm/px** on y unless overridden in `MM_PER_PX_BY_RUN` or similar), and builds **cumulative distance** along the track. The recording is then split into consecutive **segments of fixed path length**. By default the same six lengths are used everywhere: **5, 10, 15, 20, 30, and 40 mm**. Each segment is one row in the clustering table (one statistical unit).

### 2. Segment-level features (exact definitions)

Each **segment** is a slice of one trajectory between two cumulative-distance boundaries (see “From trajectory to segments” above). For that slice the code aggregates frame-level rows, then computes five columns in **`FEATURE_COLS`**. Intermediate quantities:

- **`seg_path_mm`** — sum of **`step_dist_mm`** over all frames in the segment (contour / path length along the sampled track in millimetres).
- **`net_disp_mm`** — straight-line distance in millimetres from the **first** `(x, y)` to the **last** `(x, y)` in the segment, using separate **`mm_per_px_x`** and **`mm_per_px_y`** so horizontal and vertical distances are scaled independently (anisotropic pixels).

#### `straightness`

**straightness = net_disp_mm / seg_path_mm**

This is the ratio of **chord length** (start-to-end displacement) to **integrated path length**. It lies in **(0, 1]** for typical non-degenerate paths: a perfectly straight walk gives **1**; the more the animal doubles back or meanders, the **smaller** straightness becomes because the chord is short relative to the distance actually walked. Rows with zero path length are dropped or treated as missing.

#### `tortuosity`

**tortuosity = seg_path_mm / net_disp_mm**

This is the **reciprocal** of straightness (with **`net_disp_mm`** clipped to a small floor to avoid division by zero). Values near **1** indicate an almost straight segment; **larger** values indicate a more convoluted path. In the literature “tortuosity” is defined in several ways; here it is explicitly **path length divided by net displacement**, matching the code.

#### `path_per_40` (name vs formula)

Despite the name, the implementation computes:

**path_per_40 = seg_path_mm / SEGMENT_LENGTH_MM**

where **`SEGMENT_LENGTH_MM`** is the **current** analysis window (5, 10, 15, … mm) for that run—not a fixed 40 mm divisor. So this quantity is **actual contour length in the segment bucket, normalised by the nominal segment length** for that analysis. It is useful for comparing how “filled” or distance-efficient the tracked motion is **within** each segment-length setting. Values clustered around **1** mean the summed step lengths in that bin match the nominal window; systematic deviations reflect how segmentation and sampling interact with real motion.

#### `mean_abs_turn_deg`

For every consecutive pair of steps inside the segment, the heading is atan2(Δy, Δx) for each displacement. Successive headings are differenced and wrapped to **(−π, π]**, then converted to **absolute degrees**. **`mean_abs_turn_deg`** is the **mean** of those unsigned turn angles over the segment. It captures **average sharpness** of heading changes at each step: smooth gentle curves tend toward **smaller** means; jittery or rapidly turning motion yields **larger** means. It does not, by itself, count discrete “events”—that is what **`n_turns`** is for.

#### `n_turns` and `TURN_THRESHOLD_DEG`

Using the same wrapped heading differences in degrees, **`n_turns`** counts how many consecutive steps exceed **`TURN_THRESHOLD_DEG`** (default **15°** in the notebooks). Only angles **strictly above** the threshold count. This is a **discrete** count of relatively sharp turning events, useful for separating **zig-zag**-like bouts from paths that turn slowly. The default (15°) is on the conservative side compared with some gait papers that use 22–45°; you can change **`TURN_THRESHOLD_DEG`** in the notebook if you want fewer or more events per segment.

Together, these five descriptors separate **direct** vs **winding** geometry (straightness / tortuosity / path ratio), **fine-scale heading volatility** (mean_abs_turn_deg), and **discrete sharp turns** (n_turns).

---

### 3. Principal component analysis (PCA)

**Role in this pipeline:** PCA is used for **visualisation and reporting**, not as the space in which clusters are found.

**Procedure (as in the notebooks):**

1. Build matrix **`X`** with one row per segment and columns **`FEATURE_COLS`** (raw feature scales differ a lot between columns, so the next step matters).
2. Fit **`StandardScaler`** on **`X`** and transform to **`X_scaled`** (zero mean, unit variance per column across segments in that run). **K-means** and **silhouette** scores are computed on **`X_scaled`** only—i.e. in **five-dimensional scaled feature space**.
3. Fit **`sklearn.decomposition.PCA`** with **`n_components=2`** and **`random_state=42`** on the **same** **`X_scaled`**, and transform to **(PC1, PC2)** for scatter plots.

**Interpretation:**

- **PC1** and **PC2** are orthogonal directions of maximum variance **after** z-scoring. Each component is a linear combination of the five features; the **loadings** (rows of `pca.components_`) tell you which raw (scaled) features push a segment left/right or up/down on the PCA plot.
- **Explained variance ratio** per component is reported in **`pca_specification.csv`** (and related export helpers in `pattern_export_stats.py`): it answers how much of the total variance in scaled features is captured by this 2D view. Often PC1+PC2 explain only a **fraction** of variance—so the PCA scatter is a **projection** for intuition; cluster membership still reflects **all five** scaled dimensions.

**Reproducibility:** `export_pca_specification_csv` writes **`pca_specification.csv`** with, per feature: scaler mean/scale, PCA center on scaled input, and loadings for PC1 and PC2, plus header lines with explained variance ratios. The file documents the reconstruction consistent with scikit-learn: scaled features are z-scores, then PCA applies its internal centering and projection (see comments in `pattern_export_stats.py`).

---

### 4. Clustering and choice of *k*

- **Input to clustering:** **`X_scaled`** (standardised **`FEATURE_COLS`**), **not** PCA coordinates.
- **K-means** (`random_state=42`, `n_init=10`) is run for candidate **`k`** in a range (typically **2 … 10**, upper bound capped when there are very few segments).
- The **silhouette score** on **`X_scaled`** selects **`k`** with the best average cohesion/separation for that segment length.
- **PCA** is fit **after** the scaler is known, on the same **`X_scaled`**, purely for **2D figures**.
- Clusters receive **text labels** (e.g. straight, winding, meandering) by comparing cluster **mean** features to a priority list in the notebook (`assign_best`-style rules), so legend names match biology-friendly language rather than arbitrary cluster ids.

### 5. Outputs — what each file reports (and which parameters it uses)

Outputs are organised **per analysis step** and **per segment length** (e.g. `pattern_distance_output/0_global/10mm/` for Exp1 Step 0 global, or `pattern_distance_output/w1118_dehydration_step0/10mm/` for Exp2). Unless noted, **clustering inputs** are always the five **`FEATURE_COLS`** on **`StandardScaler`**-transformed values: **`straightness`**, **`tortuosity`**, **`path_per_40`**, **`mean_abs_turn_deg`**, **`n_turns`**. **K-means** assigns integer **`pattern`**; the notebook then assigns string **`pattern_label`** and coarse **`pattern_group`** using the same priority rules as `NAMING_LABEL_SPECS` in `pattern_export_stats.py` (zig-zag → `n_turns`; straight → `straightness`; meandering → `path_per_40`; curved → `mean_abs_turn_deg`; winding → `tortuosity`; direct → `straightness`; exploratory → `path_per_40` — each step picks an unassigned cluster by max or min of that feature’s cluster mean).

**Notebook toggles:** **`SKIP_PATTERN_FIGURES`** skips PNG generation only (clustering and CSVs that do not depend on plots still run). **`ONLY_CLUSTER_NAMING_ORDER_CSV`** writes only **`cluster_naming_order.csv`** (skips large **`segment_measures.csv`** and aggregate tables) to save disk I/O.

---

#### `step0_pattern_summary.csv` (Step 0–style global summary)

**Location:** e.g. `0_global/step0_pattern_summary.csv` (Exp1 global Step 0), or `w1118_dehydration_step0/step0_pattern_summary.csv` (Exp2 dehydration run). Not inside each `Nmm` folder.

**What it tells you:** For **each** segment length run, a single-row summary of how many patterns the data support and how confident the silhouette score is.

**Columns (parameters):**

| Column | Meaning |
|--------|---------|
| `segment_length_mm` | Nominal segment length (5, 10, …) for that row. |
| `n_segments` | Number of segments after quality filters (e.g. valid `straightness` / `tortuosity` / `path_per_40`). |
| `n_patterns_k` | Chosen **K-means k** (best silhouette over the searched range). |
| `n_distinct_labels` | How many unique **`pattern_label`** strings were assigned after post-hoc naming. |
| `best_silhouette` | Maximum silhouette score over candidate *k* (computed on **`X_scaled`**, same space as K-means). |
| `labels` | Semicolon-separated list of distinct **`pattern_label`** values for that length. |

---

#### `segment_measures.csv`

**What it tells you:** The **segment-level dataset** used for clustering: every row is one spatial segment of one video, with measured kinematics and the **final cluster assignment**. Use this for downstream statistics, mixed models, or replotting without re-running the notebook.

**Parameters / columns (in order of export):**

- **`step_name`**, **`segment_length_mm`** — which analysis and distance window produced the row (e.g. `step0_global`, `10`).
- **IDs / grouping (included when present on `seg_clean`):** `experiment_id`, `detection_run_id`, `category`, `age_group`, `sex`.
- **Segment geometry / timing:** `seg_40` (segment index along cumulative distance), `n_rows` (frames in segment), `t_start_s`, `t_end_s`, `seg_path_mm`, `seg_dur_s`, `speed_mean_mm_s`.
- **Features used for K-means:** the five **`FEATURE_COLS`** (raw scale, not z-scores — scaling is internal to the notebook).
- **Cluster outputs:** `pattern` (integer cluster id from K-means), `pattern_label` (human-readable name), `pattern_group` (coarser legend bucket, e.g. “Straight”, “Zig-zag / reversing”).

PCA coordinates are **not** stored in this file (they are plot-only in the notebook).

---

#### `cluster_means_and_labels.csv`

**What it tells you:** **Cluster centroids** in feature space: for each K-means cluster (`pattern`), what the **average** locomotion profile looks like, how many segments fell in that cluster, and what label was attached.

**Parameters / columns:**

- **`step_name`**, **`segment_length_mm`**, optional **`experiment_id`** / **`detection_run_id`** for scoped runs.
- **`pattern`** — cluster id; **`n_segments`** — count of segments in that cluster.
- **`mean_<feature>`** for each of **`FEATURE_COLS`** — arithmetic mean of raw features over segments in that cluster (same units as `segment_measures`; not the scaler-centroid unless you re-derive).
- **`pattern_label`**, **`pattern_group`** — post-hoc names from the priority-based assignment.

Use this file to **justify** pattern names in text (“the cluster labelled winding has the highest mean tortuosity …”).

---

#### `cluster_naming_order.csv`

**What it tells you:** The **exact order** in which clusters received their **`pattern_label`**: which naming step ran first, which **decision feature** and **max/min rule** picked which **`pattern_cluster_id`**, and the cluster mean of that decision feature. Also includes **`cluster_mean_<feature>`** for **all** five features at that step so you can audit the full mean vector of the cluster that was named.

**Parameters / columns:** `naming_step`, `pattern_label_assigned`, `decision_feature`, `decision_op` (`max` or `min`), `pattern_cluster_id`, `cluster_mean_of_decision_feature`, plus `cluster_mean_*` for each feature in **`FEATURE_COLS`**.

This matches the notebook’s **`assign_best`** logic and is the right file to cite when explaining **why** a cluster is called “zig-zag” vs “meandering”.

---

#### `pattern_feature_means.csv`

**What it tells you:** For each **named** pattern (`pattern`, `pattern_label`, `pattern_group`), the **mean and standard deviation** of every feature across all segments assigned to that pattern — i.e. **distribution shape** within each pattern class.

**Parameters / columns:** `step_name`, `segment_length_mm`, then grouping keys `pattern`, `pattern_label`, `pattern_group`, then for each feature in **`FEATURE_COLS`**: `mean_<feature>`, `std_<feature>`, and **`n_segments`** (how many segments contributed).

---

#### `pattern_feature_means_by_split.csv`

**What it tells you:** Same as **`pattern_feature_means.csv`**, but **stratified** so you can compare patterns between cohorts. Written only when the segment table has usable **`age_group`** and/or **`sex`** columns.

**Extra grouping parameters:** adds **`age_group`** and/or **`sex`** to the group-by keys, so each row is a unique combination of (pattern identity × cohort slice). Still aggregates **`FEATURE_COLS`** with `mean` / `std` / segment counts.

Use this to ask: “within the **meandering** pattern, do young males differ from old females in **mean_abs_turn_deg** variance?” without re-aggregating from `segment_measures.csv`.

---

#### `pca_specification.csv`

**What it tells you:** How **PC1** and **PC2** were constructed from the **scaled** five features: documented **reproducibility** for the 2D PCA figures, not the K-means partition.

**Parameters:** Comment header lines record `explained_variance_ratio` for PC1 and PC2, cumulative variance for the plane, `n_segments`, `k_patterns`, `analysis_id`. The table has one row per feature with **`scaler_mean`**, **`scaler_scale`** (StdScaler), **`pca_mean_on_scaled_input`**, **`loading_PC1`**, **`loading_PC2`**. See `export_pca_specification_csv` in `pattern_export_stats.py` for the reconstruction formulas.

---

#### Figure outputs (PNG) — Step 0 naming (`step0_*.png` in each `Nmm/` folder)

Figures are **readouts** of the same clustering; axes are drawn from **segment-level** fields after clustering.

| File (typical) | What it shows | Main parameters on axes / hue |
|----------------|----------------|--------------------------------|
| **`step0_silhouette_k.png`** | **Left:** silhouette vs candidate *k*; **right:** K-means inertia vs *k*. Vertical line at chosen **`n_patterns_k`**. | *k*; silhouette on **`X_scaled`** assignments; inertia in scaled space. |
| **`step0_pattern_counts.png`** | Bar chart: how many segments per **`pattern_label`**. | Count vs label (derived from **`FEATURE_COLS`** via clusters). |
| **`step0_pca_patterns.png`** | Scatter of **PC1** vs **PC2** (PCA on **`X_scaled`**), colour = **`pattern_label`**. | 2D projection of the same five scaled features; cluster hue. |
| **`step0_movements_per_pattern.png`** | Small multiples: overlaid **trajectories** (x,y translated to segment start) for examples per **`pattern_group`**. Line colour = **YM/YF/OM/OF** if `sex`+`age_group` allow cohort tags, else **young vs old** by `age_group`. | Raw **pixel** trajectories from `df_seg`; not the scaled feature space. |
| **`step0_movements_avg_turn_per_pattern.png`** | Boxplot: **`mean_abs_turn_deg`** vs **`pattern_group`**, split by cohort or **`age_group`**. | Turn statistic vs pattern; hue = sex×age cohorts or age only. |
| **`step0_movements_n_turns_per_pattern.png`** | Boxplot: **`n_turns`** vs **`pattern_group`**, same hue logic. | Discrete turn count vs pattern. |
| **`step0_movements_turn_angle_vs_n_turns.png`** | Scatter: **`mean_abs_turn_deg`** vs **`n_turns`**, coloured by **`pattern_group`**. | Shows how smooth turning vs sharp-turn counts co-vary across segments. |
| **`step0_actual_positions.png`** | **Full-field** trajectory snippets per pattern in **absolute** image coordinates (not centred). | Raw `x`, `y` with cohort colouring where available. |

**Steps 1–3 and cohort cells** use the same **CSV** names (`segment_measures.csv`, etc.) under their own output folders, and **analogous PNGs** with prefixes like **`step1_`** or cohort-specific names (`pattern_k_selection.png`, `pattern_by_age_group_stacked.png`, …). The **quantities** on the axes remain the **`FEATURE_COLS`** and derived **`pattern` / `pattern_label` / `pattern_group`**, plus whatever **comparison dimension** that step defines (per-fly, male/female pools, young/old, etc.).

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
