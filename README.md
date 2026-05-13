# Fly trajectory pattern analysis (60 fps)

This repository contains two related Jupyter analyses that turn **2D fly trajectories** from automated detection into **discrete movement patterns** (clusters) and compare those patterns across biological conditions. Both pipelines share the same **per-segment kinematic features**, **standardization**, **PCA for visualization**, and **K-means** clustering machinery; they differ in **which experiments are pooled** and **how videos are labeled**.

---

## What the analyses do (conceptually)

1. **Trajectory → segments**  
   For each video, cumulative path length along the track is used to chop the recording into consecutive **segments of fixed length** (by default **5, 10, 15, 20, 30, and 40 mm**). Each segment is one statistical unit.

2. **Segment → features**  
   For every segment, the code computes a small vector of **locomotion descriptors** (see below). These are the inputs to clustering.

3. **Features → patterns**  
   Features are **z-scored** (`StandardScaler`), projected with **PCA** (mainly for 2D plots and interpretability), and **K-means** is run for a range of candidate cluster counts **k**. A score such as the **silhouette** metric helps choose how many distinct **movement patterns** best summarize the data at that segment length.

4. **Patterns → figures and tables**  
   Each segment length gets its own output folder with **plots** (PCA scatter, cluster summaries, condition-wise breakdowns where applicable) and **CSV exports** documenting segment-level assignments, cluster means, pattern labels, and (optionally) the exact scaler + PCA specification for reproducibility (see `pattern_export_stats.py`).

**Feature columns (both experiments):**

| Feature | Meaning (short) |
|--------|------------------|
| `straightness` | How direct the path is within the segment |
| `tortuosity` | How winding the path is |
| `path_per_40` | Path length relative to a 40 mm reference |
| `mean_abs_turn_deg` | Mean absolute heading change between steps (degrees) |
| `n_turns` | Count of sharp turns above a fixed threshold (default **15°** between successive headings) |

Default **mm per pixel** scales (used when not overridden per run): **0.04 mm/px (x)**, **0.05 mm/px (y)**.

---

## Experiment 1 — `Exp1/pattern_distance.ipynb`

**Biological question (typical use):** How do **locomotor movement patterns** differ between **young vs old** *Drosophila*, and between **males vs females**, for the **WIG** genotype, when pooling multiple recording days?

**Data flow:**

1. Raw detector exports live under `Exp1/sd_*` folders (one folder per recording session). Each session is configured in `Exp1/merge_detection_csvs.py` (`EXPERIMENT_CONFIG`) with **age** (`young` / `old`) and **ID ranges** that map `detection_output_<N>.csv` fly indices to **male** vs **female**.
2. Run **`python merge_detection_csvs.py`** from **`Exp1/`** to build **`merged_detection_all.csv`** — one long table with consistent columns plus metadata (`experiment_id`, fly id, sex, age group, genotype label such as `WIG`, etc.).
3. Open **`pattern_distance.ipynb`**, select the project **virtualenv** kernel (see [Environment](#environment)), and run **top to bottom**.

**Analysis steps inside the notebook:**

| Step | What it does |
|------|----------------|
| **Load / setup** | Reads `merged_detection_all.csv`, normalizes columns, recomputes timing and distances (`dt_s`, `step_dist_mm`, `cum_dist_mm`, `total_dist_mm`, speeds). |
| **Step 0 — global pattern discovery** | Pools **all** trajectories, picks **k** patterns per segment length, writes summary tables and figures under `pattern_distance_output/0_global/`. If `sex` and `age_group` are present, figures can label cohorts **YM, YF, OM, OF** (young/old × male/female). |
| **Steps 1–3** | Use the Step-0 pattern structure for finer splits: per-fly / per-CSV views (**Step 1**), pooled males vs females with young vs old (**Step 2**), pooled young vs old with male vs female (**Step 3**). |

Optional **`USER_VIDEO_METADATA_CSV`**: merge extra columns (or overrides) keyed by `(experiment_id, detection_run_id)`.

**Outputs:** Under `Exp1/pattern_distance_output/` (PNG figures plus CSVs such as `segment_measures.csv`, `cluster_means_and_labels.csv`, `cluster_naming_order.csv`, `pattern_feature_means*.csv`, `pca_specification.csv`). Regenerating everything can be large and slow; the notebook exposes flags like `SKIP_PATTERN_FIGURES` and `ONLY_CLUSTER_NAMING_ORDER_CSV` for partial runs.

---

## Experiment 2 — `Exp2/pattern_distance_main.ipynb`

**Biological question:** For **w1118** flies, how do movement patterns compare **control (on media)** vs **dehydrated** animals when pooling two recording dates?

**Data flow:**

1. There is **no** merged master CSV required for the main dehydration workflow. The notebook reads **`detection_output_*_filtered.csv`** (or equivalent naming as set in the notebook) directly from **`Exp2/sd_09_08/`** and **`Exp2/sd_09_09/`**.
2. **`detection_run_id`** is the video index **`N`** in the filename.
3. Condition labels follow the **`Exp2/Sign-up list (1).pdf`** convention described in the notebook:
   - **`sd_09_09`**: videos **1, 3, 5** → w1118 control; **2, 4, 6** → dehydrated w1118.
   - **`sd_09_08`**: videos **1–3** → w1118 control; **4–7** → dehydrated w1118.

**Analysis:** The notebook inlines a **Step-0–style** routine (`run_dehydration_step0_style`) that uses the same feature set and clustering approach as Exp1, but **only two hue groups** (control vs dehydrated). For plotting, the code maps these onto the same machinery used elsewhere (e.g. `age_group`–like fields with display names such as **w1118** vs **Dehydrated w1118**).

**Outputs:** `Exp2/pattern_distance_output/w1118_dehydration_step0/<N>mm/` for each segment length **N**, plus a run-level summary CSV where configured.

`Exp2/merge_detection_csvs.py` exists for parity with Exp1 if you ever want a merged table from Exp2 folders; the **primary** dehydration analysis is self-contained in **`pattern_distance_main.ipynb`**.

---

## Repository layout

```
60fps_pattern/
├── README.md                 # this file
├── requirements.txt          # Python dependencies
├── Exp1/
│   ├── pattern_distance.ipynb
│   ├── merge_detection_csvs.py
│   ├── pattern_export_stats.py
│   ├── sd_*/                  # raw detection CSVs per session (tracked if present)
│   └── pattern_distance_output/   # generated (ignored by git by default)
└── Exp2/
    ├── pattern_distance_main.ipynb  # main dehydration notebook
    ├── pattern_distance.ipynb       # additional notebook in Exp2
    ├── Sign-up list (1).pdf         # video ↔ condition key for dehydration runs
    ├── merge_detection_csvs.py
    ├── pattern_export_stats.py
    ├── sd_09_08/, sd_09_09/
    └── pattern_distance_output/
```

---

## Environment

Python **3.10+** recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=60fps-pattern --display-name='Python (.venv 60fps pattern)'
```

In Jupyter / VS Code / Cursor, choose that kernel when running either notebook. The notebooks check that the interpreter matches a project `.venv` when one exists (to avoid missing `sklearn` / `seaborn`).

---

## License and data

Add a **LICENSE** if the repository is public. Raw trajectory CSVs can be large; this repo’s **`.gitignore`** excludes the merged Exp1 table and generated `pattern_distance_output/` trees so `git push` stays within typical GitHub limits. Keep heavy artifacts on **Zenodo**, **Drive**, or **Git LFS** if collaborators need byte-for-byte reproduction without re-running the merge and notebooks.

---

## Publishing to GitHub (quick checklist)

1. Create an empty repository on GitHub (no README if you already have one here).
2. In a terminal:

```bash
cd /path/to/60fps_pattern
git init
git add README.md requirements.txt .gitignore Exp1/*.py Exp1/*.ipynb Exp2/*.py Exp2/*.ipynb
# add raw sd_* data as needed (respect .gitignore and GitHub file size limits)
git add Exp1/sd_* Exp2/sd_*   # if those folders are not ignored and you want them tracked
git commit -m "Add fly trajectory pattern analyses (Exp1 WIG age/sex, Exp2 w1118 dehydration)"
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

Replace `<YOUR_USER>` / `<YOUR_REPO>` with your account and repository name. Use **SSH** (`git@github.com:...`) instead of HTTPS if that is how your machine is configured.
