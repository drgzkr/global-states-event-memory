# Global Brain States and Hippocampal Memory

**Memory and Hippocampal Responses to Event Boundaries are Modulated by Global Brain States**
Gözükara, D., Oetringer, D., Ahmad, N., & Geerligs, L. (2026)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/global-states-event-memory/blob/main/Global_Brain_States_and_Hippocampal_Memory.ipynb)

---

## Overview

Our experiences unfold continuously, yet we remember them as discrete episodes. This repository contains all code to reproduce the analyses in the paper above, which examines how **large-scale brain state dynamics modulate hippocampal memory encoding at event boundaries** during naturalistic movie-watching.

Using fMRI data from two independent datasets — **Sherlock** (BBC series) and **StudyForrest** (Forrest Gump) — we show that:

1. Event boundaries are associated with an increased probability of being in a **Task-Positive Network (TPN)** global brain state.
2. Hippocampal activity is systematically higher during **TPN** states than during **Default Mode Network (DMN)** states.
3. The hippocampal response to event boundaries appears largest during TPN states, but is present in both states once baseline differences are controlled for.
4. Individual differences in the tendency to **shift toward the TPN at event boundaries**, and overall time spent in the DMN, predict **subsequent memory** for narrative content — while univariate hippocampal activity at boundaries does not.

---

## Repository Structure

```
global-states-event-memory/
│
├── Global_Brain_States_and_Hippocampal_Memory.ipynb   # Main analysis notebook (open in Colab)
├── run_gsbs_sherlock.py                    # Preprocessing: GSBS on Sherlock dataset
├── run_gsbs_studyforrest.py                # Preprocessing: GSBS on StudyForrest dataset
└── README.md                               # This file
```

---

## Quick Start

### Run the analysis notebook in Google Colab

Click the badge at the top of this README, or go directly to:

> https://colab.research.google.com/github/drgzkr/global-states-event-memory/blob/main/Global_States_Hippocampus_fixed.ipynb

The notebook downloads all required pre-computed data automatically from a public Google Drive folder. You only need a Google account and a browser if you prefer not to install locally.

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/drgzkr/global-states-event-memory.git
cd global-states-event-memory

# 2. Install dependencies
pip install nilearn gdown joblib scipy tqdm numpy pandas matplotlib seaborn
pip install git+https://github.com/drgzkr/statesegmentation

# 3. Open the notebook
jupyter notebook Global_States_Hippocampus_fixed.ipynb
```

When running locally, set `DATA_ROOT` in Section 1.2 of the notebook to a local directory and either download the data there manually or use `gdown` with the folder ID.

---

## Data

### Datasets

| Dataset | Stimulus | Subjects | TRs | TR (s) | Reference |
|---|---|---|---|---|---|
| **Sherlock** | BBC Sherlock, Episode 1, Part 1 (48 min) | 17 | 1,976 | 1.5 | Chen et al., 2017 |
| **StudyForrest** | Forrest Gump dubbed in German (120 min) | 15 | 3,599 | 2.0 | Hanke et al., 2016 |

Both datasets are publicly available. The Sherlock dataset can be obtained from [OpenNeuro](https://openneuro.org/datasets/ds001132). The StudyForrest dataset is available at [studyforrest.org](http://studyforrest.org) and on [OpenNeuro](https://openneuro.org/datasets/ds000113).

### Pre-computed data (required to run the notebook)

The notebook loads pre-computed intermediate data that avoids re-running the computationally expensive GSBS preprocessing (which takes several hours on a multi-core machine). The following files are hosted in a public Google Drive folder and downloaded automatically in Section 1.2:

| File | Description |
|---|---|
| `Sherlock_all_subs_hippocampus_response.pkl` | Per-subject hippocampal timeseries, Sherlock (list of arrays, shape `[n_voxels, n_TRs]`) |
| `StudyForrest_all_subs_hippocampus_response.pkl` | Per-subject hippocampal timeseries, StudyForrest |
| `Sherlock_SingleSubGlobalGSBS_Results/` | Per-subject, per-run GSBS objects for Sherlock (`.npy`) |
| `StudyForrest_SingleSubGlobalGSBS_Results/` | Per-subject, per-run GSBS objects for StudyForrest (`.npy`) |
| `Sherlock_Annotations/movie_scenes_continued.csv` | Sherlock event boundary onsets (TRs) |
| `Sherlock_Annotations/Sherlock_timing_data.csv` | Per-subject scene recall data |
| `StudyForrest_Annotations/events run{1–8}.tsv` | StudyForrest event boundary onsets (seconds), from Ben-Yakov & Henson (2018) |

To regenerate the GSBS objects from raw fMRI data, see [Preprocessing](#preprocessing) below.

### Atlas

The **Schaefer 400-ROI** cortical parcellation (7 Yeo networks) is used throughout. It is fetched automatically by `nilearn` — no manual download needed.

- [ ] The **hippocampal mask** is derived from the WFU PickAtlas toolbox (Maldjian et al., 2003), resampled to match each participant's MNI-space data. A template mask is included in the data folder.

---

## Preprocessing: GSBS Scripts

The two preprocessing scripts run the [Greedy State Boundary Search (GSBS)](https://github.com/drgzkr/statesegmentation) algorithm on single-subject whole-brain fMRI data to produce the neural state objects loaded by the notebook.

> **You do not need to run these scripts** if you use the pre-computed data downloaded by the notebook. They are provided for full reproducibility and for researchers who wish to apply the pipeline to new data.

### `run_gsbs_sherlock.py`

Processes the **Sherlock** dataset.

**Input:** One 4-D NIfTI per subject (`SherlockSub{sub}.nii`) containing the full 1,976-TR timeseries (both runs concatenated).

**What it does:**
1. Fetches the Schaefer 400-ROI atlas via `nilearn` and resamples it to match subject space.
2. Extracts the mean timeseries for each of the 400 ROIs.
3. Splits the timeseries at TR 946, corresponding to the original two-part acquisition.
4. Fits GSBS independently on run 1 (TRs 0–945) and run 2 (TRs 946–1975), with `kmax = n_TRs / 2` and `statewise_detection=True`.
5. Saves each fitted GSBS object as a `.npy` file.

**Parallelism:** Subjects are processed in parallel (default: 2 threads). Raise `N_JOBS` to match your machine.

**Output files** (one per subject per run):
```
Sherlock_GSBS_sub{sub}_run{1|2}_Schaefer_400_ROIs.npy
```

**Usage:**
```bash
# Edit DATA_DIR, OUTPUT_DIR, and REFERENCE_NIFTI at the top of the script, then:
python run_gsbs_sherlock.py
```

**Expected runtime:** ~30–60 minutes per subject on a modern CPU (2 subjects in parallel ≈ 3–6 hours total).

---

### `run_gsbs_studyforrest.py`

Processes the **StudyForrest** dataset.

**Input:** One 4-D NIfTI per subject per run (`waligned_{sub}_ses-movie_task-movie_run-{run}_space-T1w_desc-unsmDenoised_bold.nii`), reflecting the partially preprocessed data from Liu et al. (2019) with additional SPM12 coregistration and MNI normalisation applied by the authors.

**What it does:**
1. Fetches the Schaefer 400-ROI atlas via `nilearn` and resamples it to match subject space.
2. For each of the 8 runs and 15 subjects:
   - Loads the NIfTI, z-scores voxel timeseries.
   - Extracts mean timeseries for each of 400 ROIs.
   - Fits GSBS with `kmax = n_TRs / 2` and `statewise_detection=True`.
   - Saves the fitted object as a `.npy` file.

**Parallelism:** Runs are processed in parallel (default: 8 threads, one per run), with subjects looped inside each worker.

**Output files** (one per subject per run):
```
GSBS_{sub}_run{run}Schaefer_400_ROIs.npy
```

**Usage:**
```bash
# Edit DATA_DIR, OUTPUT_DIR, and REFERENCE_NIFTI at the top of the script, then:
python run_gsbs_studyforrest.py
```

**Expected runtime:** ~15–30 minutes per subject per run (8 runs in parallel ≈ 2–4 hours per subject; all 15 subjects ≈ 30–60 hours total on 8 cores).

---

## Analysis Notebook

`Global_Brain_States_and_Hippocampal_Memory.ipynb` is the single notebook that reproduces all main figures and statistical results in the paper. It is structured as follows:

| Section | Contents |
|---|---|
| **1. Setup** | Install packages, download data from Google Drive, configure output paths |
| **2. Parameters** | Dataset selection, all analysis parameters, figure style |
| **3. Schaefer Atlas** | Load atlas, build network-membership masks for DBSCAN orientation |
| **4. Utility Functions** | NIfTI projection, surface plotting, legend helpers |
| **5. Hippocampus Data** | Load pre-extracted hippocampal timeseries; subject-level QC plot |
| **6. Event Boundaries** | Load behavioural boundary annotations; apply 4.5 s HRF lag |
| **7. Global Brain States** | Load GSBS objects → LOO reliability → DBSCAN grid search → DMN/TPN labelling → state timeseries visualisation |
| **8. State Occurrence Statistics** | Proportion of time in DMN vs TPN per subject; Wilcoxon tests |
| **9. Hippocampus at Event Boundaries** | Event-window averaging of hippocampal activity and global state probability |
| **10. GLM and FIR Models** | All-boundaries GLM/FIR; state-split GLM/FIR without and with state regressor; global-state GLM/FIR |
| **11. Memory Analysis** | Sherlock only: memory scores, DMN time vs memory, TPN-at-boundary vs memory, hippocampal activity vs memory, high/low memory subgroup FIR |
| **Appendix** | Note on Wilcoxon z-statistic reporting |

**Estimated runtime (Colab free tier):** Sections 1–9 run in a few minutes. Section 10 (GLM/FIR loops) takes ~1 minute. The DBSCAN grid search in Section 7.5 takes ~5 minutes and is wrapped in `%%time`.

---

## Methods Summary

### Global brain state identification

1. **ROI extraction:** Mean timeseries extracted for each of 400 Schaefer cortical ROIs.
2. **GSBS:** The [Greedy State Boundary Search](https://github.com/drgzkr/statesegmentation) algorithm (Geerligs et al., 2021; 2022) segments each subject's timeseries into discrete neural states by maximising within-state Pearson correlations and minimising between-state correlations. Applied independently per subject and per run.
3. **DBSCAN clustering:** State activity patterns from all subjects and runs are pooled and clustered using DBSCAN with cosine distance. A grid search over `eps` (0.01–1.0) and `min_samples` (2–2,000) selects parameters via Silhouette and Calinski–Harabász scores. In both datasets this yields exactly two clusters, labelled **DMN** and **TPN** by comparing mean DAN vs DMN ROI activity.
4. **TR-level labels:** Each TR is assigned the label (0 = DMN, 1 = TPN) of its enclosing GSBS state.

### GLM and FIR models

A general linear model (GLM) is fitted per subject to estimate the mean response at event boundaries. A finite impulse response (FIR) model, with one predictor per TR in a ±10.5 s window, characterises the full temporal response profile. Both are implemented via `nilearn.glm.first_level.make_first_level_design_matrix` and `run_glm`. For the state-dependent analyses, boundaries are classified as DMN- or TPN-labelled based on the dominant state in a ±3 TR window (70% threshold). The same GLMs are re-estimated with the binary state timeseries as a nuisance regressor to control for baseline differences between states.

Group-level inference uses the **Wilcoxon signed-rank test** against zero (or against each other), implemented via `scipy.stats.wilcoxon` with the normal approximation (`method='asymptotic'`). See the Appendix for a note on z-statistic sign convention.

---

## Dependencies

| Package | Purpose |
|---|---|
| `nilearn` | Atlas fetching, NIfTI I/O, GLM |
| `numpy`, `scipy` | Numerics, statistics |
| `pandas` | Event files, memory data |
| `matplotlib`, `seaborn` | Plotting |
| `scikit-learn` | DBSCAN, silhouette score |
| `tqdm` | Progress bars |
| `joblib` | Parallelism in preprocessing scripts |
| `gdown` | Downloading data from Google Drive (notebook only) |
| `statesegmentation` | GSBS algorithm |

Install everything at once:
```bash
pip install nilearn gdown joblib scipy tqdm numpy pandas matplotlib seaborn scikit-learn
pip install git+https://github.com/drgzkr/statesegmentation
```

---

## Citation

If you use this code or the pre-computed data, please cite:

```bibtex
@article{gozukara2026global,
  title   = {Memory and Hippocampal Responses to Event Boundaries are
             Modulated by Global Brain States},
  author  = {Gözükara, Dora and Oetringer, Djamari and Ahmad, Nasir
             and Geerligs, Linda},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {}
}
```

Please also cite the GSBS method:

```bibtex
@article{geerligs2021detecting,
  title   = {Detecting neural state transitions underlying event segmentation},
  author  = {Geerligs, Linda and van Gerven, Marcel and Güçlü, Umut},
  journal = {NeuroImage},
  volume  = {236},
  pages   = {118085},
  year    = {2021}
}

@article{geerligs2022partially,
  title   = {A partially nested cortical hierarchy of neural states underlies
             event segmentation in the human brain},
  author  = {Geerligs, Linda and Gözükara, Dora and Oetringer, Djamari
             and Campbell, Karen L and van Gerven, Marcel and Güçlü, Umut},
  journal = {eLife},
  volume  = {11},
  year    = {2022}
}
```

And the datasets:

```bibtex
@article{chen2017shared,
  title   = {Shared memories reveal shared structure in neural activity
             across individuals},
  author  = {Chen, Janice and Leong, Yuan Chang and Honey, Christopher J
             and Yong, Chung Han and Norman, Kenneth A and Hasson, Uri},
  journal = {Nature Neuroscience},
  volume  = {20},
  pages   = {115--125},
  year    = {2017}
}

@article{hanke2016studyforrest,
  title   = {A studyforrest extension, simultaneous {fMRI} and eye gaze
             recordings during prolonged natural stimulation},
  author  = {Hanke, Michael and Adelhöfer, Nico and Kottke, Daniel
             and Iacovella, Vittorio and Sengupta, Ayan and Kaule, Falko R
             and Nigbur, Robert and Waite, Alexander Q and Baumgartner, Felix
             and Stadler, Jörg},
  journal = {Scientific Data},
  volume  = {3},
  pages   = {160092},
  year    = {2016}
}
```

---

## License

Code is released under the [MIT License](LICENSE). Pre-computed data files follow the terms of the original dataset licences (see OpenNeuro pages linked above).

---

## Contact

Dora Gözükara — [d.gozukara@donders.ru.nl](mailto:d.gozukara@donders.ru.nl)
Donders Institute for Brain, Cognition and Behaviour, Radboud University, Nijmegen, Netherlands# Global Brain States and Hippocampal Memory

**Memory and Hippocampal Responses to Event Boundaries are Modulated by Global Brain States**
Gözükara, D., Oetringer, D., Ahmad, N., & Geerligs, L. (2026)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drgzkr/global-states-event-memory/blob/main/Global_Brain_States_and_Hippocampal_Memory.ipynb)

---
