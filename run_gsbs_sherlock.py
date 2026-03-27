"""
run_gsbs_sherlock.py
--------------------
Fits the Greedy State Boundary Search (GSBS) algorithm to single-subject
whole-brain fMRI data from the Sherlock dataset (Chen et al., 2017).

Each subject's full movie (1,976 TRs) is already concatenated into a single
4-D NIfTI file, split here at TR 946 to match the original two-part
acquisition. GSBS is fitted independently on each half and the resulting
objects are saved to disk.

Subjects are processed in parallel (2 threads by default — raise n_jobs
if your machine has more cores available).

Usage
-----
    python run_gsbs_sherlock.py

Configure the three path variables at the top of the script before running:
    DATA_DIR    – directory containing the preprocessed per-subject NIfTIs
    OUTPUT_DIR  – directory where GSBS .npy objects will be written
    REFERENCE_NIFTI – any one subject NIfTI, used to resample the atlas

Requirements
------------
    nilearn, numpy, scipy, tqdm, joblib
    statesegmentation  (pip install git+https://github.com/drgzkr/statesegmentation)
"""

import os
import numpy as np
from scipy.stats import zscore
from nilearn import image, datasets
from tqdm import tqdm
from statesegmentation import gsbs
from joblib import Parallel, delayed

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR       = "/path/to/Sherlock_fMRI_Dataset/"
OUTPUT_DIR     = "/path/to/Sherlock_SingleSubGlobalGSBS_Results"
REFERENCE_NIFTI = os.path.join(DATA_DIR, "SherlockSub1.nii")

# Atlas parameters (must match the notebook)
N_ROIS             = 400
YEO_NETWORKS       = 17
ATLAS_RESOLUTION_MM = 2

# Sherlock run split: the movie was acquired in two parts
RUN1_END_TR = 946   # TRs 0–945  → run 1
# run 2 is everything from RUN1_END_TR onward

# Parallelism: set to number of available CPU cores (≤ n_subjects)
N_JOBS = 2

SUB_LIST = [str(i) for i in range(1, 18)]   # '1' … '17'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Atlas setup ───────────────────────────────────────────────────────────────

print("Fetching Schaefer atlas …")
schaefer    = datasets.fetch_atlas_schaefer_2018(
    n_rois=N_ROIS, yeo_networks=YEO_NETWORKS, resolution_mm=ATLAS_RESOLUTION_MM)
atlas_img   = image.load_img(schaefer["maps"])
atlas_labels = schaefer["labels"]
if isinstance(atlas_labels[0], bytes):
    atlas_labels = [lbl.decode("utf-8") for lbl in atlas_labels]

print(f"Loading reference NIfTI: {REFERENCE_NIFTI}")
reference_img   = image.load_img(REFERENCE_NIFTI)
atlas_resampled = image.resample_to_img(
    atlas_img, reference_img, interpolation="nearest")
print("Atlas resampled.")

# ── Helper ────────────────────────────────────────────────────────────────────

def extract_roi_timeseries(whole_brain_data, atlas_resampled):
    """Return (n_rois, n_timepoints) array of mean ROI timeseries."""
    atlas_data = atlas_resampled.get_fdata()
    roi_ts = []
    for roi_idx in np.unique(atlas_data):
        roi_ts.append(np.nanmean(whole_brain_data[atlas_data == roi_idx], axis=0))
    roi_ts = np.vstack(roi_ts)
    roi_ts = np.nan_to_num(roi_ts)
    return roi_ts

# ── Per-subject GSBS ──────────────────────────────────────────────────────────

def run_gsbs_subject(sub):
    """Load data, extract ROIs, fit GSBS on both runs, save results."""
    nifti_path = os.path.join(DATA_DIR, f"SherlockSub{sub}.nii")
    print(f"[sub-{sub}] Loading NIfTI …")
    run_data = image.load_img(nifti_path).get_fdata()
    run_data = zscore(run_data, axis=-1)

    print(f"[sub-{sub}] Extracting ROI timeseries …")
    roi_ts = extract_roi_timeseries(run_data, atlas_resampled)

    for run_idx, (start, end, run_label) in enumerate([
        (0,          RUN1_END_TR, "run1"),
        (RUN1_END_TR, None,       "run2"),
    ]):
        segment = roi_ts[:, start:end]
        print(f"[sub-{sub}] Fitting GSBS {run_label} "
              f"({segment.shape[1]} TRs) …")
        gsbs_obj = gsbs.GSBS(
            x=segment.T,
            kmax=int(segment.shape[1] / 2),
            statewise_detection=True,
        )
        gsbs_obj.fit()

        out_path = os.path.join(
            OUTPUT_DIR,
            f"Sherlock_GSBS_sub{sub}_{run_label}"
            f"_Schaefer_{N_ROIS}_ROIs.npy",
        )
        np.save(out_path, gsbs_obj, allow_pickle=True)
        print(f"[sub-{sub}] Saved → {out_path}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running GSBS on {len(SUB_LIST)} Sherlock subjects "
          f"({N_JOBS} parallel threads) …")
    Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(run_gsbs_subject)(sub) for sub in SUB_LIST
    )
    print("All done.")
