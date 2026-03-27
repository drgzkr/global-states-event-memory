"""
run_gsbs_studyforrest.py
------------------------
Fits the Greedy State Boundary Search (GSBS) algorithm to single-subject
whole-brain fMRI data from the StudyForrest dataset (Hanke et al., 2016).

The movie (Forrest Gump) was presented in 8 separate runs. GSBS is fitted
independently on each run and for each subject. Parallelism is over runs
(8 threads by default), with subjects looped inside each worker.

Usage
-----
    python run_gsbs_studyforrest.py

Configure the three path variables at the top of the script before running:
    DATA_DIR   – directory containing the preprocessed per-subject NIfTIs
    OUTPUT_DIR – directory where GSBS .npy objects will be written
    REFERENCE_NIFTI – any one subject/run NIfTI, used to resample the atlas

Requirements
------------
    nilearn, numpy, scipy, tqdm, joblib
    statesegmentation  (pip install git+https://github.com/drgzkr/statesegmentation)
"""

import os
import numpy as np
from scipy.stats import zscore
from nilearn import image, datasets
from statesegmentation import gsbs
from joblib import Parallel, delayed

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = "/path/to/StudyForrest_fMRI_Dataset/preprocessed"
OUTPUT_DIR = "/path/to/StudyForrest_SingleSubGlobalGSBS_Results"
REFERENCE_NIFTI = os.path.join(
    DATA_DIR,
    "waligned_sub-01_ses-movie_task-movie_run-1_space-T1w_desc-unsmDenoised_bold.nii")

# Atlas parameters (must match the notebook)
N_ROIS              = 400
YEO_NETWORKS        = 17
ATLAS_RESOLUTION_MM = 2

# Parallelism: set to number of available CPU cores (≤ n_runs)
N_JOBS = 8

RUNS    = [1, 2, 3, 4, 5, 6, 7, 8]
SUB_LIST = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06",
    "sub-09", "sub-10", "sub-14", "sub-15", "sub-16", "sub-17",
    "sub-18", "sub-19", "sub-20",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Atlas setup ───────────────────────────────────────────────────────────────

print("Fetching Schaefer atlas …")
schaefer     = datasets.fetch_atlas_schaefer_2018(
    n_rois=N_ROIS, yeo_networks=YEO_NETWORKS, resolution_mm=ATLAS_RESOLUTION_MM)
atlas_img    = image.load_img(schaefer["maps"])
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

# ── Per-run GSBS (loops over subjects internally) ─────────────────────────────

def run_gsbs_for_run(run):
    """For a single run, loop over all subjects and fit GSBS."""
    for sub in SUB_LIST:
        nifti_path = os.path.join(
            DATA_DIR,
            f"waligned_{sub}_ses-movie_task-movie_run-{run}"
            f"_space-T1w_desc-unsmDenoised_bold.nii",
        )
        print(f"[{sub} run-{run}] Loading NIfTI …")
        run_data = image.load_img(nifti_path).get_fdata()
        run_data = zscore(run_data, axis=-1)

        print(f"[{sub} run-{run}] Extracting ROI timeseries …")
        roi_ts = extract_roi_timeseries(run_data, atlas_resampled)

        print(f"[{sub} run-{run}] Fitting GSBS ({roi_ts.shape[1]} TRs) …")
        gsbs_obj = gsbs.GSBS(
            x=roi_ts.T,
            kmax=int(roi_ts.shape[1] / 2),
            statewise_detection=True,
        )
        gsbs_obj.fit()

        out_path = os.path.join(
            OUTPUT_DIR,
            f"GSBS_{sub}_run{run}Schaefer_{N_ROIS}_ROIs.npy",
        )
        np.save(out_path, gsbs_obj, allow_pickle=True)
        print(f"[{sub} run-{run}] Saved → {out_path}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running GSBS on {len(SUB_LIST)} StudyForrest subjects × "
          f"{len(RUNS)} runs ({N_JOBS} parallel threads over runs) …")
    Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(run_gsbs_for_run)(run) for run in RUNS
    )
    print("All done.")
