"""
Practical Assignment 6 - Step 1: inspect the data before modelling anything.

Reports geometry, intensity ranges, lesion burden and inter-subject intensity
scale differences, which decide the preprocessing in step 2.

Usage: python -I 01_explore.py <data_dir> <out_dir>
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nii_io import load_nii   # noqa: E402  (our own reader; see nii_io docstring)

DATA = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
os.makedirs(OUT, exist_ok=True)


def subject_ids(folder):
    pat = os.path.join(DATA, folder, "*.mri_at2.nii")
    return sorted(os.path.basename(p).replace(".mri_at2.nii", "") for p in glob.glob(pat))


def load(folder, sid, suffix):
    path = os.path.join(DATA, folder, sid + suffix)
    if not os.path.exists(path):
        return None, None
    arr, hdr = load_nii(path)
    return arr, hdr


report = {"training": [], "testing": []}

for folder in ("training", "testing"):
    print("=" * 78)
    print(folder.upper())
    for sid in subject_ids(folder):
        t1, img1 = load(folder, sid, ".mri_at1_t2space.nii")
        t2, img2 = load(folder, sid, ".mri_at2.nii")
        les, _ = load(folder, sid, ".t2_lesions.nii")

        zooms = img2["zooms"][:3]
        vox_ml = float(np.prod(zooms)) / 1000.0

        entry = {
            "id": sid,
            "shape": list(map(int, t2.shape)),
            "dtype_t1": str(t1.dtype), "dtype_t2": str(t2.dtype),
            "voxel_size_mm": [round(float(z), 4) for z in zooms],
            "voxel_volume_ml": vox_ml,
            "t1_min": float(t1.min()), "t1_max": float(t1.max()),
            "t2_min": float(t2.min()), "t2_max": float(t2.max()),
            "t1_mean_nonzero": round(float(t1[t1 > 0].mean()), 2),
            "t2_mean_nonzero": round(float(t2[t2 > 0].mean()), 2),
            "t1_p99": round(float(np.percentile(t1[t1 > 0], 99)), 2),
            "t2_p99": round(float(np.percentile(t2[t2 > 0], 99)), 2),
            "nonzero_voxels": int((t2 > 0).sum()),
            "has_lesion_mask": les is not None,
        }

        if les is not None:
            vals = np.unique(les)
            entry["lesion_mask_values"] = [float(v) for v in vals[:10]]
            entry["lesion_voxels"] = int((les > 0).sum())
            entry["lesion_volume_ml"] = round(float((les > 0).sum()) * vox_ml, 2)
            entry["lesion_fraction_of_nonzero"] = round(
                float((les > 0).sum()) / max(int((t2 > 0).sum()), 1), 6)
            inside = les > 0
            outside = (t2 > 0) & ~inside
            entry["lesion_T1_mean"] = round(float(t1[inside].mean()), 2)
            entry["lesion_T2_mean"] = round(float(t2[inside].mean()), 2)
            entry["nonlesion_T1_mean"] = round(float(t1[outside].mean()), 2)
            entry["nonlesion_T2_mean"] = round(float(t2[outside].mean()), 2)

        report[folder].append(entry)

        print("")
        print(sid)
        print("  shape %s  voxel %s mm  (%s / %s)"
              % (entry["shape"], entry["voxel_size_mm"], entry["dtype_t1"], entry["dtype_t2"]))
        print("  T1 range [%.0f, %.0f]  mean(nz) %.1f  p99 %.1f"
              % (entry["t1_min"], entry["t1_max"], entry["t1_mean_nonzero"], entry["t1_p99"]))
        print("  T2 range [%.0f, %.0f]  mean(nz) %.1f  p99 %.1f"
              % (entry["t2_min"], entry["t2_max"], entry["t2_mean_nonzero"], entry["t2_p99"]))
        if les is not None:
            print("  lesion mask values %s" % entry["lesion_mask_values"])
            print("  lesion %d voxels = %.2f ml  (%.4f%% of head voxels)"
                  % (entry["lesion_voxels"], entry["lesion_volume_ml"],
                     100 * entry["lesion_fraction_of_nonzero"]))
            print("  lesion  T1 %.1f  T2 %.1f   |  non-lesion  T1 %.1f  T2 %.1f"
                  % (entry["lesion_T1_mean"], entry["lesion_T2_mean"],
                     entry["nonlesion_T1_mean"], entry["nonlesion_T2_mean"]))

print("")
print("=" * 78)
print("INTER-SUBJECT INTENSITY SCALE (decides whether normalisation is needed)")
allsubs = report["training"] + report["testing"]
for key in ("t1_p99", "t2_p99", "t1_mean_nonzero", "t2_mean_nonzero"):
    vals = [s[key] for s in allsubs]
    print("  %-18s min %8.1f  max %8.1f  ratio %.2fx"
          % (key, min(vals), max(vals), max(vals) / max(min(vals), 1e-9)))

with open(os.path.join(OUT, "01_explore.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print("")
print("wrote %s" % os.path.join(OUT, "01_explore.json"))
