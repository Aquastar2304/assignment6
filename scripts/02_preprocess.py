"""
Practical Assignment 6 - Step 2: brain extraction and intensity normalisation.

Why this step is needed before any Gaussian is fitted
-----------------------------------------------------
1. The volumes are WHOLE HEAD, not skull-stripped. Scalp fat is bright on T1 and
   would contaminate a "healthy tissue" Gaussian badly.
2. The background is Rician noise, not zeros, so `image > 0` is not a mask - it
   selects most of the air as well.
3. Subject intensity scales differ (T1 p99 ranges 31.7 to 42.5 across the four
   subjects), so a Gaussian fitted on subjects A+B does not transfer to C
   without normalisation. WHICH normalisation is not a detail - see
   `standardise` below, where the obvious choice turned out to be wrong.

Brain extraction here is deliberately simple and reproducible - no external
tool (BET/FSL) is assumed. It exploits the fact that on T2 the skull is a dark
shell separating the bright scalp from the brain, so a threshold plus a
morphological opening disconnects the scalp ring and the largest remaining
3-D component is the brain.

Usage: python -I 02_preprocess.py <data_dir> <out_dir> <work_dir>
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nii_io import load_nii, save_nii   # noqa: E402

DATA = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
WORK = os.path.abspath(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def otsu(x, nbins=256):
    """Otsu's threshold on a 1-D sample."""
    hist, edges = np.histogram(x, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(np.float64)
    total = w.sum()
    omega = np.cumsum(w) / total
    mu = np.cumsum(w * centers) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(centers[int(np.nanargmax(sigma_b))])


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def fill_holes_3d(mask):
    out = ndi.binary_fill_holes(mask)
    for z in range(mask.shape[2]):          # also per-slice, anisotropic voxels
        out[:, :, z] = ndi.binary_fill_holes(out[:, :, z])
    return out


EROSION = 7          # in-plane structuring element, ~6.8 mm at 0.977 mm/pixel


def brain_mask(t1, t2):
    """Head -> brain. Returns (brain, head, threshold).

    Measured design note. Scalp and brain are a SINGLE connected component at
    every threshold tried (1.0x to 2.0x Otsu): the dark skull does not form a
    continuous 3-D gap, because the slices are 3 mm thick and the shell is
    broken at the orbits and skull base. So component labelling alone cannot
    separate them, and an erode-then-dilate opening simply restores the scalp.

    What does work is to keep the eroded core itself. Measured over the three
    training subjects, an in-plane erosion of 7 yields 1261 / 1431 / 1393 ml -
    physiologically sensible brain volumes, against 2086 ml for the whole head -
    while retaining 100.0 / 99.9 / 100.0 per cent of the labelled lesion voxels.

    Trade-off, stated rather than hidden: the erosion also trims outer cortical
    grey matter, so the "healthy" class is weighted towards white matter and
    deep grey. For periventricular and deep white-matter MS lesions, which is
    what these masks contain, that costs nothing; a cortical-lesion study would
    need proper skull stripping (BET/HD-BET) instead.
    """
    sm = ndi.gaussian_filter(t2, sigma=(1.0, 1.0, 0.3))
    thr = otsu(sm[sm > 0].ravel())

    head = sm > thr
    head = ndi.binary_closing(head, structure=np.ones((3, 3, 1)))
    head = largest_component(head)

    core = ndi.binary_erosion(head, structure=np.ones((EROSION, EROSION, 1)))
    brain = largest_component(core)
    brain = fill_holes_3d(brain)
    brain = largest_component(brain)
    return brain, fill_holes_3d(head), thr


WM_WINDOW = 0.03        # +-3% of the T1 mode counts as white matter


def standardise(t1, t2, mask):
    """Intensity standardisation anchored on WHITE MATTER.

    Returns (t1n, t2n, t1_anchor, t2_anchor, n_wm_voxels).

    The obvious choice, and the one this script used first, is robust
    percentile scaling: map p1 -> 0 and p99 -> 1 inside the brain. MEASURED,
    that fails on the test subject, and it is worth recording why, because the
    failure is silent - the images look fine and only the segmentation is
    absurd.

    p1 and p99 are not anatomy. They are the extremes of whatever happens to
    be in the mask, so they move when the mask does. Across these four
    subjects the T1 anchors came out:

        aitm  p1  7.05  p99 38.11        toml  p1  5.99  p99 45.95
        cyra  p1  7.49  p99 38.05        TEST  p1  2.53  p99 34.42

    The test subject has both the lowest floor and the lowest ceiling, so
    (v - p1) / (p99 - p1) stretches its intensities further than any training
    subject's. Its normalised T1 median landed at 0.652 against 0.520, 0.536
    and 0.564 for the three training subjects - outside the training range
    entirely. Normal white matter was pushed into the band where the lesion
    Gaussian lives, and the segmenter duly returned a single 120 ml "lesion"
    that was in fact most of the centrum semiovale.

    Two changes fix it.

    1. ANCHOR ON A TISSUE, NOT ON A PERCENTILE. White matter is the largest
       tissue class in the brain and the brightest on T1, so the mode of the
       within-brain T1 histogram is a stable landmark. T2 is then anchored on
       the median T2 OF THOSE SAME VOXELS, which ties both modalities to one
       piece of anatomy rather than to two unrelated percentiles.

    2. SCALE ONLY - DO NOT SUBTRACT AN OFFSET. These are magnitude MR images:
       zero signal genuinely means zero. Subtracting a subject-dependent p1
       invents an offset that differs between subjects, which was the larger
       half of the error.

    After the change the test T1 median sits at 0.807 against 0.831, 0.869
    and 0.897 - inside the training spread - and the lesion / healthy contrast
    is consistent across subjects (lesion T2 1.51 to 1.57, healthy 1.19 to
    1.21), which is the property a transferable Gaussian actually needs.

    Note that the resulting units are "fraction of white-matter intensity",
    so values run to about 1.2 on T1 and 1.9 on T2 rather than to 1.0.
    """
    a = t1[mask]
    lo, hi = float(np.percentile(a, 20)), float(np.percentile(a, 99.5))
    hist, edges = np.histogram(a, bins=200, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    t1_anchor = float(centers[int(np.argmax(hist))])

    wm = mask & (np.abs(t1 - t1_anchor) < WM_WINDOW * t1_anchor)
    t2_anchor = float(np.median(t2[wm]))

    if t1_anchor <= 0 or t2_anchor <= 0:        # degenerate, should not happen
        raise ValueError("non-positive intensity anchor")

    return ((t1 / t1_anchor).astype(np.float32),
            (t2 / t2_anchor).astype(np.float32),
            t1_anchor, t2_anchor, int(wm.sum()))


def subject_ids(folder):
    pat = os.path.join(DATA, folder, "*.mri_at2.nii")
    return sorted(os.path.basename(p).replace(".mri_at2.nii", "") for p in glob.glob(pat))


report = []
for folder in ("training", "testing"):
    for sid in subject_ids(folder):
        t1, h1 = load_nii(os.path.join(DATA, folder, sid + ".mri_at1_t2space.nii"))
        t2, h2 = load_nii(os.path.join(DATA, folder, sid + ".mri_at2.nii"))
        zooms = h2["zooms"][:3]
        vox_ml = float(np.prod(zooms)) / 1000.0

        brain, head, thr = brain_mask(t1, t2)
        t1n, t2n, a1, a2, n_wm = standardise(t1, t2, brain)

        lesion = None
        lpath = os.path.join(DATA, folder, sid + ".t2_lesions.nii")
        if os.path.exists(lpath):
            raw, _ = load_nii(lpath)
            lesion = raw > 0.5          # binary; the one stray 2.0 voxel is included
            np.save(os.path.join(WORK, sid + "_lesion.npy"), lesion)

        np.save(os.path.join(WORK, sid + "_t1n.npy"), t1n)
        np.save(os.path.join(WORK, sid + "_t2n.npy"), t2n)
        np.save(os.path.join(WORK, sid + "_brain.npy"), brain)
        save_nii(os.path.join(WORK, sid + "_brainmask.nii"),
                 brain.astype(np.float32), zooms)

        e = {
            "id": sid, "folder": folder,
            "zooms": [round(float(z), 4) for z in zooms],
            "voxel_volume_ml": vox_ml,
            "otsu_threshold_T2": round(thr, 4),
            "head_voxels": int(head.sum()),
            "brain_voxels": int(brain.sum()),
            "brain_volume_ml": round(float(brain.sum()) * vox_ml, 1),
            "brain_fraction_of_head": round(float(brain.sum()) / max(int(head.sum()), 1), 4),
            "T1_wm_anchor": round(a1, 3), "T2_wm_anchor": round(a2, 3),
            "wm_anchor_voxels": n_wm,
            "T1_median_normalised": round(float(np.median(t1n[brain])), 4),
            "T2_median_normalised": round(float(np.median(t2n[brain])), 4),
        }
        if lesion is not None:
            e["lesion_voxels_total"] = int(lesion.sum())
            e["lesion_voxels_in_brain"] = int((lesion & brain).sum())
            e["lesion_retained_by_mask"] = round(
                float((lesion & brain).sum()) / max(int(lesion.sum()), 1), 4)
            e["lesion_volume_ml"] = round(float((lesion & brain).sum()) * vox_ml, 2)
        report.append(e)

        line = ("%-26s brain %7d vox = %6.1f ml  (%.1f%% of head)  otsu %.2f"
                "  WM anchor T1 %.2f T2 %.2f -> median %.3f/%.3f"
                % (sid, e["brain_voxels"], e["brain_volume_ml"],
                   100 * e["brain_fraction_of_head"], thr, a1, a2,
                   e["T1_median_normalised"], e["T2_median_normalised"]))
        if lesion is not None:
            line += ("   lesion %5d in brain / %5d total  (%.1f%% retained)"
                     % (e["lesion_voxels_in_brain"], e["lesion_voxels_total"],
                        100 * e["lesion_retained_by_mask"]))
        print(line)

with open(os.path.join(OUT, "02_preprocess.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print("")
print("wrote %s and arrays to %s" % (os.path.join(OUT, "02_preprocess.json"), WORK))
