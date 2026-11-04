"""
Practical Assignment 6 - Step 6: figures for the SELECTED model, and the
deliverable lesion map.

Step 4 drew the same kinds of picture for the fixed three-component model of
Step 3. This script redraws them for the model actually chosen in Step 5 and
adds the one figure Step 4 could not have: the model-selection curve itself.

Produces
  fig1_feature_space.png     the (T1, T2) joint histogram with the fitted
                             Gaussian ellipses - the picture that shows WHY two
                             modalities are needed: the lesion class separates
                             in the joint plane but overlaps in either alone.
  fig2_marginals.png         per-modality histograms, the same point in 1-D.
  fig3_mrf_effect.png        MAP versus MAP+MRF across the LOSO folds.
  fig4_training_overlay.png  prediction vs ground truth, TP/FP/FN coloured.
  fig5_test_result.png       the test subject's detected lesions.
  fig6_model_selection.png   LOSO Dice against K, with and without the
                             white-matter constraint.
  <test>.t2_lesions_predicted.nii   the lesion map as NIfTI.

Usage: python -I 06_figures.py <work_dir> <out_dir>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import Ellipse   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nii_io import save_nii              # noqa: E402

WORK = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

TRAIN = ["mp_aitm5157.2002-09-19", "mp_cyra4454.1999-08-23", "mp_toml5255.1999-08-16"]
TEST = "mp_royr6403.1998-12-07"
ZOOMS = (0.9765625, 0.9765625, 3.0)

# Axis ranges. Intensities are in units of white-matter signal (see
# `standardise` in 02_preprocess.py), so they run past 1.0: white matter sits
# at 1.0 on T1 by construction and CSF reaches about 1.9 on T2.
T1_RANGE = (0.0, 1.5)
T2_RANGE = (0.0, 2.2)
GREY_MAX = 1.6          # display window for the T1/T2 greyscale panels

with open(os.path.join(OUT, "05_model_selection.json"), encoding="utf-8") as fh:
    rep = json.load(fh)
sel = rep["final_loso"]["selected"]


def load(sid, what):
    return np.load(os.path.join(WORK, "%s_%s.npy" % (sid, what)))


def ellipse(ax, mean, cov, n_std, **kw):
    vals, vecs = np.linalg.eigh(np.array(cov))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse(xy=mean, width=w, height=h, angle=ang, fill=False, **kw))


# ------------------------------------------------------------------ gather voxels
Xh, Xl = [], []
for sid in TRAIN:
    t1, t2 = load(sid, "t1n"), load(sid, "t2n")
    br = load(sid, "brain")
    le = load(sid, "lesion") & br
    hh = br & ~le
    Xh.append(np.stack([t1[hh], t2[hh]], 1))
    Xl.append(np.stack([t1[le], t2[le]], 1))
Xh, Xl = np.concatenate(Xh), np.concatenate(Xl)
rng = np.random.default_rng(0)
Hs = Xh[rng.choice(len(Xh), min(200000, len(Xh)), replace=False)]

mix = rep["healthy_mixture_params"]
les = rep["lesion_params"]

# ------------------------------------------------------------------ fig 1
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
for a, title in zip(ax, ["Healthy voxels + fitted %d-component mixture" % sel["K"],
                         "Lesion voxels + fitted Gaussian"]):
    a.set_xlabel("T1 (fraction of white-matter signal)")
    a.set_ylabel("T2 (fraction of white-matter signal)")
    a.set_title(title)
    a.set_xlim(*T1_RANGE)
    a.set_ylim(*T2_RANGE)

ax[0].hist2d(Hs[:, 0], Hs[:, 1], bins=200, range=[T1_RANGE, T2_RANGE],
             cmap="Blues", norm=matplotlib.colors.LogNorm())
for i, c in enumerate(sorted(mix, key=lambda d: d["mean"][1])):
    ellipse(ax[0], c["mean"], c["cov"], 2, edgecolor="crimson", lw=1.8,
            label="healthy mixture components (2$\\sigma$)" if i == 0 else None)
    ax[0].plot(c["mean"][0], c["mean"][1], "x", color="crimson", ms=8, mew=2)
    ax[0].annotate("%.0f%%" % (100 * c["weight"]), c["mean"],
                   textcoords="offset points", xytext=(6, 6),
                   fontsize=8, color="darkred")
ellipse(ax[0], les["mean"], les["cov"], 2, edgecolor="k", lw=2.0, ls="--",
        label="lesion Gaussian (2$\\sigma$), for contrast")
ax[0].legend(loc="upper left", fontsize=8)

ax[1].hist2d(Xl[:, 0], Xl[:, 1], bins=140, range=[T1_RANGE, T2_RANGE],
             cmap="Reds", norm=matplotlib.colors.LogNorm())
ellipse(ax[1], les["mean"], les["cov"], 2, edgecolor="k", lw=2.2,
        label="lesion Gaussian (2$\\sigma$)")
for i, c in enumerate(sorted(mix, key=lambda d: d["mean"][1])):
    ellipse(ax[1], c["mean"], c["cov"], 2, edgecolor="steelblue", lw=1.3, ls="--",
            label="healthy components, for contrast" if i == 0 else None)
ax[1].legend(loc="upper left", fontsize=8)
fig.suptitle("Multimodal (T1,T2) Gaussian tissue models - joint feature space", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_feature_space.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig 2
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for i, name in enumerate(["T1", "T2"]):
    ax[i].hist(Hs[:, i], bins=160, density=True, alpha=.6, color="steelblue",
               label="healthy")
    ax[i].hist(Xl[:, i], bins=160, density=True, alpha=.6, color="crimson",
               label="lesion")
    ax[i].set_title("%s marginal - overlap in one modality alone" % name)
    ax[i].set_xlabel("%s (fraction of white-matter signal)" % name)
    ax[i].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_marginals.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig 3
rows = rep["final_loso"]["per_subject"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
labels = [r["subject"][3:11] for r in rows]
x = np.arange(len(rows))
for i, (key, title) in enumerate([("dice", "Dice"), ("ppv", "Precision (PPV)"),
                                  ("sensitivity", "Sensitivity")]):
    ax[i].bar(x - .18, [r["no_mrf"][key] or 0.0 for r in rows], .36,
              label="MAP only", color="lightsteelblue")
    ax[i].bar(x + .18, [r["with_mrf"][key] or 0.0 for r in rows], .36,
              label="MAP + MRF", color="darkslateblue")
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels, fontsize=8)
    ax[i].set_title(title)
    ax[i].set_ylim(0, 1)
    ax[i].grid(axis="y", alpha=.3)
ax[0].legend(fontsize=9)
fig.suptitle("Effect of the MRF prior (leave-one-subject-out, K=%d, beta=%.2f, tau=%.1f)"
             % (sel["K"], sel["beta"], sel["tau"]), y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_mrf_effect.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig 6
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
for restrict, colour, name in ((False, "steelblue", "no constraint"),
                               (True, "crimson", "white-matter constraint")):
    pts = [s for s in rep["k_sweep"] if s["rim_constraint"] == restrict]
    pts.sort(key=lambda s: s["K"])
    ks = [s["K"] for s in pts]
    ax[0].plot(ks, [s["mean_dice"] for s in pts], "o-", color=colour, label=name)
    ax[1].plot(ks, [s["mean_predicted_volume_ml"] for s in pts], "o-",
               color=colour, label=name)
ax[0].set_xlabel("K, number of healthy Gaussian components")
ax[0].set_ylabel("mean leave-one-subject-out Dice")
ax[0].set_title("Model selection: Dice against K")
ax[0].grid(alpha=.3)
ax[0].legend(fontsize=9)
true_vols = [r["true_volume_ml"] for r in rows]
ax[1].axhspan(min(true_vols), max(true_vols), color="k", alpha=.10,
              label="range of true lesion loads")
ax[1].set_xlabel("K, number of healthy Gaussian components")
ax[1].set_ylabel("mean predicted lesion volume (ml)")
ax[1].set_title("Predicted lesion load against K")
ax[1].grid(alpha=.3)
ax[1].legend(fontsize=9)
fig.suptitle("Choosing the healthy-class model by leave-one-subject-out", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig6_model_selection.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig 4
sid = TRAIN[1]
t2, br = load(sid, "t2n"), load(sid, "brain")
le = load(sid, "lesion") & br
pred = load(sid, "pred")
zs = [18, 24, 30, 36]

fig, ax = plt.subplots(3, len(zs), figsize=(3.2 * len(zs), 9.6))
for c, z in enumerate(zs):
    ax[0, c].imshow(t2[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=GREY_MAX)
    ax[0, c].set_title("T2  z=%d" % z, fontsize=9)
    ax[1, c].imshow(t2[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=GREY_MAX)
    m = np.ma.masked_where(~le[:, :, z].T, le[:, :, z].T)
    ax[1, c].imshow(m, cmap="autumn", alpha=.75, origin="lower")
    ax[1, c].set_title("ground-truth lesions", fontsize=9)
    ax[2, c].imshow(t2[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=GREY_MAX)
    tp = (pred & le)[:, :, z].T
    fp = (pred & ~le)[:, :, z].T
    fn = (~pred & le)[:, :, z].T
    for arr, col in ((tp, (0, 1, 0)), (fp, (1, 0, 0)), (fn, (0, .5, 1))):
        rgba = np.zeros(arr.shape + (4,))
        rgba[..., :3] = col
        rgba[..., 3] = arr * .8
        ax[2, c].imshow(rgba, origin="lower")
    ax[2, c].set_title("green TP / red FP / blue FN", fontsize=9)
for a in ax.ravel():
    a.axis("off")
fig.suptitle("%s  -  Gaussian + MRF segmentation (K=%d, beta=%.2f, tau=%.1f)"
             % (sid, sel["K"], sel["beta"], sel["tau"]), y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_training_overlay.png"), dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ fig 5 - test
t1t, t2t = load(TEST, "t1n"), load(TEST, "t2n")
brt, pt = load(TEST, "brain"), load(TEST, "pred")
zs = [14, 20, 26, 32, 38]
fig, ax = plt.subplots(3, len(zs), figsize=(3.1 * len(zs), 9.4))
for c, z in enumerate(zs):
    ax[0, c].imshow(t1t[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=GREY_MAX)
    ax[0, c].set_title("T1  z=%d" % z, fontsize=9)
    ax[1, c].imshow(t2t[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=GREY_MAX)
    ax[1, c].set_title("T2  z=%d" % z, fontsize=9)
    ax[2, c].imshow(t2t[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=GREY_MAX)
    m = np.ma.masked_where(~pt[:, :, z].T, pt[:, :, z].T)
    ax[2, c].imshow(m, cmap="autumn", alpha=.8, origin="lower")
    ax[2, c].set_title("detected lesions", fontsize=9)
for a in ax.ravel():
    a.axis("off")
fig.suptitle("TEST subject %s  -  detected lesions, %.2f ml (%.2f%% lesion load)"
             % (TEST, rep["test"]["lesion_volume_ml"],
                rep["test"]["lesion_load_percent"]), y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_test_result.png"), dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ NIfTI output
save_nii(os.path.join(OUT, TEST + ".t2_lesions_predicted.nii"),
         pt.astype(np.float32), ZOOMS)

print("figures written to %s" % FIG)
for f in sorted(os.listdir(FIG)):
    print("   %s" % f)
print("lesion map : %s" % os.path.join(OUT, TEST + ".t2_lesions_predicted.nii"))
