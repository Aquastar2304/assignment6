"""
Practical Assignment 6 - Step 4: figures and the deliverable lesion map.

Produces
  fig1_feature_space.png     the (T1, T2) joint histogram with the fitted
                             Gaussian ellipses - the picture that shows WHY two
                             modalities are needed: the lesion class separates
                             in the joint plane but overlaps in either alone.
  fig2_marginals.png         per-modality histograms, the same point in 1-D.
  fig3_mrf_effect.png        MAP versus MAP+MRF across the LOSO folds.
  fig4_training_overlay.png  prediction vs ground truth, TP/FP/FN coloured.
  fig5_test_result.png       the test subject's detected lesions.
  <test>.t2_lesions_predicted.nii   the lesion map as NIfTI.

Usage: python -I 04_figures.py <work_dir> <out_dir>
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

rep = json.load(open(os.path.join(OUT, "03_models.json"), encoding="utf-8"))


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

mA = rep["models_all_training"]

# ------------------------------------------------------------------ fig 1
fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
for a, title in zip(ax, ["Healthy voxels + fitted models", "Lesion voxels + fitted model"]):
    a.set_xlabel("T1 (normalised)")
    a.set_ylabel("T2 (normalised)")
    a.set_title(title)
    a.set_xlim(-0.05, 1.25)
    a.set_ylim(-0.05, 1.45)

ax[0].hist2d(Hs[:, 0], Hs[:, 1], bins=200, range=[[-0.05, 1.25], [-0.05, 1.45]],
             cmap="Blues", norm=matplotlib.colors.LogNorm())
ellipse(ax[0], mA["model_A_healthy"]["mean"], mA["model_A_healthy"]["cov"], 2,
        edgecolor="k", lw=2.2, ls="--", label="Model A: single Gaussian (2$\\sigma$)")
for i, c in enumerate(sorted(mA["model_B_healthy_mixture"], key=lambda d: d["mean"][1])):
    ellipse(ax[0], c["mean"], c["cov"], 2, edgecolor="crimson", lw=1.8,
            label="Model B: mixture components (2$\\sigma$)" if i == 0 else None)
    ax[0].plot(c["mean"][0], c["mean"][1], "x", color="crimson", ms=8, mew=2)
ax[0].legend(loc="upper left", fontsize=8)

ax[1].hist2d(Xl[:, 0], Xl[:, 1], bins=140, range=[[-0.05, 1.25], [-0.05, 1.45]],
             cmap="Reds", norm=matplotlib.colors.LogNorm())
ellipse(ax[1], mA["model_A_lesion"]["mean"], mA["model_A_lesion"]["cov"], 2,
        edgecolor="k", lw=2.2, label="lesion Gaussian (2$\\sigma$)")
ellipse(ax[1], mA["model_A_healthy"]["mean"], mA["model_A_healthy"]["cov"], 2,
        edgecolor="steelblue", lw=1.8, ls="--", label="healthy (Model A), for contrast")
ax[1].legend(loc="upper left", fontsize=8)
fig.suptitle("Multimodal (T1,T2) Gaussian tissue models - joint feature space", y=0.99)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_feature_space.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig 2
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for i, name in enumerate(["T1", "T2"]):
    ax[i].hist(Hs[:, i], bins=160, density=True, alpha=.6, color="steelblue", label="healthy")
    ax[i].hist(Xl[:, i], bins=160, density=True, alpha=.6, color="crimson", label="lesion")
    ax[i].set_title("%s marginal - overlap in one modality alone" % name)
    ax[i].set_xlabel("%s (normalised)" % name)
    ax[i].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_marginals.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig 4
best_key = rep["test"]["model"]
det = rep["loso_detail"][best_key]
sid = TRAIN[1]
t2, br = load(sid, "t2n"), load(sid, "brain")
le = load(sid, "lesion") & br
pred = load(sid, "pred")
zs = [18, 24, 30, 36]

fig, ax = plt.subplots(3, len(zs), figsize=(3.2 * len(zs), 9.6))
for c, z in enumerate(zs):
    ax[0, c].imshow(t2[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1.2)
    ax[0, c].set_title("T2  z=%d" % z, fontsize=9)
    ax[1, c].imshow(t2[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1.2)
    m = np.ma.masked_where(~le[:, :, z].T, le[:, :, z].T)
    ax[1, c].imshow(m, cmap="autumn", alpha=.75, origin="lower")
    ax[1, c].set_title("ground-truth lesions", fontsize=9)
    ax[2, c].imshow(t2[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1.2)
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
fig.suptitle("%s  -  Gaussian + MRF segmentation (beta=%.2f, tau=%.1f)"
             % (sid, det["beta"], det["tau"]), y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig4_training_overlay.png"), dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ fig 3
rows = det["per_subject"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
labels = [r["subject"][3:11] for r in rows]
x = np.arange(len(rows))
for i, (key, title) in enumerate([("dice", "Dice"), ("ppv", "Precision (PPV)"),
                                  ("sensitivity", "Sensitivity")]):
    ax[i].bar(x - .18, [r["no_mrf"][key] for r in rows], .36,
              label="MAP only", color="lightsteelblue")
    ax[i].bar(x + .18, [r["with_mrf"][key] for r in rows], .36,
              label="MAP + MRF", color="darkslateblue")
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(labels, fontsize=8)
    ax[i].set_title(title)
    ax[i].set_ylim(0, 1)
    ax[i].grid(axis="y", alpha=.3)
ax[0].legend(fontsize=9)
fig.suptitle("Effect of the MRF prior (leave-one-subject-out)", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_mrf_effect.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ fig 5 - test
t1t, t2t = load(TEST, "t1n"), load(TEST, "t2n")
brt, pt = load(TEST, "brain"), load(TEST, "pred")
zs = [14, 20, 26, 32, 38]
fig, ax = plt.subplots(3, len(zs), figsize=(3.1 * len(zs), 9.4))
for c, z in enumerate(zs):
    ax[0, c].imshow(t1t[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1.2)
    ax[0, c].set_title("T1  z=%d" % z, fontsize=9)
    ax[1, c].imshow(t2t[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1.2)
    ax[1, c].set_title("T2  z=%d" % z, fontsize=9)
    ax[2, c].imshow(t2t[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1.2)
    m = np.ma.masked_where(~pt[:, :, z].T, pt[:, :, z].T)
    ax[2, c].imshow(m, cmap="autumn", alpha=.8, origin="lower")
    ax[2, c].set_title("detected lesions", fontsize=9)
for a in ax.ravel():
    a.axis("off")
fig.suptitle("TEST subject %s  -  detected lesions, %.2f ml"
             % (TEST, rep["test"]["lesion_volume_ml"]), y=0.995)
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
