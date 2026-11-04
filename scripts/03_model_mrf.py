"""
Practical Assignment 6 - Step 3: fit the Gaussian tissue models, tune and
evaluate the MRF by leave-one-subject-out, then segment the test image.

What the assignment asks for, and how each piece is answered
------------------------------------------------------------
"train Gaussian models for the two classes - healthy and diseased"
    -> MODEL A: one bivariate Gaussian per class over x = (T1, T2).

"create multimodal Gaussians to create the models for the tissues"
    -> MODEL B: the healthy class is a 3-component mixture, one component per
       tissue (CSF / grey matter / white matter), the lesion class stays a
       single Gaussian. Still a two-class decision, but the healthy likelihood
       is no longer a single blob straddling three tissues.

"use the MRI across two modalities and MRFs to detect the lesions"
    -> Potts-prior MRF over the label field, minimised by checkerboard ICM.

Honest protocol. Only three labelled subjects exist, so beta and the decision
offset tau are chosen by LEAVE-ONE-SUBJECT-OUT: for each held-out subject the
models are fitted on the other two only. The reported LOSO numbers therefore
never see the subject they are scored on. The final test segmentation uses all
three training subjects with the LOSO-chosen hyperparameters.

Usage: python -I 03_model_mrf.py <work_dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (GaussianClass, GaussianMixture, icm,      # noqa: E402
                    lesion_wise, seg_metrics)

WORK = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
os.makedirs(OUT, exist_ok=True)

TRAIN = ["mp_aitm5157.2002-09-19", "mp_cyra4454.1999-08-23", "mp_toml5255.1999-08-16"]
TEST = "mp_royr6403.1998-12-07"
ZOOMS = (0.9765625, 0.9765625, 3.0)
N_HEALTHY_COMPONENTS = 3
SEED = 0

BETAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
TAUS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def load(sid, with_lesion=True):
    t1 = np.load(os.path.join(WORK, sid + "_t1n.npy"))
    t2 = np.load(os.path.join(WORK, sid + "_t2n.npy"))
    br = np.load(os.path.join(WORK, sid + "_brain.npy"))
    le = None
    if with_lesion:
        le = np.load(os.path.join(WORK, sid + "_lesion.npy")) & br
    return t1, t2, br, le


def features(t1, t2, mask):
    return np.stack([t1[mask], t2[mask]], axis=1).astype(np.float64)


def fit_models(sids, n_components=N_HEALTHY_COMPONENTS):
    """Fit healthy and lesion models on the union of the given subjects."""
    Xh, Xl = [], []
    for sid in sids:
        t1, t2, br, le = load(sid)
        healthy = br & ~le
        Xh.append(features(t1, t2, healthy))
        Xl.append(features(t1, t2, le))
    Xh = np.concatenate(Xh)
    Xl = np.concatenate(Xl)

    prior_l = len(Xl) / (len(Xh) + len(Xl))
    single = GaussianClass().fit(Xh)                             # MODEL A healthy
    mixture = GaussianMixture(n_components, seed=SEED).fit(Xh)   # MODEL B healthy
    lesion = GaussianClass().fit(Xl)
    return {"healthy_single": single, "healthy_mixture": mixture,
            "lesion": lesion, "prior_lesion": prior_l,
            "n_healthy": len(Xh), "n_lesion": len(Xl)}


def unary_terms(m, t1, t2, brain, healthy_model, tau):
    """Return (2, X, Y, Z) of -log p(x|k) - log pi_k, with tau shifting the
    lesion class. tau > 0 makes lesion cheaper, i.e. raises sensitivity."""
    X = features(t1, t2, brain)
    lp_h = healthy_model.log_pdf(X) + np.log(1.0 - m["prior_lesion"])
    lp_l = m["lesion"].log_pdf(X) + np.log(m["prior_lesion"]) + tau

    u = np.full((2,) + brain.shape, 1e6, dtype=np.float64)
    u[0][brain] = -lp_h
    u[1][brain] = -lp_l
    return u


def segment(m, sid, healthy_key, tau, beta, with_lesion=True):
    t1, t2, br, le = load(sid, with_lesion)
    u = unary_terms(m, t1, t2, br, m[healthy_key], tau)
    if beta == 0.0:
        lab = np.argmin(u, axis=0).astype(np.int8)
        lab[~br] = 0
        info = {"iterations": 0, "converged": True, "changed_per_iter": []}
    else:
        lab, info = icm(u, br, ZOOMS, beta=beta, n_iter=8)
    return (lab == 1), br, le, info


report = {"config": {"train": TRAIN, "test": TEST, "zooms": list(ZOOMS),
                     "healthy_components": N_HEALTHY_COMPONENTS,
                     "betas": BETAS, "taus": TAUS, "seed": SEED}}

# ---------------------------------------------------------------- 1. models on all 3
print("=" * 78)
print("GAUSSIAN TISSUE MODELS  (fitted on all three training subjects)")
full = fit_models(TRAIN)
print("  healthy voxels %d   lesion voxels %d   empirical lesion prior %.5f"
      % (full["n_healthy"], full["n_lesion"], full["prior_lesion"]))
print("")
print("  MODEL A - single bivariate Gaussian per class, features (T1n, T2n)")
ph = full["healthy_single"].params()
pl = full["lesion"].params()
print("    healthy  mean %s  corr %+.3f" % (ph["mean"], ph["correlation"]))
print("             cov  %s" % ph["cov"])
print("    lesion   mean %s  corr %+.3f" % (pl["mean"], pl["correlation"]))
print("             cov  %s" % pl["cov"])
print("")
print("  MODEL B - healthy as a %d-component mixture (one per tissue)"
      % N_HEALTHY_COMPONENTS)
for i, c in enumerate(sorted(full["healthy_mixture"].params(), key=lambda d: d["mean"][1])):
    print("    comp %d  w %.3f  mean (T1 %.3f, T2 %.3f)  corr %+.3f"
          % (i, c["weight"], c["mean"][0], c["mean"][1], c["correlation"]))
print("    EM iterations %d   mean log-likelihood %.4f"
      % (full["healthy_mixture"].n_iter_run_, full["healthy_mixture"].loglik_))

report["models_all_training"] = {
    "n_healthy_voxels": full["n_healthy"], "n_lesion_voxels": full["n_lesion"],
    "prior_lesion": round(full["prior_lesion"], 6),
    "model_A_healthy": ph, "model_A_lesion": pl,
    "model_B_healthy_mixture": full["healthy_mixture"].params(),
    "model_B_em_iterations": full["healthy_mixture"].n_iter_run_,
}

# ---------------------------------------------------------------- 2. LOSO tuning
print("")
print("=" * 78)
print("LEAVE-ONE-SUBJECT-OUT TUNING  (models never see the scored subject)")
loso_models = {}
for held in TRAIN:
    loso_models[held] = fit_models([s for s in TRAIN if s != held])

grid = {}
for healthy_key in ("healthy_single", "healthy_mixture"):
    print("")
    print("  %s" % ("MODEL A (single Gaussian healthy)" if healthy_key == "healthy_single"
                    else "MODEL B (mixture healthy)"))
    best = (-1.0, None)
    for beta in BETAS:
        for tau in TAUS:
            dices = []
            for held in TRAIN:
                pred, br, le, _ = segment(loso_models[held], held, healthy_key, tau, beta)
                dices.append(seg_metrics(pred, le, br)["dice"])
            mean_d = float(np.mean(dices))
            grid[(healthy_key, beta, tau)] = (mean_d, dices)
            if mean_d > best[0]:
                best = (mean_d, (beta, tau))
    print("    mean LOSO Dice, rows = beta, cols = tau")
    print("              " + "  ".join("%5.1f" % t for t in TAUS))
    for beta in BETAS:
        print("    beta %-4.2f " % beta
              + "  ".join("%.3f" % grid[(healthy_key, beta, t)][0] for t in TAUS))
    print("    BEST: beta %.2f  tau %.1f  mean LOSO Dice %.4f"
          % (best[1][0], best[1][1], best[0]))
    report.setdefault("loso_grid", {})[healthy_key] = {
        "best_beta": best[1][0], "best_tau": best[1][1],
        "best_mean_dice": round(best[0], 4),
        "grid": {"beta=%.2f,tau=%.1f" % (b, t): round(grid[(healthy_key, b, t)][0], 4)
                 for b in BETAS for t in TAUS},
    }

# ---------------------------------------------------------------- 3. LOSO detail
print("")
print("=" * 78)
print("LEAVE-ONE-SUBJECT-OUT RESULTS at the chosen settings")
report["loso_detail"] = {}
for healthy_key in ("healthy_single", "healthy_mixture"):
    beta = report["loso_grid"][healthy_key]["best_beta"]
    tau = report["loso_grid"][healthy_key]["best_tau"]
    name = "MODEL A" if healthy_key == "healthy_single" else "MODEL B"
    print("")
    print("  %s   beta=%.2f  tau=%.1f" % (name, beta, tau))
    rows = []
    for held in TRAIN:
        m = loso_models[held]
        p_map, br, le, _ = segment(m, held, healthy_key, tau, 0.0)
        p_mrf, _, _, info = segment(m, held, healthy_key, tau, beta)
        a = seg_metrics(p_map, le, br)
        b = seg_metrics(p_mrf, le, br)
        lw = lesion_wise(p_mrf, le)
        rows.append({"subject": held, "no_mrf": a, "with_mrf": b, "lesion_wise": lw,
                     "icm": info})
        print("    %-26s MAP Dice %.4f sens %.3f ppv %.3f | "
              "MRF Dice %.4f sens %.3f ppv %.3f | lesions %d/%d  FP %d"
              % (held, a["dice"], a["sensitivity"], a["ppv"],
                 b["dice"], b["sensitivity"], b["ppv"],
                 lw["detected"], lw["true_lesions"], lw["false_positive_clusters"]))
    md_a = float(np.mean([r["no_mrf"]["dice"] for r in rows]))
    md_b = float(np.mean([r["with_mrf"]["dice"] for r in rows]))
    print("    mean Dice  without MRF %.4f   with MRF %.4f   (%+.4f)"
          % (md_a, md_b, md_b - md_a))
    report["loso_detail"][healthy_key] = {
        "beta": beta, "tau": tau, "per_subject": rows,
        "mean_dice_no_mrf": round(md_a, 4), "mean_dice_with_mrf": round(md_b, 4),
        "mrf_gain": round(md_b - md_a, 4),
    }

# ---------------------------------------------------------------- 4. test image
best_key = max(("healthy_single", "healthy_mixture"),
               key=lambda k: report["loso_detail"][k]["mean_dice_with_mrf"])
beta = report["loso_detail"][best_key]["beta"]
tau = report["loso_detail"][best_key]["tau"]
print("")
print("=" * 78)
print("TEST SEGMENTATION  -  %s, beta=%.2f, tau=%.1f, models fitted on all 3 subjects"
      % ("MODEL B (mixture)" if best_key == "healthy_mixture" else "MODEL A (single)",
         beta, tau))

pred_test, br_test, _, info = segment(full, TEST, best_key, tau, beta, with_lesion=False)
vox_ml = float(np.prod(ZOOMS)) / 1000.0
np.save(os.path.join(WORK, TEST + "_pred.npy"), pred_test)

lab, n_clusters = ndi.label(pred_test)
sizes = (np.array(ndi.sum(pred_test, lab, range(1, n_clusters + 1)))
         if n_clusters else np.array([]))
big = int((sizes >= 3).sum())
print("  detected lesion voxels : %d   = %.2f ml"
      % (pred_test.sum(), pred_test.sum() * vox_ml))
print("  connected clusters     : %d  (%d with >= 3 voxels)" % (n_clusters, big))
print("  ICM                    : %d iterations, converged=%s, changes %s"
      % (info["iterations"], info["converged"], info["changed_per_iter"]))
print("  brain volume           : %.1f ml   lesion load %.2f%% of brain"
      % (br_test.sum() * vox_ml, 100.0 * pred_test.sum() / br_test.sum()))

report["test"] = {
    "subject": TEST, "model": best_key, "beta": beta, "tau": tau,
    "lesion_voxels": int(pred_test.sum()),
    "lesion_volume_ml": round(float(pred_test.sum()) * vox_ml, 2),
    "clusters": int(n_clusters), "clusters_ge_3_voxels": big,
    "brain_volume_ml": round(float(br_test.sum()) * vox_ml, 1),
    "lesion_load_percent_of_brain": round(100.0 * pred_test.sum() / br_test.sum(), 3),
    "icm": info,
}

# training-subject predictions from the full model, for the figures
for sid in TRAIN:
    p, _, _, _ = segment(full, sid, best_key, tau, beta)
    np.save(os.path.join(WORK, sid + "_pred.npy"), p)

with open(os.path.join(OUT, "03_models.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, default=str)
print("")
print("wrote %s" % os.path.join(OUT, "03_models.json"))
