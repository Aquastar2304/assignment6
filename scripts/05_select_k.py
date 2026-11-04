"""
Practical Assignment 6 - Step 5: choose the number of healthy tissue
components, then produce the final segmentation of the test image.

Why this step exists
--------------------
Step 3 fixed the healthy class at three Gaussians (CSF / grey / white matter)
and the test segmentation came out at 165 ml, roughly four times the largest
training lesion load, with false positives smeared along the cortical ribbon.

This script sweeps two things by leave-one-subject-out, the same protocol
already used for beta and tau:

  K        the number of components in the healthy mixture;
  RIM_MM   a white-matter spatial constraint - voxels within RIM_MM of the
           brain surface cannot be lesion. MS lesions are white-matter
           lesions, which is a genuine domain fact rather than a fit to these
           three subjects, and it is checkable: 96.2 / 98.6 / 99.8 per cent of
           labelled lesion voxels lie inside that region.

Both variants are reported, so the cost of the constraint is visible rather
than assumed.

WHAT THE SWEEP ACTUALLY SHOWED - the original hypothesis was wrong
------------------------------------------------------------------
The hypothesis behind varying K was partial volume: a voxel on a grey-matter /
CSF boundary is a mixture of the two, so its (T1, T2) lands between the tissue
clusters, near where the lesion cluster sits; with only three healthy
components nothing owns those boundary voxels, so the lesion Gaussian claims
them. Giving the mixture more components should therefore fix the
over-detection.

MEASURED, it does not. With the constraint on, K = 3 scores 0.6262 and K = 6
scores 0.6597 - about +0.03 - while K = 6 and K = 8 differ by 0.0003, which is
noise over three folds. The constraint is worth roughly +0.15 Dice at EVERY K.

The real cause of the 165 ml result was not the mixture at all: it was the
intensity standardisation. Percentile (p1/p99) scaling pushed the TEST
subject's normal white matter into the lesion band - see `standardise` in
02_preprocess.py, which documents the measurement. Once that is anchored on
white matter instead, the test segmentation drops to 38.5 ml, inside the
training range, and the residual cortical false positives are removed by the
spatial constraint rather than by extra Gaussians.

K = 6 is selected here because it scores highest, but K = 3 with the
constraint would be a defensible and simpler choice.

Note on cost
------------
The first version of this script re-fitted and re-segmented inside the whole
beta/tau grid and was killed for memory. Two facts make that unnecessary:

  * tau is an additive constant on the log-posterior difference, so the
    expensive per-voxel likelihood map is computed ONCE per (K, subject) and
    reused across the entire grid;
  * with two labels the MRF needs one neighbourhood pass per sweep, not K
    (see `icm_binary` in models.py).

Usage: python -I 05_select_k.py <work_dir> <out_dir>
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (GaussianClass, GaussianMixture, axis_weights,    # noqa: E402
                    checkerboard, icm, icm_binary, lesion_wise,
                    neighbour_sum, seg_metrics)

WORK = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
os.makedirs(OUT, exist_ok=True)

TRAIN = ["mp_aitm5157.2002-09-19", "mp_cyra4454.1999-08-23", "mp_toml5255.1999-08-16"]
TEST = "mp_royr6403.1998-12-07"
ALL = TRAIN + [TEST]
ZOOMS = (0.9765625, 0.9765625, 3.0)
VOX_ML = float(np.prod(ZOOMS)) / 1000.0
SEED = 0

# Grid. K and the white-matter constraint are the two axes under test, so
# they keep their full range. beta and tau are pinned to a narrow band around
# the values an earlier full sweep (5 x 2 x 4 x 4 = 480 cells) selected: that
# sweep chose beta = 1.0, tau = 2.0 in 8 of its 10 rows and never chose
# tau = 0, so re-searching them costs ten times the runtime to re-confirm a
# result already in hand. 3 x 2 x 1 x 3 = 18 cells, 54 ICM runs.
KS = [3, 6, 8]
BETAS = [1.0]
TAUS = [1.0, 2.0, 3.0]
RIM_MM = 6.0                 # exclude voxels this close to the brain surface
N_ICM = 12
FORBID = np.float32(-1e9)    # du value that makes the lesion label impossible


# ---------------------------------------------------------------- subject data
class Subject(object):
    """Everything about one subject that does not depend on the model."""

    def __init__(self, sid):
        self.sid = sid
        self.t1 = np.load(os.path.join(WORK, sid + "_t1n.npy"))
        self.t2 = np.load(os.path.join(WORK, sid + "_t2n.npy"))
        self.brain = np.load(os.path.join(WORK, sid + "_brain.npy"))
        lp = os.path.join(WORK, sid + "_lesion.npy")
        self.lesion = (np.load(lp) & self.brain) if os.path.exists(lp) else None
        # MRF constants: parity and the neighbour sum of the mask itself
        self.parity = checkerboard(self.brain.shape)
        self.nb_mask = neighbour_sum(self.brain.astype(np.float32),
                                     axis_weights(ZOOMS))
        # white-matter candidate region
        dist = ndi.distance_transform_edt(self.brain, sampling=ZOOMS)
        self.interior = self.brain & (dist >= RIM_MM)

    def feats(self, mask):
        return np.stack([self.t1[mask], self.t2[mask]], axis=1).astype(np.float64)


print("loading subjects ...")
S = {sid: Subject(sid) for sid in ALL}
for sid in ALL:
    s = S[sid]
    extra = ""
    if s.lesion is not None:
        extra = ("   lesion %5d vox, %.1f%% inside the interior region"
                 % (s.lesion.sum(),
                    100.0 * (s.lesion & s.interior).sum() / s.lesion.sum()))
    print("  %-26s brain %7d vox   interior %7d (%.1f%%)%s"
          % (sid, s.brain.sum(), s.interior.sum(),
             100.0 * s.interior.sum() / s.brain.sum(), extra))


# ---------------------------------------------------------------- models
def fit(sids, K):
    """Fit the healthy K-mixture and the single lesion Gaussian on `sids`."""
    Xh = np.concatenate([S[s].feats(S[s].brain & ~S[s].lesion) for s in sids])
    Xl = np.concatenate([S[s].feats(S[s].lesion) for s in sids])
    return {"healthy": GaussianMixture(K, seed=SEED).fit(Xh),
            "lesion": GaussianClass().fit(Xl),
            "prior_lesion": float(len(Xl)) / (len(Xh) + len(Xl)),
            "n_healthy": int(len(Xh)), "n_lesion": int(len(Xl))}


def du_base(m, sub):
    """u_0 - u_1 at tau = 0: the log-posterior preference for 'lesion'.

    Computed once per (model, subject); tau only shifts it by a constant and
    the rim constraint only forces some entries to -inf, so the whole beta /
    tau / constraint grid reuses this one array.
    """
    X = sub.feats(sub.brain)
    d = ((m["lesion"].log_pdf(X) + np.log(m["prior_lesion"]))
         - (m["healthy"].log_pdf(X) + np.log(1.0 - m["prior_lesion"])))
    out = np.full(sub.brain.shape, FORBID, dtype=np.float32)
    out[sub.brain] = d.astype(np.float32)
    return out


def segment(du0, sub, tau, beta, restrict, n_iter=N_ICM):
    du = du0 + np.float32(tau)
    if restrict:
        du = np.where(sub.interior, du, FORBID).astype(np.float32)
    if beta == 0.0:
        lab = (du > 0) & sub.brain
        return lab, {"iterations": 0, "changed_per_iter": [], "converged": True}
    return icm_binary(du, sub.brain, ZOOMS, beta=beta, n_iter=n_iter,
                      parity=sub.parity, nb_mask=sub.nb_mask)


report = {"config": {"ks": KS, "betas": BETAS, "taus": TAUS, "rim_mm": RIM_MM,
                     "icm_iters": N_ICM, "seed": SEED}}

# ---------------------------------------------------------------- solver check
# The sweep uses the specialised two-class solver. Before trusting several
# hundred of its results, check it against the general K-class reference
# implementation from Step 3 on a real case.
print("")
print("solver equivalence check (icm_binary vs the general K-class icm)")
_sub = S[TRAIN[0]]
_m = fit([s for s in TRAIN if s != _sub.sid], 3)
_d0 = du_base(_m, _sub)
_chk = []
for _beta in (0.5, 1.0, 2.0):
    _d = _d0 + np.float32(1.0)
    _fast, _ = icm_binary(_d, _sub.brain, ZOOMS, beta=_beta, n_iter=6,
                          parity=_sub.parity, nb_mask=_sub.nb_mask)
    _u = np.full((2,) + _sub.brain.shape, 1e6, dtype=np.float64)
    _u[0][_sub.brain] = 0.0
    _u[1][_sub.brain] = -_d[_sub.brain]
    _slow, _ = icm(_u, _sub.brain, ZOOMS, beta=_beta, n_iter=6)
    _agree = int((_fast == (_slow == 1)).sum())
    _tot = int(_fast.size)
    _chk.append({"beta": _beta, "identical_voxels": _agree, "total": _tot,
                 "match": _agree == _tot})
    print("  beta %.1f : %d / %d voxels identical  -> %s"
          % (_beta, _agree, _tot, "MATCH" if _agree == _tot else "DIFFER"))
report["solver_equivalence_check"] = _chk
del _d0, _d, _u, _fast, _slow, _m

# ---------------------------------------------------------------- the sweep
print("")
print("=" * 86)
print("MODEL SELECTION BY LEAVE-ONE-SUBJECT-OUT")
print("  K   = number of Gaussian components for the HEALTHY class")
print("  rim = white-matter constraint: no lesion within %.0f mm of the brain surface"
      % RIM_MM)
print("")
print("  %-4s %-5s %-6s %-5s %-9s %-9s %-9s %-10s"
      % ("K", "rim", "beta", "tau", "meanDice", "meanPPV", "meanSens", "meanPredML"))

best = {"dice": -1.0}
sweep = []
for K in KS:
    loso_du = {}
    for held in TRAIN:
        m = fit([s for s in TRAIN if s != held], K)
        loso_du[held] = du_base(m, S[held])
    for restrict in (False, True):
        bestK = {"dice": -1.0}
        for beta in BETAS:
            for tau in TAUS:
                dice, ppv, sens, vol = [], [], [], []
                for held in TRAIN:
                    pred, _ = segment(loso_du[held], S[held], tau, beta, restrict)
                    mt = seg_metrics(pred, S[held].lesion, S[held].brain)
                    dice.append(mt["dice"])
                    ppv.append(mt["ppv"] or 0.0)
                    sens.append(mt["sensitivity"] or 0.0)
                    vol.append(float(pred.sum()) * VOX_ML)
                md = float(np.mean(dice))
                if md > bestK["dice"]:
                    bestK = {"dice": md, "beta": beta, "tau": tau,
                             "dices": dice, "ppv": float(np.mean(ppv)),
                             "sens": float(np.mean(sens)), "vol": float(np.mean(vol))}
        print("  %-4d %-5s %-6.2f %-5.1f %-9.4f %-9.4f %-9.4f %-10.1f"
              % (K, "yes" if restrict else "no", bestK["beta"], bestK["tau"],
                 bestK["dice"], bestK["ppv"], bestK["sens"], bestK["vol"]))
        sweep.append({"K": K, "rim_constraint": restrict, "beta": bestK["beta"],
                      "tau": bestK["tau"], "mean_dice": round(bestK["dice"], 4),
                      "per_subject_dice": [round(d, 4) for d in bestK["dices"]],
                      "mean_ppv": round(bestK["ppv"], 4),
                      "mean_sensitivity": round(bestK["sens"], 4),
                      "mean_predicted_volume_ml": round(bestK["vol"], 2)})
        if bestK["dice"] > best["dice"]:
            best = {"dice": bestK["dice"], "K": K, "restrict": restrict,
                    "beta": bestK["beta"], "tau": bestK["tau"]}
    del loso_du

report["k_sweep"] = sweep
print("")
print("  SELECTED: K=%d  rim=%s  beta=%.2f  tau=%.1f  mean LOSO Dice %.4f"
      % (best["K"], best["restrict"], best["beta"], best["tau"], best["dice"]))

# ---------------------------------------------------------------- LOSO detail
print("")
print("=" * 86)
print("LEAVE-ONE-SUBJECT-OUT DETAIL at the selected configuration")
rows = []
for held in TRAIN:
    m = fit([s for s in TRAIN if s != held], best["K"])
    d0 = du_base(m, S[held])
    sub = S[held]
    p_map, _ = segment(d0, sub, best["tau"], 0.0, best["restrict"])
    p_mrf, info = segment(d0, sub, best["tau"], best["beta"], best["restrict"])
    a = seg_metrics(p_map, sub.lesion, sub.brain)
    b = seg_metrics(p_mrf, sub.lesion, sub.brain)
    lw = lesion_wise(p_mrf, sub.lesion)
    rows.append({"subject": held, "no_mrf": a, "with_mrf": b, "lesion_wise": lw,
                 "icm": info,
                 "true_volume_ml": round(float(sub.lesion.sum()) * VOX_ML, 2),
                 "pred_volume_ml": round(float(p_mrf.sum()) * VOX_ML, 2)})
    print("  %-26s MAP Dice %.4f | MRF Dice %.4f sens %.3f ppv %.3f | "
          "vol true %5.1f pred %5.1f ml | lesions %d/%d FPclust %d"
          % (held, a["dice"], b["dice"], b["sensitivity"], b["ppv"],
             rows[-1]["true_volume_ml"], rows[-1]["pred_volume_ml"],
             lw["detected"], lw["true_lesions"], lw["false_positive_clusters"]))
    del d0
md_a = float(np.mean([r["no_mrf"]["dice"] for r in rows]))
md_b = float(np.mean([r["with_mrf"]["dice"] for r in rows]))
print("  mean Dice  without MRF %.4f   with MRF %.4f   (MRF gain %+.4f)"
      % (md_a, md_b, md_b - md_a))
report["final_loso"] = {"selected": best, "per_subject": rows,
                        "mean_dice_no_mrf": round(md_a, 4),
                        "mean_dice_with_mrf": round(md_b, 4),
                        "mrf_gain": round(md_b - md_a, 4)}

# ---------------------------------------------------------------- final test
print("")
print("=" * 86)
print("FINAL TEST SEGMENTATION  (models fitted on all three training subjects)")
full = fit(TRAIN, best["K"])
dt = du_base(full, S[TEST])
pred, info = segment(dt, S[TEST], best["tau"], best["beta"], best["restrict"],
                     n_iter=25)
np.save(os.path.join(WORK, TEST + "_pred.npy"), pred)
brt = S[TEST].brain
lab, n_cl = ndi.label(pred)
sizes = np.array(ndi.sum(pred, lab, range(1, n_cl + 1))) if n_cl else np.array([])
print("  lesion volume : %.2f ml  (%d voxels, %.2f%% of a %.0f ml brain)"
      % (pred.sum() * VOX_ML, pred.sum(), 100.0 * pred.sum() / brt.sum(),
         brt.sum() * VOX_ML))
print("  clusters      : %d total, %d with >= 3 voxels, largest %.2f ml"
      % (n_cl, int((sizes >= 3).sum()), (sizes.max() * VOX_ML) if n_cl else 0.0))
print("  ICM           : %d iters, converged=%s" % (info["iterations"], info["converged"]))
print("  training lesion volumes for comparison: %s ml"
      % [r["true_volume_ml"] for r in rows])

report["test"] = {
    "subject": TEST, "K": best["K"], "rim_constraint": best["restrict"],
    "beta": best["beta"], "tau": best["tau"],
    "lesion_voxels": int(pred.sum()),
    "lesion_volume_ml": round(float(pred.sum()) * VOX_ML, 2),
    "brain_volume_ml": round(float(brt.sum()) * VOX_ML, 1),
    "lesion_load_percent": round(100.0 * pred.sum() / brt.sum(), 3),
    "clusters": int(n_cl), "clusters_ge_3": int((sizes >= 3).sum()),
    "largest_cluster_ml": round(float(sizes.max()) * VOX_ML, 2) if n_cl else 0.0,
    "icm": info,
}
report["healthy_mixture_params"] = full["healthy"].params()
report["lesion_params"] = full["lesion"].params()
report["training_counts"] = {"n_healthy_voxels": full["n_healthy"],
                             "n_lesion_voxels": full["n_lesion"],
                             "prior_lesion": round(full["prior_lesion"], 6)}

# predictions on the training subjects, from the all-training model, for figures
for sid in TRAIN:
    d = du_base(full, S[sid])
    p, _ = segment(d, S[sid], best["tau"], best["beta"], best["restrict"])
    np.save(os.path.join(WORK, sid + "_pred.npy"), p)
    del d

with open(os.path.join(OUT, "05_model_selection.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, default=str)
print("")
print("wrote %s" % os.path.join(OUT, "05_model_selection.json"))
