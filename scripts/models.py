"""
Practical Assignment 6 - multimodal Gaussian tissue models and the MRF.

Two things live here:

1. GAUSSIAN TISSUE MODELS over the joint (T1, T2) intensity vector.
   "Multimodal" in this assignment means multi-MODALITY: each voxel carries a
   2-vector x = (T1, T2), and each tissue class is a BIVARIATE Gaussian with a
   full 2x2 covariance. The off-diagonal terms matter - T1 and T2 are
   correlated within a tissue, and a diagonal-covariance model throws away
   exactly the joint structure that separates lesion from normal white matter.

       log p(x | k) = -0.5 [ (x-mu_k)' S_k^-1 (x-mu_k) + log|S_k| + d log 2pi ]

   `GaussianClass`   - one bivariate Gaussian (used for the lesion class).
   `GaussianMixture` - a K-component mixture fitted by EM (used for the healthy
                       class, because healthy brain is not one tissue: CSF, grey
                       matter and white matter occupy three well-separated lobes
                       of the (T1,T2) plane and a single Gaussian straddling all
                       three fits none of them).

2. THE MARKOV RANDOM FIELD that turns independent per-voxel decisions into a
   spatially coherent segmentation.

   Posterior energy to minimise, over the label field L:

       U(L) = sum_i [ -log p(x_i | L_i) - log pi_{L_i} ]        (data term)
            + beta * sum_{i~j} w_ij * 1[L_i != L_j]             (Potts prior)

   The Potts prior says neighbouring voxels tend to share a label, which is
   what suppresses the isolated salt-and-pepper false positives that a pure
   maximum-a-posteriori voxel classifier produces in noisy MRI.

   Neighbourhood: 6-connected, but ANISOTROPICALLY WEIGHTED. The voxels here
   are 0.977 x 0.977 x 3.0 mm, so a through-plane neighbour is three times
   further away than an in-plane one and should not carry equal influence.
   Weights are set proportional to 1/distance.

   Minimisation is by ICM (Iterated Conditional Modes), updated on a
   checkerboard so that no voxel is updated simultaneously with any of its own
   neighbours - a synchronous sweep can oscillate instead of converging.
"""
import numpy as np
from scipy import ndimage as ndi

LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------
#  Gaussian models
# --------------------------------------------------------------------------
class GaussianClass:
    """A single multivariate (here bivariate) Gaussian with full covariance."""

    def __init__(self, reg=1e-6):
        self.reg = reg
        self.mean = None
        self.cov = None
        self._chol = None
        self._logdet = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        d = X.shape[1]
        cov = np.cov(X, rowvar=False)
        self.cov = np.atleast_2d(cov) + self.reg * np.eye(d)
        self._prepare()
        return self

    def set_params(self, mean, cov):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        self._prepare()
        return self

    def _prepare(self):
        self._chol = np.linalg.cholesky(self.cov)
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._chol))))

    def log_pdf(self, X):
        X = np.asarray(X, dtype=np.float64)
        d = X.shape[1]
        diff = X - self.mean
        sol = np.linalg.solve(self._chol, diff.T)       # (d, n)
        maha = np.sum(sol ** 2, axis=0)
        return -0.5 * (maha + self._logdet + d * LOG2PI)

    def params(self):
        return {"mean": [round(float(v), 5) for v in self.mean],
                "cov": [[round(float(v), 6) for v in row] for row in self.cov],
                "correlation": round(float(self.cov[0, 1] /
                                           np.sqrt(self.cov[0, 0] * self.cov[1, 1])), 4)}


class GaussianMixture:
    """K-component full-covariance mixture, fitted by EM with k-means++ init."""

    def __init__(self, n_components=3, reg=1e-6, n_iter=100, tol=1e-6, seed=0):
        self.K = n_components
        self.reg = reg
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.weights = None
        self.comps = None
        self.n_iter_run_ = 0
        self.loglik_ = None

    def _init(self, X):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        idx = [int(rng.integers(n))]
        for _ in range(1, self.K):
            d2 = np.min(((X[:, None, :] - X[idx][None, :, :]) ** 2).sum(-1), axis=1)
            tot = d2.sum()
            probs = d2 / tot if tot > 0 else np.full(n, 1.0 / n)
            idx.append(int(rng.choice(n, p=probs)))
        centers = X[idx].copy()
        lab = np.zeros(n, dtype=int)
        for _ in range(10):
            lab = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
            for k in range(self.K):
                if np.any(lab == k):
                    centers[k] = X[lab == k].mean(axis=0)
        resp = np.zeros((n, self.K))
        resp[np.arange(n), lab] = 1.0
        return resp

    def fit(self, X, max_samples=200000):
        X = np.asarray(X, dtype=np.float64)
        if len(X) > max_samples:                       # EM on a random subsample
            rng = np.random.default_rng(self.seed)
            X = X[rng.choice(len(X), max_samples, replace=False)]
        n, d = X.shape
        resp = self._init(X)
        prev = -np.inf
        for it in range(self.n_iter):
            nk = resp.sum(axis=0) + 1e-12
            self.weights = nk / n
            self.comps = []
            for k in range(self.K):
                mu = (resp[:, k][:, None] * X).sum(axis=0) / nk[k]
                diff = X - mu
                cov = (resp[:, k][:, None] * diff).T @ diff / nk[k]
                cov = np.atleast_2d(cov) + self.reg * np.eye(d)
                self.comps.append(GaussianClass().set_params(mu, cov))
            lp = np.stack([np.log(self.weights[k]) + self.comps[k].log_pdf(X)
                           for k in range(self.K)], axis=1)
            m = lp.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(lp - m).sum(axis=1))
            resp = np.exp(lp - lse[:, None])
            ll = float(lse.mean())
            self.n_iter_run_ = it + 1
            if abs(ll - prev) < self.tol:
                prev = ll
                break
            prev = ll
        self.loglik_ = prev
        return self

    def log_pdf(self, X):
        lp = np.stack([np.log(self.weights[k]) + self.comps[k].log_pdf(X)
                       for k in range(self.K)], axis=1)
        m = lp.max(axis=1, keepdims=True)
        return (m[:, 0] + np.log(np.exp(lp - m).sum(axis=1)))

    def params(self):
        return [{"weight": round(float(self.weights[k]), 5), **self.comps[k].params()}
                for k in range(self.K)]


# --------------------------------------------------------------------------
#  Markov Random Field
# --------------------------------------------------------------------------
def neighbour_kernel(zooms):
    """6-connected kernel with weights proportional to 1/voxel-spacing."""
    kx, ky, kz = (1.0 / float(z) for z in zooms[:3])
    s = max(kx, ky, kz)
    kx, ky, kz = kx / s, ky / s, kz / s
    k = np.zeros((3, 3, 3), dtype=np.float64)
    k[0, 1, 1] = k[2, 1, 1] = kx
    k[1, 0, 1] = k[1, 2, 1] = ky
    k[1, 1, 0] = k[1, 1, 2] = kz
    return k


def icm(unary, mask, zooms, beta=1.0, n_iter=8, init=None):
    """Minimise the Potts posterior energy by checkerboard ICM.

    unary : (K, X, Y, Z) float array of -log p(x_i | k) - log pi_k
    mask  : (X, Y, Z) bool, voxels to label (outside stays 0)
    Returns (labels, info).
    """
    K = unary.shape[0]
    kernel = neighbour_kernel(zooms)
    total_w = float(kernel.sum())

    labels = (np.argmin(unary, axis=0) if init is None else init.copy()).astype(np.int8)
    labels[~mask] = 0

    gx, gy, gz = np.indices(mask.shape)
    parity = ((gx + gy + gz) % 2).astype(bool)

    history = []
    for _ in range(n_iter):
        changed = 0
        for colour in (True, False):
            sel = mask & (parity == colour)
            if not sel.any():
                continue
            agree = np.empty((K,) + mask.shape, dtype=np.float64)
            for k in range(K):
                ind = ((labels == k) & mask).astype(np.float64)
                agree[k] = ndi.convolve(ind, kernel, mode="constant", cval=0.0)
            energy = unary + beta * (total_w - agree)
            new = np.argmin(energy, axis=0).astype(np.int8)
            diff = sel & (new != labels)
            changed += int(diff.sum())
            labels[diff] = new[diff]
        history.append(changed)
        if changed == 0:
            break
    return labels, {"iterations": len(history), "changed_per_iter": history,
                    "converged": bool(history[-1] == 0) if history else True}


# --------------------------------------------------------------------------
#  Evaluation
# --------------------------------------------------------------------------
def dice(pred, truth):
    p, t = pred.astype(bool), truth.astype(bool)
    denom = int(p.sum()) + int(t.sum())
    return float(2.0 * (p & t).sum() / denom) if denom else float("nan")


def seg_metrics(pred, truth, mask):
    p = pred.astype(bool) & mask
    t = truth.astype(bool) & mask
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t & mask).sum())
    return {
        "dice": round(dice(p, t), 4),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "ppv": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "tp": tp, "fp": fp, "fn": fn,
        "pred_voxels": int(p.sum()), "true_voxels": int(t.sum()),
    }


def lesion_wise(pred, truth, min_voxels=3):
    """Per-connected-component detection: a true lesion counts as detected if
    any predicted voxel overlaps it."""
    lab_t, n_t = ndi.label(truth)
    lab_p, n_p = ndi.label(pred)
    detected = 0
    counted = 0
    for i in range(1, n_t + 1):
        comp = lab_t == i
        if comp.sum() < min_voxels:
            continue
        counted += 1
        if (comp & pred).any():
            detected += 1
    false_pos = 0
    for i in range(1, n_p + 1):
        comp = lab_p == i
        if comp.sum() < min_voxels:
            continue
        if not (comp & truth).any():
            false_pos += 1
    return {"true_lesions": counted, "detected": detected,
            "lesion_sensitivity": round(detected / counted, 4) if counted else None,
            "false_positive_clusters": false_pos}


# --------------------------------------------------------------------------
#  Fast two-class MRF
# --------------------------------------------------------------------------
#  `icm` above is the general K-class reference implementation and is what
#  Step 3 used. For the Step-5 model-selection sweep it is the wrong tool:
#  every call allocates three (K, X, Y, Z) float64 volumes plus the three
#  int64 index grids from np.indices - about 180 MB per call, times several
#  hundred calls - and runs a general 27-point convolution when only six of
#  the 27 weights are non-zero.
#
#  The two routines below compute exactly the same thing for K = 2 at a
#  fraction of the cost:
#
#    * the neighbourhood sum becomes six shifted additions instead of a
#      convolution;
#    * only ONE neighbourhood pass per sweep is needed, because with two
#      labels agree_0 = nb(mask) - agree_1;
#    * everything is float32 and the only full-size arrays are du, a and
#      nb_mask (13 MB each at 256x256x50) instead of 52 MB each.
#
#  The decision rule follows from the same Potts energy. With
#  a = nb(labels == 1), m = nb(mask) and du = u_0 - u_1,
#
#      E_0 - E_1 = du + beta * (agree_1 - agree_0) = du + beta * (2a - m)
#
#  so label 1 wins exactly when du + beta * (2a - m) > 0.

def axis_weights(zooms):
    """Per-axis Potts weights, proportional to 1/voxel-spacing, max scaled to 1."""
    kx, ky, kz = (1.0 / float(z) for z in zooms[:3])
    s = max(kx, ky, kz)
    return (kx / s, ky / s, kz / s)


def neighbour_sum(ind, w):
    """Weighted 6-neighbour sum of `ind`, by array shifts.

    Identical to convolving with `neighbour_kernel`, but six shifted adds
    instead of a 27-point convolution.
    """
    wx, wy, wz = w
    out = np.zeros_like(ind)
    out[1:, :, :] += wx * ind[:-1, :, :]
    out[:-1, :, :] += wx * ind[1:, :, :]
    out[:, 1:, :] += wy * ind[:, :-1, :]
    out[:, :-1, :] += wy * ind[:, 1:, :]
    out[:, :, 1:] += wz * ind[:, :, :-1]
    out[:, :, :-1] += wz * ind[:, :, 1:]
    return out


def checkerboard(shape):
    """Parity mask, built by broadcasting three 1-D ranges.

    np.indices on a 256x256x50 volume costs 78 MB of int64; this costs 3 MB.
    """
    i = (np.arange(shape[0]) % 2)[:, None, None]
    j = (np.arange(shape[1]) % 2)[None, :, None]
    k = (np.arange(shape[2]) % 2)[None, None, :]
    return ((i + j + k) % 2).astype(bool)


def icm_binary(du, mask, zooms, beta=1.0, n_iter=12, parity=None, nb_mask=None):
    """Two-class checkerboard ICM.

    du   : float32 (X, Y, Z) of u_0 - u_1; POSITIVE favours the lesion label.
    mask : bool (X, Y, Z), voxels that may be labelled.
    parity, nb_mask : optional precomputed per-subject constants (both are
           independent of beta, tau and the model, so the sweep computes
           them once per subject rather than once per call).
    """
    w = axis_weights(zooms)
    if nb_mask is None:
        nb_mask = neighbour_sum(mask.astype(np.float32), w)
    if parity is None:
        parity = checkerboard(mask.shape)

    lab = (du > 0) & mask
    b = np.float32(beta)
    history = []
    for _ in range(n_iter):
        changed = 0
        for colour in (True, False):
            a = neighbour_sum(lab.astype(np.float32), w)
            new = (du + b * (2.0 * a - nb_mask)) > 0
            sel = mask & (parity == colour)
            diff = sel & (new != lab)
            c = int(diff.sum())
            if c:
                lab[diff] = new[diff]
            changed += c
        history.append(changed)
        if changed == 0:
            break
    lab &= mask
    return lab, {"iterations": len(history), "changed_per_iter": history,
                 "converged": bool(history[-1] == 0) if history else True}
