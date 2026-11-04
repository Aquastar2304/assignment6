# Practical Assignment 6 — MS lesion detection with multimodal Gaussians and MRFs

**Task.** Train Gaussian models for two classes — healthy tissue and lesion — from the
three training MRIs, using **both** modalities (T1 and T2) jointly, then use those
models together with a Markov Random Field to detect the lesions in the test image.

**Data.** Four subjects, each with a T1 volume (resampled into T2 space), a T2 volume,
and — for the three training subjects — a binary lesion mask.

| subject | set | volume | voxel size | lesion load |
|---|---|---|---|---|
| `mp_aitm5157.2002-09-19` | training | 256×256×50 | 0.977 × 0.977 × 3.0 mm | 13.2 ml |
| `mp_cyra4454.1999-08-23` | training | 256×256×50 | 0.977 × 0.977 × 3.0 mm | 43.1 ml |
| `mp_toml5255.1999-08-16` | training | 256×256×50 | 0.977 × 0.977 × 3.0 mm | 49.2 ml |
| `mp_royr6403.1998-12-07` | **test** | 256×256×50 | 0.977 × 0.977 × 3.0 mm | *to be predicted* |

**Headline result.** Leave-one-subject-out Dice **0.660**, of which **+0.106 comes from the
MRF** over the per-voxel classifier. On the test image the method finds **38.53 ml** of
lesion (2.11 % lesion load) in 52 clusters — squarely inside the 13–49 ml range seen in
training.

---

## 1. Reading the data — two header traps

Both of these corrupt everything downstream and neither announces itself, so they are
worth recording.

**Trap 1: `vox_offset = 348`.** The NIfTI-1 standard requires 352 for a single-file
`.nii`, and `nibabel` refuses to load the file, then "fixes" the value to 352 if forced.
That fix would be **wrong here**. The arithmetic settles it:

```
file size            13,107,548 bytes
minus header offset       −  348
                     = 13,107,200 = 256 × 256 × 50 × 4 bytes   (exactly)
```

The data really does start at byte 348. Accepting nibabel's correction shifts every
voxel by 4 bytes. `scripts/nii_io.py` is a small reader that honours the declared offset.

**Trap 2: stale `scl_slope` / `scl_inter`.** The standard says to apply
`value × slope + inter`. Doing so collapses the entire intensity range: the images are
already stored in [0, 100], and after scaling every voxel lands in [50.00, 50.15] — all
tissue contrast destroyed. The scaling fields are leftovers from an earlier integer
representation of these files. The loader therefore defaults to `apply_scaling=False`,
which is a deliberate, documented deviation from the spec rather than an oversight.

## 2. Brain extraction and intensity standardisation

### Brain extraction

The volumes are **whole head**, not skull-stripped, and the background is Rician noise
rather than zeros — so neither `image > 0` nor a simple threshold gives a brain mask.

I measured that scalp and brain form a **single connected component at every threshold
from 1.0× to 2.0× Otsu**: with 3 mm slices the dark skull is not a continuous 3-D
barrier, as it is broken at the orbits and skull base. Connected-component labelling
alone therefore cannot separate them, and an erode-then-dilate opening simply restores
the scalp. What works is to keep the **eroded core** (7-pixel in-plane structuring
element, ≈ 6.8 mm). Result: 1282 / 1438 / 1469 ml for the training subjects against
2086 ml for the whole head, while retaining **100.0 / 100.0 / 100.0 %** of labelled
lesion voxels.

*Stated trade-off:* the erosion also trims outer cortical grey matter, so the healthy
class is weighted towards white matter and deep grey. For the periventricular and deep
white-matter lesions these masks contain, that costs nothing. A cortical-lesion study
would need proper skull stripping (BET / HD-BET) instead.

### Intensity standardisation — where the first attempt failed

Subject intensity scales differ, so a Gaussian fitted on subjects A+B does not transfer
to C without normalisation. My first choice was the obvious one — robust percentile
scaling, p1 → 0 and p99 → 1 inside the brain. **It silently broke the test subject.**

p1 and p99 are not anatomy; they are the extremes of whatever is in the mask, so they
move when the mask does:

| T1 anchors | aitm | cyra | toml | **test** |
|---|---|---|---|---|
| p1 | 7.05 | 7.49 | 5.99 | **2.53** |
| p99 | 38.11 | 38.05 | 45.95 | **34.42** |

The test subject has both the lowest floor and the lowest ceiling, so
`(v − p1)/(p99 − p1)` stretches its intensities further than any training subject's. Its
normalised T1 median landed at **0.652** against 0.520 / 0.536 / 0.564 — outside the
training range entirely. Normal white matter was pushed into the band where the lesion
Gaussian lives, and the segmenter returned a single **120 ml "lesion"** that was in fact
most of the centrum semiovale.

Two changes fix it:

1. **Anchor on a tissue, not on a percentile.** White matter is the largest tissue class
   and the brightest on T1, so the mode of the within-brain T1 histogram is a stable
   landmark. T2 is then anchored on the median T2 **of those same voxels**, tying both
   modalities to one piece of anatomy rather than two unrelated percentiles.
2. **Scale only — do not subtract an offset.** These are magnitude MR images: zero
   signal genuinely means zero. Subtracting a subject-dependent p1 invents an offset
   that differs between subjects, and this was the larger half of the error.

| T1 median in brain | aitm | cyra | toml | **test** | test in range? |
|---|---|---|---|---|---|
| p1/p99 anchors | 0.520 | 0.536 | 0.564 | **0.652** | no |
| white-matter anchor | 0.869 | 0.831 | 0.897 | **0.807** | **yes** |

Lesion-versus-healthy contrast also becomes consistent across subjects (lesion T2
1.51–1.57, healthy 1.19–1.21), which is the property a transferable Gaussian needs.
Units are now *fraction of white-matter signal*, so values run to ≈ 1.2 on T1 and ≈ 1.9
on T2 rather than to 1.0.

## 3. The multimodal Gaussian models

"Multimodal" here means **multi-modality**: each voxel carries a 2-vector
**x** = (T1, T2), and each class is a **bivariate Gaussian with full 2×2 covariance**:

```
log p(x | k) = -0.5 [ (x - mu_k)' S_k^-1 (x - mu_k) + log|S_k| + d log 2pi ]
```

The off-diagonal terms matter. Every fitted class has substantial T1–T2 correlation — the
lesion class sits at **−0.662** — so a diagonal-covariance model would discard exactly
the joint structure that separates lesion from normal white matter.

**Healthy tissue is not one Gaussian.** CSF, grey matter and white matter occupy
well-separated lobes of the (T1, T2) plane, and a single Gaussian straddling all three
fits none of them. The healthy class is therefore a **K-component mixture fitted by EM**
with k-means++ initialisation; the lesion class remains a single Gaussian.

Fitted on all three training subjects (1,427,173 healthy and 36,864 lesion voxels,
lesion prior 0.0252), with K = 6 selected:

| component | weight | mean (T1, T2) | corr | interpretation |
|---|---|---|---|---|
| 1 | 0.085 | (0.364, 1.738) | −0.21 | CSF |
| 2 | 0.131 | (0.594, 1.503) | −0.55 | CSF / grey partial volume |
| 3 | 0.188 | (0.801, 1.175) | +0.58 | grey matter, broad |
| 4 | 0.292 | (0.848, 1.145) | −0.43 | grey matter, tight |
| 5 | 0.299 | (1.005, 0.973) | −0.28 | **white matter** (1.0 by construction) |
| 6 | 0.006 | (1.305, 1.703) | −0.70 | outlier catch-all |
| **lesion** | — | **(0.899, 1.536)** | **−0.66** | darker than WM on T1, much brighter on T2 |

The white-matter component landing at T1 = 1.005 is a useful internal check: the
standardisation anchor was defined to put it at 1.0, and EM — which knew nothing about
that — recovered it.

`fig1_feature_space.png` shows why **both** modalities are needed: the lesion cluster
sits above the healthy ridge at comparable T1, so it separates in the joint plane while
overlapping heavily in either marginal alone (`fig2_marginals.png`).

## 4. The Markov Random Field

A per-voxel classifier produces salt-and-pepper false positives in noisy MRI. The MRF
adds a spatial prior. Energy to minimise over the label field L:

```
U(L) = sum_i [ -log p(x_i | L_i) - log pi_{L_i} ]        (data term)
     + beta * sum_{i~j} w_ij * 1[L_i != L_j]             (Potts prior)
```

The second term is the **Potts prior**: neighbouring voxels tend to share a label.

**Anisotropy.** Voxels are 0.977 × 0.977 × 3.0 mm, so a through-plane neighbour is three
times further away than an in-plane one and must not carry equal influence. The
6-neighbour weights are set proportional to 1/spacing.

**Minimisation** is by **ICM** (Iterated Conditional Modes) on a **checkerboard**, so no
voxel updates simultaneously with any of its own neighbours — a synchronous sweep can
oscillate instead of converging. On the test image ICM converged in 10 iterations
(changes per sweep: 5459 → 845 → 156 → 44 → 19 → 10 → 6 → 5 → 2 → 0).

A τ term biases the lesion class; it is a single additive constant on the log-posterior
difference, tuned alongside β.

**Implementation note.** The sweep uses a specialised two-class solver: with two labels
`agree_0 = nb(mask) − agree_1`, so one neighbourhood pass per sweep suffices instead of
K, and the 6-neighbour sum is six shifted additions rather than a 27-point convolution.
Before trusting several hundred of its results I checked it against the general K-class
implementation at three β values — **3,276,800 / 3,276,800 voxels identical in every
case** (printed in `outputs/05_select_log.txt`).

## 5. Model selection

β, τ, K and the spatial constraint are all chosen by **leave-one-subject-out** — never on
the test image, and never on the subject being scored. With three subjects, each fold
trains on two and is evaluated on the third.

The spatial constraint encodes a genuine domain fact rather than a fit to these three
subjects: **MS lesions are white-matter lesions**, so voxels within 6 mm of the brain
surface are excluded from the lesion class. This is checkable — 96.2 %, 98.6 % and 99.8 %
of labelled lesion voxels lie inside that interior region, so the constraint discards
almost no true lesion.

| K | constraint | mean Dice | mean PPV | mean sens. | mean predicted vol. |
|---|---|---|---|---|---|
| 3 | no | 0.4838 | 0.440 | 0.693 | 51.0 ml |
| 3 | **yes** | 0.6262 | 0.673 | 0.690 | 32.5 ml |
| 6 | no | 0.5004 | 0.427 | 0.729 | 55.0 ml |
| **6** | **yes** | **0.6597** | **0.675** | **0.726** | **34.6 ml** |
| 8 | no | 0.4976 | 0.434 | 0.710 | 52.3 ml |
| 8 | yes | 0.6594 | 0.696 | 0.706 | 32.0 ml |

**Selected: K = 6, white-matter constraint, β = 1.0, τ = 2.0.**

Read honestly, this table says the **constraint** is what matters (+0.15 Dice at every K)
and **K barely does**: K = 6 and K = 8 differ by 0.0003, which is noise across three
folds. My original hypothesis — that more mixture components would absorb the
partial-volume voxels and fix the over-detection — is **not supported**. Going from K = 3
to K = 6 buys only +0.03 Dice. The over-detection was caused by the intensity
standardisation (§2), and the residual cortical false positives are removed by the
spatial constraint, not by extra Gaussians. K = 6 is selected because it scores highest,
but K = 3 with the constraint would be a defensible simpler choice.

β and τ were fixed at 1.0 and 2.0, the values an earlier full 480-cell sweep selected in
8 of its 10 rows (it never selected τ = 0), so the trimmed grid re-searches only the axes
genuinely in question.

## 6. Results

### Leave-one-subject-out

| held-out subject | Dice (MAP) | Dice (MAP+MRF) | sens. | spec. | PPV | true vol | pred vol | lesions found |
|---|---|---|---|---|---|---|---|---|
| `mp_aitm5157` | 0.4420 | **0.6301** | 0.923 | 0.9896 | 0.478 | 13.2 ml | 25.4 ml | 43 / 54 |
| `mp_cyra4454` | 0.6203 | **0.5978** | 0.467 | 0.9971 | 0.831 | 43.1 ml | 24.2 ml | 46 / 172 |
| `mp_toml5255` | 0.5986 | **0.7511** | 0.789 | 0.9892 | 0.717 | 49.2 ml | 54.2 ml | 57 / 112 |
| **mean** | **0.5536** | **0.6597** | — | — | — | — | — | — |

**The MRF contributes +0.106 Dice** on average — the central claim of the assignment, and
it holds. But not uniformly: it gains +0.188 on `aitm` and +0.153 on `toml`, and **loses
0.023 on `cyra`**. On `cyra` the per-voxel classifier was already conservative
(sensitivity 0.467), and the smoothing prior erodes small lesions further. Reporting the
mean alone would hide that.

### Test image — `mp_royr6403.1998-12-07`

| quantity | value |
|---|---|
| lesion volume | **38.53 ml** (13,468 voxels) |
| lesion load | 2.11 % of a 1829 ml brain |
| clusters | 52 (all ≥ 3 voxels) |
| largest cluster | 15.06 ml |
| ICM | 10 iterations, converged |

Three independent sanity checks:

- **Volume** sits inside the training range (13.2 / 43.1 / 49.2 ml) rather than 3–4× above it.
- **Spatial distribution** (`fig5_test_result.png`) is periventricular and bilateral —
  the classic MS pattern — concentrated at z = 26 and z = 32 with almost nothing at
  z = 14 or z = 38.
- **Fragmentation** is plausible: 52 clusters with a 15 ml largest, against 278 clusters
  with a 120 ml largest before the standardisation fix.

Output map: `outputs/mp_royr6403.1998-12-07.t2_lesions_predicted.nii`.

## 7. Limitations

1. **Three training subjects.** Each LOSO fold trains on two. All reported numbers carry
   wide error bars, and the K = 6 vs K = 8 distinction is certainly within noise.
2. **Lesion-wise sensitivity is mediocre** — 43/54, 46/172, 57/112 lesions detected. The
   method finds lesion *burden* far better than it finds every individual small lesion.
   Voxel-wise Dice flatters this: large periventricular confluent lesions dominate the
   score while small isolated ones are missed.
3. **No test ground truth**, so the test number is validated only by the three indirect
   checks above, not measured.
4. **Brain mask is an eroded core**, so cortical lesions are out of scope by construction.
5. **The 6 mm constraint is a blunt proxy** for white matter. A proper white-matter
   segmentation or an atlas-based prior would be better.
6. **The intensity standardisation is a one-landmark scale.** A full Nyúl–Udupa
   piecewise-linear histogram match would be more robust still.

## 8. Reproducing

```
python -I scripts/01_explore.py    data/assignment6 outputs
python -I scripts/02_preprocess.py data/assignment6 outputs work
python -I scripts/05_select_k.py   work outputs
python -I scripts/06_figures.py    work outputs
```

| file | role |
|---|---|
| `scripts/nii_io.py` | NIfTI reader/writer honouring the real `vox_offset`, ignoring stale `scl_` fields |
| `scripts/01_explore.py` | geometry, intensity ranges, lesion burden → `outputs/01_explore.json` |
| `scripts/02_preprocess.py` | brain extraction + white-matter-anchored standardisation |
| `scripts/models.py` | `GaussianClass`, `GaussianMixture` (EM), Potts MRF, ICM, metrics |
| `scripts/03_model_mrf.py` | first modelling pass, K fixed at 3 (kept as the record of that stage) |
| `scripts/04_figures.py` | figures for that first pass |
| `scripts/05_select_k.py` | model selection by LOSO + final test segmentation |
| `scripts/06_figures.py` | figures for the selected model |

Figures: `outputs/figures/fig1`–`fig6`. Machine-readable records: `outputs/*.json`.
Seed fixed at 0 throughout; EM uses a 200,000-voxel subsample.
