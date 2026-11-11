"""
Feature selection algorithms used in the heart disease identification study.

Implemented methods
-------------------
* ReliefF  - instance-based relevance weighting (Kononenko, 1994)
* mRMR     - minimum Redundancy Maximum Relevance (Peng et al., 2005)
* LASSO    - L1-regularised logistic regression coefficients
* LLBFS    - Local-Learning-Based Feature Selection (Sun et al., 2010), margin-based
* FCMIM    - Fast Conditional Mutual Information Maximisation (Fleuret, 2004)
* EEMIFS   - Enhanced Ensemble Mutual Information Feature Selection (proposed)

Every selector exposes the same interface::

    scores, ranking = selector(X, y, **kwargs)

``scores``  : 1-D array with one importance score per feature (higher = better)
``ranking`` : feature indices ordered from most to least important
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _discretize(X: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Equal-frequency discretisation (needed by entropy-based estimators)."""
    Xd = np.empty_like(X, dtype=int)
    for j in range(X.shape[1]):
        col = X[:, j:j + 1]
        if len(np.unique(col)) <= n_bins:
            _, Xd[:, j] = np.unique(col.ravel(), return_inverse=True)
        else:
            kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            with warnings.catch_warnings():  # ties can merge quantile bins
                warnings.simplefilter("ignore")
                Xd[:, j] = kb.fit_transform(col).ravel().astype(int)
    return Xd


def _entropy(*cols: np.ndarray) -> float:
    """Joint Shannon entropy (in nats) of one or more discrete columns."""
    joint = np.stack(cols, axis=1)
    _, counts = np.unique(joint, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _mi(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information I(a; b) for discrete variables."""
    return _entropy(a) + _entropy(b) - _entropy(a, b)


def _cmi(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Conditional mutual information I(a; b | c) for discrete variables."""
    return _entropy(a, c) + _entropy(b, c) - _entropy(a, b, c) - _entropy(c)


def _ranking(scores: np.ndarray) -> np.ndarray:
    return np.argsort(scores)[::-1]


# --------------------------------------------------------------------------- #
# 1. ReliefF
# --------------------------------------------------------------------------- #

def relief_f(X: np.ndarray, y: np.ndarray, n_neighbors: int = 10,
             n_samples: int | None = None, random_state: int = 0):
    """ReliefF weights: features that separate nearby samples of different
    classes (misses) more than samples of the same class (hits) score higher."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    span = X.max(axis=0) - X.min(axis=0)
    span[span == 0] = 1.0
    Xn = (X - X.min(axis=0)) / span
    idx = rng.permutation(n)[: n_samples or n]
    classes, counts = np.unique(y, return_counts=True)
    prior = dict(zip(classes, counts / n))
    w = np.zeros(d)
    for i in idx:
        dist = np.abs(Xn - Xn[i]).sum(axis=1)
        dist[i] = np.inf
        for c in classes:
            mask = np.where(y == c)[0]
            mask = mask[mask != i]
            nn = mask[np.argsort(dist[mask])[:n_neighbors]]
            diff = np.abs(Xn[nn] - Xn[i]).mean(axis=0)
            if c == y[i]:
                w -= diff
            else:
                w += prior[c] / (1 - prior[y[i]]) * diff
    scores = w / len(idx)
    return scores, _ranking(scores)


# --------------------------------------------------------------------------- #
# 2. mRMR
# --------------------------------------------------------------------------- #

def mrmr(X: np.ndarray, y: np.ndarray, n_bins: int = 5):
    """Greedy mRMR (MID criterion): relevance I(f;y) minus mean redundancy
    with the already selected features."""
    Xd = _discretize(X, n_bins)
    d = X.shape[1]
    relevance = np.array([_mi(Xd[:, j], y) for j in range(d)])
    selected = [int(np.argmax(relevance))]
    scores = np.zeros(d)
    scores[selected[0]] = relevance[selected[0]]
    remaining = set(range(d)) - set(selected)
    while remaining:
        best, best_val = None, -np.inf
        for j in remaining:
            red = np.mean([_mi(Xd[:, j], Xd[:, s]) for s in selected])
            val = relevance[j] - red
            if val > best_val:
                best, best_val = j, val
        selected.append(best)
        scores[best] = best_val
        remaining.remove(best)
    # Convert greedy order into monotonically decreasing scores
    order = np.array(selected)
    rank_scores = np.zeros(d)
    rank_scores[order] = np.linspace(1.0, 1.0 / d, d)
    return rank_scores, order


# --------------------------------------------------------------------------- #
# 3. LASSO
# --------------------------------------------------------------------------- #

def lasso(X: np.ndarray, y: np.ndarray, C: float = 0.1, random_state: int = 0):
    """Absolute L1-regularised logistic-regression coefficients."""
    with warnings.catch_warnings():          # 'penalty' is deprecated in sklearn >= 1.8
        warnings.simplefilter("ignore", FutureWarning)
        model = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                   max_iter=5000, random_state=random_state)
        model.fit(X, y)
    scores = np.abs(model.coef_).ravel()
    return scores, _ranking(scores)


# --------------------------------------------------------------------------- #
# 4. Local-Learning-Based Feature Selection
# --------------------------------------------------------------------------- #

def llbfs(X: np.ndarray, y: np.ndarray, sigma: float = 2.0, lam: float = 1.0,
          n_iter: int = 20, lr: float = 0.05):
    """Simplified LLBFS (Sun, Todorovic & Goodison, 2010).

    The data is decomposed into local problems: for every sample the nearest
    hit / miss are found under the current feature weighting (kernel weighted),
    and feature weights are learned by maximising the logistic margin with an
    L1 penalty.  Weights are kept non-negative.
    """
    n, d = X.shape
    w = np.ones(d) / np.sqrt(d)
    for _ in range(n_iter):
        diffs = np.abs(X[:, None, :] - X[None, :, :])          # n x n x d
        dist = diffs @ w
        np.fill_diagonal(dist, np.inf)
        same = y[:, None] == y[None, :]
        k = np.exp(-dist / sigma)
        k[np.isinf(dist)] = 0
        k_hit = np.where(same, k, 0)
        k_miss = np.where(~same, k, 0)
        k_hit /= k_hit.sum(axis=1, keepdims=True) + 1e-12
        k_miss /= k_miss.sum(axis=1, keepdims=True) + 1e-12
        m_hit = np.einsum("ij,ijd->id", k_hit, diffs)
        m_miss = np.einsum("ij,ijd->id", k_miss, diffs)
        z = m_miss - m_hit                                      # local margin vectors
        margin = z @ w
        grad = -(z * (1 / (1 + np.exp(margin)))[:, None]).mean(axis=0) + lam * np.sign(w) / n
        w = np.maximum(w - lr * grad * d, 0)
        if w.sum() == 0:
            w = np.ones(d) / d
        w /= np.linalg.norm(w) + 1e-12
    return w, _ranking(w)


# --------------------------------------------------------------------------- #
# 5. Fast Conditional Mutual Information Maximisation
# --------------------------------------------------------------------------- #

def fcmim(X: np.ndarray, y: np.ndarray, n_bins: int = 5):
    """Fleuret's CMIM: pick the feature f maximising min_s I(f; y | s)."""
    Xd = _discretize(X, n_bins)
    d = X.shape[1]
    partial = np.array([_mi(Xd[:, j], y) for j in range(d)], dtype=float)
    selected: list[int] = []
    scores = np.zeros(d)
    for _ in range(d):
        cand = [j for j in range(d) if j not in selected]
        best = max(cand, key=lambda j: partial[j])
        scores[best] = partial[best]
        selected.append(best)
        for j in cand:
            if j != best:
                partial[j] = min(partial[j], _cmi(Xd[:, j], y, Xd[:, best]))
    order = np.array(selected)
    rank_scores = np.zeros(d)
    rank_scores[order] = np.linspace(1.0, 1.0 / d, d)
    return rank_scores, order


# --------------------------------------------------------------------------- #
# 6. EEMIFS (proposed)
# --------------------------------------------------------------------------- #

def _kernel_mi(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Kernel dependence (normalised HSIC with RBF kernels), used as a smooth
    non-linear mutual-information surrogate."""
    n = len(x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    Kx = np.exp(-gamma * (x[:, None] - x[None, :]) ** 2)
    Ky = (y[:, None] == y[None, :]).astype(float)
    H = np.eye(n) - np.ones((n, n)) / n
    Kxc, Kyc = H @ Kx @ H, H @ Ky @ H
    hsic = np.sum(Kxc * Kyc)
    return float(hsic / (np.sqrt(np.sum(Kxc * Kxc) * np.sum(Kyc * Kyc)) + 1e-12))


def eemifs(X: np.ndarray, y: np.ndarray, n_bootstrap: int = 30, n_bins: int = 5,
           redundancy_weight: float = 0.5, interaction_weight: float = 0.25,
           random_state: int = 0):
    """Enhanced Ensemble Mutual Information Feature Selection.

    1. **Ensemble relevance** - three complementary information measures:
       discrete (histogram) MI, k-NN (KSG) MI and kernel MI (HSIC).  Each is
       min-max normalised and combined with *adaptive weights* proportional to
       its stability (inverse coefficient of variation across bootstraps).
    2. **Stability enhancement** - all measures are averaged over
       ``n_bootstrap`` bootstrap resamples.
    3. **Interaction detection** - rewards features whose joint information
       with another feature exceeds the sum of individual information
       (synergy, I(f_i, f_j; y) - I(f_i; y) - I(f_j; y) > 0).
    4. **Adaptive redundancy control** - greedy forward selection penalising
       mean normalised MI with already selected features.
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    measures = np.zeros((3, n_bootstrap, d))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        Xbd = _discretize(Xb, n_bins)
        measures[0, b] = [_mi(Xbd[:, j], yb) for j in range(d)]
        measures[1, b] = mutual_info_classif(Xb, yb, n_neighbors=5,
                                             random_state=int(rng.integers(1e9)))
        measures[2, b] = [_kernel_mi(Xb[:, j], yb) for j in range(d)]

    # Normalise every measure to [0, 1] per bootstrap
    mn = measures.min(axis=2, keepdims=True)
    mx = measures.max(axis=2, keepdims=True)
    norm = (measures - mn) / (mx - mn + 1e-12)
    mean = norm.mean(axis=1)                          # 3 x d
    cv = norm.std(axis=1).mean(axis=1) / (mean.mean(axis=1) + 1e-12)
    alpha = 1.0 / (cv + 1e-6)
    alpha /= alpha.sum()                              # adaptive weights
    relevance = (alpha[:, None] * mean).sum(axis=0)

    # Feature interaction (synergy) score
    Xd = _discretize(X, n_bins)
    ind = np.array([_mi(Xd[:, j], y) for j in range(d)])
    inter = np.zeros(d)
    for i in range(d):
        syn = []
        for j in range(d):
            if i == j:
                continue
            pair = Xd[:, i] * (Xd[:, j].max() + 1) + Xd[:, j]
            syn.append(max(_mi(pair, y) - ind[i] - ind[j], 0.0))
        inter[i] = np.mean(syn)
    inter = inter / (inter.max() + 1e-12)
    base = relevance + interaction_weight * inter

    # Normalised pairwise redundancy
    H = np.array([_entropy(Xd[:, j]) for j in range(d)])
    red = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            red[i, j] = red[j, i] = _mi(Xd[:, i], Xd[:, j]) / (min(H[i], H[j]) + 1e-12)

    selected = [int(np.argmax(base))]
    final = np.zeros(d)
    final[selected[0]] = base[selected[0]]
    while len(selected) < d:
        cand = [j for j in range(d) if j not in selected]
        # adaptive: redundancy penalty grows as more features are selected
        lam = redundancy_weight * (len(selected) / d) ** 0.5
        vals = {j: base[j] - lam * red[j, selected].mean() for j in cand}
        best = max(vals, key=vals.get)
        final[best] = vals[best]
        selected.append(best)
    order = np.array(selected)
    rank_scores = np.zeros(d)
    rank_scores[order] = np.linspace(1.0, 1.0 / d, d)
    return rank_scores, order


SELECTORS = {
    "ReliefF": relief_f,
    "mRMR": mrmr,
    "LASSO": lasso,
    "LLBFS": llbfs,
    "FCMIM": fcmim,
    "EEMIFS (Proposed)": eemifs,
}
