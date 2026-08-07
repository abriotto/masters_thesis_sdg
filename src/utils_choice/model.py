"""McFadden's conditional logit, and the checks that it is implemented right.

The estimator is written here rather than taken from a library because
statsmodels is not in the project environment. That makes validation
non-optional: :func:`run_validation_checks` is called from the notebooks so the
evidence appears in their output.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


class ConditionalLogit:
    """Conditional (multinomial) logit for choice sets of varying size::

        P(alternative j chosen from set g) = exp(x_j'b) / sum_k exp(x_k'b)

    ``counts`` says how many times each alternative was chosen in its set, so 3
    stochastic runs of a game enter as counts summing to 3. That likelihood is
    exactly the cross-entropy against the observed vote shares: pooling repeated
    choices and modelling the vote distribution are the same estimator.

    There is no intercept, and anything constant within a choice set cancels --
    which is why an outside option needs an alternative-specific constant.

    Penalties: ``l2`` ridge, ``l1`` lasso via the standard positive/negative
    split so the objective stays smooth for L-BFGS-B. Indices in
    ``unpenalized`` (the alternative-specific constant) are never penalised.
    """

    def __init__(self, l2=0.0, l1=0.0, unpenalized=()):
        self.l2, self.l1 = l2, l1
        self.unpenalized = set(unpenalized)

    @staticmethod
    def _index(groups):
        _, first, inv = np.unique(groups, return_index=True, return_inverse=True)
        return inv, len(first)

    def _neg_ll(self, b, X, counts, inv, n_groups):
        v = X @ b
        mx = np.full(n_groups, -np.inf)
        np.maximum.at(mx, inv, v)
        ex = np.exp(v - mx[inv])
        se = np.bincount(inv, weights=ex, minlength=n_groups)
        n_g = np.bincount(inv, weights=counts, minlength=n_groups)
        nll = -(counts @ v) + n_g @ (mx + np.log(se))
        p = ex / se[inv]
        return nll, X.T @ (n_g[inv] * p - counts)

    def fit(self, X, counts, groups):
        X, counts = np.asarray(X, float), np.asarray(counts, float)
        inv, n_groups = self._index(groups)
        k = X.shape[1]
        pen = np.array([j not in self.unpenalized for j in range(k)], float)
        if self.l1 > 0:
            def obj(z):
                b = z[:k] - z[k:]
                nll, g = self._neg_ll(b, X, counts, inv, n_groups)
                nll += (self.l1 * np.sum(pen * (z[:k] + z[k:]))
                        + 0.5 * self.l2 * np.sum(pen * b ** 2))
                gb = g + self.l2 * pen * b
                return nll, np.r_[gb + self.l1 * pen, -gb + self.l1 * pen]
            r = minimize(obj, np.zeros(2 * k), jac=True, method="L-BFGS-B",
                         bounds=[(0, None)] * (2 * k))
            self.coef_ = r.x[:k] - r.x[k:]
            self.coef_[np.abs(self.coef_) < 1e-7] = 0.0
        else:
            def obj(b):
                nll, g = self._neg_ll(b, X, counts, inv, n_groups)
                return (nll + 0.5 * self.l2 * np.sum(pen * b ** 2),
                        g + self.l2 * pen * b)
            r = minimize(obj, np.zeros(k), jac=True, method="L-BFGS-B")
            self.coef_ = r.x
        self.converged_ = bool(r.success)
        return self

    def predict_proba(self, X, groups):
        """Choice probabilities, summing to 1 within each choice set."""
        v = np.asarray(X, float) @ self.coef_
        inv, n_groups = self._index(groups)
        mx = np.full(n_groups, -np.inf)
        np.maximum.at(mx, inv, v)
        ex = np.exp(v - mx[inv])
        return ex / np.bincount(inv, weights=ex, minlength=n_groups)[inv]


# ------------------------------------------------------------- validation ----
def _check_recovery(seed=0, n_groups=4000, k=4):
    """Simulate choices from a known beta; the estimator must recover it."""
    rng = np.random.default_rng(seed)
    beta = np.array([1.5, -0.8, 0.4, 0.0])[:k]
    Xs, gs, cs = [], [], []
    for g in range(n_groups):
        J = rng.integers(3, 7)
        Xg = rng.normal(size=(J, k))
        p = np.exp(Xg @ beta); p /= p.sum()
        Xs.append(Xg); gs.append(np.full(J, g)); cs.append(rng.multinomial(3, p))
    m = ConditionalLogit().fit(np.vstack(Xs), np.concatenate(cs), np.concatenate(gs))
    err = np.abs(m.coef_ - beta).max()
    assert err < 0.12, err
    return f"recovers a known beta from simulated choices (max |error| = {err:.3f})"


def _check_binary_equivalence(seed=1, n_groups=3000, k=3):
    """With choice sets of size 2 the conditional logit reduces exactly to
    logistic regression on the difference of the two alternatives, without an
    intercept. Checked against scikit-learn."""
    rng = np.random.default_rng(seed)
    beta = np.array([0.9, -1.3, 0.5])[:k]
    X = rng.normal(size=(2 * n_groups, k))
    groups = np.repeat(np.arange(n_groups), 2)
    d = X[0::2] - X[1::2]
    first = (rng.random(n_groups) < 1 / (1 + np.exp(-(d @ beta)))).astype(int)
    counts = np.zeros(2 * n_groups)
    counts[0::2], counts[1::2] = first, 1 - first
    ours = ConditionalLogit().fit(X, counts, groups).coef_
    sk = LogisticRegression(penalty=None, fit_intercept=False,
                            max_iter=5000).fit(d, first).coef_[0]
    err = np.abs(ours - sk).max()
    assert err < 1e-3, err
    return f"matches scikit-learn on the J=2 reduction (max |difference| = {err:.1e})"


def _check_gradient(seed=2, n_groups=200, k=4):
    """The analytic gradient must match a numerical one, or the optimiser is
    solving a different problem than the likelihood."""
    from scipy.optimize import check_grad
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * 4, k))
    groups = np.repeat(np.arange(n_groups), 4)
    counts = np.zeros(len(X))
    counts[rng.integers(0, 4, n_groups) + 4 * np.arange(n_groups)] = 3
    m = ConditionalLogit()
    inv, n_g = m._index(groups)
    err = check_grad(lambda b: m._neg_ll(b, X, counts, inv, n_g)[0],
                     lambda b: m._neg_ll(b, X, counts, inv, n_g)[1],
                     rng.normal(size=k))
    assert err < 1e-4, err
    return f"analytic gradient matches the numerical one (error = {err:.1e})"


def _check_proper_distribution(seed=3, n_groups=50, k=3):
    """Predictions must sum to 1 within every choice set -- the property that
    makes this a choice model rather than a normalised score."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(3, 7, n_groups)
    groups = np.repeat(np.arange(n_groups), sizes)
    X = rng.normal(size=(len(groups), k))
    counts = np.zeros(len(groups))
    for g in range(n_groups):
        counts[np.flatnonzero(groups == g)[0]] = 3
    m = ConditionalLogit().fit(X, counts, groups)
    err = np.abs(np.bincount(groups, weights=m.predict_proba(X, groups)) - 1).max()
    assert err < 1e-10, err
    return f"choice probabilities sum to 1 in every set (max error = {err:.1e})"


def _check_l1_sparsifies(seed=4, n_groups=400, k=6):
    """A strong L1 penalty must set coefficients to exactly zero, otherwise the
    selection results mean nothing."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * 4, k))
    groups = np.repeat(np.arange(n_groups), 4)
    beta = np.array([2.0, 0, 0, 0, 0, 0])[:k]
    counts = np.zeros(len(X))
    for g in range(n_groups):
        s = np.flatnonzero(groups == g)
        p = np.exp(X[s] @ beta); p /= p.sum()
        counts[s] = rng.multinomial(3, p)
    nz = np.sum(ConditionalLogit(l1=50.0).fit(X, counts, groups).coef_ != 0)
    assert nz < k, nz
    return f"a strong L1 penalty zeroes coefficients ({nz} of {k} survive)"


def run_validation_checks():
    """All five checks; raises if any fails, returns the messages to print."""
    return [_check_recovery(), _check_binary_equivalence(), _check_gradient(),
            _check_proper_distribution(), _check_l1_sparsifies()]
