"""
The handful of statistics this paper reports, in one place.

Five figure scripts had grown five identical copies of `wilson`, which is a
small problem now and a large one the moment one of them is edited. Every
inference in the paper comes from here.
"""

import math
import random

__all__ = ["wilson", "mcnemar_exact", "permutation_diff", "fisher_exact_2x2",
           "cohen_kappa"]


def wilson(k, n, z=1.96):
    """
    Wilson score interval for k successes in n trials.

    Not the normal approximation: at n = 4 per cell (which is what a 100-trial
    run gives per mode x delta) the normal interval runs off the ends of [0, 1]
    and reports negative accuracy.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(a_correct, b_correct):
    """
    Two-sided exact McNemar on paired binary outcomes -- the same trials
    answered by two observers.

    Paired is the point: the observers here differ by orders of magnitude in
    overall accuracy, and an unpaired test on 100 trials would be swamped by
    that difference rather than by the trial-level disagreement that matters.
    Returns (b, c, p) where b and c are the discordant counts.
    """
    a, b_ = list(a_correct), list(b_correct)
    if len(a) != len(b_):
        raise ValueError(f"unpaired: {len(a)} vs {len(b_)}")
    b = sum(1 for x, y in zip(a, b_) if x and not y)
    c = sum(1 for x, y in zip(a, b_) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    # Binomial(n, 0.5) tail, doubled.
    def _pmf(k):
        return math.comb(n, k) * 0.5 ** n
    k = min(b, c)
    p = 2.0 * sum(_pmf(i) for i in range(k + 1))
    return b, c, min(1.0, p)


def permutation_diff(x, y, n_iter=20000, seed=0):
    """
    Two-sided permutation test on a difference of means between two unpaired
    samples. Used where a t-test's normality assumption is not available --
    per-trial foil distances, which are bounded and skewed.
    """
    rng = random.Random(seed)
    x, y = list(x), list(y)
    obs = abs(sum(x) / len(x) - sum(y) / len(y))
    pool = x + y
    nx = len(x)
    hits = 0
    for _ in range(n_iter):
        rng.shuffle(pool)
        d = abs(sum(pool[:nx]) / nx - sum(pool[nx:]) / (len(pool) - nx))
        if d >= obs - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_iter + 1)


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact on [[a, b], [c, d]]. Returns p."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def _p(x):
        return (math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1))
    obs = _p(a)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return min(1.0, sum(_p(x) for x in range(lo, hi + 1) if _p(x) <= obs + 1e-12))


def cohen_kappa(a, b):
    """
    Agreement between two observers on the same trials, corrected for the
    agreement two independent observers would reach by chance.

    Raw agreement is useless here: two observers who are both at 20% accuracy
    agree on ~68% of trials purely by both being wrong a lot. Kappa asks whether
    they are wrong on the *same* trials, which is the question -- whether the
    models share one failure mode or merely share a low score.

    Kappa is 0 at chance agreement, 1 at perfect, and negative when two
    observers disagree more than independence predicts.
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError(f"unpaired: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)
