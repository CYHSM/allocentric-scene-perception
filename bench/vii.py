"""
One axis from ResNet-50 to a frontier VLM.

The three read-outs in this paper are not comparable as raw accuracy. Retrieval
over a 100-scene gallery has a chance level of 1%, the 4AFC arm 25%, the 2AFC
arm 50%; 54% on the 2AFC arm is barely off the floor while 54% on retrieval
would be extraordinary. Worse, a model that is simply *better at everything*
scores higher on all of them, so a raw ranking measures general competence and
not the thing the benchmark exists to isolate.

So each cell is converted to **d'** under the unbiased m-alternative forced
choice model -- retrieval with a gallery of N is m = N -- and then normalised by
the model's own performance at zero viewpoint change:

    VII(delta) = d'(delta) / d'(0)

`delta = 0` is the appearance gate: same place, same bearing, different
appearance sample. It is what the model can do when no viewpoint change is asked
of it, so dividing by it removes general competence and leaves viewpoint
invariance. VII = 1 is a representation the walk does not disturb; VII = 0 is
chance. A model with no appearance gate at all (d'(0) = 0) has no VII, and this
module returns None rather than a number, because the ratio would be noise over
noise.

Nothing here corrects for response bias, which m-AFC d' assumes away. That
assumption is violated in this dataset -- Qwen2.5-VL-7B answers "4" on 59% of
trials -- so `constant_answer_rate` computes what each model's own choice
distribution would score while ignoring the images entirely, and it belongs
beside every number VII produces.
"""

import numpy as np

__all__ = ["pc_from_dprime", "dprime", "vii", "constant_answer_rate",
           "vii_curve"]

# Gauss-Hermite nodes; 64 is far more than this integrand needs.
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(64)


def _phi(x):
    """Standard normal CDF, without pulling in scipy for one function."""
    from math import erf, sqrt
    return np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))(x)


def pc_from_dprime(d, m):
    """
    Proportion correct for an unbiased m-AFC observer with sensitivity `d`.

        P(c) = INT phi(x - d) Phi(x)^(m-1) dx

    evaluated by Gauss-Hermite after x = d + sqrt(2) t.
    """
    if m < 2:
        raise ValueError(f"m-AFC needs at least 2 alternatives, got {m}")
    z = d + np.sqrt(2.0) * _GH_X
    return float(np.sum(_GH_W * _phi(z) ** (m - 1)) / np.sqrt(np.pi))


def dprime(p_correct, m, n_trials=None, max_d=8.0):
    """
    Invert `pc_from_dprime`. Returns d' >= 0.

    At or below chance returns exactly 0.0 rather than a negative number: a
    below-chance cell here means the model is following a positional prior, not
    that it discriminates in reverse, and a negative d' would propagate into VII
    as a sign flip that means nothing.

    A perfect cell has infinite d', so when `n_trials` is given the estimate is
    capped at the highest rate that many trials can resolve, (n - 0.5) / n --
    the usual log-linear guard. Without `n_trials` a perfect cell returns inf.
    """
    chance = 1.0 / m
    if p_correct <= chance:
        return 0.0
    if n_trials:
        p_correct = min(p_correct, (n_trials - 0.5) / n_trials)
    if p_correct >= 1.0:
        return float("inf")

    lo, hi = 0.0, max_d
    if pc_from_dprime(hi, m) < p_correct:
        return float("inf")
    for _ in range(80):                      # bisection: ~1e-24 on this range
        mid = 0.5 * (lo + hi)
        if pc_from_dprime(mid, m) < p_correct:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def vii(p_delta, p_zero, m, n_trials=None):
    """
    Viewpoint-Invariance Index: d'(delta) / d'(0).

    None when the model has no appearance gate to normalise by (d'(0) = 0), or
    when d'(0) is infinite and the ratio is undefined.
    """
    d0 = dprime(p_zero, m, n_trials)
    if d0 <= 0.0 or not np.isfinite(d0):
        return None
    dd = dprime(p_delta, m, n_trials)
    if not np.isfinite(dd):
        return None
    return dd / d0


def vii_curve(by_delta, m, n_trials=None, zero_key=0):
    """
    `{delta: p_correct}` -> `{delta: VII}`. `by_delta` must contain `zero_key`.
    """
    if zero_key not in by_delta:
        raise KeyError(f"no delta={zero_key} cell to normalise by; "
                       f"have {sorted(by_delta)}")
    p0 = by_delta[zero_key]
    return {d: vii(p, p0, m, n_trials) for d, p in by_delta.items()}


def constant_answer_rate(model_choices, correct_choices):
    """
    What this model's own answer distribution scores while ignoring the images.

    Two numbers, because they answer different objections:

    `best_fixed`   the accuracy of always emitting the single best option --
                   what a reviewer computes to check whether a result is real.
    `distribution` the accuracy of sampling the model's observed choice
                   distribution independently of the trial -- what the model
                   would score if its images carried no information at all but
                   its positional prior stayed exactly as observed.

    Any reported accuracy that does not clear both of these is not evidence of
    anything.
    """
    correct_choices = list(correct_choices)
    model_choices = [c for c in model_choices if c is not None]
    n = len(correct_choices)
    if n == 0:
        raise ValueError("no trials")

    options = sorted(set(correct_choices) | set(model_choices))
    freq = {o: correct_choices.count(o) / n for o in options}
    best_fixed = max(freq.values())

    m = len(model_choices)
    prior = {o: (model_choices.count(o) / m if m else 0.0) for o in options}
    distribution = sum(prior[o] * freq[o] for o in options)
    return {"best_fixed": best_fixed, "distribution": distribution,
            "answer_freq": freq, "model_prior": prior}
