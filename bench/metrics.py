"""
Graded read-outs for the 4MT dataset: retrieval, invariance margin, and RSA.

    python bench/metrics.py --bank data/grid/c0_shape_colour_n4 \
        --model vit_base_patch14_dinov2.lvd142m

Why these and not 4AFC. The 4AFC score bottoms out at 25% and, on this dataset,
every interesting cell is already there -- so nothing can be ordered and no
trend is visible. All three metrics below run on a pool of every scene in the
bank, so their floor is 1/S rather than 1/4, and all three are continuous.

They answer three different questions about the same embeddings:

* **retrieval** -- can the model find the scene again from a new viewpoint;
* **NVM** -- by how much it wins or loses. This is exactly the margin a
  nearest-neighbour rule thresholds, so it *explains* a chance-level 4AFC
  instead of restating it;
* **RSA** -- whether the geometry of the scene manifold survives the rotation
  at all, even where the nearest neighbour does not. That is the question a
  linear probe is usually reached for, without the probe's free parameters.

Every scene contributes one image per (azimuth x appearance). The query is the
target under appearance A, the gallery the target under appearance B, so the
same nuisance variation the 4AFC items carry is present here -- otherwise Delta-0
would be a byte-identical match and the whole curve would be anchored to a
number that measures nothing.
"""

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Pure-numpy metric cores -- no model, no I/O, so they can be unit-tested
# against inputs whose answers are known analytically.
# --------------------------------------------------------------------------- #

def _unit(mat):
    mat = np.asarray(mat, dtype=np.float64)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(n, 1e-12)


def retrieval(queries, gallery, ks=(1, 5)):
    """
    Rank `gallery` against each query. Row i of both is scene i.

    Returns {"recall@k": ..., "map": ..., "median_rank": ..., "n": S}. With
    exactly one relevant item per query, mean average precision is the mean
    reciprocal rank; it is reported under the usual name because that is what
    the retrieval literature would call it.
    """
    q, g = _unit(queries), _unit(gallery)
    sims = q @ g.T
    correct = np.diag(sims).copy()

    # Rank of the correct item, 1-based, counting a tie AGAINST the model. A
    # degenerate embedding that maps every image to the same point produces an
    # all-equal similarity matrix; scoring ties optimistically would report that
    # as 100% retrieval. Only off-diagonal entries are compared, so the correct
    # item is never counted as beating itself.
    off = sims.copy()
    np.fill_diagonal(off, -np.inf)
    ranks = (off >= correct[:, None]).sum(axis=1) + 1
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["map"] = float((1.0 / ranks).mean())
    out["median_rank"] = float(np.median(ranks))
    out["n"] = int(len(ranks))
    out["chance_recall@1"] = 1.0 / max(len(ranks), 1)
    return out


def nvm(queries, gallery, hardest=False):
    """
    Normalised viewpoint-invariance margin, in [-1, 1].

        NVM = E[ (d_diff - d_same) / (d_diff + d_same) ]

    with cosine distance, `d_same` the distance to the same scene seen from the
    other viewpoint and `d_diff` the distance to other scenes -- averaged over
    distractors, or the nearest one when `hardest` is set.

    Reading the sign:

    * **> 0** -- the same place is closer than a different place despite the
      viewpoint change. This is invariance, and 1.0 is perfect.
    * **~ 0** -- same and different places are equidistant. The embedding
      carries no usable scene identity across the rotation; a representation
      that collapses to one point per viewpoint lands here, not below zero.
    * **< 0** -- the same place is reliably *further* than a different one. Not
      mere absence of signal but anti-correlated signal, which is what a 4AFC
      score below chance looks like from the inside.

    Scale-free, so it is comparable across architectures and embedding widths.
    """
    q, g = _unit(queries), _unit(gallery)
    d = 1.0 - (q @ g.T)
    s = len(d)
    if s < 2:
        return float("nan")
    d_same = np.diag(d)
    off = d.copy()
    np.fill_diagonal(off, np.nan)
    d_diff = np.nanmin(off, axis=1) if hardest else np.nanmean(off, axis=1)
    denom = d_diff + d_same
    ok = denom > 1e-12
    return float(np.mean((d_diff[ok] - d_same[ok]) / denom[ok]))


def _spearman(a, b):
    """Spearman rho without scipy, with average ranks for ties."""
    def rank(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(1, len(v) + 1, dtype=np.float64)
        # Average the ranks within each tied run.
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra @ rb) / denom) if denom > 1e-12 else float("nan")


def rsa(view_x, view_y):
    """
    Spearman rho between the scene x scene distance matrices at two viewpoints.

    Asks whether the *relative* geometry of the scene manifold is preserved
    under rotation -- a weaker and more forgiving question than whether the
    nearest neighbour is right, and the one that separates "the information is
    gone" from "the information is there but the metric cannot use it".
    """
    x, y = _unit(view_x), _unit(view_y)
    if len(x) < 3:
        return float("nan")
    dx = 1.0 - (x @ x.T)
    dy = 1.0 - (y @ y.T)
    iu = np.triu_indices(len(x), k=1)
    return _spearman(dx[iu], dy[iu])


def all_metrics(queries, gallery, view_x=None, view_y=None):
    """Every metric for one (viewpoint X -> viewpoint Y) comparison."""
    out = retrieval(queries, gallery)
    out["nvm"] = nvm(queries, gallery)
    out["nvm_hardest"] = nvm(queries, gallery, hardest=True)
    if view_x is not None and view_y is not None:
        out["rsa"] = rsa(view_x, view_y)
    return out


# --------------------------------------------------------------------------- #
# Running them over a rendered bank
# --------------------------------------------------------------------------- #

def bank_frames(bank, role=None):
    """
    {(scene, appearance, azimuth): relative image path}.

    `role="anchor"` restricts to the retrieval gallery. Probes are deliberately
    excluded from it: a probe is the same place as its anchor with one landmark
    nudged, so putting probes in the gallery would make Recall@1 depend on how
    many near-duplicates the pool happens to hold rather than on the model, and
    the number would stop being comparable between modes. Probes are what
    `exchange.py` measures scene distance with; they are not places to retrieve.
    """
    out = {}
    for sid, scene in sorted(bank["scenes"].items()):
        if role is not None and scene.get("role") != role:
            continue
        for key, frame in scene["frames"].items():
            app, az = key.split("_")
            out[(sid, app, int(az))] = os.path.join(scene["dir"],
                                                    frame["image"])
    return out


def embed_bank(bank_dir, bank, model_id, batch=64, workers=8, role=None):
    """One embedding pass over every frame in the bank."""
    from backends import EmbeddingBackend

    frames = bank_frames(bank, role=role)
    paths = sorted({os.path.join(bank_dir, p) for p in frames.values()})
    be = EmbeddingBackend(model_id, batch=batch, workers=workers)
    fake = [{"study": {"image": os.path.relpath(p, bank_dir)}, "options": []}
            for p in paths]
    be.precompute(fake, bank_dir)
    vec = {k: be.feats[be.index[os.path.join(bank_dir, p)]]
           for k, p in frames.items()}
    return vec


def angular_separation(a, b):
    d = abs(int(a) - int(b)) % 360
    return min(d, 360 - d)


def score_bank(bank_dir, model_id, appearance="changed", **kw):
    """
    Metrics for every viewpoint pair in a bank, aggregated by Delta-azimuth.

    `appearance="changed"` queries under A and searches a gallery under B, which
    is the benchmark condition. `"same"` uses B for both -- the control that
    isolates viewpoint from the nuisance variation, and whose Delta-0 cell is a
    byte-identical match and must therefore come out at exactly 1.0.
    """
    with open(os.path.join(bank_dir, "bank.json")) as f:
        bank = json.load(f)
    vec = embed_bank(bank_dir, bank, model_id, role="anchor", **kw)

    scenes = sorted({k[0] for k in vec})
    azimuths = sorted({k[2] for k in vec})
    q_app, g_app = ("A", "B") if appearance == "changed" else ("B", "B")

    def stack(app, az):
        rows = [vec.get((s, app, az)) for s in scenes]
        keep = [i for i, r in enumerate(rows) if r is not None]
        return np.stack([rows[i] for i in keep]), keep

    by_delta = defaultdict(list)
    for x, y in itertools.product(azimuths, azimuths):
        if appearance == "same" and x == y:
            pass                       # keep it: this is the identity check
        qs, kq = stack(q_app, x)
        gs, kg = stack(g_app, y)
        if len(kq) != len(kg) or len(kq) < 3:
            continue
        vx, _ = stack(g_app, x)
        m = all_metrics(qs, gs, view_x=vx, view_y=gs)
        by_delta[angular_separation(x, y)].append(m)

    out = {}
    for delta, runs in sorted(by_delta.items()):
        out[delta] = {k: float(np.nanmean([r[k] for r in runs]))
                      for k in runs[0]}
        out[delta]["n_pairs"] = len(runs)
    return {"bank": os.path.basename(os.path.abspath(bank_dir)),
            "model": model_id, "appearance": appearance,
            "n_scenes": len(scenes),
            "stimulus_mode": bank.get("stimulus_mode"),
            "n_objects": bank.get("n_objects"),
            "by_delta": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--appearance", default="changed",
                    choices=["changed", "same", "both"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    modes = ["changed", "same"] if args.appearance == "both" else [args.appearance]
    results = [score_bank(args.bank, args.model, appearance=m, batch=args.batch)
               for m in modes]

    for res in results:
        print(f"\n{res['bank']}  |  {res['model']}  |  appearance {res['appearance']}"
              f"  |  {res['n_scenes']} scenes"
              f"  (mode {res['stimulus_mode']}, n_objects {res['n_objects']})")
        print(f"  {'delta':>6s} {'R@1':>7s} {'R@5':>7s} {'mAP':>7s} "
              f"{'medrank':>8s} {'NVM':>7s} {'NVM-hard':>9s} {'RSA':>7s}")
        for delta, m in sorted(res["by_delta"].items()):
            print(f"  {delta:>6d} {m['recall@1']*100:>6.1f}% {m['recall@5']*100:>6.1f}%"
                  f" {m['map']*100:>6.1f}% {m['median_rank']:>8.1f}"
                  f" {m['nvm']:>+7.3f} {m['nvm_hardest']:>+9.3f} {m['rsa']:>+7.3f}")
        chance = 100.0 / max(res["n_scenes"], 1)
        print(f"  chance Recall@1 = {chance:.1f}%   "
              f"(NVM > 0 means the same place is closer than a different one)")
        if res["appearance"] == "same" and 0 in res["by_delta"]:
            r0 = res["by_delta"][0]["recall@1"]
            flag = "OK" if r0 > 0.999 else "FAILED"
            print(f"  identity check (delta 0, appearance same): "
                  f"Recall@1 = {r0:.3f}  [{flag}]")

    out = args.out or os.path.join(args.bank, "metrics.json")
    # Merge rather than overwrite. One cell is scored by several models in
    # sequence, and grid_report.py reads them all out of this one file; a plain
    # write would leave only whichever model happened to run last, with nothing
    # to show that the others had been dropped.
    kept = []
    if os.path.exists(out):
        try:
            with open(out) as f:
                prior = json.load(f)
            fresh = {(r["model"], r["appearance"]) for r in results}
            kept = [r for r in prior
                    if (r.get("model"), r.get("appearance")) not in fresh]
        except (ValueError, KeyError, TypeError):
            kept = []          # unreadable or foreign schema: start clean
    merged = kept + results
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n[metrics] -> {out}  ({len(results)} new, {len(kept)} kept)")


if __name__ == "__main__":
    main()
