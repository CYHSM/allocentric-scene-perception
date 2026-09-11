"""
How far apart two scene layouts are, in metres, after the best global rotation.

The benchmark's foils are drawn uniformly at random from the other 99 scenes, so
each trial's difficulty is a lottery. Gemini 3.8 Flash's errors are almost
entirely the trials where that lottery produced a near-duplicate: its failures
sit at a mean 17.8 deg from the nearest foil against 36.2 deg for its successes
(permutation p = 0.0001), and in 12 of 16 failures it chose the single most
confusable foil. So difficulty needs to be a controlled axis, not an accident.

The distance below is the natural one for this task. The camera rotates around
a fixed world, so a layout that is the target turned by phi is *not* a different
place to an allocentric observer looking from a correspondingly shifted
viewpoint -- it is the hardest possible foil. Hence: minimise over phi.

    D(i, j) = min over phi   mean over peaks k   || R(phi) p_ik  -  p_jk ||

Peaks correspond **by landmark identity in every mode**, so `k` is that
correspondence and a close foil is one that is near-rotated *with the same
landmark assignment*.

This was previously done only for c0/c1/c2, with c3/c4 minimised over all 24
permutations as well, on the belief that those modes gave every scene unique
mountains. They do not. `fm_stimulus.assign_objects` replaces the per-seed
landforms with `canonical_forms(n)` at render time for the peak modes, so all
100 c3/c4 scenes are built from ONE set of four landforms, permuted per scene --
exactly like c0/c1/c2. What misled the earlier version is `layout["morphologies"]`,
a label field written by `sample_layout` *before* that substitution and never
updated: it lists 106 distinct strings describing forms that were never
rendered. The peaks' actual `form` vectors resolve to a single set.

Taking a minimum over 24 relabelings found alignments that identity matching
forbids, which compressed the c3/c4 distances to a median of 7.3 m against
32.4 m for c0-c2 -- from identical peak coordinates. That made percentile bands
mean different things per mode and left c4 barely touched by hard-foil
selection.

D is in the same units as the probe ladder that was configured and never
rendered (`probe_ladder: [1, 2, 4, 8, 16, 24]` metres), so selecting existing
scene pairs by D gives that displacement axis with no Blender time at all.
"""

import glob
import itertools
import json

import numpy as np

def _identity(peak):
    """
    What this peak *is*, as a sortable key.

    Object modes carry it in `obj` ({shape, colour}); peak modes carry it in the
    `form` vector. Never `type`, which is the stale pre-assignment label.
    """
    if peak.get("obj"):
        return (0, tuple(sorted(peak["obj"].items())))
    return (1, tuple(round(peak["form"][k], 6) for k in sorted(peak["form"])))


def load_layouts(mode, root="data/scenes_100"):
    """{scene_id: (4, 2) array of peak positions}, row k the same landmark everywhere."""
    bank = {}
    for f in sorted(glob.glob(f"{root}/{mode}/bank.shard*.json")):
        bank.update(json.load(open(f))["scenes"])
    out = {}
    for sid, v in bank.items():
        peaks = sorted(v["layout"]["peaks"], key=_identity)
        out[sid] = np.array([[p["x"], p["y"]] for p in peaks], dtype=float)
    return out


def distance_matrix(layouts, mode, phi_step=1.0):
    """Pairwise D in metres. Symmetric, zero diagonal."""
    ids = sorted(layouts)
    P = np.stack([layouts[s] for s in ids])              # (S, 4, 2)
    S, K, _ = P.shape

    phis = np.deg2rad(np.arange(0.0, 360.0, phi_step))
    c, s = np.cos(phis), np.sin(phis)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)   # (F, 2, 2)

    perms = [np.arange(K)]

    D = np.zeros((S, S))
    Prot = np.einsum("fab,skb->sfka", R, P)              # (S, F, K, 2)
    for i in range(S):
        best = None
        for perm in perms:
            # (F, K, 2) vs (S, K, 2) -> (S, F, K)
            d = np.linalg.norm(Prot[i][None, :, :, :] - P[:, None, perm, :], axis=-1)
            cand = d.mean(axis=2).min(axis=1)            # mean over peaks, min over phi
            best = cand if best is None else np.minimum(best, cand)
        D[i] = best
    D = np.minimum(D, D.T)                               # enforce symmetry
    np.fill_diagonal(D, 0.0)
    return ids, D


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/scenes_100")
    ap.add_argument("--out", default="data/layout_distance.npz")
    ap.add_argument("--modes", nargs="+",
                    default=["c0_shape_colour", "c1_shape", "c2_colour",
                             "c3_peaks_bare", "c4_valley"])
    args = ap.parse_args()

    store = {}
    for mode in args.modes:
        lay = load_layouts(mode, args.root)
        ids, D = distance_matrix(lay, mode, phi_step=1.0)
        store[f"{mode}__ids"] = np.array(ids)
        store[f"{mode}__D"] = D
        off = D[~np.eye(len(ids), dtype=bool)]
        nn = np.min(D + np.eye(len(ids)) * 1e9, axis=1)
        print(f"{mode:18s} {len(ids)} scenes | all pairs: median {np.median(off):5.1f} m, "
              f"min {off.min():4.1f} | nearest neighbour: median {np.median(nn):4.1f} m, "
              f"min {nn.min():4.1f} m")
    np.savez_compressed(args.out, **store)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
