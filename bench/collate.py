#!/usr/bin/env python3
"""
Every number in the paper, produced once, from the result files.

    python3 bench/collate.py            # writes paper/*.csv and paper/runs.json
    python3 bench/collate.py --check    # validate only, non-zero exit on problems

Tables and figures read the CSVs this writes. None of them opens a result file,
globs `results/`, or re-derives an accuracy, because that is how the same model
ended up with two different numbers in two different figures.

Three checks run on every collation and are the reason this script exists
rather than a dict of numbers in a plotting file:

1. **Config.** Each run is diffed against `paper_spec.ARM`. Deviations are
   named in `runs.json` and the run is marked ineligible for the headline
   table, not dropped -- a partial human arm is still worth plotting, it just
   must not be reported as if it were 100 trials.
2. **Trial identity.** Every eligible run must have answered the *same* 100
   trial ids. The stratified subsample is deterministic, so this holds by
   construction and fails loudly when someone changes the sampler.
3. **Positional prior.** Every accuracy is reported next to what the observer's
   own answer distribution would score with the images removed. An accuracy
   that does not clear that number is not evidence of anything.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_spec as SPEC
import stats as S
import vii as V

OUT_DIR = os.path.join(SPEC.REPO, "paper")


def _load(path):
    with open(os.path.join(SPEC.REPO, path)) as f:
        return json.load(f)


def _bench_digest():
    p = os.path.join(SPEC.REPO, SPEC.ARM["benchmark"])
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _provenance():
    p = os.path.join(OUT_DIR, "provenance.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def trial_geometry():
    """
    trial_id -> the foil geometry of that trial, from the benchmark file.

    Difficulty on this benchmark is a distance in metres, not a label: each
    trial's foils are drawn from a percentile band of rotation-optimal layout
    distance to the target. Joining it onto the per-trial records is what lets
    a figure ask whether an observer's errors track foil geometry, which is the
    one place where the frontier models and the open ones behave differently.
    """
    p = os.path.join(SPEC.REPO, SPEC.ARM["benchmark"])
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        b = json.load(f)
    return {t["id"]: {
        "min_foil_distance_m": t.get("min_foil_distance_m"),
        "foil_band_lo_m": (t.get("foil_distance_band_m") or [None, None])[0],
        "foil_band_hi_m": (t.get("foil_distance_band_m") or [None, None])[1],
    } for t in b["trials"]}


def load_runs():
    """One dict per observer in the roster, plus whatever went wrong loading it."""
    prov = _provenance()
    runs = []
    for obs in SPEC.OBSERVERS:
        rec = dict(obs)
        path = os.path.join(SPEC.REPO, obs["path"])
        if not os.path.exists(path):
            rec.update(present=False, blocking=["result file not found"],
                       gaps=[], eligible=False, trials=[])
            runs.append(rec)
            continue
        d = _load(obs["path"])
        summary, trials = d["summary"], d["results"]
        answered = [t for t in trials if t.get("model_choice") is not None]
        rec.update(
            present=True,
            summary=summary,
            run_config=summary.get("run_config"),
            trials=trials,
            n_trials=len(trials),
            n_answered=len(answered),
            n_unparsed=len(trials) - len(answered),
            accuracy=(sum(bool(t["is_correct"]) for t in trials) / len(trials)
                      if trials else 0.0),
            api_usage=summary.get("api_usage"),
        )
        rec["blocking"], rec["gaps"] = SPEC.deviations(summary, obs, prov)
        # An unparsed reply is scored wrong, which is the honest default, but a
        # run where it happens often is measuring the parser, not the model.
        if rec["n_unparsed"] > 0.05 * max(rec["n_trials"], 1):
            rec["gaps"].append(
                f"{rec['n_unparsed']}/{rec['n_trials']} replies did not parse "
                f"and are scored wrong")
        rec["eligible"] = not rec["blocking"]
        runs.append(rec)
    return runs


def cells(run):
    """{(mode, delta): (k, n)} for one run."""
    out = collections.defaultdict(lambda: [0, 0])
    for t in run["trials"]:
        key = (t["mode"], int(t["delta"]))
        out[key][1] += 1
        out[key][0] += bool(t["is_correct"])
    return {k: tuple(v) for k, v in out.items()}


def by_delta(run):
    out = collections.defaultdict(lambda: [0, 0])
    for t in run["trials"]:
        out[int(t["delta"])][1] += 1
        out[int(t["delta"])][0] += bool(t["is_correct"])
    return {k: tuple(v) for k, v in sorted(out.items())}


def by_mode(run):
    out = collections.defaultdict(lambda: [0, 0])
    for t in run["trials"]:
        out[t["mode"]][1] += 1
        out[t["mode"]][0] += bool(t["is_correct"])
    return {k: tuple(v) for k, v in out.items()}


def priors(run, rotated_only=False):
    """
    The positional-prior controls, from this run's own answers.

    `rotated_only` restricts both the observer's answer distribution and the
    answer key to the delta >= 45 trials, which is the only fair comparison for
    the rotated accuracy column. It matters here more than it usually would: the
    answer key of the 100-trial slice is **not balanced** -- 16/29/29/26 overall
    and 11/22/26/21 on the rotated trials -- even though the 500-trial bank it
    was drawn from is exactly 125 each. The stratified subsample balances mode
    and delta and takes no care of the answer slot. So an observer that happens
    to avoid answering "1" is rewarded and one that favours it is punished, by a
    few points, for no reason connected to the task.

    That is a defect in the slice, not in the observers, and the fix that costs
    nothing is to report each observer's accuracy against its own prior rather
    than against the nominal 25%.
    """
    trials = ([t for t in run["trials"] if int(t["delta"]) > 0]
              if rotated_only else run["trials"])
    if not trials:
        return None
    return V.constant_answer_rate(
        [t.get("model_choice") for t in trials],
        [t["correct_choice"] for t in trials])


def vii_curve(run):
    """
    VII by delta, with the floor made explicit.

    VII is d'(delta) / d'(0). When an observer is at or below chance with no
    viewpoint change its d'(0) is 0, the ratio is undefined, and `vii.vii`
    returns None. That is not a missing value to be imputed -- it is the finding
    that the observer has no appearance gate to be invariant about.
    """
    bd = by_delta(run)
    if 0 not in bd:
        return {}, None
    p0 = bd[0][0] / bd[0][1]
    n0 = bd[0][1]
    m = SPEC.ARM["n_options"]
    d0 = V.dprime(p0, m, n0)
    curve = {}
    for delta, (k, n) in bd.items():
        curve[delta] = V.vii(k / n, p0, m, min(n, n0))
    return curve, d0


def _benchmark_items():
    """trial_id -> the item the current benchmark specifies for that id."""
    p = os.path.join(SPEC.REPO, SPEC.ARM["benchmark"])
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        b = json.load(f)
    return {t["id"]: {
        "mode": t["mode"], "delta": t["delta"],
        "correct_choice": t["correct_choice"],
        "study": t.get("study_relpath"),
        "options": [o.get("rel_path") for o in t.get("options", [])],
    } for t in b["trials"]}


def check_trial_identity(runs):
    """
    Every eligible run answered the same trials -- checked against the
    benchmark, not against the other runs.

    Trial ids are positional (`c4_valley_d135_t01`), so rebuilding the benchmark
    reassigns which scene sits at each index while every id still matches by
    name. Comparing id sets between runs therefore passes a file whose trials
    point at entirely different scenes, which is what happened: p01's first
    session was generated 72 minutes before the benchmark was rebuilt, and 40 of
    its 100 trials were different scenes drawn from a different foil band
    (1.5-4.8 m against the current 2.7-12.3 m). The id sets were identical, so
    the old check passed it.

    A result file records ids and choices, not stimuli, so the strongest test it
    supports is the answer slot: the slot a run recorded must be the slot the
    benchmark specifies. That catches a stale build wherever the rebuild moved
    an answer -- 17 of those 40 trials -- which is enough to fail the run rather
    than average it in. `check_task_items` is the exact test, for runs that ship
    the task file they came from.
    """
    items = _benchmark_items()
    bad = []
    for r in [x for x in runs if x.get("eligible")]:
        ids = {t["trial_id"] for t in r["trials"]}
        missing = ids - set(items)
        if missing:
            bad.append(f"{r['key']}: {len(missing)} trial ids are not in "
                       f"{os.path.basename(SPEC.ARM['benchmark'])}")
            continue
        moved = [t["trial_id"] for t in r["trials"]
                 if t.get("correct_choice") is not None
                 and t["correct_choice"] != items[t["trial_id"]]["correct_choice"]]
        if moved:
            bad.append(
                f"{r['key']}: {len(moved)}/{len(ids)} trials record a different "
                f"answer slot than the benchmark -- built from a superseded "
                f"version (e.g. {', '.join(sorted(moved)[:3])})")
    return bad


def check_task_items(runs):
    """
    Where a run ships the task file it was generated from, compare the stimuli.

    Only the human arm has one. `task.json` carries the study image and the
    ordered option list, so this is the exact test the result files cannot
    support: same id, same pixels, same order, same key.
    """
    items = _benchmark_items()
    if not items:
        return []
    want = _bench_digest()
    bad = []
    for r in [x for x in runs if x.get("present") and x.get("access") == "human"]:
        # Cheapest check first: the page stamps the digest of the benchmark it
        # was generated from into every saved result.
        got = (r.get("summary") or {}).get("benchmark_sha256_12")
        if got and want and got != want:
            bad.append(f"{r['key']}: generated from benchmark {got}, "
                       f"the arm is {want}")
        task = os.path.join(SPEC.REPO, os.path.dirname(r["path"]), "task.json")
        if not os.path.exists(task):
            continue
        with open(task) as f:
            tk = json.load(f)
        spec = tk["trials"]
        tsha = tk.get("benchmark_sha256_12")
        if tsha and want and tsha != want:
            bad.append(f"{r['key']}: task.json was built from benchmark "
                       f"{tsha}, the arm is {want}")
        off = [t["id"] for t in spec
               if t["id"] not in items
               or t.get("study_relpath") != items[t["id"]]["study"]
               or [o.get("rel_path") for o in t.get("options", [])]
                  != items[t["id"]]["options"]
               or t["correct_choice"] != items[t["id"]]["correct_choice"]]
        if off:
            modes = sorted({t.split("_d")[0] for t in off})
            bad.append(f"{r['key']}: {len(off)}/{len(spec)} task trials differ "
                       f"from the benchmark (modes: {', '.join(modes)})")
    return bad


def unclaimed():
    """
    Result files that look like this arm but are in no roster row.

    A superseded `_n500` or a `_partial` snapshot left in `results/` is how a
    dead run gets back into a plot, so they are named rather than ignored.
    """
    claimed = {o["path"] for o in SPEC.OBSERVERS}
    found = []
    for root in ("results", "human_task_hard"):
        d = os.path.join(SPEC.REPO, root)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json") or fn == "task.json":
                continue
            rel = f"{root}/{fn}"
            if rel in claimed:
                continue
            try:
                s = _load(rel).get("summary", {})
            except Exception:
                continue
            if os.path.basename(s.get("benchmark") or "") == \
               os.path.basename(SPEC.ARM["benchmark"]):
                found.append((rel, s.get("model"), s.get("prompt_style"),
                              s.get("total_trials")))
    return found


def ladder_rows():
    """
    One row per observer x foil bank, for the difficulty axis.

    The banks are matched: same trial ids, same study scenes, same answer key,
    only the foils differ. So an observer's accuracy across banks is a
    within-items psychometric curve in metres, and the trials it is averaged
    over are the same trials at every rung.

    A missing bank for an observer is skipped, not zero-filled -- the ladder is
    filled in as runs land and a hole must not read as a floor.
    """
    primary = {b["key"]: b for b in SPEC.BANKS}[SPEC.PRIMARY_BANK]
    out = []
    for obs in SPEC.OBSERVERS:
        for bank in SPEC.BANKS:
            path = obs["path"].replace(primary["tag"], bank["tag"])
            # The human export is named by session, not by bank, so the
            # substitution is a no-op and the same 40 trials would be counted
            # once per rung -- a flat line across the ladder that is an artefact
            # of the filename, not a measurement. A non-primary bank has to
            # actually name itself in the path.
            if bank["key"] != SPEC.PRIMARY_BANK and path == obs["path"]:
                continue
            full = os.path.join(SPEC.REPO, path)
            if not os.path.exists(full):
                continue
            d = _load(path)
            trials = d["results"]
            if not trials:
                continue
            cell = collections.defaultdict(lambda: [0, 0])
            for t in trials:
                cell[int(t["delta"]) > 0][1] += 1
                cell[int(t["delta"]) > 0][0] += bool(t["is_correct"])
            k0, n0 = cell[False]
            kr, nr = cell[True]
            expect = SPEC.ARM["n_trials"]
            out.append(dict(
                key=obs["key"], label=obs["label"], family=obs["family"],
                access=obs["access"], bank=bank["key"],
                complete=int(len(trials) >= expect
                             and not d["summary"].get("in_progress")),
                median_foil_m=bank["median_foil_m"], pct_lo=bank["pct"][0],
                pct_hi=bank["pct"][1], path=path, n=len(trials),
                acc=sum(bool(t["is_correct"]) for t in trials) / len(trials),
                acc_d0=(k0 / n0 if n0 else None), n_d0=n0,
                acc_drot=(kr / nr if nr else None), n_drot=nr))
    return out


def setsize_rows():
    """
    One row per observer x landmark-count bank.

    These runs are not on the locked benchmark, so they are collated separately
    and never enter `observers.csv` or the headline table. The banks are not
    matched to each other the way the foil ladder is: they are different scenes
    with different landmark counts, so this is a between-banks comparison and
    the only thing held fixed is the sampling rule and every render setting.

    Files are found by model slug rather than by roster path, because a set-size
    run has no entry in `OBSERVERS` and inventing one would put it in the
    headline table.
    """
    out = []
    rdir = os.path.join(SPEC.REPO, "results")
    if not os.path.isdir(rdir):
        return out
    for bank in SPEC.SETSIZE_BANKS:
        for fn in sorted(os.listdir(rdir)):
            if bank["tag"] not in fn or not fn.endswith(".json"):
                continue
            if "smoke" in fn:
                continue
            rel = f"results/{fn}"
            try:
                d = _load(rel)
            except Exception:
                continue
            trials = d.get("results") or []
            if not trials:
                continue
            model = (d.get("summary") or {}).get("model", fn)
            cell = collections.defaultdict(lambda: [0, 0])
            for t in trials:
                rot = int(t["delta"]) > 0
                cell[rot][1] += 1
                cell[rot][0] += bool(t["is_correct"])
            k0, n0 = cell[False]
            kr, nr = cell[True]
            unparsed = sum(1 for t in trials if t.get("model_choice") is None)
            out.append(dict(
                model=model, bank=bank["key"], n_landmarks=bank["n_landmarks"],
                median_foil_m=bank["median_foil_m"],
                anchor_separation_m=bank["anchor_separation_m"],
                complete=int(len(trials) >= SPEC.ARM["n_trials"]
                             and not (d.get("summary") or {}).get("in_progress")),
                n=len(trials), n_unparsed=unparsed,
                acc=sum(bool(t["is_correct"]) for t in trials) / len(trials),
                acc_d0=(k0 / n0 if n0 else None), n_d0=n0,
                acc_drot=(kr / nr if nr else None), n_drot=nr,
                path=rel))
    return out


def write(runs):
    os.makedirs(OUT_DIR, exist_ok=True)

    # runs.json -- provenance. Everything a caption or a reviewer needs.
    prov = []
    for r in runs:
        prov.append({
            "key": r["key"], "label": r["label"], "family": r["family"],
            "access": r["access"], "params_b": r.get("params_b"),
            "active_b": r.get("active_b"), "role": r.get("role", "primary"),
            "path": r["path"], "present": r.get("present", False),
            "eligible": r.get("eligible", False),
            "blocking_deviations": r.get("blocking", []),
            "provenance_gaps": r.get("gaps", []),
            "n_trials": r.get("n_trials"), "n_answered": r.get("n_answered"),
            "accuracy": r.get("accuracy"),
            "run_config": r.get("run_config"),
            "api_usage": r.get("api_usage"),
        })
    meta = {
        "arm": SPEC.ARM,
        "benchmark_sha256_12_on_disk": _bench_digest(),
        "trial_identity_problems": check_trial_identity(runs) + check_task_items(runs),
        "unclaimed_result_files": [
            {"path": p, "model": m, "prompt": pr, "n": n}
            for p, m, pr, n in unclaimed()],
        "runs": prov,
    }
    with open(f"{OUT_DIR}/runs.json", "w") as f:
        json.dump(meta, f, indent=2)

    # observers.csv -- one row per observer, the headline table's source.
    with open(f"{OUT_DIR}/observers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "family", "access", "params_b", "role",
                    "eligible", "n", "acc", "acc_lo", "acc_hi",
                    "acc_d0", "n_d0", "acc_drot", "n_drot",
                    "dprime_d0", "vii_180", "prior_best_fixed",
                    "prior_distribution", "prior_rot", "excess_rot",
                    "top_choice_share_rot", "blocking", "gaps"])
        for r in runs:
            if not r.get("present"):
                continue
            bd = by_delta(r)
            k0, n0 = bd.get(0, (0, 0))
            krot = sum(k for d, (k, n) in bd.items() if d > 0)
            nrot = sum(n for d, (k, n) in bd.items() if d > 0)
            p, lo, hi = S.wilson(sum(bool(t["is_correct"]) for t in r["trials"]),
                                 len(r["trials"]))
            curve, d0 = vii_curve(r)
            pr = priors(r)
            prr = priors(r, rotated_only=True)
            w.writerow([
                r["key"], r["label"], r["family"], r["access"],
                r.get("params_b") or "", r.get("role", "primary"),
                int(bool(r.get("eligible"))), len(r["trials"]),
                f"{p:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                f"{k0 / n0:.4f}" if n0 else "", n0,
                f"{krot / nrot:.4f}" if nrot else "", nrot,
                f"{d0:.3f}" if d0 is not None else "",
                f"{curve.get(180):.3f}" if curve.get(180) is not None else "",
                f"{pr['best_fixed']:.4f}", f"{pr['distribution']:.4f}",
                f"{prr['distribution']:.4f}" if prr else "",
                (f"{krot / nrot - prr['distribution']:+.4f}"
                 if prr and nrot else ""),
                (f"{max(prr['model_prior'].values()):.4f}" if prr else ""),
                "; ".join(r.get("blocking", [])),
                "; ".join(r.get("gaps", [])),
            ])

    # cells.csv -- one row per observer x mode x delta, for every figure.
    with open(f"{OUT_DIR}/cells.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "family", "access", "eligible",
                    "mode", "delta", "k", "n", "acc", "lo", "hi"])
        for r in runs:
            if not r.get("present"):
                continue
            for (mode, delta), (k, n) in sorted(cells(r).items()):
                p, lo, hi = S.wilson(k, n)
                w.writerow([r["key"], r["label"], r["family"], r["access"],
                            int(bool(r.get("eligible"))), mode, delta, k, n,
                            f"{p:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    # ladder.csv -- one row per observer x foil bank.
    lad = ladder_rows()
    with open(f"{OUT_DIR}/ladder.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["key", "label", "family", "access", "bank", "complete",
                "median_foil_m",
                "pct_lo", "pct_hi", "n", "acc", "acc_d0", "n_d0",
                "acc_drot", "n_drot", "path"]
        w.writerow(cols)
        for r in lad:
            w.writerow(["" if r[c] is None else r[c] for c in cols])

    # setsize.csv -- one row per observer x landmark-count bank. Separate from
    # ladder.csv because these are different scenes, not the same trials with
    # different foils.
    ss = setsize_rows()
    with open(f"{OUT_DIR}/setsize.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["model", "bank", "n_landmarks", "median_foil_m",
                "anchor_separation_m", "complete", "n", "n_unparsed", "acc",
                "acc_d0", "n_d0", "acc_drot", "n_drot", "path"]
        w.writerow(cols)
        for r in ss:
            w.writerow(["" if r[c] is None else r[c] for c in cols])

    # trials.csv -- the long form, with each trial's foil geometry joined on.
    geo = trial_geometry()
    with open(f"{OUT_DIR}/trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "trial_id", "mode", "delta", "correct_choice",
                    "model_choice", "is_correct", "min_foil_distance_m"])
        for r in runs:
            if not r.get("present"):
                continue
            for t in r["trials"]:
                g = geo.get(t["trial_id"], {})
                d = g.get("min_foil_distance_m")
                w.writerow([r["key"], t["trial_id"], t["mode"], int(t["delta"]),
                            t["correct_choice"], t.get("model_choice"),
                            int(bool(t["is_correct"])),
                            f"{d:.3f}" if d is not None else ""])
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report problems and exit non-zero; write nothing")
    args = ap.parse_args()

    runs = load_runs()
    meta = None if args.check else write(runs)

    on_disk = _bench_digest()
    problems = []
    if on_disk != SPEC.ARM["benchmark_sha256_12"]:
        problems.append(f"benchmark on disk is {on_disk}, spec says "
                        f"{SPEC.ARM['benchmark_sha256_12']}")
    problems += check_trial_identity(runs) + check_task_items(runs)

    print(f"arm      {SPEC.ARM['name']}  ({SPEC.ARM['benchmark']}, "
          f"sha {on_disk}, n={SPEC.ARM['n_trials']}, "
          f"prompt={SPEC.ARM['prompt_style_model']}, "
          f"max_tokens={SPEC.ARM['max_tokens']})")
    print()
    print(f"{'observer':28s} {'n':>4s} {'acc':>6s}  {'d0':>5s}  {'rot':>6s}  notes")
    print("-" * 92)
    for r in runs:
        if not r.get("present"):
            print(f"{r['label']:28s} {'--':>4s} {'--':>6s}  {'--':>5s}  "
                  f"{'--':>6s}  MISSING {r['path']}")
            continue
        bd = by_delta(r)
        k0, n0 = bd.get(0, (0, 0))
        krot = sum(k for d, (k, n) in bd.items() if d > 0)
        nrot = sum(n for d, (k, n) in bd.items() if d > 0)
        print(f"{r['label']:28s} {len(r['trials']):4d} "
              f"{r['accuracy']:6.1%}  "
              f"{(k0 / n0 if n0 else 0):5.0%}  "
              f"{(krot / nrot if nrot else 0):6.1%}  "
              f"{'; '.join(r['blocking']) or 'locked config'}"
              f"{'  [' + '; '.join(r['gaps']) + ']' if r['gaps'] else ''}")

    unc = unclaimed()
    if unc:
        print(f"\n{len(unc)} result file(s) on this benchmark are in no roster row:")
        for p, m, pr, n in unc:
            print(f"  {p}  ({m}, {pr}, n={n})")

    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("  " + p)
    if not args.check:
        lad = collections.Counter(r["bank"] for r in ladder_rows())
    print(f"\nladder: " + ", ".join(f"{b['key']} {lad.get(b['key'], 0)} runs"
                                    for b in SPEC.BANKS))
    ss = collections.Counter(r["bank"] for r in setsize_rows() if r["complete"])
    if ss:
        print("set size: " + ", ".join(f"N={b['n_landmarks']} {ss.get(b['key'], 0)} runs"
                                       for b in SPEC.SETSIZE_BANKS))
    print(f"wrote paper/runs.json, observers.csv, cells.csv, trials.csv, "
          f"ladder.csv, setsize.csv")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
