"""
The same task the models take, for people.

    python bench/build_human_task.py --out human_task
    python bench/build_human_task.py --benchmark data/vlm_benchmark_4afc.json \
        --per_cell 2 --out human_task_4afc

Produces a directory holding `index.html`, an `images/` folder, and
`task.json`. Open the HTML in any browser -- no server, no install. At the end
the participant saves a JSON file.

**The point of this script is that the comparison is exact.** Human and model
must not differ in anything except who is answering, so:

* The trials are **drawn from the same benchmark file the models run**, keeping
  each trial's id, its images, its option order and its `correct_choice`. The
  slice is written back out as `task.json`, and `bench/evaluate_vlm.py
  --benchmark <that file>` runs a model on exactly those trials.
* The instructions are `get_prompt_text(n, "neutral")`, read from
  `evaluate_vlm.py` rather than retyped, so the two cannot drift apart. Use
  `--prompt_style neutral` for the model arm.
* The output is written in the **same schema as a model result file**, so
  `bench/analyze_vii.py` reads a person with no changes and puts them on the
  same VII axis. `model_choice` holds the person's choice; `latency` is their
  response time in seconds.
* No feedback is given, because the models get none. Getting a trial right must
  not change how the next one is answered.

Trial *order* is shuffled per participant, which the models' fixed order is not.
That is deliberate: a person doing 50 trials in a fixed order accumulates
practice effects aligned with condition, and a model with no memory between
calls cannot. The trial ids record which is which, so the pairing survives.
"""

import argparse
import base64
import importlib.util
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_evaluate_vlm():
    """Import for the prompt text alone; the module pulls in torch lazily."""
    spec = importlib.util.spec_from_file_location(
        "evaluate_vlm", os.path.join(HERE, "evaluate_vlm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def balanced_slice(trials, per_cell, n_options, seed):
    """
    `per_cell` trials from every (mode, delta), with the answer position
    balanced inside the slice.

    Taking the first k of each cell would inherit whatever answer-position
    pattern happened to sit at the front of it. The whole benchmark is balanced
    per cell; a subsample is not balanced unless it is made so.
    """
    rng = random.Random(seed)
    by_cell = {}
    for t in trials:
        by_cell.setdefault((t["mode"], t["delta"]), []).append(t)

    out = []
    for cell, pool in sorted(by_cell.items()):
        by_pos = {}
        for t in pool:
            by_pos.setdefault(t["correct_choice"], []).append(t)
        for v in by_pos.values():
            rng.shuffle(v)
        picked, i = [], 0
        # round-robin over answer positions until the cell quota is filled
        while len(picked) < per_cell:
            positions = [p for p in sorted(by_pos) if len(by_pos[p]) > i]
            if not positions:
                raise ValueError(f"{cell}: only {len(picked)} of {per_cell} trials available")
            for p in positions:
                if len(picked) == per_cell:
                    break
                picked.append(by_pos[p][i])
            i += 1
        out.extend(picked)
    return out


def stage_images(trials, out_dir, embed, quality):
    """
    Copy every image the slice needs into `out_dir/images` under a flat name,
    and rewrite the trials to point at it. Returns {flat_name: data-uri} when
    embedding.

    Flattening matters: the bank nests by mode and scene, and two modes hold a
    file of the same name. `a000_A_az000.png` alone is ambiguous.
    """
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    blobs, seen = {}, {}

    def stage(path, mode):
        if path in seen:
            return seen[path]
        flat = f"{mode}__{os.path.basename(path)}"
        dst = os.path.join(img_dir, flat)
        if embed:
            from PIL import Image
            im = Image.open(path).convert("RGB")
            im.save(dst.rsplit(".", 1)[0] + ".jpg", "JPEG", quality=quality)
            dst = dst.rsplit(".", 1)[0] + ".jpg"
            flat = os.path.basename(dst)
            with open(dst, "rb") as f:
                blobs[flat] = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        elif not os.path.exists(dst):
            shutil.copy2(path, dst)
        seen[path] = flat
        return flat

    for t in trials:
        t["study_web"] = stage(t["study_image"], t["mode"])
        for o in t["options"]:
            o["web"] = stage(o["image_path"], t["mode"])
    return blobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="data/vlm_benchmark_2afc.json",
                    help="the same file the models are run on")
    ap.add_argument("--out", default="human_task")
    ap.add_argument("--per_cell", type=int, default=2,
                    help="trials per (mode x delta); 2 gives 50 trials, ~6 min")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--embed", action="store_true",
                    help="inline the images as JPEG data URIs -- one shareable "
                         "file, but the model arm must then run on the same JPEGs")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality when embedding")
    ap.add_argument("--title", default="Which one is the same place?")
    args = ap.parse_args()

    blob = json.load(open(args.benchmark))
    n_options = blob["n_options"]
    trials = balanced_slice(blob["trials"], args.per_cell, n_options, args.seed)

    os.makedirs(args.out, exist_ok=True)
    blobs = stage_images(trials, args.out, args.embed, args.quality)

    instructions = _load_evaluate_vlm().get_prompt_text(n_options, style="neutral")
    # The models are told to answer "Final Answer: Option X"; a person clicks a
    # button. Strip the output-format lines and nothing else.
    human_instructions = instructions.split("Answer on the last line")[0].strip()

    # `task.json` is a benchmark file: run the models on it, not on the full set.
    slim = [{k: v for k, v in t.items() if k not in ("study_web",)} for t in trials]
    for t in slim:
        t["options"] = [{k: v for k, v in o.items() if k != "web"} for o in t["options"]]
    with open(os.path.join(args.out, "task.json"), "w") as f:
        json.dump({**{k: v for k, v in blob.items() if k != "trials"},
                   "total_trials": len(slim),
                   "derived_from": os.path.basename(args.benchmark),
                   "per_cell": args.per_cell, "slice_seed": args.seed,
                   "trials": slim}, f, indent=2)

    payload = {
        "n_options": n_options,
        "benchmark": os.path.basename(args.benchmark),
        "embedded": bool(args.embed),
        "instructions": human_instructions,
        "trials": [{
            "trial_id": t["id"], "mode": t["mode"], "delta": t["delta"],
            "correct_choice": t["correct_choice"],
            "study": t["study_web"],
            "options": [o["web"] for o in t["options"]],
        } for t in trials],
        "blobs": blobs,
    }

    html = TEMPLATE.replace("__TITLE__", args.title).replace(
        "__PAYLOAD__", json.dumps(payload))
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(html)

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(args.out) for f in fs)
    print(f"{len(trials)} trials ({n_options}AFC), "
          f"{len(set(sum([[t['study_web']] + [o['web'] for o in t['options']] for t in trials], [])))} images")
    print(f"wrote {args.out}/index.html  and  {args.out}/task.json")
    print(f"directory is {size/1e6:.1f} MB")
    print(f"\nmatched model arm:\n"
          f"  python bench/evaluate_vlm.py --benchmark {args.out}/task.json \\\n"
          f"      --model <id> --prompt_style neutral --out results/<name>.json")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--paper:#f6f4ee;--sheet:#fff;--ink:#1a1f1d;--dim:#5b625c;--rule:#ddd8cc;
      --accent:#2f6690;--ok:#4f6a54;--warn:#9e2a2b;--focus:#2f6690}
@media (prefers-color-scheme:dark){:root{--paper:#14181a;--sheet:#1b2023;--ink:#e8e6df;
      --dim:#9aa39c;--rule:#2e353a;--accent:#7fb3d5;--ok:#93b79a;--warn:#e08a83;--focus:#7fb3d5}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px 20px 64px}
h1{font-size:1.6rem;margin:0 0 .5em;letter-spacing:-.01em}
h2{font-size:1.1rem;margin:1.6em 0 .5em}
p{margin:0 0 1em;max-width:66ch}
button{font:inherit;cursor:pointer;border-radius:6px;border:1px solid var(--rule);
  background:var(--sheet);color:var(--ink);padding:10px 18px}
button:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.primary:disabled{opacity:.45;cursor:not-allowed}
.card{background:var(--sheet);border:1px solid var(--rule);border-radius:8px;padding:22px}
.pre{white-space:pre-wrap;font-size:.97rem;color:var(--dim)}
label{display:block;margin:14px 0 6px;font-size:.9rem;color:var(--dim)}
input{font:inherit;padding:9px 12px;border:1px solid var(--rule);border-radius:6px;
  background:var(--paper);color:var(--ink);width:280px;max-width:100%}
/* --- trial ------------------------------------------------------------ */
#bar{height:4px;background:var(--rule);border-radius:2px;overflow:hidden;margin-bottom:6px}
#bar i{display:block;height:100%;background:var(--accent);width:0;transition:width .2s}
.count{font-size:.82rem;color:var(--dim);margin-bottom:16px;
  font-variant-numeric:tabular-nums;display:flex;justify-content:space-between}
.study{text-align:center;margin-bottom:8px}
.study img{max-width:min(100%,560px);border-radius:6px;border:1px solid var(--rule)}
.tag{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);
  display:block;margin-bottom:6px}
.opts{display:grid;gap:14px;margin-top:14px;
  grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.opt{padding:0;overflow:hidden;border:2px solid var(--rule);background:var(--sheet);
  border-radius:8px;display:block;width:100%}
.opt:hover{border-color:var(--accent)}
.opt img{display:block;width:100%}
.opt .n{padding:9px 12px;font-weight:600;font-size:.9rem;text-align:left;
  border-top:1px solid var(--rule)}
.opt kbd{font:inherit;background:var(--paper);border:1px solid var(--rule);
  border-radius:4px;padding:1px 6px;margin-right:8px;font-size:.85em}
/* --- done ------------------------------------------------------------- */
textarea{width:100%;height:190px;font-family:ui-monospace,SFMono-Regular,monospace;
  font-size:.78rem;border:1px solid var(--rule);border-radius:6px;padding:12px;
  background:var(--paper);color:var(--ink)}
.row{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}
.note{font-size:.88rem;color:var(--dim)}
[hidden]{display:none!important}
</style></head><body>
<div class="wrap">

<section id="intro" class="card">
  <h1>__TITLE__</h1>
  <p class="pre" id="instr"></p>
  <p>Each option is photographed from the <b>same direction</b>, so you cannot
  tell them apart by the camera angle &mdash; only by the arrangement of the
  landmarks. Some trials are easy and some are close to impossible. Answer as
  best you can; there is no feedback, and guessing when unsure is expected.</p>
  <p><b>Use the number keys</b> to answer, or click. About <span id="mins"></span> minutes.</p>
  <label for="pid">Participant ID (anything that is not your name)</label>
  <input id="pid" autocomplete="off" placeholder="e.g. p01">
  <div class="row"><button class="primary" id="start">Start</button></div>
</section>

<section id="task" hidden>
  <div id="bar"><i></i></div>
  <div class="count"><span id="prog"></span><span id="hint"></span></div>
  <div class="study"><span class="tag">Study image</span><img id="study" alt="the place to remember"></div>
  <div class="opts" id="opts"></div>
</section>

<section id="done" hidden class="card">
  <h1>Finished &mdash; thank you</h1>
  <p id="summary" class="note"></p>
  <p><b>Send this file back.</b> If the download does not start, copy the text
  below instead.</p>
  <div class="row">
    <button class="primary" id="dl">Download results</button>
    <button id="copy">Copy to clipboard</button>
  </div>
  <textarea id="out" readonly></textarea>
</section>

</div>
<script>
const DATA = __PAYLOAD__;
const $ = s => document.querySelector(s);
const src = name => DATA.embedded ? DATA.blobs[name] : "images/" + name;

document.getElementById("instr").textContent = DATA.instructions;
$("#mins").textContent = Math.max(3, Math.round(DATA.trials.length * 7 / 60));

// Trial order is shuffled per participant; the recorded trial_id keeps the
// pairing with the model runs, which see a fixed order.
const order = DATA.trials.map((t, i) => i);
for (let i = order.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [order[i], order[j]] = [order[j], order[i]];
}

let k = 0, t0 = 0, pid = "anon";
const results = [];

function render() {
  const t = DATA.trials[order[k]];
  $("#prog").textContent = `Trial ${k + 1} of ${DATA.trials.length}`;
  $("#hint").textContent = `press 1–${t.options.length}`;
  $("#bar i").style.width = (100 * k / DATA.trials.length) + "%";
  $("#study").src = src(t.study);

  const box = $("#opts");
  box.innerHTML = "";
  t.options.forEach((name, i) => {
    const b = document.createElement("button");
    b.className = "opt"; b.type = "button";
    b.innerHTML = `<img alt="option ${i + 1}"><span class="n"><kbd>${i + 1}</kbd>Option ${i + 1}</span>`;
    b.querySelector("img").src = src(name);
    b.addEventListener("click", () => answer(i + 1));
    box.appendChild(b);
  });
  t0 = performance.now();
}

function answer(choice) {
  const t = DATA.trials[order[k]];
  results.push({
    trial_id: t.trial_id, mode: t.mode, delta: t.delta,
    correct_choice: t.correct_choice,
    model_choice: choice,               // the schema analyze_vii.py reads
    is_correct: choice === t.correct_choice,
    latency: (performance.now() - t0) / 1000,
    reply: String(choice), error: null,
    presentation_index: k
  });
  k++;
  if (k < DATA.trials.length) render(); else finish();
}

document.addEventListener("keydown", e => {
  if ($("#task").hidden) return;
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= DATA.trials[order[k]].options.length) { e.preventDefault(); answer(n); }
});

function finish() {
  $("#task").hidden = true; $("#done").hidden = false;
  const acc = results.filter(r => r.is_correct).length / results.length;
  const byDelta = {}, byMode = {};
  for (const r of results) {
    (byDelta[r.delta] = byDelta[r.delta] || []).push(r.is_correct);
    (byMode[r.mode] = byMode[r.mode] || []).push(r.is_correct);
  }
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const dist = {};
  for (const r of results) dist[r.model_choice] = (dist[r.model_choice] || 0) + 1;

  const blob = {
    summary: {
      model: "human:" + pid, benchmark: DATA.benchmark,
      prompt_style: "neutral", n_options: DATA.n_options,
      chance_level: 1 / DATA.n_options,
      total_trials: results.length, overall_accuracy: acc,
      choice_distribution: dist,
      by_mode_delta: Object.fromEntries(Object.entries(byDelta).map(
        ([d, v]) => [d, {accuracy: mean(v), total: v.length}])),
      user_agent: navigator.userAgent,
      finished_at: new Date().toISOString()
    },
    results
  };
  const text = JSON.stringify(blob, null, 2);
  $("#out").value = text;
  $("#summary").textContent =
    `${results.length} trials, ${(100 * acc).toFixed(1)}% correct ` +
    `(chance ${(100 / DATA.n_options).toFixed(0)}%). ` +
    Object.entries(byDelta).sort((a, b) => a[0] - b[0])
      .map(([d, v]) => `${d}° ${(100 * mean(v)).toFixed(0)}%`).join("  ·  ");

  $("#dl").onclick = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], {type: "application/json"}));
    a.download = `human_${pid}_${DATA.n_options}afc.json`;
    document.body.appendChild(a); a.click(); a.remove();
  };
  $("#copy").onclick = async () => {
    try { await navigator.clipboard.writeText(text); $("#copy").textContent = "Copied"; }
    catch (e) { $("#out").select(); $("#copy").textContent = "Select all and copy"; }
  };
}

$("#start").onclick = () => {
  pid = ($("#pid").value || "anon").replace(/[^A-Za-z0-9_.-]/g, "") || "anon";
  $("#intro").hidden = true; $("#task").hidden = false;
  render();
};
</script></body></html>
"""

if __name__ == "__main__":
    main()
