# Superseded and smoke-test runs

Nothing here is in the paper. `bench/collate.py` reports any file in
`results/` that matches the locked arm but is in no roster row, because a
leftover `_n1` smoke test or a superseded `_n500` sitting next to the real run
is how a dead result gets back into a figure. Moving them here keeps
`results/` equal to what the paper uses.

They are kept rather than deleted: the smoke tests are the evidence that a
model's reply format was checked before money was spent on it, and the
superseded runs are the evidence for what changed and why.

| file | why it is not in the paper |
|---|---|
| `or_*_n1.json`, `or_*_n2.json`, `or_*_n4.json` | smoke tests: one to four trials to prove the request format, image encoding and answer parsing before a paid run |
| `or_qwen_qwen3-vl-235b-a22b-instruct_cot_anyview_4afc_hard_n500.json` | 500 trials, superseded by the 100-trial slice every other observer ran; keeping it would break the paired comparison |
| `or_google_gemini-3.8-flash_cot_anyview_hardfoils_n100.json` | written before the output filename carried the benchmark name; same content as the `_4afc_hard_` file, kept as the collision's evidence |
| `human_p01_4afc_partial{10,20,30}.json` | earlier autosaves of the same session; `partial40` supersedes them |
