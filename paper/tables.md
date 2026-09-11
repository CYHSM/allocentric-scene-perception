### Table 1 — 4AFC, hard foils, chance 25%, same 100 trials

| Observer            | Params | Accuracy | 95% CI     | Δ=0 gate | Δ≥45    | own prior | vs prior | Notes |
|---------------------|--------|----------|------------|----------|---------|-----------|----------|-------|
| Human               | —      | **86%**  | [78%, 91%] | **100%** | **82%** | 26%       | +57%     |       |
| InternVL3.5-4B      | 4B     | 17%      | [11%, 26%] | 35%      | 12%     | 21%       | -9%      |       |
| InternVL3.5-2B      | 2B     | 21%      | [14%, 30%] | 25%      | 20%     | 24%       | -4%      |       |
| InternVL3.5-1B      | 1B     | 23%      | [16%, 32%] | 35%      | 20%     | 19%       | +1%      |       |
| InternVL3.5-38B     | 38B    | 28%      | [20%, 37%] | 65%      | 19%     | 21%       | -2%      |       |
| InternVL3.5-8B      | 8B     | 28%      | [20%, 37%] | 55%      | 21%     | 25%       | -4%      |       |
| InternVL3.5-30B-A3B | 30B    | 29%      | [21%, 39%] | 55%      | 22%     | 27%       | -4%      |       |
| Qwen2.5-VL-32B      | 32B    | 29%      | [21%, 39%] | 75%      | 18%     | 24%       | -7%      |       |
| Qwen2.5-VL-3B       | 3B     | 29%      | [21%, 39%] | 30%      | 29%     | 27%       | +1%      |       |
| Qwen3-VL-235B (t)   | 235B   | 30%      | [22%, 40%] | 75%      | 19%     | 26%       | -8%      |       |
| InternVL3.5-14B     | 14B    | 31%      | [23%, 41%] | 50%      | 26%     | 27%       | -1%      |       |
| Qwen2.5-VL-72B      | 72B    | 31%      | [23%, 41%] | 75%      | 20%     | 26%       | -6%      |       |
| Qwen3-VL-235B (i)   | 235B   | 33%      | [25%, 43%] | 90%      | 19%     | 22%       | -3%      |       |
| InternVL3-38B       | 38B    | 34%      | [25%, 44%] | 85%      | 21%     | 24%       | -2%      |       |
| Qwen2.5-VL-7B       | 7B     | 36%      | [27%, 46%] | 55%      | 31%     | 27%       | +4%      |       |
| Gemini 3.8 Flash    | —      | 44%      | [35%, 54%] | 65%      | **39%** | 24%       | +15%     |       |
| GPT-5.6 Luna        | —      | **45%**  | [36%, 55%] | **100%** | 31%     | 25%       | +7%      |       |

*own prior* = what this observer's own answer distribution scores on the rotated trials with the images removed. *vs prior* is the Δ≥45 column minus it, and is the number to read: the answer key of the 100-trial slice is 11/22/26/21 on rotated trials rather than balanced, so raw accuracy carries a few points of position luck.

### Table 2 — rotated trials (Δ≥45) by stimulus mode

| Observer                 | c0      | c1      | c2      | c3      | c4      |
|--------------------------|---------|---------|---------|---------|---------|
| Human                    | **94%** | **75%** | **81%** | **81%** | **81%** |
| InternVL3.5-4B           | 19%     | 6%      | 25%     | **0%**  | 12%     |
| InternVL3.5-2B           | 12%     | **12%** | 19%     | 25%     | 31%     |
| InternVL3.5-1B           | **12%** | 19%     | 25%     | 25%     | 19%     |
| InternVL3.5-38B          | 19%     | **6%**  | 19%     | 19%     | 31%     |
| InternVL3.5-8B           | 19%     | **6%**  | 19%     | 44%     | 19%     |
| InternVL3.5-30B-A3B      | **6%**  | 19%     | 19%     | 31%     | 38%     |
| Qwen2.5-VL-32B           | 12%     | **0%**  | 31%     | 6%      | 38%     |
| Qwen2.5-VL-3B            | 38%     | **12%** | 31%     | 38%     | 25%     |
| Qwen3-VL-235B (t)        | 12%     | **12%** | 19%     | 19%     | 31%     |
| InternVL3.5-14B          | 25%     | **6%**  | 31%     | 38%     | 31%     |
| Qwen2.5-VL-72B           | 19%     | 12%     | **12%** | 31%     | 25%     |
| Qwen3-VL-235B (i)        | 12%     | **6%**  | 19%     | 25%     | 31%     |
| InternVL3-38B            | 25%     | 12%     | **12%** | 38%     | 19%     |
| Qwen2.5-VL-7B            | 50%     | 19%     | 25%     | **19%** | 44%     |
| Gemini 3.8 Flash         | 38%     | 38%     | 44%     | 44%     | **31%** |
| GPT-5.6 Luna             | 38%     | 19%     | 50%     | 31%     | **19%** |
| **All 16 models pooled** | **22%** | **13%** | **25%** | **27%** | **28%** |

\* partial run; see paper/runs.json for what is missing.

### Table 3 — accuracy by viewpoint change

| Observer              | Δ=0°    | Δ=45°   | Δ=90°   | Δ=135°  | Δ=180°  |
|-----------------------|---------|---------|---------|---------|---------|
| Human                 | 100%    | 90%     | 80%     | 85%     | 75%     |
| InternVL3.5-4B        | 35%     | 20%     | 20%     | 5%      | 5%      |
| InternVL3.5-2B        | 25%     | **10%** | 25%     | 20%     | 25%     |
| InternVL3.5-1B        | 35%     | 15%     | 30%     | 25%     | **10%** |
| InternVL3.5-38B       | 65%     | 15%     | 15%     | 25%     | 20%     |
| InternVL3.5-8B        | 55%     | 40%     | 20%     | **5%**  | 20%     |
| InternVL3.5-30B-A3B   | 55%     | 35%     | 15%     | 25%     | 15%     |
| Qwen2.5-VL-32B        | 75%     | 30%     | 15%     | **0%**  | 25%     |
| Qwen2.5-VL-3B         | 30%     | 50%     | **15%** | 25%     | 25%     |
| Qwen3-VL-235B (t)     | 75%     | 30%     | 25%     | 10%     | 10%     |
| InternVL3.5-14B       | 50%     | 30%     | 25%     | **20%** | 30%     |
| Qwen2.5-VL-72B        | 75%     | 40%     | 15%     | **10%** | 15%     |
| Qwen3-VL-235B (i)     | 90%     | 40%     | 15%     | **5%**  | 15%     |
| InternVL3-38B         | 85%     | 40%     | **5%**  | 15%     | 25%     |
| Qwen2.5-VL-7B         | 55%     | 45%     | **15%** | 25%     | 40%     |
| Gemini 3.8 Flash      | 65%     | 45%     | 50%     | **15%** | 45%     |
| GPT-5.6 Luna          | 100%    | 35%     | 35%     | **10%** | 45%     |
| **All models pooled** | **61%** | **32%** | **21%** | **15%** | **23%** |

Pooled, the floor is Δ=135° — below chance — and accuracy *recovers* at a half turn. 7 of 16 models have their own worst cell at 135°, and 4 tie for worst across two viewpoints.

### Table 5 — Qwen2.5-VL-32B across instruction styles (rotated trials, chance 25%)

| Instruction                           | Rotated accuracy | 95% CI     | Unparsed |
|---------------------------------------|------------------|------------|----------|
| chain of thought, viewpoint unstated  | 18%              | [11%, 27%] | 0        |
| chain of thought, viewpoint asserted  | 14%              | [8%, 23%]  | 0        |
| instructed to mentally rotate         | 22%              | [15%, 33%] | 0        |
| instructed to pick an anchor landmark | 22%              | [15%, 33%] | 0        |
| instructed to imagine a plan view     | 24%              | [16%, 34%] | 0        |
| instructed to eliminate alternatives  | 19%              | [12%, 29%] | 0        |

### Table 4 — rotated accuracy at three foil difficulties

| Observer            | hard  (7 m) | mid  (31 m) | easy  (43 m) |
|---------------------|-------------|-------------|--------------|
| Human               | 82%         | —           | —            |
| InternVL3.5-4B      | 12%         | —           | —            |
| InternVL3.5-2B      | 20%         | —           | —            |
| InternVL3.5-1B      | 20%         | —           | —            |
| InternVL3.5-38B     | 19%         | —           | —            |
| InternVL3.5-8B      | 21%         | —           | —            |
| InternVL3.5-30B-A3B | 22%         | —           | —            |
| Qwen2.5-VL-32B      | 18%         | 28%         | 24%          |
| Qwen2.5-VL-3B       | 29%         | 25%         | 24%          |
| Qwen3-VL-235B (t)   | 19%         | —           | —            |
| InternVL3.5-14B     | 26%         | —           | —            |
| Qwen2.5-VL-72B      | 20%         | 24%         | 31%          |
| Qwen3-VL-235B (i)   | 19%         | 26%         | 24%          |
| InternVL3-38B       | 21%         | 18%         | 24%          |
| Qwen2.5-VL-7B       | 31%         | 25%         | 29%          |
| Gemini 3.8 Flash    | 39%         | 85%         | 82%          |
| GPT-5.6 Luna        | 31%         | 55%         | 49%          |

The same trials throughout: same study scene, same azimuths, same answer slot. Only the foils differ.


11 run(s) carry provenance gaps; see paper/runs.json.

wrote paper/table1.tex, table2.tex, table3.tex
