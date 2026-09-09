# CLUSTER.md — dgx2 (primary), Marvin (Uni Bonn), Leonardo (CINECA)

Everything learned by actually running jobs on these machines. Every number here was
read off the live system, not from vendor docs — where the official documentation
disagreed with the machine, the machine won and the doc is corrected below.

**Scope.** This is operational knowledge: access, storage, environment, submission,
and the specific ways jobs fail silently. Scientific results live in the branch README.

> **Adapted for `allocentric-scene-perception`.** This file was written for `tevol`
> (a text-only LLM sweep). The machine-level facts carry over unchanged and are the
> reason it is worth keeping. What is *different* for this project is collected in
> §0.1 and §10: this repo runs **vision**-language models over rendered PNGs, so it
> needs image data staged on the cluster, needs far more disk than a text sweep, and
> hits a storage trap that `tevol` never did (§10.1). Sections that are purely about
> `tevol`'s SLURM sweeps (§6.2, §7.2, §9) have not been re-verified here.

---

## 0. Which machine should I use?

| | **dgx2** | Marvin | Leonardo |
|---|---|---|---|
| **GPU** | **A100-SXM4 80GB ×8** | A40 48GB ×8/node | A100 64GB ×4/node |
| **Measured s/program** (sweep, n=64) | **0.484** | 1.050 | not re-measured |
| **Time to *start*** | **immediate** — no scheduler | seconds to hours, depends on request shape | hours to unknown (2192 pending observed) |
| **Exclusivity** | ⚠️ **none** — 20+ users share it | guaranteed by SLURM | guaranteed by SLURM |
| **Setup fragility** | Medium — CUDA wheel trap (§4.4) | High — Intel/AMD split (§4.3) | Low |
| **Getting code in** | `git pull` (HTTPS token) | `rsync` from the laptop | `git pull` (SSH key) |
| **Network** | public SSH; **no outbound port 22** | **Uni Bonn VPN required** | public SSH |

### 0.1 For this project specifically

The 4MT benchmark evaluates VLMs over ~1,500 rendered items. Measured on dgx2:

| | Measured |
|---|---|
| Qwen3-VL-32B-Instruct, bf16, 2× A100 | **0.8–1.2 s/item** (5 images/item) |
| Model load from `/raid` cache | ~15 s (1058 shards) |
| 63 GB model download | ~8 min |
| Full sweep, 1,500 items × 5 channels | **~25 min** |

For comparison, Qwen2.5-VL-7B-4bit on the laptop's MLX/Metal ran **9.4 s/item** — dgx2
is ~10× faster *and* runs a 32B model unquantised. The whole overnight laptop sweep
(2,303 calls, ~4 h) re-runs here in well under an hour.

**No SLURM is involved.** dgx2 has no scheduler, so everything below about partitions,
QoS and `sbatch` applies only to Marvin and Leonardo.

**Rule of thumb: prefer dgx2 for anything that fits in ≤4 GPUs and ≤a few hours.**
It is 2.2× faster per GPU than Marvin (§7.1, measured across three values of n) and has
no queue at all, which together beat Marvin by roughly 4× in time-to-result.

**Use Marvin instead when the run is long or must not be disturbed.** dgx2 has no
scheduler, so nothing reserves a GPU: another user can start work on the same device
mid-run and both slow down. SLURM's queue is the price of exclusivity. For a multi-hour
run, either accept that risk or take the queue.

> Queue depth and dgx2 occupancy are the two most volatile numbers in this file.
> Re-measure (§7.1) before believing any of the above.

---

## 1. Access

### dgx2
```bash
ssh dgx2                                        # user markus_frey, configured in ~/.ssh/config
```
Verified working for this project on 2026-09-07: 8× A100-SXM4 80GB, driver **570.86.15**
(CUDA 12.8), system Python **3.12.3**, 256 cores. GPUs 5–7 were occupied by another
user throughout; 0–4 were free. Confirm before every run — there is no queue.
Public SSH, no VPN. **No scheduler** — you are on the machine, `nvidia-smi` is the
whole queue system. Check occupancy before starting anything:
```bash
ssh dgx2 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'
```
Pick free devices explicitly and stay inside the 4-GPU budget:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python sweep.py ...
```
Run long jobs under `nohup`/`tmux`; there is no scheduler to survive your disconnect.

### Leonardo
```bash
ssh mfrey000@login01-ext.leonardo.cineca.it     # login02, login05, ... also valid
```
Public internet, no VPN. Login nodes are slow and flaky under load — prefer running
monitors from the laptop over holding an interactive session.

### Marvin
```bash
ssh mfrey_hpc@marvin.hpc.uni-bonn.de            # gpu.marvin.hpc.uni-bonn.de for builds
```

**Requires the Uni Bonn network or VPN.** Off-network, SSH does not refuse — it
**times out**, which looks like the cluster being down. On this Mac the VPN is a Cisco
AnyConnect profile named `UniBonn`:

```bash
scutil --nc list                  # shows the profile and whether it is connected
scutil --nc start "UniBonn"       # pops the credential/2FA dialog
```

Claude cannot authenticate the VPN — a human has to complete the prompt.

**Key setup (already done, recorded in case it must be redone).** Keys go into
FreeIPA, not `authorized_keys`; propagation took **~44 minutes**. If an SSH agent
holds a certificate for another cluster, Marvin may reject the connection because the
certificate's principal (`mfrey000`) does not match the Marvin username.

---

## 2. Storage

### Leonardo — fixed path
```
/leonardo_work/AIFAC_S07_154/mfrey/tevol      # repo (TEVOL_ROOT)
/leonardo_work/AIFAC_S07_154/mfrey/hf_cache   # HF_HOME
```

### Marvin — allocated workspace, not a fixed path
`/home/mfrey_hpc` is capped at 100 GB and is **not** for venvs, model caches, or
artifacts. Everything lives in a Lustre workspace:

```bash
ws_allocate tevol 90     # 90 days, 3 extensions available
ws_find tevol            # -> /lustre/scratch/data/mfrey_hpc-tevol
ws_list                  # shows remaining time and expiration date
ws_extend tevol 30
```

Current workspace: `/lustre/scratch/data/mfrey_hpc-tevol`, expires **2026-11-25**.

> **Workspaces are deleted 21 days after expiry.** Anything not committed to GitHub or
> copied back to the laptop is gone. Set a reminder before the expiration date.

---

## 3. Getting code onto each cluster

**dgx2 uses an HTTPS token** (see below) — `git pull` works:
```bash
ssh dgx2 'cd ~/tevol && git pull'
```
Its repo lives at `~/tevol`, venv at `~/tevol/.venv`, HF cache at `~/tevol/hf_cache`
(export `HF_HOME=~/tevol/hf_cache`; it is **not** set in the shell profile).

> ⚠️ **Do not follow that layout for this project.** `/home` is on the root array,
> which is at **99% (30 GB free)** — and `/home/markus_frey` is already 98 GB of it.
> A single VLM checkpoint is larger than the free space. See §10.1.

> ⚠️ **dgx2 has no outbound port 22.** `github.com:22` resolves to a dead address
> (85.215.1.202), so `git@github.com:` clones and SSH agent forwarding both fail with
> `Connection refused`. HTTPS to github.com / pypi.org / huggingface.co all work.
> The remote must therefore be `https://github.com/CYHSM/tevol.git`.

Credentials are a **fine-grained PAT** scoped to this one repo, stored at
`~/.git-credentials` (0600) with `credential.helper store`. dgx2 is shared by 20+ users;
`/home/markus_frey` is `drwxr-x---` so other users cannot read it, but **root can**.
Keep the token narrow (one repo) and short-lived, and never widen it to a classic token.

To re-create it, run this *on dgx2* so the secret never enters a terminal log:
```bash
read -rs -p 'token: ' T && \
  printf 'https://CYHSM:%s@github.com\n' "$T" > ~/.git-credentials && \
  chmod 600 ~/.git-credentials && unset T && \
  git config --global credential.helper store
```

**Leonardo has a GitHub deploy key** — `git pull` works:
```bash
cd /leonardo_work/AIFAC_S07_154/mfrey/tevol && git pull
```

**Marvin does not**, and SSH agent forwarding does not help (the GitHub key is not in
the agent). Push from the laptop instead:
```bash
rsync -av --exclude .venv --exclude hf_cache --exclude .git \
  ~/Documents/Github/antigravity/tevol/ \
  mfrey_hpc@marvin.hpc.uni-bonn.de:/lustre/scratch/data/mfrey_hpc-tevol/
```
Consequence: **fixes made on Marvin are stranded there.** Copy them back to the laptop
and commit, or they are lost when the workspace expires.

---

## 4. Software environment

All three machines use a pinned venv rather than a vendor AI module, because the
executor calls `self_attn(..., past_key_values=...)` — the **transformers 5.x**
signature (4.x uses `past_key_value`, singular). The version is load-bearing.

| | torch | transformers | notes |
|---|---|---|---|
| Leonardo | 2.6.0+cu124 | 5.16.1 | Environment Modules (§4.1) |
| Marvin | 2.6.0+cu124 | 5.16.1 | Lmod, AMD tree (§4.2–4.3) |
| **dgx2** | **2.11.0+cu128** | 5.16.1 | no modules, system py3.12 (§4.4) |

Leonardo and Marvin set `HF_HUB_OFFLINE=1` — **their compute nodes have no internet**,
so model and dataset caches must be populated from a login node first. **dgx2 does have
outbound HTTPS**, so it downloads on demand and needs no offline flag; just make sure
`HF_HOME` points into `~/tevol/hf_cache` (§3) and not the small default in `~/.cache`.

### 4.1 Leonardo modules
Environment Modules. `slurm/env_setup.sh`:
```bash
module load python/3.11.7      # the only python module; (default)
module load cuda/12.3
```

### 4.2 Marvin modules
Lmod, EasyBuild naming — the short names in the official docs do not exist:
```bash
module load Python/3.11.5-GCCcore-13.2.0     # NOT "Python/3.11"
module load CUDA/12.4.0                      # NOT "CUDA/12"
```

### 4.3 ⚠️ Marvin's Intel/AMD split — the worst trap on either cluster

**Login nodes are Intel Xeon Platinum 8468 (full AVX-512). GPU compute nodes are AMD
EPYC 7713 (Zen 3, no AVX-512 at all).** There are two EasyBuild trees:

```
/opt/software/easybuild-INTEL/modules/all     # login nodes
/opt/software/easybuild-AMD/modules/all       # compute nodes
```

SLURM propagates the submitting shell's environment, so **`MODULEPATH` arrives on the
compute node still pointing at `easybuild-INTEL`.** The venv's interpreter is the AMD
build, but it then loads the Intel-compiled `libpython3.11.so` and dies with
**SIGILL, exit code 132, before writing a single line to the log.**

`slurm/marvin/env_setup.sh` now selects the tree explicitly rather than inheriting it:
```bash
module unuse /opt/software/easybuild-INTEL/modules/all
module use   /opt/software/easybuild-AMD/modules/all
```
AMD-built binaries also run on the Intel login nodes, so this is correct in both places.

> **This cannot be caught by testing on a login node.** The venv passed all 43 tests
> there *because* the login node is the architecture the Intel libraries were built
> for. Marvin had "passed its test suite" for hours while being incapable of running a
> single compute job. **Always validate on a compute node:**
> ```bash
> srun -p mlgpu_devel --account=ag_ifi_wrobel -n1 --mem=8G --time=00:05:00 \
>   --pty bash -c 'source slurm/marvin/env_setup.sh; python -c "import torch; print(torch.__version__)"'
> ```

---

### 4.4 ⚠️ dgx2's CUDA wheel trap — torch installs but sees no GPU

dgx2 has **no module system**; it is a plain Ubuntu 24.04 box with system Python 3.12.
There is no `uv`. The venv is built with `python3 -m venv` and plain `pip`.

The trap: **`pip install torch` resolves to a CUDA 13.0 build, and dgx2's driver is
12.8 (570.86.15).** It installs cleanly, imports cleanly, and then reports:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
torch 2.13.0+cu130   cuda False   device_count 8
```

`cuda False` **alongside** `device_count 8` — it silently runs on CPU. There is no
error and no non-zero exit; a sweep would just be ~100× slower and nobody would know.
This is the same failure class as Marvin's Intel/AMD split (§4.3): an environment that
looks installed but is not usable.

Pin the CUDA build explicitly:
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install "transformers>=5.0,<6" "datasets>=2.20" accelerate scipy numpy matplotlib pytest
# datasets pins fsspec<=2026.6.0 and the torch install pulls a newer one:
.venv/bin/pip install "fsspec[http]<=2026.6.0,>=2023.1.0"
```

**Always assert the GPU is real before launching anything long:**
```bash
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```
`pytest tests/` is the stronger check — the identity gate revalidates the executor
against the stock forward pass on the new hardware and torch version (23 tests, ~50 s).

---

## 5. Hardware and partitions (all values read from the live scheduler)

### Leonardo — `boost_usr_prod`
Node: **4× A100 64GB**, 32 CPU cores, 514 GB RAM (`Gres=gpu:a100:4`, `cpu=32`).
Partition `MaxTime=1-00:00:00`, `DefaultTime=00:30:00`, `MaxNodes=64`.

`boost_usr_prod` is a **partition**, not a QoS. Valid QoS values:

| QoS | MaxWall | Limits | Priority |
|---|---|---|---|
| `normal` | (partition default, 1 day) | — | 40 |
| `boost_qos_dbg` | **00:30:00** | cpu=256, gpu=32 | **80** |
| `boost_qos_lprod` | 4-00:00:00 | — | 40 |
| `boost_qos_bprod` | 1-00:00:00 | node=256 | 60 |

**`boost_qos_dbg` has double the priority of `normal`** and is the only practical way
to jump the queue — at the cost of a hard 30-minute wall. Shard work into ≤30 min
chunks to exploit it.

### Marvin — four walltime tiers per GPU type

| Partition | MaxWall | Nodes | Per node |
|---|---|---|---|
| `sgpu_devel` | 1:00:00 | 32 | **4× A100**, 128 cores, 1000 GB |
| `sgpu_short` | 8:00:00 | 32 | " |
| `sgpu_medium` | 1-00:00:00 | 32 | " |
| `sgpu_long` | 7-00:00:00 | 32 | " |
| `mlgpu_devel` | 1:00:00 | 24 | **8× A40 48GB**, 128 cores, 500 GB |
| `mlgpu_short` | 8:00:00 | 24 | " |
| `mlgpu_medium` | 1-00:00:00 | 24 | " |
| `mlgpu_long` | 7-00:00:00 | 24 | " |

No QoS flags. **`--account=ag_ifi_wrobel` is required.**

The bare names `sgpu` and `mlgpu` from the official docs **do not exist**.

---

## 6. Submitting jobs

### 6.1 Traps that apply to both clusters

**`$(dirname "$0")` does not work in an sbatch script.** SLURM copies the script to
`/var/spool/slurmd/job<ID>/slurm_script`, so `$0` is the spool path and any
sibling-file `source` fails. Use an absolute path, or `readlink -f` with a fallback.

**`RANK` is not set by `srun`** — only `SLURM_PROCID` is. PyTorch's `env://`
rendezvous reads `RANK` directly, so a distributed job hangs or dies without it.
`tevol/distributed.py` exports `RANK`/`LOCAL_RANK`/`WORLD_SIZE` from the SLURM
variables *and* passes rank/world_size explicitly to `init_process_group`.

Standard preamble, correct on both:
```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export WORLD_SIZE=$SLURM_NTASKS
srun --cpu-bind=cores python -u <script>.py ...
```

**A failed-looking `ssh` may still have submitted the job.** A retry then produces a
duplicate that races on the same artifact path. Always `squeue -u $USER` before
resubmitting.

### 6.2 Marvin-specific: three ways to sit in `PENDING (Resources)` forever

All three look identical in `squeue`. In the observed case the *third* was the real one.

**(a) Asking for a whole node.** Every `mlgpu` node sits in state `mixed` essentially
always, so `--gres=gpu:a40:8` waits for a full node to drain — measured ETA **77
minutes**. A 3- or 4-GPU request backfills onto an already-mixed node in seconds.

**(b) Not setting `--mem`.** Marvin defaults to **3900 MB per CPU**, so
`--cpus-per-task=8 --ntasks=4` silently requests **122 GB**, more than the ~53 GB free
on the only node with spare GPUs. Four ranks of a 0.5B bf16 model need a couple of GB
each. **Always set `--mem` explicitly.**

**(c) A node with hung `/opt/software`.** See §8.

Check what is genuinely free before sizing a request:
```bash
for n in $(sinfo -p mlgpu_short -h -N -o "%N %T" | awk '$2 !~ /drain|down|reserved/ {print $1}'); do
  d=$(scontrol show node $n)
  ag=$(echo "$d" | grep -o 'AllocTRES=[^ ]*' | grep -o 'gres/gpu=[0-9]*' | cut -d= -f2)
  ca=$(echo "$d" | grep -o 'CPUAlloc=[0-9]*' | cut -d= -f2)
  rm=$(echo "$d" | grep -o 'RealMemory=[0-9]*' | cut -d= -f2)
  am=$(echo "$d" | grep -o 'AllocMem=[0-9]*' | cut -d= -f2)
  printf "%-10s free_gpu=%s free_cpu=%s free_mem=%sG\n" \
    "$n" "$((8-${ag:-0}))" "$((128-ca))" "$(((rm-am)/1024))"
done
```

---

## 7. Measured performance

### 7.1 Exhaustive NLL sweep — dgx2 vs Marvin, like-for-like

Same `bench_nll.py`, same model (Qwen2.5-1.5B-Instruct, 56 units), same bf16, same
question sets. `s/prog` is one full 56-unit teacher-forced forward over `n` questions.

| n | Marvin A40 | **dgx2 A100-SXM4** | speedup |
|---|---|---|---|
| 64 | 1.050 s | **0.484 s** | 2.17× |
| 128 | 2.108 s | **0.964 s** | 2.19× |
| 256 | 4.402 s | **1.960 s** | 2.25× |
| per unit-execution | 17.2 ms | **7.8 ms** | 2.2× |
| per-program overhead `a` | 0.083 s | **0.045 s** | 1.8× |

Full 15,951-program sweep (1,047,256 unit-executions ≈ 18,701 identity-equivalents):

| n | Marvin, 4× A40 | **dgx2, 4× A100** |
|---|---|---|
| 64 | 1.3 h | **0.6 h** |
| 128 | 2.7 h | **1.2 h** |
| 256 | 5.4 h | **2.5 h** |
| 512 | ~11 h | **~5 h** |

Two facts that generalise:

* **Scaling is exactly linear in `n`** (n=128 measured 2.01× n=64) and **batch size is
  irrelevant** — 8/16/32 differ by <2% on both machines. The GPU is saturated at 8
  sequences, so tune `n`, never `batch_size`.
* **The 2.2× ratio held across all three values of `n`**, which is what marks it as a
  real hardware difference rather than a measurement artifact.

### 7.2 Older per-eval numbers (generation-based scans, Qwen2.5-0.5B)

| What | Where | Measured |
|---|---|---|
| `baseline.py`, GSM8K train labels | Leonardo, 1× A100 | **15 min** |
| k=1 scan, 96 evals @ n=512 | Leonardo, 4× A100 | **22 min** — 24 evals/GPU, ~56 s/eval |
| fine gain sweep, 192 evals @ n=512 | Marvin, 3× A40 | **1:07:51** — 64 evals/GPU, ~59 s/eval |
| model load + dataset + identity eval | Marvin A40 | ~5–6 min of the above |
| identity accuracy @ n=512 elite | Leonardo A100 | **0.5195**, 1 SE = 0.0221 |
| identity accuracy @ n=512 elite | Marvin A40 | **0.5156**, 1 SE = 0.0221 |

**The 0.0039 identity gap is 2 questions out of 512** — A100-vs-A40 bf16 numerics on
identical questions, not a bug. `merge_gain.py` warns on this rather than failing.
Treat a gap of more than a few questions as a real problem (wrong split, wrong labels).

### ⚠️ Do not extrapolate between clusters from memory bandwidth

The A40 has roughly **⅓** the HBM bandwidth of an A100, so a bandwidth-bound decode
should cost ~3× per eval. **Measured, it costs ~1.1–1.35×** — near parity. The
spec-sheet prediction was wrong by a factor of two to three, and predicting from it
would have wrongly ruled Marvin out.

The reason is that this workload is *not* purely bandwidth-bound at these settings.
With `batch_size=64` and a 0.5B model, per-decode-step Python and kernel-launch
overhead is a large share of the step, and it does not scale with bandwidth at all —
consistent with the earlier finding that our executor runs at 1.11× HF's fused loop.
Prefill over 2-shot GSM8K prompts is compute-bound rather than bandwidth-bound too.

**Benchmark one real eval on new hardware rather than reasoning from specs.** The
per-eval numbers above are the only trustworthy basis for budgeting.

The dgx2 measurement makes the same point from the other direction: A100-SXM4 has
**2.9×** the A40's bandwidth (2039 vs 696 GB/s) but delivered **2.2×**. Per-program
overhead `a` fell by 1.8× as well — the whole pipeline scaled together, which is not
what a purely bandwidth-bound workload does. Spec ratios have now over-predicted in one
direction and under-predicted in the other; neither is a substitute for `bench_nll.py`.

**Wall-clock budgeting:** `scan_gain.py` writes its artifact **once, at the end**. A
timeout loses every eval. Overshoot the wall deliberately, and prefer a longer-tier
partition over a tight fit in a `_devel` tier.

---

## 8. Playbook: a job that is `RUNNING` but producing nothing

Observed on Marvin `mlgpu015`: job `RUNNING`, **zero-byte logs**, no progress, forever.
Cause was a hung `/opt/software` NFS mount — Lustre was fine, the EasyBuild tree was
not, so `module load` blocked in state `D` (uninterruptible I/O) indefinitely.

Inspect a running job's node without killing it:
```bash
srun --jobid=<ID> --overlap -N1 -n1 bash -c 'ps -u $USER -o pid,etime,stat,cmd --sort=start_time'
srun --jobid=<ID> --overlap -N1 -n1 bash -c 'timeout 10 stat /opt/software >/dev/null && echo OK || echo HUNG'
```
`STAT=D` means blocked on I/O, not busy. Work around with `sbatch --exclude=<node>`,
and report the node to support.

### Failure signature quick-reference

| Symptom | Cause |
|---|---|
| SSH to Marvin **times out** | Not on Uni Bonn VPN (§1) |
| Exit **132**, `Illegal instruction`, logs cut off after the `nvidia-smi` line | Intel/AMD `MODULEPATH` (§4.3) |
| `error while loading shared libraries: libpython3.11.so.1.0` | Module tree not loaded, or `/opt/software` hung |
| `PENDING (Resources)` with idle GPUs visible | Whole-node request, or implicit `--mem` (§6.2) |
| `RUNNING`, zero-byte logs, indefinitely | Hung node filesystem (§8) |
| Distributed job hangs at startup | `RANK` unset (§6.1) |
| `Invalid qos specification` on Leonardo | Used `boost_usr_prod` as a QoS; it is a partition (§5) |
| `torch.cuda.is_available()` False **but** `device_count()` 8, on dgx2 | cu130 wheel vs 12.8 driver (§4.4) |
| `could not read Username for 'https://github.com'` on dgx2 | No token in `~/.git-credentials` (§3) |
| `ssh: connect to host 85.215.1.202 port 22: Connection refused` from dgx2 | No outbound port 22; use the HTTPS remote (§3) |
| dgx2 job suddenly ~2× slower mid-run | Another user took the same GPU — no scheduler (§0) |
| `sbatch` job dies instantly, `env_setup.sh: No such file or directory` | Used `$(dirname $0)`; SLURM copies the script to `/var/spool/slurmd`. Use `$SLURM_SUBMIT_DIR` |

### Known-bad nodes
- **`mlgpu015`** (Marvin) — `/opt/software` hung as of 2026-08-27. Use `--exclude=mlgpu015`.

---

## 9. Keeping the machines consistent

The two clusters must evaluate **identical questions** for their results to be
mergeable. `train_labels.json` (the 7473-question baseline correctness labels) is
generated on Leonardo and **copied** to Marvin — never regenerated independently,
since regeneration on different silicon would shift the labels and silently change the
splits.

Verify after any sync: identity accuracy on the n=512 elite set should match across
clusters to within a couple of questions (§7).

---

## 10. This project on dgx2 — VLM evaluation

`tevol` streamed text through a 0.5–1.5B model. This repo pushes **five 640×440 PNGs
per item** through a 32B vision-language model, which changes the storage picture and
adds one trap that a text sweep never meets.

### 10.1 ⚠️ `/home` is full — everything lives on `/raid`

```
/dev/md0    1.8T  1.7T   30G  99% /          <- /home is here. Unusable.
/dev/md127   28T   24T  3.0T  89% /raid      <- use this
```

`/raid` itself is root-owned and `mkdir /raid/<user>` is **denied**. Two directories
are world-writable (`drwxrwxrwx`): `/raid/nbe_tmp` and `/raid/s3`. This project uses:

```
/raid/nbe_tmp/markus_frey/asp/          # project root
  ├── .venv/                            # cu128 venv (§4.4)
  ├── hf_cache/                         # HF_HOME - 63 GB for Qwen3-VL-32B alone
  ├── data/bench_v2/                    # rendered items + images (~1.7 GB)
  └── *.py                              # scripts, scp'd from the laptop
```

`nbe_tmp` is a shared scratch directory with no quota and **no backup or retention
guarantee** — it is not owned by us and another user could clear it. Treat it exactly
like Marvin's expiring workspace (§2): results must be copied back to the laptop, and
nothing there is authoritative.

### 10.2 Environment

The CUDA wheel trap (§4.4) applies unchanged and is the main hazard. Verified recipe:

```bash
BASE=/raid/nbe_tmp/markus_frey/asp
mkdir -p $BASE/hf_cache && cd $BASE
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
./.venv/bin/pip install -q "transformers>=5.0,<6" accelerate pillow numpy qwen-vl-utils
./.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
# -> 2.11.0+cu128 NVIDIA A100-SXM4-80GB
```

`torchvision` is required here but was not for `tevol` — some VLM image processors
(SmolVLM's, for one) fail to load without it, with an error that names only the
processor class and not the missing dependency.

### 10.3 Getting data across

There is no git-LFS story for 1.7 GB of renders; `rsync` over SSH is fine and takes
about a minute:

```bash
rsync -a --include="*/" --include="*_az*.png" --exclude="*" \
  data/bench_v2/ dgx2:/raid/nbe_tmp/markus_frey/asp/data/bench_v2/
scp bench/*.py dgx2:/raid/nbe_tmp/markus_frey/asp/
```

> Note the include/exclude ordering: `--include="*_az*.png"` also matches
> `*_az000_mask.png`, so instance masks come along unless excluded *before* it. Costs
> disk, not correctness.

### 10.4 Blender on dgx2 — works, but not with OptiX

Rendering also runs here, ~2x faster per GPU than the laptop and 4x that again
across GPUs. No sudo needed: Blender ships a portable tarball.

```bash
BASE=/raid/nbe_tmp/markus_frey/asp
curl -sL https://download.blender.org/release/Blender5.2/blender-5.2.1-linux-x64.tar.xz \
  -o blender.tar.xz && tar -xJf blender.tar.xz -C $BASE && rm blender.tar.xz
$BASE/blender-5.2.1-linux-x64/blender -b -v      # runs as-is; no missing libs
```

> `libXkbcommon.so.0` is genuinely absent from this box, but Blender only needs
> it for the GUI — `blender -b` does not touch it. Checking with
> `ldconfig -p | grep " libfoo "` reports *everything* as missing because of the
> `name (libc6,x86-64) => path` format; test with `[ -e /usr/lib/x86_64-linux-gnu/... ]`.

⚠️ **OptiX does not work on this driver.** Cycles spends ~5 minutes compiling and
then dies with `OPTIX_ERROR_INTERNAL_COMPILER_ERROR` loading `kernel_optix.ptx.zst`.
CUDA works fine, so force it:

```bash
CYCLES_GPU=CUDA CUDA_VISIBLE_DEVICES=0 $BASE/blender-5.2.1-linux-x64/blender -b -P fm/render_bank.py -- ...
```

`generate_scene._enable_gpu` tries METAL → OPTIX → CUDA and honours `CYCLES_GPU`,
so the same scene code renders on the laptop and here.

**Measured** (640x440, 32 samples): 3.1 s/frame on one A100 vs 5.7 s on the
laptop's M-series Metal — only ~1.8x per GPU. The win is parallelism: four
shards pinned to GPUs 0-3 with disjoint seed ranges give ~4x on top, ~13 min per
scene of 160 frames, so a 60-scene bank lands in ~3.2 h.

```bash
for i in 0 1 2 3; do
  CYCLES_GPU=CUDA CUDA_VISIBLE_DEVICES=$i nohup $BASE/blender-5.2.1-linux-x64/blender \
    -b -P fm/render_bank.py -- --out data/scenes/shard$i --scenes 15 --seed0 $((i*15)) \
    > render_v3_$i.log 2>&1 &
done
```

Shards write separate `bank.json` files deliberately — one shared manifest would
race, since each process rewrites it after every scene.

### 10.5 Running a sweep

```bash
ssh dgx2 'cd /raid/nbe_tmp/markus_frey/asp && \
  HF_HOME=$PWD/hf_cache CUDA_VISIBLE_DEVICES=0,1 nohup bash dgx_sweep.sh > sweep.log 2>&1 &'
```

`bench/exchange.py` is now the single entry point for every backend
(`--backend auto|mlx|hf|timm`); the separate `run_eval_hf.py` and
`run_eval_embed.py` are gone. Prompts live in `bench/prompts.py` and model
loading in `bench/backends.py`, so the cluster asks exactly what the laptop
asked — otherwise the two runs are not comparable, which is the whole point of
running on both.

```bash
# a VLM on images
./.venv/bin/python exchange.py --items data/scenes/items.json \
  --backend hf --model Qwen/Qwen3-VL-32B-Instruct --channel V --limit 400
# a frozen encoder swept over latent noise (the delay analogue)
./.venv/bin/python exchange.py --items data/scenes/items.json \
  --backend timm --model vit_large_patch14_dinov2.lvd142m --channel EMB \
  --sigmas 0,0.25,0.5,1,2 --noise-seeds 3
```

Runs are **resumable**: each writes one JSON line per item and skips item ids already
present in the output file. A killed job loses at most one item.

Model choice, measured: `Qwen/Qwen3-VL-32B-Instruct` (dense, bf16, ~64 GB) sits on two
A100s with room to spare. The flagship `Qwen3-VL-235B-A22B-Instruct` is ~470 GB in
bf16 — beyond the 4-GPU budget — and its FP8 release is **not usable on A100**, which
is Ampere and has no native FP8. An AWQ/INT4 build would fit; it was not needed.

### 10.5 Failure signatures specific to this project

| Symptom | Cause |
|---|---|
| `No space left on device` while downloading a model | Wrote to `/home` or default `~/.cache`; `HF_HOME` not exported (§10.1) |
| `mkdir: cannot create directory '/raid/markus_frey': Permission denied` | `/raid` root is not writable; use `/raid/nbe_tmp/...` |
| `Could not load any image processor class` naming torchvision | `torchvision` missing from the venv (§10.2) |
| `HfApi.list_models() got an unexpected keyword argument 'direction'` | Newer `huggingface_hub`; `sort=` alone, no `direction=` |
| Cycles dies with `OPTIX_ERROR_INTERNAL_COMPILER_ERROR` | OptiX is broken on driver 570.86.15; set `CYCLES_GPU=CUDA` (§10.4) |
| `ldconfig -p` claims every library is missing | Wrong grep pattern for its `name (libc6,x86-64) => path` format; stat the file instead |
| Eval runs but every reply is the same digit | Not a bug — small VLMs genuinely do this. The answer key is uniform by construction, so a constant reply scores exactly chance. `analyze.py` flags it as DEGENERATE |
