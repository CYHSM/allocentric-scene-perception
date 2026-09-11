#!/usr/bin/env bash
# Render the c0 bank at other landmark counts, everything else held fixed.
#
#   nohup bash blender/four_mountains/render_n_sweep.sh > logs/n_sweep.log 2>&1 &
#   N_LIST="6 8" bash blender/four_mountains/render_n_sweep.sh    # a subset
#
# `data/scenes_100` is N=4 and is never touched. Each count gets its own tree
# with the identical internal structure, so the family can be published as one
# dataset:
#
#   data/scenes_100        <- N=4, already rendered
#   data/scenes_100_n01    <- this script
#   data/scenes_100_n02
#   data/scenes_100_n06
#
# Every render parameter is copied from `data/scenes_100/c0_shape_colour/
# bank.json`: 100 anchor layouts and no probe ladder, seed0 0, azimuths every 45
# degrees, 24 Cycles samples, 640x440, instance masks written, 4 shards, and the
# same bank.json merge. Layouts are a deterministic function of
# (seed0, anchors, n_peaks), so a000 in every tree is the same seed drawn with a
# different number of peaks. Masks are kept even though the benchmark ignores
# them, because the point is a dataset that matches what is already published.
#
# TWO THINGS ARE NOT UNIFORM ACROSS N, and both are recorded in bank.json.
#
# 1. Anchor separation. Two anchors must be at least SEP[N] metres apart in
#    layout distance to count as different places, and three of the five counts
#    cannot reach 100 anchors at the standard 12 m -- see the table by
#    `sep_for` below. Read `anchor_separation_m` out of bank.json before
#    comparing distances across banks; it sets the floor of the scale that foil
#    bands are measured on.
#
# 2. Peak size. Peaks must shrink to pack onto the ring. Mean height is 18.7,
#    17.7, 17.9, 15.1 and 12.2 m at N = 1, 2, 4, 6, 8, and the minimum spacing
#    between peaks falls 75 -> 47 -> 33 -> 25 m from N = 2 to N = 8. Set size
#    and apparent object size are therefore not independent, which any
#    across-N comparison has to state. See data/DATASET.md.
set -uo pipefail
cd "$(dirname "$0")/../.."

BLENDER="${BLENDER:-$PWD/blender-5.2.1-linux-x64/blender}"
SCRIPT="blender/four_mountains/render_bank.py"
MODE="${MODE:-c0_shape_colour}"
N_LIST="${N_LIST:-1 2 6}"
SHARDS="${SHARDS:-4}"
ANCHORS=100
PROBES=0
SEED0=0
AZ_STEP=45
SAMPLES=24
RES="640 440"
mkdir -p logs

# Cycles prefers OPTIX over CUDA when both are exposed, and on this box the
# OptiX kernel fails to compile ("OPTIX_ERROR_INTERNAL_COMPILER_ERROR" loading
# kernel_optix.ptx.zst). data/scenes_100 was rendered on CUDA -- its log says so
# -- so pin CUDA both to make the render work and to keep the new trees on the
# same backend as the existing one. The first CUDA render of a session spends
# about five minutes compiling kernels before the first image appears; that is
# not a hang.
export CYCLES_GPU="${CYCLES_GPU:-CUDA}"

# N -> minimum anchor separation in metres. 12 is the project default and the
# value data/scenes_100 (N=4) used. Anchor capacity is not monotonic in N and
# three counts cannot reach 100 anchors at 12 m; measured with the same greedy
# packer fm_bank.sample_anchors uses, at its own 400*n_anchors seed budget:
#
#     N      1    2    4    6    8
#     @12m  20   92  100  100   61
#
# N=8 reaches 100 at a 10 m floor (92 at 11 m); capacity is not its blocker.
#
# Few peaks give a layout space too small to pack 100 places into; many peaks
# shrink and crowd onto the ring, so the layouts concentrate and stop being
# 12 m apart. Both low counts need a lower floor.
#
# N > 6 is not renderable in c0 at all: every landmark must be unique in both
# shape and colour and the canonical inventory holds six of each, so
# fm_stimulus.canonical_objects raises above six. That ceiling is the reason
# this sweep stops at 6 and not a choice about coverage.
sep_for() {
    case "$1" in
        1) echo 4.0 ;;
        2) echo 10.0 ;;
        *) echo 12.0 ;;
    esac
}

if [ ! -x "$BLENDER" ]; then echo "no blender at $BLENDER"; exit 1; fi

# One sweep at a time. Two copies write the same scene directories and the same
# per-shard bank files, so the merge sees half of each and the tree is quietly
# wrong rather than obviously broken. This has already happened twice, from a
# retry loop that fired more than once.
exec 9>"logs/.n_sweep.lock"
if ! flock -n 9; then
    echo "another render_n_sweep.sh already holds logs/.n_sweep.lock -- exiting"
    exit 0
fi

# Fewest-megabytes-first among idle GPUs, one per shard. This deliberately does
# NOT take logs/.vlm_gpu.lock: that lock serialises VLM evaluation, and a
# multi-hour render holding it would block every evaluation on the box.
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                    | awk -F', ' '$2 < 2000 {print $2","$1}' | sort -n | cut -d, -f2)
if [ "${#FREE[@]}" -lt "$SHARDS" ]; then
    echo "only ${#FREE[@]} idle GPUs, need $SHARDS"
    nvidia-smi --query-gpu=index,memory.used --format=csv
    exit 1
fi
echo "idle GPUs: ${FREE[*]}  |  using the first $SHARDS  |  N_LIST=$N_LIST"

for N in $N_LIST; do
    NN=$(printf "%02d" "$N")
    SEP=$(sep_for "$N")
    OUT="data/scenes_100_n${NN}/${MODE}"
    if [ -f "$OUT/bank.json" ]; then
        echo "[skip] N=$N already merged at $OUT/bank.json"; continue
    fi
    mkdir -p "$OUT"
    echo "=========================================================="
    echo "N=$N  separation=${SEP} m  ->  $OUT   |  $(date)"
    echo "=========================================================="

    pids=()
    for ((s=0; s<SHARDS; s++)); do
        LOG="logs/n${NN}_${MODE}_shard${s}.log"
        CUDA_VISIBLE_DEVICES="${FREE[$s]}" "$BLENDER" -b -P "$SCRIPT" -- \
            --out "$OUT" --mode "$MODE" \
            --anchors "$ANCHORS" --probes_per_anchor "$PROBES" \
            --seed0 "$SEED0" --n_peaks "$N" --separation_m "$SEP" \
            --azimuth_step "$AZ_STEP" --samples "$SAMPLES" \
            --resolution $RES \
            --shard "$s" --shards "$SHARDS" > "$LOG" 2>&1 &
        pids+=($!)
        echo "  shard $s -> GPU ${FREE[$s]}  (pid ${pids[-1]}, log $LOG)"
    done

    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    if [ "$fail" -ne 0 ]; then
        echo "[warn] a shard for N=$N exited non-zero; see logs/n${NN}_*.log"
    fi

    # Merge the per-shard banks into bank.json, in the layout the existing tree
    # uses: one shard's metadata, all shards' scenes, n_shards.
    python3 - "$OUT" "$SHARDS" <<'PY'
import glob, json, os, sys
out, shards = sys.argv[1], int(sys.argv[2])
files = sorted(glob.glob(os.path.join(out, "bank.shard*.json")))
if len(files) != shards:
    print(f"  MERGE SKIPPED: {len(files)} shard banks, expected {shards}")
    sys.exit(1)
bank = json.load(open(files[0]))
scenes = {}
for f in files:
    scenes.update(json.load(open(f))["scenes"])
bank["scenes"] = dict(sorted(scenes.items()))
bank["n_shards"] = shards
with open(os.path.join(out, "bank.json"), "w") as fh:
    json.dump(bank, fh, indent=1)
imgs = len(glob.glob(os.path.join(out, "a*", "*_az*.png")))
print(f"  merged {len(scenes)} scenes, n_objects={bank['n_objects']}, "
      f"anchor_separation_m={bank.get('anchor_separation_m')}, "
      f"{imgs} png ({imgs // 2} images + {imgs // 2} masks)")
PY
    echo "N=$N finished at $(date)"
done
echo "N SWEEP DONE at $(date)"
