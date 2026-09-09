#!/usr/bin/env bash
set -euo pipefail

cd /raid/nbe_tmp/markus_frey/asp
mkdir -p data/scenes_100

B=./blender-5.2.1-linux-x64/blender

echo "=== Launching 100-scene dataset on GPUs 4, 6, 7 ==="

# GPU 6: c0_shape_colour (4 shards)
for s in 0 1 2 3; do
  CYCLES_GPU=CUDA CUDA_VISIBLE_DEVICES=6 setsid nohup $B -t 4 -b \
    -P fm/render_bank.py -- \
    --out data/scenes_100/c0_shape_colour --mode c0_shape_colour \
    --anchors 100 --probes_per_anchor 0 --azimuth_step 45 --samples 24 \
    --shard $s --shards 4 \
    >> bank100_c0_${s}.log 2>&1 < /dev/null &
done

# GPU 6: c2_colour (4 shards)
for s in 0 1 2 3; do
  CYCLES_GPU=CUDA CUDA_VISIBLE_DEVICES=6 setsid nohup $B -t 4 -b \
    -P fm/render_bank.py -- \
    --out data/scenes_100/c2_colour --mode c2_colour \
    --anchors 100 --probes_per_anchor 0 --azimuth_step 45 --samples 24 \
    --shard $s --shards 4 \
    >> bank100_c2_${s}.log 2>&1 < /dev/null &
done

# GPU 7: c1_shape (4 shards)
for s in 0 1 2 3; do
  CYCLES_GPU=CUDA CUDA_VISIBLE_DEVICES=7 setsid nohup $B -t 4 -b \
    -P fm/render_bank.py -- \
    --out data/scenes_100/c1_shape --mode c1_shape \
    --anchors 100 --probes_per_anchor 0 --azimuth_step 45 --samples 24 \
    --shard $s --shards 4 \
    >> bank100_c1_${s}.log 2>&1 < /dev/null &
done

# GPU 7: c3_peaks_bare (4 shards)
for s in 0 1 2 3; do
  CYCLES_GPU=CUDA CUDA_VISIBLE_DEVICES=7 setsid nohup $B -t 4 -b \
    -P fm/render_bank.py -- \
    --out data/scenes_100/c3_peaks_bare --mode c3_peaks_bare \
    --anchors 100 --probes_per_anchor 0 --azimuth_step 45 --samples 24 \
    --shard $s --shards 4 \
    >> bank100_c3_${s}.log 2>&1 < /dev/null &
done

# GPU 4: c4_valley (4 shards)
for s in 0 1 2 3; do
  CYCLES_GPU=CUDA CUDA_VISIBLE_DEVICES=4 setsid nohup $B -t 8 -b \
    -P fm/render_bank.py -- \
    --out data/scenes_100/c4_valley --mode c4_valley \
    --anchors 100 --probes_per_anchor 0 --azimuth_step 45 --samples 24 \
    --shard $s --shards 4 \
    >> bank100_c4_${s}.log 2>&1 < /dev/null &
done

sleep 4
echo "Total 100-scene render processes launched: \$(ps -u markus_frey -o cmd | grep -c \"[r]ender_bank.*scenes_100\")"
