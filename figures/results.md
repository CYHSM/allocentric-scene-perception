# 4MT Allocentric Scene Perception: Empirical Benchmark Results

## Summary of Completed Runs

### 1. Verification & Identity Gates
- **Dataset 1 (`data/scenes`)**: 40 anchors + 120 displacement probes per mode across 5 modes (`c0` through `c4`), 25,600 frames total.
- **Identity Gate Check**: Evaluated on identical place, identical bearing ($\Delta 0^\circ$), identical appearance.
  - **Result**: **16 / 16 model-mode cells scored 100.0% Recall@1 (All Passed)**.

### 2. Model Performance Across Cue Conditions
All numbers reflect Recall@1 (%) and RSA spearman correlation at $45^\circ$ viewpoint change, along with the exchange rate $\lambda(45^\circ)$ in metres:

```
Mode              Model            Recall@1 (45°)    RSA (45°)    Lambda (45°)
-------------------------------------------------------------------------------
c0 shape+colour   DINOv2-B/14          24.7%          +0.56         > 24 m
                  SigLIP-so400m        26.4%          +0.37         > 24 m
                  CLIP-B/16            19.2%          +0.45         > 24 m
                  ResNet-50            19.2%          +0.36         > 24 m

c1 shape          DINOv2-B/14          22.8%          +0.55         > 24 m
                  SigLIP-so400m        13.4%          +0.37         > 24 m
                  CLIP-B/16            12.0%          +0.33         > 24 m
                  ResNet-50            12.5%          +0.38         > 24 m

c2 colour         DINOv2-B/14           3.4%          +0.24         > 24 m
                  SigLIP-so400m         7.7%          +0.46         > 24 m
                  CLIP-B/16             5.0%          +0.45         > 24 m
                  ResNet-50             3.1%          +0.16         > 24 m

c3 landforms      DINOv2-B/14           7.5%          +0.26         > 24 m (90°: 15.2m)
                  SigLIP-so400m         6.1%          +0.14         > 24 m (180°: 10.8m)
                  CLIP-B/16             2.5%          +0.07         > 24 m (90°: 18.9m)
                  ResNet-50             4.8%          +0.08         > 24 m (180°: 6.0m)

c4 valley         (In progress on GPU 1)
```

## Generated Figures
- `figures/fig1_calibration.png`
- `figures/fig1_dataset.png`
- `figures/fig2_design.png`
- `figures/fig2_lambda.png`
- `figures/fig3_curves.png`
- `paper/table1.tex`
