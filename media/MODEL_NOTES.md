# The model, from the code: talk notes

Everything below is read off `model.py`, `train.py`, `data.py` and `wandb_helper.py`
on `main`, with the default arguments of `train.py` unless stated. Where the
paper (arXiv 2303.06367) says something different, the difference is flagged;
section 4 collects them, because those are the questions a careful audience asks.

Figures: `architecture_mse.pdf` (encoder + decoder), `architecture_triplet_mse.pdf`
(same encoder, triplet objective).

---

## 0. One-paragraph version

A sample is **T views of one scene** (default T = 6, 64×64 RGB). A small conv
stack turns each view into an 8×8 grid of features. The grid is split two ways:
**averaged over views** it gives one vector per screen location (MEC, "what is
where, regardless of view"); **averaged over locations** it gives one vector per
view (LEC, "what this view is like"). The hippocampus multiplies every location
vector with every view vector, then runs **self-attention over all view×location
tokens** (CA3 at 8×8, CA1 at 4×4). The result is split the same two ways again
into **16 object latents** (constant across views) and **T view latents**
(constant across space). A **pixel-wise MLP** takes one object latent, one view
latent and one pixel coordinate and outputs a colour and a logit; a softmax over
objects turns the logits into masks, and the mask-weighted colours are the
reconstruction. The only loss is the pixel reconstruction error. The masks are
never supervised; their argmax is the segmentation.

---

## 1. Data, as the model sees it

| | |
|---|---|
| scenes | 5,000 train (scene 0–4,999), 1,000 test (5,000–5,999), fixed in `DataModule.setup` |
| views per scene | 16 renders around the arena (`surround`), 16 from inside (`within`) |
| sample | T consecutive rows of the scene's CSV, resized to 64×64, values in [0, 1] |
| order of views | the CSV order, which is **not** sorted by azimuth (README: sampled at random around the centre) |
| which views | `get_samples_within_scene` seeds its RNG with the scene index, so **a scene always yields the same T views** (comment in code: "the same scene always uses the same images") |
| masks | instance masks, one-hot, used **only** for ARI at validation |

Worth knowing for the talk: the model never receives camera pose, azimuth or
any geometric label. View identity enters only as an index t in [0, 1].

---

## 2. Encoder (`VisualCortex`, `Split`, `Hippocampus`)

Shapes for batch b, T views.

| stage | operation | output |
|---|---|---|
| input | — | b × T × 3 × 64 × 64 |
| V1 | conv 4×4, stride 2, pad 1, 128 ch, LeakyReLU | 32 × 32 |
| V2 | same | 16 × 16 |
| V4 | same, **from V2** | 8 × 8 → b × T × 64 × 128 tokens |
| IT | same, **also from V2** (parallel to V4, `self.it(v2)`) | 8 × 8 → b × T × 64 × 128 |
| MEC | V4 **mean over T** → b × 64 × 128; concat RSC (learned 32-d vector); MLP 160→1024→320 | b × 64 × 320 |
| LEC | IT **mean over the 64 locations** → b × T × 128; MLP 128→1024→320 | b × T × 320 |
| conjunction | `einsum("bkc,btc->btkc")`: h[t,k] = MEC[k] ⊙ LEC[t] | b × T × 64 × 320 |
| into CA3 | linear 320→320 (no bias) + 3D sinusoidal position (t, row, col) | T·64 tokens |
| CA3 | Transformer encoder, L layers (default 4), d = 320, 5 heads, FF 1024, pre-norm, no dropout | T·64 tokens |
| pool | reshape to 8×8 per view, avg-pool 2×2, ×2 | 4 × 4 |
| CA1 | + 3D position again; second Transformer stack, L layers | b × T × 16 × 320 |
| object latents | CA1 **mean over T** → b × 16 × 320; concat RSC (learned 80-d); MLP 400→1024→32 | b × 16 × 32 |
| view latents | CA1 **mean over the 16 locations** → b × T × 320; MLP 320→1024→32 | b × T × 32 |

### What each piece is for

* **The two averages are the whole trick.** Anything that survives an average
  over views is, by construction, shared across views; anything that survives
  an average over locations is, by construction, a property of a whole view.
  The factorisation into "what/where" and "which view" is enforced by the
  architecture, not learned or regularised.
* **The conjunction** gives every (view, location) pair its own token before
  attention, the analogue of TEM's binding of structure (MEC) and sensory
  content (LEC) in hippocampus. In the code it is an element-wise product per
  channel, i.e. an outer product over the (t, k) indices but **not** over
  channels.
* **Self-attention as CA3**: every token can read from every view and every
  location, which is how information about an object in one view reaches the
  same object in another view. Formally, one attention step is one retrieval
  step of a modern Hopfield network, which is the pattern-completion story.
* **The second split** is what produces object slots: each of the 16 object
  latents is one cell of the 4×4 CA1 grid, averaged over views.

### Size (computed from the layer shapes, biases included)

| part | parameters |
|---|---|
| visual cortex (4 convs) | ≈ 0.79 M |
| forward split (2 MLPs + RSC) | ≈ 0.95 M |
| linear into CA3 | ≈ 0.10 M |
| CA3 + CA1, L = 4 each | ≈ 8.5 M (≈ 1.07 M per layer) |
| backward split | ≈ 0.81 M |
| decoder MLP | ≈ 1.09 M |
| **total** | **≈ 12.3 M** at L = 4; **≈ 5.9 M** at L = 1 (the README example) |

---

## 3. Pixel-wise decoder (`Decoder`)

### The computation

1. **Pick what to decode.** Training: a random half of the views
   (`Td = T // 2`, sorted random sample) at half resolution (every other
   pixel, 32×32). Validation: all views, full 64×64.
2. **Pick the slots.** The first `K_down` object latents (default 8; the
   README example uses 10) — `range(Kd)`, deterministic. The remaining
   latents of the 16 are computed and **never decoded**.
3. **Build one input per (slot k, view t, pixel x, y):**
   `[ o_k (32) | f_t (32) | t | x | y ]` → 67 numbers. t is `linspace(0,1,T)`
   at the chosen views, x and y are `linspace(-1,1,64)`. Raw coordinates, no
   Fourier features.
4. **One shared MLP**: 67 → 512 → 512 → 512 → 512 → 512 → 4, LeakyReLU, no
   output non-linearity. Outputs 3 colour channels c_k (unbounded, no sigmoid)
   and 1 logit a_k.
5. **Turn logits into masks.**
   * `a_1 := −Σ_{k≥2} a_k` — the first slot's own logit is thrown away and
     replaced by minus the sum of the others.
   * LayerNorm (no learned affine) over all of (views, slots, height, width) of
     one sample.
   * Softmax over slots → m_k, which sum to 1 at every pixel.
6. **Reconstruct**: x̂ = Σ_k m_k · c_k.
7. **Loss**: |x̂ − x|^p summed over pixels, channels and views, averaged over
   the batch; p = 2 by default (so a summed squared error), optional annealing
   of p. Two L1 penalties on the latents exist and default to weight 0.
8. **Segment** (evaluation only): argmax_k m_k per pixel. For the colour
   coding, the slot with the largest average mask is treated as background.
   Scored with ARI and foreground-ARI against the instance masks.

### Why it segments at all

* **Competition**: the softmax makes slots bid for every pixel. A slot gets
  credit for a pixel only by explaining it better than every other slot.
* **A cheap way to explain a pixel**: o_k is constant across views and only 32
  numbers. The cheapest code that renders an object correctly in all T views
  is one slot per object with its appearance and position in o_k, and the
  view latent telling the MLP how that projects into view t.
* **Shared decoder**: all slots use the same MLP, so a slot cannot specialise
  by having its own weights. It can only differ by its latent.
* **The first slot**: defined as minus the others' logits, it wins where the
  others are confidently low and loses where any of them is confident, so it
  tends to become the background/default slot.
* **LayerNorm** fixes the scale of the logits per sample, so the softmax can be
  neither saturated (one slot takes everything early) nor flat (all slots tie).

### Compute

The MLP runs once per (slot, view, pixel). Training: 8 slots × 3 views × 32 ×
32 ≈ 25k evaluations per sample; validation: 8 × 6 × 64 × 64 ≈ 197k. That is
why training decodes half the views at half resolution.

---

## 4. Code vs. paper: say these before someone else does

| topic | paper | code on `main` |
|---|---|---|
| visual cortex | pre-trained CORnet-Z (ASP) | 4 conv layers trained from scratch; IT reads V2, parallel to V4 |
| PH / PR | separate areas with strong and weak connections | not present; the split goes straight to MEC / LEC |
| RSC | auxiliary head-direction loss ℒ_HD | a learned vector, identical for every input — equivalent to an extra bias in the first MLP layer; no ℒ_HD |
| CA1 | outer product + feed-forward | second self-attention stack; the product comes **before** CA3 |
| decoder | 5 × 128 units (ASP), periodic positional encodings | 5 × 512 units, raw linear coordinates |
| loss | "1/n Σ (x − x̂)²" (a mean) | summed over pixels, views and channels; averaged over the batch |
| slots | — | 16 object latents computed, only the first `K_down` = 8 decoded |
| alternative | — | `use_transformer=False` (GRU) branch references undefined names (`F`, `xy_after_conv`) and would crash |

---

## 5. Questions to expect, with answers

**Encoder**

* *"How does the model know where the camera is?"* — It doesn't. There is no
  pose input; view identity is only the index t. Any allocentric structure has
  to be inferred from how the images co-vary across views. That is the
  interesting claim, and also why "novel view synthesis" should be phrased
  carefully (below).
* *"Averaging V4 over views at a fixed screen location mixes different parts of
  the world when the camera moves. Why is that sensible?"* — For a static camera
  (CATER, MOVi) it is exactly right; for an orbiting camera it is a blurry
  summary. It is an inductive bias that guarantees the MEC code is
  view-invariant, not one that is already allocentric. The cross-view mixing
  that makes it useful happens later, in attention, before the second average.
* *"Is this just SIMONe?"* — The skeleton is: transformer over all
  view×location tokens, 2×2 pooling, splitting by averaging over time and
  space, a pixel-wise decoder with a softmax over slots (the code acknowledges
  the SIMONe repository). The differences are the split **before** the
  transformer with the multiplicative conjunction, the learned RSC vector,
  deterministic latents with no KL term or sampling, and the p-loss.
* *"Why deterministic, not a VAE?"* — The paper's argument: it is unclear how a
  circuit would sample from a learned distribution, so only means are used, as
  in TEM. The cost is no generative prior, so you cannot sample new scenes.
* *"Why V4 → MEC and IT → LEC?"* — Biological motivation, roughly a spatial and
  an object route. In the code V4 and IT are identical layers fed by the same
  V2; what differs is only the averaging applied downstream.
* *"What do the positional encodings encode?"* — Screen row, screen column and
  the view's index in the sample. Nothing about pose.
* *"Is the RSC doing anything?"* — Not as implemented: a constant vector
  concatenated to every token. Be upfront that the head-direction version in
  the paper's Fig. 1 is not what `main` trains.

**Decoder**

* *"Why are the masks object-shaped if they are never supervised?"* —
  Competition between slots, a shared decoder, and a tight per-slot bottleneck
  that must hold across all views (section 3). One object per slot is the
  cheapest way to reconstruct every view.
* *"How are slots assigned? Is the model permutation-invariant like slot
  attention?"* — No. Each object latent is one cell of the 4×4 CA1 grid, so
  slots start with a spatial prior. After training a slot need not stay near its
  cell, since attention mixes information, but there is no iterative
  competition for inputs as in slot attention.
* *"What if a scene has more objects than slots?"* — 8 decoded slots cover up to
  4 objects plus background in ASP and most CATER scenes; MOVi-C has up to 10
  objects, part of why performance drops there.
* *"Why does background end up in one slot?"* — The first slot is defined by the
  others (−Σ), which makes it the default. In evaluation the slot with the
  largest mean mask is labelled background; FG-ARI ignores background pixels.
* *"Why no sigmoid on colour?"* — With a squared error the decoder learns the
  range; unbounded outputs avoid gradient saturation. It means a slot's colour
  can be anything where its mask is ~0, which is why per-slot colour images
  look odd outside the object.
* *"Is this novel view synthesis?"* — Only in a limited sense. All T views are
  encoded; training decodes a random half of them, so the model reconstructs
  views it was given. Rendering a view it never saw would need that view's
  latent f_t, which has no pose argument to set. Say "reconstruction across
  views", not free-viewpoint rendering.
* *"Why is ASP so much harder than CATER?"* — Same-colour objects and objects
  coloured like the ground (green, white variants), a large textured background
  and distal landmark competing for slots, and a moving camera rather than
  moving objects. Table 1's FG-ARI on ASP is 0.05–0.30.
* *"What fails?"* — Shadows are assigned to their object (visible in Fig. 5),
  same-colour objects merge, and complexity hurts (MOVi-A 0.79 → MOVi-C 0.32
  FG-ARI).
* *"Why decode only part of the image during training?"* — Cost: the MLP runs
  per slot, view and pixel; half the views at half resolution is 8× cheaper.

**Both**

* *"Is it biologically plausible?"* — The encoder's area mapping is a
  hypothesis about roles, not anatomy. The pixel-wise decoder is the least
  plausible part (the code says so in a comment). Backpropagation applies
  throughout.
* *"How does this relate to the triplet model?"* — Same encoder in
  `architecture_triplet_mse.pdf`; the triplet loss on CA1 replaces the decoder.
  The paper's scene-discrimination results (CA3 0.89, CA1 0.88) come from the
  triplet-trained model, not the reconstruction model.
* *"How big is it?"* — ≈ 12 M parameters with the default 4 layers per
  attention stack; ≈ 6 M with 1 layer.
