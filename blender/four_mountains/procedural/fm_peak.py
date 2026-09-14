"""
Parametric landmark peaks: a small, explicit shape space.

Pure numpy -- no `bpy` -- so every peak can be generated, measured and
unit-tested outside Blender.

Why this replaces `fm_morphology`
---------------------------------
The twelve named morphologies (`horn`, `mesa`, `caldera`, ...) were categorical
wrappers around domain-warped ridged-multifractal noise. Two consequences made
them unusable as the substrate for an *identity* manipulation:

* shape was whatever the noise happened to do, so two peaks of the same
  morphology with different seeds read as different mountains -- which makes
  the `binding` family (recognising *which* landmark stands where) ill-posed;
* switching morphology changed size, summit sharpness, flank profile and
  surface texture simultaneously and by an uncontrolled amount, so `identity`
  had no magnitude -- only a count of how many peaks were swapped.

Here a peak is a short vector instead, split into three groups by the role each
attribute plays in the benchmark.

Size, form, pose
----------------
``size``  ``height``, ``width``
          What makes M1 bigger than M2. Part of a peak's identity -- it travels
          with the peak when `binding` permutes which landmark stands where --
          but **frozen under identity substitution**, because apparent size is
          the image's distance cue. A peak that got taller and a peak that moved
          closer produce the same angular-size change, so letting identity foils
          alter size would make `identity` and `metric` visually confusable and
          the two families would stop measuring different things.

``form``  ``peak_angle``, ``flank``, ``offset``, ``elongation``, ``ridges``,
          ``roughness``
          Identity proper: scale-free shape. This is what an identity foil
          changes, and the distance moved through this space is the family's
          difficulty parameter -- the analogue of metres for `metric`.

``pose``  ``x``, ``y``, ``rot_z``
          Not identity. Position is what `metric` moves; `rot_z` orients the
          summit offset and the arete phase in the world and is held fixed.

Geometry
--------
The base stays an exact circle of radius ``width`` with ``Z == 0`` on the
outermost ring (no flat overlapping skirt, no Z-fighting against the valley
floor). Inside it, height is a function of one normalised coordinate ``t``:
0 at the summit, 1 at the base ring, measured *from the offset summit* along the
ray that reaches the base circle. That keeps a leaning peak watertight.

    z = height * (1 - t**peak_angle) ** flank

``peak_angle`` sets the summit: < 1 is a needle (vertical at the apex), 1 a
straight cone, > 1 a plateau. ``flank`` sets the base: > 1 flares into a talus
apron, < 1 drops as a cliff. Aretes displace ``t`` with an envelope that
vanishes at both ends, so ridges bulge mid-flank -- where real aretes are --
while the summit height and the base ring stay exact.
"""

import math

import numpy as np

from fm_noise import fbm, ridged_fbm

# name -> (low, high). Ranges are the sampling envelope *and* the normaliser
# for shape distance, so a distance of 1.0 spans the full space.
FORM_SPEC = {
    "peak_angle": (0.70, 2.00),   # sharp .. straight cone .. broad-topped
    "flank": (1.15, 2.60),        # flared into a talus apron (>1 = meets the
                                  # ground tangentially, so there is no crease)
    "offset": (0.00, 0.20),       # summit displacement, as a fraction of width
    "crest": (0.00, 0.55),        # summit ridge length, as a fraction of width
    "roughness": (0.10, 0.55),    # relief amplitude
    "ruggedness": (0.55, 1.70),   # relief frequency: few broad spurs .. many
}
FORM_KEYS = tuple(FORM_SPEC)

# Peaks in one scene must be at least this far apart in form space. Measured
# against the within-identity noise floor (how much a peak's own silhouette
# changes as the camera walks around it): at 0.5 two peaks differ by only 1.4x
# that floor and are effectively the same landmark from a new angle, which makes
# `binding` unanswerable. At 1.0 the ratio is 3.8x.
MIN_SEPARATION = 1.0

# Size is sampled as a base radius plus an ASPECT (height / base radius), not as
# two independent ranges. Independent ranges let a 35 m summit sit on an 11 m
# base -- a 3:1 spire, which is what made the peaks read as tipis no matter what
# surface detail was applied. Real mountains are wide: the Matterhorn is roughly
# 0.5:1. This was the single largest realism factor found.
SIZE_SPEC = {
    "width": (15.0, 21.0),
    "aspect": (0.75, 1.35),
}

# Surface texture. RIDGED noise, not smooth fbm: smooth value noise produces
# rolling hills, and a peak covered in rolling hills reads as a draped tent.
# Ridged noise creases, and creases are what make a surface read as rock. Deliberately NOT part of identity: it is the same
# in character for every peak and constant for a given peak across viewpoints,
# so it adds realism without moving anything through form space.
TEXTURE_AMPLITUDE = 0.34
TEXTURE_FREQ = 0.17
TEXTURE_FINE_FREQ = 0.62
TEXTURE_FINE_MIX = 0.40

# Angular warp applied before aretes are laid down, so ridge spacing is
# irregular. Large enough to break the pinwheel, small enough that the
# ridge count stays readable.
RIDGE_JITTER = 0.38

# --- Erosion character. None of this is identity: it is the same in kind for
# --- every peak, and constant for a given peak across viewpoints. It exists
# --- because a peak built from a smooth profile plus a cosine reads as a tipi.

# Radians an arete rotates between summit and base. Straight radial ridges are
# the single strongest cue that a shape was generated rather than eroded.
RIDGE_TWIST = 0.55

# Exponent on the arete waveform. 1.0 is a plain cosine -- ridges and valleys
# equally wide. Above 1 the crests narrow and the ground between them broadens
# into a concave cirque, which is the actual arete/cirque morphology.
RIDGE_SHARPNESS = 2.4

# Couloirs. Sampled on a circle so the pattern is periodic in bearing (no seam
# at +/-pi) and constant along the fall line, so they run downhill like real
# gullies instead of wrapping around the cone.
# How hard the aretes bite. This is the expensive knob for identity stability:
# deep crests make the silhouette depend on whether one happens to sit on the
# skyline, which is viewpoint noise. Sharpness is nearly free by comparison, so
# aretes stay sharp and shallow rather than blunt and deep.
CREST_GAIN = 0.8

# A shoulder / buttress band partway up the flank. Surface texture alone leaves
# the *outline* a smooth cone, and the outline is what reads as "tipi". This
# pushes the silhouette out over a seeded height band so peaks acquire shoulders
# and spurs. Not identity: seeded per peak, same in kind for all of them.
# How far the toe of the mountain pulls in from the base circle, and at what
# scale. Depth 0.24 means the footprint wanders between 76% and 100% of the
# nominal base radius.
APRON_DEPTH = 0.24
APRON_SCALE = 2.2

SHOULDER_AMPLITUDE = 0.20
SHOULDER_SCALE = 1.7
SHOULDER_WIDTH = 0.26

GULLY_SCALE = 3.1
GULLY_AMPLITUDE = 0.055


# --------------------------------------------------------------------------- #
# Shape space
# --------------------------------------------------------------------------- #

def normalise(form):
    """Form dict -> vector in [0, 1]^5, the space distances are measured in."""
    return np.array([(float(form[k]) - FORM_SPEC[k][0]) /
                     (FORM_SPEC[k][1] - FORM_SPEC[k][0]) for k in FORM_KEYS])


def denormalise(vec):
    """Inverse of `normalise`; `ridges` is snapped back to a whole count."""
    out = {}
    for k, v in zip(FORM_KEYS, np.asarray(vec, dtype=float)):
        lo, hi = FORM_SPEC[k]
        val = lo + float(np.clip(v, 0.0, 1.0)) * (hi - lo)
        out[k] = round(val) if k == "ridges" else val
    return out


def shape_distance(a, b):
    """Euclidean distance in the normalised form space (0 .. sqrt(5))."""
    return float(np.linalg.norm(normalise(a) - normalise(b)))


def sample_form(rng):
    return denormalise(rng.random(len(FORM_KEYS)))


def sample_size(rng):
    """A peak's size as {height, width}, drawn through a plausible aspect."""
    w = float(rng.uniform(*SIZE_SPEC["width"]))
    a = float(rng.uniform(*SIZE_SPEC["aspect"]))
    return {"width": w, "height": w * a}


def sample_distinct_forms(rng, n, min_distance=MIN_SEPARATION, max_tries=4000):
    """
    `n` forms that are all at least `min_distance` apart.

    Peaks a model cannot tell apart make `binding` unanswerable, so separation
    is enforced at sampling time rather than hoped for.
    """
    forms = []
    for _ in range(max_tries):
        if len(forms) == n:
            return forms
        cand = sample_form(rng)
        if all(shape_distance(cand, f) >= min_distance for f in forms):
            forms.append(cand)
    raise RuntimeError(
        f"could not place {n} forms at least {min_distance} apart "
        f"(got {len(forms)}); loosen min_distance or widen FORM_SPEC")


def substitute_form(rng, form, distance, avoid=(), min_distance=MIN_SEPARATION,
                    tolerance=0.12, max_tries=3000):
    """
    A form about `distance` away from `form`, and `min_distance` from `avoid`.

    This is what an identity foil applies: a landmark the target does not
    contain, at a *controlled* dissimilarity, so identity has a difficulty dial
    rather than only a count of changed peaks.
    """
    base = normalise(form)
    best, best_err = None, None
    for _ in range(max_tries):
        direction = rng.normal(size=len(FORM_KEYS))
        direction /= np.linalg.norm(direction)
        cand_vec = np.clip(base + direction * distance, 0.0, 1.0)
        cand = denormalise(cand_vec)
        if any(shape_distance(cand, f) < min_distance for f in avoid):
            continue
        err = abs(shape_distance(cand, form) - distance)
        if best_err is None or err < best_err:
            best, best_err = cand, err
        if err <= tolerance:
            return cand
    if best is not None and best_err <= tolerance * 2.5:
        return best
    return None


def max_form_distance():
    """Largest achievable separation, for clamping a ladder rung."""
    return math.sqrt(len(FORM_KEYS))


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _crest_distance(X, Y, width, offset, crest):
    """
    Distance from the summit CREST -- a segment, not a point.

    A single apex is what produced the "lines running to the top": every ridge
    and gully, however it was generated, had to converge there. A crest of
    length `crest * width` gives ridges somewhere else to end, and `crest = 0`
    still recovers a point summit for a clean cone.
    """
    cx, cy = offset * width, 0.0                 # local frame; rot_z orients it
    half = 0.5 * crest * width
    # segment runs along local +/-X through the offset centre
    dx = np.clip(X - cx, -half, half)
    return np.hypot(X - cx - dx, Y - cy)


def build_peak(form, height, width, rot_z=0.0, seed=0, n_r=140, n_theta=320,
               texture=True):
    """
    Evaluate a parametric peak on a polar grid.

    Returns (X, Y, Z) in the peak's local frame, with Z == `height` at the
    summit and Z == 0 on the outermost ring, exactly.

    The grid runs from the base centre outward, and every relief term is a
    function of 2D position rather than of bearing. Bearing-driven relief is
    what makes ridges radiate from the apex like tent seams; 2D noise has no
    such focus, so spurs and gullies wander the way eroded ones do.
    """
    W = float(width)
    r = np.linspace(0.0, W, n_r)
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(r, th, indexing="ij")
    X, Y = R * np.cos(T), R * np.sin(T)

    off = float(form["offset"])
    crest = float(form.get("crest", 0.0))
    d = _crest_distance(X, Y, W, off, crest)
    # Normalise by the crest distance at the base ring along the same bearing,
    # so Z reaches exactly 0 on the outermost ring whatever the crest does.
    d_edge = _crest_distance(W * np.cos(th), W * np.sin(th), W, off, crest)

    # Irregular footprint. The mountain stops at a radius that wanders with 2D
    # noise instead of at the base circle; everything outside that is clipped to
    # t == 1, i.e. Z == 0, and lies flat on the valley floor. Without this the
    # toe is a perfect circle and the peak reads as an object set down on the
    # ground rather than part of it.
    apron = ridged_fbm(X * (APRON_SCALE / W), Y * (APRON_SCALE / W),
                       freq=1.0, octaves=3, seed=int(seed) + 271)
    apron = (apron - apron.min()) / max(float(np.ptp(apron)), 1e-9)
    shrink = 1.0 - APRON_DEPTH * apron
    t = np.clip(d / np.maximum(d_edge[None, :] * shrink, 1e-9), 0.0, 1.0)

    # Relief: 2D ridged noise, centred so it neither inflates nor shrinks the
    # peak, with an envelope that vanishes at summit and base.
    # Aim for a handful of spurs across the peak, not a field of hoodoos:
    # `ruggedness` 0.55..1.70 maps to roughly 3..6 relief features across the
    # full 2W diameter.
    scale = float(form["ruggedness"]) / max(W, 1e-6) * 1.55
    relief = ridged_fbm(X * scale, Y * scale, freq=1.0, octaves=5,
                        seed=int(seed) + 613)
    relief = relief - relief.mean()
    # Envelope is ~1 almost everywhere and only falls at the very base ring.
    # sin(pi t) vanished at BOTH ends, which is what made flat summits perfectly
    # flat (relief switched off exactly where a plateau is) and made the toe a
    # perfect circle against the valley floor. Letting relief live at the summit
    # gives broad tops a rugged crown; letting it live near the base means the
    # mountain reaches Z == 0 at a wandering radius, so the visible toe is
    # irregular and the leftover flat annulus merges invisibly with the ground.
    env = 1.0 - np.power(t, 6.0)
    t = np.clip(t - float(form["roughness"]) * env * relief * 0.95, 0.0, 1.0)

    Z = float(height) * np.power(
        np.maximum(1.0 - np.power(t, float(form["peak_angle"])), 0.0),
        float(form["flank"]))

    if texture:
        fine = ridged_fbm(X * scale * 4.5, Y * scale * 4.5, freq=1.0, octaves=4,
                          seed=int(seed) + 991)
        fine = fine - fine.mean()
        Z = Z * (1.0 + TEXTURE_AMPLITUDE * fine * (1.0 - np.power(t, 8.0)))

    Z[-1, :] = 0.0
    Z = np.maximum(Z, 0.0)
    peak = Z.max()
    if peak > 1e-9:
        # The crest is a line, so the grid may not land exactly on its highest
        # point; rescale so `height` is the summit altitude by definition.
        Z *= float(height) / peak
    Z[-1, :] = 0.0

    if rot_z:
        c, s_ = math.cos(rot_z), math.sin(rot_z)
        X, Y = X * c - Y * s_, X * s_ + Y * c
    return X, Y, Z


def describe(form):
    """Short human-readable label, for figures and the bank viewer."""
    a, f = form["peak_angle"], form["flank"]
    summit = "sharp" if a < 0.95 else ("flat-topped" if a > 1.7 else "conical")
    base = "cliffed" if f < 1.0 else ("flared" if f > 1.7 else "even")
    lean = "" if form["offset"] < 0.08 else " leaning"
    c = form.get("crest", 0.0)
    shape = "peak" if c < 0.15 else ("ridge" if c > 0.38 else "crested")
    rug = "smooth" if form["roughness"] < 0.25 else (
        "rugged" if form["roughness"] > 0.42 else "broken")
    if a > 1.6:
        summit = "broad"
    return f"{summit} {shape}/{base}{lean}, {rug}"
