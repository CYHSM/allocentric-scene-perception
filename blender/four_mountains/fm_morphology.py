"""
Mountain morphology library for the Four Mountains Task.

Each morphology is a pure numpy heightfield evaluated on a polar grid. All of
them are clipped so that Z -> 0 exactly at the base radius, which keeps the
peaks free of flat overlapping skirts and lets them be translated anywhere on
the valley floor without Z-fighting against the terrain.

Add a new landform by writing a function with the signature below and
registering it in MORPHOLOGIES.
"""

import numpy as np

from fm_noise import (cosine_taper, domain_warp, fbm, plateau_taper,
                      ridged_fbm, smoothstep)


def _elliptic(Xw, Yw, base_radius, aspect_x, aspect_y, angle_deg):
    """Rotated elliptical radius + its matching taper (for elongated landforms)."""
    a = np.radians(angle_deg)
    xr = Xw * np.cos(a) - Yw * np.sin(a)
    yr = Xw * np.sin(a) + Yw * np.cos(a)
    r_ell = np.sqrt((xr / aspect_x) ** 2 + (yr / aspect_y) ** 2)
    norm = np.clip(r_ell / base_radius, 0.0, 1.0)
    return xr, yr, norm, cosine_taper(norm)


# --------------------------------------------------------------------------- #
# Morphologies. ctx carries the shared polar/warped grids and noise fields.
# --------------------------------------------------------------------------- #

def _horn(ctx):
    """Matterhorn-style pyramidal horn: four razor arêtes, glacial cirques between."""
    n, T, tap, ridge = ctx["norm_r"], ctx["Tw"], ctx["taper"], ctx["ridge"]
    aretes = np.abs(np.cos(2.0 * T + 0.3)) ** 3.2
    cirque = 0.34 * (1.0 - aretes) * (n ** 1.25)
    profile = (1.0 - n) ** 1.95
    return profile * (0.50 + 0.50 * aretes) - cirque + 0.32 * ridge * (1.0 - n), tap


def _ridge(ctx):
    """Elongated alpine massif with a knife-edge spine and a saddled crest."""
    xr, yr, n, tap = _elliptic(ctx["Xw"], ctx["Yw"], ctx["R0"], 1.55, 0.70, 20)
    saddle = 0.74 + 0.28 * np.cos(2.7 * xr / ctx["R0"])
    spine = np.exp(-((yr / (ctx["R0"] * 0.22)) ** 2))
    profile = (1.0 - n) ** 1.35
    return profile * saddle + 0.24 * spine + 0.30 * ctx["ridge"] * (1.0 - n), tap


def _mesa(ctx):
    """Flat-topped tableland: sheer cliff band under a level summit, talus apron."""
    n, T, ridge = ctx["norm_r"], ctx["Tw"], ctx["ridge"]
    tap = plateau_taper(n, 0.40)   # keep the summit genuinely level
    cap = 1.0 - smoothstep(0.30, 0.46, n)          # level summit plateau
    apron = 0.42 * (1.0 - smoothstep(0.46, 1.0, n))  # scree skirt below the scarp
    benches = 0.06 * np.sin(9.0 * np.pi * n) * smoothstep(0.46, 0.95, n)
    faults = 0.05 * np.sin(3.0 * T + 0.8)
    return cap * (0.94 + faults) + apron + benches + 0.18 * ridge * (1.0 - n), tap


def _dome(ctx):
    """Stratovolcano: concave-up flanks scored by radiating fluvial ravines."""
    n, T, tap, ridge = ctx["norm_r"], ctx["Tw"], ctx["taper"], ctx["ridge"]
    profile = 1.0 / (1.0 + 3.8 * (n ** 1.9))
    ravines = 0.11 * np.sin(9.0 * T) * (n ** 1.2)
    return profile + ravines + 0.20 * ridge * (1.0 - n), tap


def _caldera(ctx):
    """Collapsed volcano: raised crater rim around a sunken, snow-holding basin."""
    n, T, tap, ridge = ctx["norm_r"], ctx["Tw"], ctx["taper"], ctx["ridge"]
    rim_r = 0.26
    rim = np.exp(-((n - rim_r) / 0.13) ** 2)
    notch = 0.22 * np.cos(3.0 * T + 1.1)          # breached rim on one flank
    flank = 1.0 / (1.0 + 5.0 * (n ** 1.7))
    crater = 0.55 * np.exp(-(n / (rim_r * 0.85)) ** 2)
    return flank * 0.72 + rim * (0.42 + notch * 0.30) - crater + 0.18 * ridge * (1.0 - n), tap


def _butte(ctx):
    """Isolated rock tower / spire: near-vertical shaft on a wide debris cone."""
    n, T, tap, ridge = ctx["norm_r"], ctx["Tw"], ctx["taper"], ctx["ridge"]
    shaft = 1.0 / (1.0 + np.exp((n - 0.22) * 20.0))
    fluting = 1.0 + 0.10 * np.sin(11.0 * T)
    debris = 0.34 * (1.0 - n) ** 1.6
    return shaft * fluting * 0.92 + debris + 0.14 * ridge * (1.0 - n), tap


def _twin(ctx):
    """Two summits of unequal height joined by a col."""
    Xw, Yw, R0, tap, n = ctx["Xw"], ctx["Yw"], ctx["R0"], ctx["taper"], ctx["norm_r"]
    d1 = np.sqrt((Xw - 0.30 * R0) ** 2 + (Yw - 0.14 * R0) ** 2) / R0
    d2 = np.sqrt((Xw + 0.32 * R0) ** 2 + (Yw + 0.18 * R0) ** 2) / R0
    p1 = np.exp(-(d1 / 0.36) ** 1.7)
    p2 = 0.82 * np.exp(-(d2 / 0.40) ** 1.7)
    col = 0.30 * np.exp(-(np.abs(Yw + 0.02 * R0) / (R0 * 0.20)) ** 2) * (1.0 - n)
    return np.maximum(p1, p2) + col * 0.5 + 0.22 * ctx["ridge"] * (1.0 - n), tap


def _sawtooth(ctx):
    """Serrated crest: a row of pinnacles and notches along one strike direction."""
    xr, yr, n, tap = _elliptic(ctx["Xw"], ctx["Yw"], ctx["R0"], 1.70, 0.62, -35)
    spine = np.exp(-((yr / (ctx["R0"] * 0.17)) ** 2))
    teeth = 0.5 + 0.5 * np.abs(np.sin(3.4 * np.pi * xr / ctx["R0"]))
    profile = (1.0 - n) ** 1.20
    return profile * (0.46 + 0.54 * teeth * spine) + 0.34 * ctx["ridge"] * (1.0 - n), tap


def _cuesta(ctx):
    """Tilted plateau: a cliff scarp on one side, a long gentle dip slope opposite."""
    Xw, R0, n, ridge = ctx["Xw"], ctx["R0"], ctx["norm_r"], ctx["ridge"]
    tap = plateau_taper(n, 0.30)
    u = np.clip(Xw / R0, -1.0, 1.0)
    scarp = smoothstep(-0.42, -0.18, u)            # abrupt rise on the -X flank
    dip = 1.0 - 0.62 * smoothstep(-0.18, 1.0, u)   # long shallow back slope
    return scarp * dip * (1.0 - n) ** 0.75 + 0.20 * ridge * (1.0 - n), tap


def _cone(ctx):
    """Symmetric ash cone (Fuji-like): straight flanks, small summit crater."""
    n, T, tap, ridge = ctx["norm_r"], ctx["Tw"], ctx["taper"], ctx["ridge"]
    profile = (1.0 - n) ** 1.10
    gullies = 0.055 * np.sin(17.0 * T) * (n ** 1.4)
    crater = 0.10 * np.exp(-(n / 0.09) ** 2)
    return profile + gullies - crater + 0.12 * ridge * (1.0 - n), tap


def _whaleback(ctx):
    """Roche moutonnée: ice-smoothed stoss side, plucked and steep on the lee."""
    xr, yr, n, tap = _elliptic(ctx["Xw"], ctx["Yw"], ctx["R0"], 1.45, 0.85, 60)
    u = np.clip(xr / (ctx["R0"] * 1.45), -1.0, 1.0)
    asym = 1.0 - 0.45 * smoothstep(0.05, 0.85, u)
    profile = np.cos(0.5 * np.pi * np.clip(n, 0, 1)) ** 1.5
    return profile * asym + 0.14 * ctx["ridge"] * (1.0 - n), tap


def _massif(ctx):
    """Broad complex block with several subsidiary summits and hanging valleys."""
    Xw, Yw, R0, n = ctx["Xw"], ctx["Yw"], ctx["R0"], ctx["norm_r"]
    tap = plateau_taper(n, 0.22)
    rng = np.random.default_rng(ctx["seed"] + 77)
    acc = np.zeros_like(Xw)
    for _ in range(4):
        cx, cy = rng.uniform(-0.42, 0.42, 2) * R0
        w = rng.uniform(0.28, 0.46)
        h = rng.uniform(0.62, 1.0)
        d = np.sqrt((Xw - cx) ** 2 + (Yw - cy) ** 2) / R0
        acc = np.maximum(acc, h * np.exp(-(d / w) ** 1.8))
    return acc * (1.0 - 0.25 * n) + 0.30 * ctx["ridge"] * (1.0 - n), tap


MORPHOLOGIES = {
    "horn": _horn,
    "ridge": _ridge,
    "mesa": _mesa,
    "dome": _dome,
    "caldera": _caldera,
    "butte": _butte,
    "twin": _twin,
    "sawtooth": _sawtooth,
    "cuesta": _cuesta,
    "cone": _cone,
    "whaleback": _whaleback,
    "massif": _massif,
}

# Kept so older scripts/configs using the previous name keep working.
MORPHOLOGY_ALIASES = {"mesa_crag": "mesa"}


def morphology_names():
    return sorted(MORPHOLOGIES)


def build_heightfield(morphology, height=16.0, base_radius=12.0,
                      n_r=96, n_theta=220, seed=42, relief=1.0):
    """
    Evaluate a morphology on a polar grid.

    `relief` scales the fine erosion detail without changing the summit height.

    Returns (X, Y, Z) arrays of shape (n_r, n_theta) in the mountain's local
    frame, with Z == 0 on the outermost ring.
    """
    morphology = MORPHOLOGY_ALIASES.get(morphology, morphology)
    if morphology not in MORPHOLOGIES:
        raise ValueError(
            f"Unknown morphology {morphology!r}. Available: {morphology_names()}"
        )

    r_vals = np.linspace(0.0, base_radius, n_r)
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(r_vals, theta_vals, indexing="ij")

    X = R * np.cos(T)
    Y = R * np.sin(T)

    # Domain warping breaks the radial symmetry of the polar parameterisation.
    Xw, Yw = domain_warp(X, Y, freq=0.055, strength=base_radius * 0.16, seed=seed)
    Tw = np.arctan2(Yw, Xw)
    norm_r = np.clip(R / base_radius, 0.0, 1.0)

    ctx = {
        "X": X, "Y": Y, "Xw": Xw, "Yw": Yw, "Tw": Tw,
        "R": R, "R0": base_radius, "norm_r": norm_r,
        "taper": cosine_taper(norm_r),
        "ridge": ridged_fbm(Xw, Yw, freq=0.075, octaves=6, seed=seed),
        "fbm": fbm(Xw, Yw, freq=0.05, octaves=5, seed=seed + 5),
        "erosion": ridged_fbm(Xw, Yw, freq=0.13, octaves=3, seed=seed + 11),
        "gullies": ridged_fbm(Xw, Yw, freq=0.38, octaves=2, seed=seed + 23),
        "seed": seed,
    }

    shape, taper = MORPHOLOGIES[morphology](ctx)

    # Erosion detail: broad undulation plus fine gullies, faded at the footprint edge.
    detail = (ctx["fbm"] - 0.5) * 0.34 * (1.0 - norm_r) ** 0.6
    detail += (ctx["erosion"] - 0.40) * 0.62 * relief * (1.0 - norm_r) ** 0.60
    detail += (ctx["gullies"] - 0.42) * 0.26 * relief * (1.0 - norm_r) ** 0.90
    Z = np.maximum((shape + detail) * taper, 0.0)

    # Normalise so `height` is the true summit altitude for every morphology.
    peak = float(Z.max())
    Z = Z * (height / peak) if peak > 1e-6 else Z

    # The outermost ring must be exactly zero for a seamless join to the valley.
    Z[-1, :] = 0.0
    return X, Y, Z
