"""
Vectorised numpy noise primitives used by the Four Mountains terrain generators.

Everything here is pure numpy (no bpy), so the heightfields can be inspected,
plotted and unit-tested outside of Blender.
"""

import numpy as np

_LATTICE_CACHE = {}


def _lattice(seed, size=512):
    """Random value lattice, cached per seed (tileable via modulo indexing)."""
    key = (seed, size)
    if key not in _LATTICE_CACHE:
        _LATTICE_CACHE[key] = np.random.default_rng(seed).random((size, size))
    return _LATTICE_CACHE[key]


def value_noise(x, y, freq=1.0, seed=0, size=512):
    """Smooth-interpolated 2D value noise in [0, 1]."""
    grid = _lattice(seed, size)
    xf = np.asarray(x, dtype=np.float64) * freq
    yf = np.asarray(y, dtype=np.float64) * freq

    xi = np.floor(xf).astype(np.int64)
    yi = np.floor(yf).astype(np.int64)
    tx = xf - xi
    ty = yf - yi

    # Quintic smoothstep (C2 continuous -> no visible lattice creases)
    sx = tx * tx * tx * (tx * (tx * 6.0 - 15.0) + 10.0)
    sy = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0)

    i0, i1 = xi % size, (xi + 1) % size
    j0, j1 = yi % size, (yi + 1) % size

    v00 = grid[i0, j0]
    v10 = grid[i1, j0]
    v01 = grid[i0, j1]
    v11 = grid[i1, j1]

    a = v00 + (v10 - v00) * sx
    b = v01 + (v11 - v01) * sx
    return a + (b - a) * sy


def fbm(x, y, freq=1.0, octaves=6, lacunarity=2.03, gain=0.5, seed=0):
    """Classic fractional Brownian motion, normalised to roughly [0, 1]."""
    total = np.zeros_like(np.asarray(x, dtype=np.float64))
    amp = 1.0
    norm = 0.0
    f = freq
    for o in range(octaves):
        total += amp * value_noise(x, y, freq=f, seed=seed + 17 * o)
        norm += amp
        amp *= gain
        f *= lacunarity
    return total / max(norm, 1e-9)


def ridged_fbm(x, y, freq=1.0, octaves=6, lacunarity=2.07, gain=0.5,
               sharpness=1.9, seed=0):
    """
    Ridged multifractal: inverts and squares each octave so crests form sharp
    arêtes instead of rounded blobs. Returns roughly [0, 1].
    """
    total = np.zeros_like(np.asarray(x, dtype=np.float64))
    amp = 1.0
    norm = 0.0
    f = freq
    weight = np.ones_like(total)
    for o in range(octaves):
        n = value_noise(x, y, freq=f, seed=seed + 31 * o)
        ridge = (1.0 - np.abs(2.0 * n - 1.0)) ** sharpness
        # Feed the previous octave forward so ridges reinforce along crests
        ridge = ridge * np.clip(weight, 0.0, 1.0)
        weight = 0.65 + 0.65 * ridge
        total += amp * ridge
        norm += amp
        amp *= gain
        f *= lacunarity
    return total / max(norm, 1e-9)


def domain_warp(x, y, freq=0.05, strength=4.0, seed=0):
    """Offset sample coordinates by a noise field -> organic, non-radial shapes."""
    wx = (value_noise(x, y, freq=freq, seed=seed + 991) - 0.5) * 2.0
    wy = (value_noise(x, y, freq=freq, seed=seed + 4523) - 0.5) * 2.0
    return x + strength * wx, y + strength * wy


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cosine_taper(norm_r):
    """1 at the centre, exactly 0 at norm_r >= 1 (kills flat overlapping skirts)."""
    return 0.5 * (1.0 + np.cos(np.pi * np.clip(norm_r, 0.0, 1.0)))


def plateau_taper(norm_r, hold=0.35):
    """Footprint fade that stays flat out to `hold` before dropping to 0 at 1."""
    return 1.0 - smoothstep(hold, 1.0, norm_r)
