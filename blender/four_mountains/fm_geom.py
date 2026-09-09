"""
Camera geometry for the Four Mountains benchmark. Pure numpy -- no `bpy`.

Mirrors the conventions baked into the .blend so that metadata written by
`render_dataset.FourMountainsRenderer` can be re-derived offline:

* the camera sits on a spherical orbit around the valley centre
  (`FourMountainsRenderer.set_camera_orbit`),
* it carries a TRACK_TO constraint with track axis -Z and up axis +Y, which
  is the usual "look at the target, keep world Z up" basis,
* the sensor is Blender's 36 mm default, fit to the longer image axis.

Everything downstream -- the in-frame test used when sampling layouts, and the
egocentric `L2` text serialisation -- goes through this module, so the two can
never drift apart.
"""

import math

import numpy as np

SENSOR_MM = 36.0
WORLD_UP = np.array([0.0, 0.0, 1.0])


def camera_position(azimuth_deg, elevation_deg, radius, target_z):
    """Identical to FourMountainsRenderer.set_camera_orbit."""
    phi = math.radians(elevation_deg)
    theta = math.radians(azimuth_deg)
    return np.array([
        radius * math.cos(phi) * math.cos(theta),
        radius * math.cos(phi) * math.sin(theta),
        radius * math.sin(phi) + target_z,
    ])


def camera_basis(cam_pos, target):
    """(forward, right, up) unit vectors for a TRACK_TO(-Z, up=Y) camera."""
    cam_pos = np.asarray(cam_pos, dtype=float)
    forward = np.asarray(target, dtype=float) - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, WORLD_UP)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


def half_fov(lens_mm, res_x, res_y):
    """(horizontal, vertical) half angles in radians, sensor fit to the long axis."""
    if res_x >= res_y:
        half_w = SENSOR_MM / 2.0
        half_h = half_w * res_y / res_x
    else:
        half_h = SENSOR_MM / 2.0
        half_w = half_h * res_x / res_y
    return math.atan(half_w / lens_mm), math.atan(half_h / lens_mm)


def to_camera_frame(points, cam_pos, target):
    """
    Project world points into camera axes.

    Returns (x, y, depth) arrays: x is rightward, y is up, depth is along the
    view direction. Points behind the camera have depth <= 0.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    forward, right, up = camera_basis(cam_pos, target)
    d = points - np.asarray(cam_pos, dtype=float)
    return d @ right, d @ up, d @ forward


def peak_silhouette(x, y, base_z, height, radius, cam_pos):
    """
    World points tracing the silhouette of one peak, as seen from `cam_pos`.

    A peak is a cone-ish body of `radius` at its base rising to `height`. Its
    outline against the sky is bounded by the two base corners perpendicular to
    the line of sight, the summit, and the near base point -- checking those
    four is enough to know whether the whole peak clears the frame.

    Testing the base radius as a *disc* would be wrong: a peak extends upward
    from its base, not downward, and treating it as a sphere rejects layouts
    that render perfectly well.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    centre = np.array([float(x), float(y), float(base_z)])
    los = centre[:2] - cam_pos[:2]
    n = np.linalg.norm(los)
    perp = np.array([-los[1], los[0]]) / n if n > 1e-9 else np.array([1.0, 0.0])
    off = perp * float(radius)
    return np.array([
        [centre[0] + off[0], centre[1] + off[1], base_z],
        [centre[0] - off[0], centre[1] - off[1], base_z],
        [centre[0], centre[1], base_z + float(height)],
        [centre[0], centre[1], base_z],
    ])


def points_in_frame(points, cam_pos, target, lens_mm, res_x, res_y, margin=0.0):
    """True when every world point projects inside the frame."""
    x, y, depth = to_camera_frame(points, cam_pos, target)
    if np.any(depth <= 0):
        return False
    hfov, vfov = half_fov(lens_mm, res_x, res_y)
    hfov *= (1.0 - margin)
    vfov *= (1.0 - margin)
    safe = np.maximum(depth, 1e-6)
    return bool(np.all(np.abs(np.arctan2(x, safe)) <= hfov)
                and np.all(np.abs(np.arctan2(y, safe)) <= vfov))


def layout_in_frame(peaks, base_z, cam_pos, target, lens_mm, res_x, res_y, margin=0.02):
    """True when every peak in a layout is fully visible from this camera."""
    pts = np.vstack([
        peak_silhouette(p["x"], p["y"], base_z, p["height"], p["base_radius"], cam_pos)
        for p in peaks
    ])
    return points_in_frame(pts, cam_pos, target, lens_mm, res_x, res_y, margin)


def angular_radius(points, radii, cam_pos):
    """Half-angle each peak's base subtends, in degrees."""
    dist = distance_from_camera(points, cam_pos)
    return np.degrees(np.arctan2(np.atleast_1d(radii), np.maximum(dist, 1e-6)))


def bearing_from_camera(points, cam_pos, target):
    """Signed horizontal angle from the view axis, in degrees (right positive)."""
    x, _, depth = to_camera_frame(points, cam_pos, target)
    return np.degrees(np.arctan2(x, np.maximum(depth, 1e-6)))


def distance_from_camera(points, cam_pos):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return np.linalg.norm(points - np.asarray(cam_pos, dtype=float), axis=1)


def compass_bearing(points):
    """
    Allocentric compass bearing of each point from the valley centre, degrees
    clockwise from north. Matches the game's azimuth convention.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return np.degrees(np.arctan2(points[:, 0], points[:, 1])) % 360.0
