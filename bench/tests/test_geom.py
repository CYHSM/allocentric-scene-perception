"""Camera geometry must agree with the conventions baked into the .blend."""

import math

import numpy as np

import fm_geom as geom


def test_camera_position_matches_set_camera_orbit():
    # The same spherical formula FourMountainsRenderer.set_camera_orbit uses.
    az, elev, radius, tz = 37.0, 17.0, 94.0, 6.0
    phi, theta = math.radians(elev), math.radians(az)
    expected = np.array([radius * math.cos(phi) * math.cos(theta),
                         radius * math.cos(phi) * math.sin(theta),
                         radius * math.sin(phi) + tz])
    assert np.allclose(geom.camera_position(az, elev, radius, tz), expected)


def test_basis_is_orthonormal_and_points_at_the_target():
    cam = geom.camera_position(120.0, 17.0, 94.0, 6.0)
    tgt = np.array([0.0, 0.0, 6.0])
    f, r, u = geom.camera_basis(cam, tgt)
    for v in (f, r, u):
        assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.isclose(f @ r, 0.0, atol=1e-9)
    assert np.isclose(f @ u, 0.0, atol=1e-9)
    assert np.allclose(f, (tgt - cam) / np.linalg.norm(tgt - cam))
    assert r[2] == 0.0 or np.isclose(r[2], 0.0, atol=1e-9)   # right stays level


def test_half_fov_uses_the_long_image_axis():
    h, v = geom.half_fov(28.0, 640, 440)
    assert np.isclose(h, math.atan(18.0 / 28.0))
    assert np.isclose(v, math.atan((18.0 * 440 / 640) / 28.0))
    assert h > v


def test_valley_centre_projects_to_the_middle_of_the_frame():
    cam = geom.camera_position(210.0, 17.0, 94.0, 6.0)
    tgt = np.array([0.0, 0.0, 6.0])
    x, y, depth = geom.to_camera_frame(tgt[None, :], cam, tgt)
    assert np.isclose(x[0], 0.0, atol=1e-9)
    assert np.isclose(y[0], 0.0, atol=1e-9)
    assert depth[0] > 0


def test_compass_bearing_convention():
    # +y is north, +x is east.
    pts = np.array([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, -10.0, 0.0]])
    assert np.allclose(geom.compass_bearing(pts), [0.0, 90.0, 180.0])
