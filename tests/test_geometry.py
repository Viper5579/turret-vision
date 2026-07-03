import math

from turretvision.calib.geometry import PixelAngleMapper


def make_mapper(w=800, h=600, hfov=70.0):
    return PixelAngleMapper(w, h, intrinsics_file=None, fallback_hfov_deg=hfov)


def test_center_is_zero():
    m = make_mapper()
    az, el = m.pixel_to_angles(400, 300)
    assert abs(az) < 1e-9 and abs(el) < 1e-9


def test_right_edge_is_half_hfov():
    """WHY this is the key correctness check: the fallback focal is defined by
    'the right edge of the image = +hfov/2'. If this fails, every angle the
    pipeline emits is scaled wrong."""
    m = make_mapper(hfov=70.0)
    az, _ = m.pixel_to_angles(800, 300)
    assert math.isclose(az, 35.0, abs_tol=0.05)


def test_up_is_positive_elevation():
    m = make_mapper()
    _, el = m.pixel_to_angles(400, 100)  # above center in image coords
    assert el > 0


def test_roundtrip():
    m = make_mapper()
    u0, v0 = 612.0, 145.0
    az, el = m.pixel_to_angles(u0, v0)
    u1, v1 = m.angles_to_pixel(az, el)
    assert abs(u1 - u0) < 1e-6 and abs(v1 - v0) < 1e-6


def test_boresight_offset_applied():
    m = PixelAngleMapper(800, 600, fallback_hfov_deg=70.0,
                         boresight_yaw_deg=1.5, boresight_pitch_deg=-0.5)
    az, el = m.pixel_to_angles(400, 300)
    assert math.isclose(az, 1.5, abs_tol=1e-9)
    assert math.isclose(el, -0.5, abs_tol=1e-9)
