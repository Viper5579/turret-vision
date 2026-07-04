"""Lead prediction: constant-velocity intercept + gravity drop (D7).

WHY no drag: at 3-6 m indoor ranges drag on an airsoft BB is second-order;
modeling it before muzzle velocity consistency can even be measured is
precision theater. WHY gravity anyway: at 50 m/s over 5 m, tof ~ 0.1 s,
drop = g*t^2/2 ~ 5 cm — bigger than the acceptable miss distance. One line
of math that matters.

Geometry: the target sits at range r along the line of sight (az, el); its
angular rates x range give tangential velocity. The intercept time solves the
standard quadratic |p + v*t| = v_projectile * t (find where a bullet fired
NOW meets the target flying straight). Aim = direction of that meeting point,
plus a pitch-up of atan(drop/r) to spend the gravity drop.

The max_lead clamp is a sanity valve: a computed lead bigger than the clamp
means bad inputs (rate spike, wrong range), not a real lead — aiming 40 deg
off the target because of one noisy frame is worse than lagging it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.81


@dataclass(frozen=True)
class LeadSolution:
    yaw_deg: float          # aim azimuth   (same frame as the input az/el)
    pitch_deg: float        # aim elevation, gravity included
    tof_s: float            # projectile time of flight to intercept
    lead_deg: float         # total angular offset from the target's position
    clamped: bool


def _unit(az_rad: float, el_rad: float) -> tuple[float, float, float]:
    # x right, y up, z forward (camera frame, matching geometry.py signs)
    return (math.cos(el_rad) * math.sin(az_rad),
            math.sin(el_rad),
            math.cos(el_rad) * math.cos(az_rad))


class LeadPredictor:
    def __init__(self, projectile_speed_mps: float = 50.0,
                 gravity_comp: bool = True, max_lead_deg: float = 15.0, **_unused):
        self._vp = projectile_speed_mps
        self._gravity = gravity_comp
        self._max_lead = max_lead_deg

    def solve(self, az_deg: float, el_deg: float,
              az_rate_dps: float, el_rate_dps: float,
              range_m: float) -> LeadSolution | None:
        if self._vp <= 0 or range_m <= 0:
            return None
        az, el = math.radians(az_deg), math.radians(el_deg)
        p = tuple(range_m * c for c in _unit(az, el))
        # Tangential velocity from angular rates (small-angle basis vectors:
        # d(unit)/daz and d(unit)/del scaled by r and the rates).
        waz, wel = math.radians(az_rate_dps), math.radians(el_rate_dps)
        e_az = (math.cos(az), 0.0, -math.sin(az))              # horizontal tangent
        e_el = (-math.sin(el) * math.sin(az), math.cos(el),
                -math.sin(el) * math.cos(az))                  # vertical tangent
        v = tuple(range_m * (waz * math.cos(el) * a + wel * b)
                  for a, b in zip(e_az, e_el, strict=True))

        # |p + v t| = vp t  ->  (|v|^2 - vp^2) t^2 + 2(p.v) t + |p|^2 = 0
        vv = sum(c * c for c in v) - self._vp ** 2
        pv = 2.0 * sum(a * b for a, b in zip(p, v, strict=True))
        pp = sum(c * c for c in p)
        if abs(vv) < 1e-9:
            tof = -pp / pv if pv < 0 else None
        else:
            disc = pv * pv - 4.0 * vv * pp
            if disc < 0:
                tof = None     # target strictly outruns the projectile
            else:
                sq = math.sqrt(disc)
                roots = sorted(((-pv - sq) / (2 * vv), (-pv + sq) / (2 * vv)))
                tof = next((r for r in roots if r > 0), None)
        if tof is None or not math.isfinite(tof):
            return None

        hit = tuple(a + b * tof for a, b in zip(p, v, strict=True))
        aim_az = math.degrees(math.atan2(hit[0], hit[2]))
        aim_el = math.degrees(math.atan2(hit[1], math.hypot(hit[0], hit[2])))
        if self._gravity:
            r_hit = math.sqrt(sum(c * c for c in hit))
            aim_el += math.degrees(math.atan2(0.5 * G * tof * tof, r_hit))

        lead = math.hypot(aim_az - az_deg, aim_el - el_deg)
        clamped = lead > self._max_lead
        if clamped:
            scale = self._max_lead / lead
            aim_az = az_deg + (aim_az - az_deg) * scale
            aim_el = el_deg + (aim_el - el_deg) * scale
            lead = self._max_lead
        return LeadSolution(aim_az, aim_el, tof, lead, clamped)
