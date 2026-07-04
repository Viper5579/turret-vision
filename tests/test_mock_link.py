"""MockLink firmware behaviors: trapezoidal convergence, travel-limit clamping,
e-stop, link-dead safe-hold, and wire integrity. Simulated clock throughout —
no sleeps, fully deterministic."""
from turretvision.link import protocol
from turretvision.link.base import AimOutput
from turretvision.link.mock_link import MockLink


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def aim(t, yaw, pitch=0.0, valid=True, estop=False):
    return AimOutput(t, valid, yaw, pitch, 0.0, 0.0, 1.0 if valid else 0.0, None, estop)


def run(link, clock, seconds, yaw=None, pitch=0.0, dt=0.01, send=True, **aim_kw):
    telem = None
    for _ in range(int(seconds / dt)):
        clock.t += dt
        if send and yaw is not None:
            link.send_aim(aim(clock.t, yaw, pitch, **aim_kw))
        telem = link.poll_telemetry()
    return telem


def test_converges_to_setpoint_with_limited_velocity():
    clock = Clock()
    link = MockLink(time_fn=clock, yaw_vmax_dps=240.0)
    rates = []
    for _ in range(200):
        clock.t += 0.01
        link.send_aim(aim(clock.t, 30.0, 10.0))
        t = link.poll_telemetry()
        rates.append(abs(t.yaw_rate_dps))
    assert abs(t.yaw_deg - 30.0) < 0.5
    assert abs(t.pitch_deg - 10.0) < 0.5
    assert max(rates) <= 240.0 + 1e-6   # trapezoid respects vmax
    assert t.homed


def test_clamps_setpoints_to_travel_limits():
    clock = Clock()
    link = MockLink(time_fn=clock, yaw_limits=(-170, 170), pitch_limits=(-10, 60))
    t = run(link, clock, 4.0, yaw=500.0, pitch=90.0)
    assert t.yaw_deg <= 170.0 + 0.01
    assert t.pitch_deg <= 60.0 + 0.01
    assert t.status & protocol.STATUS_YAW_AT_LIMIT
    assert t.status & protocol.STATUS_PITCH_AT_LIMIT


def test_estop_stops_motion_and_sets_status():
    clock = Clock()
    link = MockLink(time_fn=clock)
    run(link, clock, 0.3, yaw=90.0)                      # get it moving
    t_stop = run(link, clock, 0.5, yaw=90.0, estop=True)  # estop mid-slew
    assert t_stop.estopped
    assert t_stop.yaw_rate_dps == 0.0
    pos_frozen = t_stop.yaw_deg
    t_later = run(link, clock, 0.5, yaw=90.0, estop=True)
    assert t_later.yaw_deg == pos_frozen                 # ignores setpoints while stopped
    t_resume = run(link, clock, 3.0, yaw=90.0)           # clear estop -> resumes
    assert not t_resume.estopped
    assert abs(t_resume.yaw_deg - 90.0) < 0.5


def test_link_dead_safe_hold_and_recovery():
    clock = Clock()
    link = MockLink(time_fn=clock, aim_timeout_s=0.5)
    run(link, clock, 0.2, yaw=120.0)         # moving toward a far setpoint
    t = run(link, clock, 1.0, send=False)    # link goes silent past the timeout
    assert link.safe_hold
    assert t.yaw_rate_dps == 0.0
    assert t.yaw_deg < 120.0                 # stopped short, holding
    t2 = run(link, clock, 4.0, yaw=120.0)    # frames resume -> motion resumes
    assert not link.safe_hold
    assert abs(t2.yaw_deg - 120.0) < 0.5


def test_heartbeat_holds_position_without_safe_hold():
    clock = Clock()
    link = MockLink(time_fn=clock, aim_timeout_s=0.5)
    run(link, clock, 3.0, yaw=20.0)
    # keep sending target_valid=0 heartbeats far past the timeout window
    t = run(link, clock, 2.0, yaw=0.0, valid=False)
    assert not link.safe_hold                # heartbeats prove the link is alive
    assert abs(t.yaw_deg - 20.0) < 0.5       # setpoint held, not chased to 0


def test_every_exchange_crosses_the_wire_cleanly():
    clock = Clock()
    link = MockLink(time_fn=clock)
    n = 250
    for _ in range(n):
        clock.t += 0.01
        link.send_aim(aim(clock.t, 15.0))
        assert link.poll_telemetry() is not None
    assert link.aim_packets == n
    assert link.wire_errors == 0
