"""UVC control layer: v4l2-ctl output parsing, toggle value mapping (UVC
auto_exposure menu: 3=auto, 1=manual), startup apply ordering, and how camera
controls surface through the tuning registry. All against a fake runner --
no camera or v4l2-ctl needed."""
from turretvision.capture.uvc_ctrl import UvcControls, apply_ctrls
from turretvision.tune.params import ParamRegistry, add_camera_params

LIST_CTRLS = """\
User Controls

  brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
  white_balance_automatic 0x0098090c (bool)   : default=1 value=1
  gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=10
  white_balance_temperature 0x0098091a (int) : min=2800 max=6500 step=1 default=4600 value=4600 flags=inactive
  backlight_compensation 0x0098091c (int)    : min=0 max=2 step=1 default=1 value=1

Camera Controls

  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
  exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=157 flags=inactive
"""


class FakeRunner:
    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.commands.append(args)
        if "--list-ctrls" in args:
            return LIST_CTRLS
        return ""


def test_parses_ctrl_table():
    cam = UvcControls("/dev/video0", runner=FakeRunner())
    assert cam.get("auto_exposure") == 3
    exp = cam.controls["exposure_time_absolute"]
    assert (exp["min"], exp["max"], exp["value"]) == (1, 5000, 157)
    assert exp["flags"] == "inactive"
    assert cam.controls["white_balance_automatic"]["type"] == "bool"


def test_set_issues_v4l2ctl_and_updates_cache():
    r = FakeRunner()
    cam = UvcControls("/dev/video0", runner=r)
    cam.set("exposure_time_absolute", 80)
    assert ["-d", "/dev/video0", "-c", "exposure_time_absolute=80"] in r.commands
    assert cam.get("exposure_time_absolute") == 80


def test_apply_ctrls_switches_autos_to_manual_first():
    r = FakeRunner()
    apply_ctrls("/dev/video0",
                {"exposure_time_absolute": 80, "auto_exposure": 1,
                 "white_balance_temperature": 4500, "white_balance_automatic": 0},
                runner=r)
    sets = [a[-1] for a in r.commands if "-c" in a]
    # autos first, else the manual writes bounce off flags=inactive
    assert set(sets[:2]) == {"auto_exposure=1", "white_balance_automatic=0"}
    assert set(sets[2:]) == {"exposure_time_absolute=80", "white_balance_temperature=4500"}


def test_camera_params_in_registry_toggle_and_snapshot():
    r = FakeRunner()
    cam = UvcControls("/dev/video0", runner=r)
    reg = ParamRegistry()
    n = add_camera_params(reg, cam)
    assert n == 6  # brightness is deliberately not exposed

    ae = reg.specs["camera.v4l2_ctrls.auto_exposure"]
    assert ae.kind == "toggle" and (ae.on_value, ae.off_value) == (3, 1)
    reg.queue("camera.v4l2_ctrls.auto_exposure", 1)   # switch to manual
    reg.queue("camera.v4l2_ctrls.exposure_time_absolute", 80)
    reg.apply_pending()
    assert cam.get("auto_exposure") == 1
    assert cam.get("exposure_time_absolute") == 80

    # snapshot carries RAW driver values -> what Save writes is exactly what
    # V4L2Camera re-applies at startup
    snap = reg.snapshot()
    assert snap["camera.v4l2_ctrls.auto_exposure"] == 1
    assert snap["camera.v4l2_ctrls.exposure_time_absolute"] == 80

    # slider range comes from the driver, capped to a sane tuning range
    exp = reg.specs["camera.v4l2_ctrls.exposure_time_absolute"]
    assert (exp.vmin, exp.vmax) == (1, 500)
    wb = reg.specs["camera.v4l2_ctrls.white_balance_temperature"]
    assert (wb.vmin, wb.vmax) == (2800, 6500)
