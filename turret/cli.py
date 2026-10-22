"""Command line: `python -m turret <command>` (the `tv` script wraps these in the container).

serve    run the system and the GUI (simulator until M1 brings the hardware runtime)
sim      headless simulator run, scored against ground truth (--check for CI gates)
doctor   check the machine, camera, ESP32, and compute budget
config   print the effective configuration as YAML
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import yaml

from . import __version__


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .config import load_config
    from .runtime.bus import Bus
    from .runtime.runner import SimRunner
    from .server.app import create_app

    cfg = load_config(args.config)
    bus = Bus()
    if args.hw:
        from .runtime.hardware import HardwareRunner

        try:
            runner = HardwareRunner(cfg, bus)
        except Exception as e:
            print(f"Cannot start hardware mode: {e}", file=sys.stderr)
            return 2
        source = "hardware"
        print("Hardware mode: turret control is live; the camera joins in M1.")
    else:
        runner = SimRunner(cfg, bus)
        source = "sim"
    runner.start()
    app = create_app(cfg, bus, source=source)
    print(f"Mission Control: http://<this machine>:{args.port or cfg.server.port}  ({source})")
    try:
        uvicorn.run(app, host=cfg.server.host, port=args.port or cfg.server.port, log_level="warning")
    finally:
        runner.stop()
    return 0


def _sim(args: argparse.Namespace) -> int:
    from .config import load_config
    from .runtime.pipeline import Mode
    from .runtime.runner import SimRunner

    cfg = load_config(args.config)
    cfg.sim.realtime = False
    runner = SimRunner(cfg, mode=Mode[args.mode.upper()])
    score = runner.run_headless(args.seconds).summary()
    score["pipeline"] = runner.pipeline.stats()
    print(json.dumps(score, indent=1))
    if args.check:
        fails = []
        if (score["track_coverage"] or 0) < 0.9:
            fails.append("track_coverage < 0.9")
        if (score["bearing_err_deg_p50"] or 99) > 0.5:
            fails.append("bearing_err_deg_p50 > 0.5")
        if score["false_track_frames"] > 2:
            fails.append("false_track_frames > 2")
        if score["pipeline"]["unmatched"] > 2:
            fails.append("unmatched frames > 2")
        if fails:
            print("SIM CHECK FAILED: " + "; ".join(fails), file=sys.stderr)
            return 1
        print("sim check passed")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from . import doctor

    return doctor.main(as_json=args.json, quick=args.quick)


def _config(args: argparse.Namespace) -> int:
    from .config import load_config

    print(yaml.safe_dump(load_config(args.config).model_dump(), sort_keys=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="turret", description=f"Turret Vision v{__version__}")
    ap.add_argument("--config", help="override file (default data/local.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the system and the GUI")
    which = s.add_mutually_exclusive_group()
    which.add_argument("--hw", action="store_true", help="real ESP32 over USB (camera joins in M1)")
    which.add_argument("--sim", action="store_true", help="the simulator (default)")
    s.add_argument("--port", type=int)
    s.set_defaults(fn=_serve)

    s = sub.add_parser("sim", help="headless simulator run with a score")
    s.add_argument("--seconds", type=float, default=10.0)
    s.add_argument("--mode", default="track", choices=["idle", "manual", "track"])
    s.add_argument("--check", action="store_true", help="exit 1 if the score misses the CI gates")
    s.set_defaults(fn=_sim)

    s = sub.add_parser("doctor", help="check the machine, camera, ESP32, compute")
    s.add_argument("--json", action="store_true")
    s.add_argument("--quick", action="store_true", help="skip ESP32 and compute checks")
    s.set_defaults(fn=_doctor)

    s = sub.add_parser("config", help="print the effective config")
    s.set_defaults(fn=_config)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    return args.fn(args)
