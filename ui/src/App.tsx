import { useState } from "react";
import { Charts } from "./Charts";
import { AimPad, EStop, ModeSwitch, useOperatorInputs } from "./controls";
import { HealthView } from "./Health";
import { useLink, type Telemetry } from "./link";
import { TurretDial } from "./TurretDial";
import { VideoView } from "./VideoView";

function fmt(v: number | null | undefined, digits = 1, unit = "") {
  return v == null || Number.isNaN(v) ? "n/a" : `${v.toFixed(digits)}${unit}`;
}

function StatusBar({ connected, tel, source }: { connected: boolean; tel: Telemetry | null; source?: string }) {
  const l = tel?.link;
  const linkState = !connected
    ? { cls: "bad", text: "GUI offline" }
    : l?.hash_mismatch
      ? { cls: "bad", text: "Firmware out of date: run tv flash" }
      : l?.synced
        ? { cls: "good", text: `ESP32 linked, clock ±${fmt(l.sync_residual_ms, 2)} ms` }
        : { cls: "warn", text: "Waiting for ESP32" };
  return (
    <div className="statusbar">
      <span className={`pill ${linkState.cls}`}>{linkState.text}</span>
      <span className="stat">
        <span className="num">{fmt(tel?.pipeline.fps, 0)}</span> fps
      </span>
      <span className="stat">
        <span className="num">{fmt(tel?.pipeline.latency_ms_p50, 1)}</span> ms to command
      </span>
      <span className="stat">
        <span className="num">{tel?.pipeline.unmatched ?? 0}</span> unmatched frames
      </span>
      {source === "sim" && tel?.sim && (
        <span className="stat sim">
          Simulator at <span className="num">{fmt(tel.sim.speed, 2)}x</span>, {tel.sim.throws} throws
        </span>
      )}
    </div>
  );
}

function TrackReadout({ tel }: { tel: Telemetry | null }) {
  const t = tel?.track;
  if (!t) return <p className="readout quiet">No target. Throw a plane across the view.</p>;
  return (
    <dl className="readout">
      <div>
        <dt>Target</dt>
        <dd className={t.confirmed ? "target" : "quiet"}>{t.confirmed ? "Locked" : "Confirming"}</dd>
      </div>
      <div>
        <dt>Bearing</dt>
        <dd className="num">
          {fmt(t.az)}° / {fmt(t.el)}°
        </dd>
      </div>
      <div>
        <dt>Rate</dt>
        <dd className="num">
          {fmt(t.az_rate, 0)} / {fmt(t.el_rate, 0)} °/s
        </dd>
      </div>
      <div>
        <dt>Uncertainty</dt>
        <dd className="num">±{fmt(t.sigma, 2)}°</dd>
      </div>
    </dl>
  );
}

export default function App() {
  const { connected, hello, tel } = useLink();
  const [tab, setTab] = useState<"live" | "health">("live");
  useOperatorInputs(tel);
  const limits: [number, number] = [hello?.limits.yaw_min_deg ?? -170, hello?.limits.yaw_max_deg ?? 170];

  return (
    <div className={tel?.estop ? "app stopped" : "app"}>
      <header className="top">
        <h1>
          Turret <span className="quiet">Mission Control</span>
        </h1>
        <nav aria-label="Views">
          <button aria-current={tab === "live"} onClick={() => setTab("live")}>
            Live
          </button>
          <button aria-current={tab === "health"} onClick={() => setTab("health")}>
            System check
          </button>
        </nav>
        <EStop engaged={!!tel?.estop} />
      </header>
      <StatusBar connected={connected} tel={tel} source={hello?.source} />
      {tab === "live" ? (
        <main className="live">
          <VideoView live={connected} />
          <aside className="rail">
            <ModeSwitch tel={tel} />
            <TurretDial tel={tel} hfov={hello?.camera.hfov ?? 70} limits={limits} />
            <TrackReadout tel={tel} />
            <AimPad enabled={tel?.mode === "MANUAL" && !tel?.estop} />
            <p className="keys quiet">
              Keys: <kbd>1</kbd> hold, <kbd>2</kbd> manual, <kbd>3</kbd> track, arrows to aim, <kbd>Space</kbd> stop.
              An Xbox controller works too.
            </p>
          </aside>
          <Charts tel={tel} />
        </main>
      ) : (
        <main>
          <HealthView />
        </main>
      )}
    </div>
  );
}
