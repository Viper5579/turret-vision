// Operator inputs: mode switch, e-stop, an on-screen aim pad, keyboard, and an Xbox
// controller through the browser Gamepad API. All of them send the same commands.
import { useEffect, useRef, useState } from "react";
import { commands, type Telemetry } from "./link";

const MAX_JOG = 120; // deg/s at full deflection

export function ModeSwitch({ tel }: { tel: Telemetry | null }) {
  const mode = tel?.mode ?? "IDLE";
  const items: { id: Telemetry["mode"]; label: string; hint: string }[] = [
    { id: "IDLE", label: "Hold", hint: "1" },
    { id: "MANUAL", label: "Manual", hint: "2" },
    { id: "TRACK", label: "Track", hint: "3" },
  ];
  return (
    <div className="modes" role="radiogroup" aria-label="Mode">
      {items.map((it) => (
        <button
          key={it.id}
          role="radio"
          aria-checked={mode === it.id}
          className={`mode mode-${it.id.toLowerCase()}`}
          disabled={tel?.estop}
          onClick={() => commands.mode(it.id)}
        >
          {it.label}
          <kbd>{it.hint}</kbd>
        </button>
      ))}
    </div>
  );
}

export function EStop({ engaged }: { engaged: boolean }) {
  return (
    <button
      className={engaged ? "estop engaged" : "estop"}
      onClick={() => commands.estop(!engaged)}
      aria-pressed={engaged}
    >
      {engaged ? "Release stop" : "Stop"}
      <kbd>Space</kbd>
    </button>
  );
}

export function AimPad({ enabled }: { enabled: boolean }) {
  const pad = useRef<HTMLDivElement>(null);
  const [knob, setKnob] = useState<[number, number]>([0, 0]);
  const active = useRef(false);

  const move = (e: React.PointerEvent) => {
    if (!active.current || !pad.current) return;
    const r = pad.current.getBoundingClientRect();
    let x = ((e.clientX - r.left) / r.width) * 2 - 1;
    let y = ((e.clientY - r.top) / r.height) * 2 - 1;
    const n = Math.hypot(x, y);
    if (n > 1) {
      x /= n;
      y /= n;
    }
    setKnob([x, y]);
    // Pad right = turn right = negative yaw. Pad up = pitch up.
    commands.jog(-x * Math.abs(x) * MAX_JOG, -y * Math.abs(y) * MAX_JOG * 0.5);
  };
  const end = () => {
    active.current = false;
    setKnob([0, 0]);
    commands.jog(0, 0);
  };
  return (
    <div
      ref={pad}
      className={enabled ? "aimpad" : "aimpad disabled"}
      onPointerDown={(e) => {
        if (!enabled) return;
        active.current = true;
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        move(e);
      }}
      onPointerMove={move}
      onPointerUp={end}
      onPointerCancel={end}
      aria-label="Aim pad: drag to slew in Manual mode"
    >
      <div className="aimpad-cross" />
      <div className="aimpad-knob" style={{ left: `${50 + knob[0] * 42}%`, top: `${50 + knob[1] * 42}%` }} />
      {!enabled && <span className="aimpad-note">Switch to Manual to aim</span>}
    </div>
  );
}

/** Keyboard (Space, 1-3, arrows/WASD) and gamepad (left stick, A = track, B = stop). */
export function useOperatorInputs(tel: Telemetry | null) {
  const telRef = useRef(tel);
  telRef.current = tel;

  useEffect(() => {
    const held = new Set<string>();
    const sendKeys = () => {
      const yaw = (held.has("left") ? 1 : 0) - (held.has("right") ? 1 : 0);
      const pitch = (held.has("up") ? 1 : 0) - (held.has("down") ? 1 : 0);
      commands.jog(yaw * MAX_JOG * 0.5, pitch * MAX_JOG * 0.25);
    };
    const keyName = (k: string) =>
      ({ ArrowLeft: "left", a: "left", ArrowRight: "right", d: "right", ArrowUp: "up", w: "up", ArrowDown: "down", s: "down" })[k];
    const down = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, select")) return;
      if (e.code === "Space") {
        e.preventDefault();
        if (!e.repeat) commands.estop(!telRef.current?.estop);
        return;
      }
      if (e.key === "1") commands.mode("IDLE");
      if (e.key === "2") commands.mode("MANUAL");
      if (e.key === "3") commands.mode("TRACK");
      const k = keyName(e.key);
      if (k && telRef.current?.mode === "MANUAL") {
        e.preventDefault();
        held.add(k);
        sendKeys();
      }
    };
    const up = (e: KeyboardEvent) => {
      const k = keyName(e.key);
      if (k && held.delete(k)) sendKeys();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);

    let raf = 0;
    let lastSent = 0;
    const prevButtons: boolean[] = [];
    const poll = (t: number) => {
      const gp = navigator.getGamepads?.().find((g) => g);
      if (gp) {
        const pressed = gp.buttons.map((b) => b.pressed);
        if (pressed[0] && !prevButtons[0]) commands.mode("TRACK");
        if (pressed[1] && !prevButtons[1]) commands.estop(!telRef.current?.estop);
        if (pressed[2] && !prevButtons[2]) commands.mode("MANUAL");
        if (pressed[3] && !prevButtons[3]) commands.mode("IDLE");
        pressed.forEach((p, i) => (prevButtons[i] = p));
        if (telRef.current?.mode === "MANUAL" && t - lastSent > 30) {
          const dz = (v: number) => (Math.abs(v) < 0.12 ? 0 : v);
          const x = dz(gp.axes[0] ?? 0);
          const y = dz(gp.axes[1] ?? 0);
          commands.jog(-x * Math.abs(x) * MAX_JOG, -y * Math.abs(y) * MAX_JOG * 0.5);
          lastSent = t;
        }
      }
      raf = requestAnimationFrame(poll);
    };
    raf = requestAnimationFrame(poll);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      cancelAnimationFrame(raf);
    };
  }, []);
}
