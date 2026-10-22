// WebSocket connection to the Jetson. Frames go straight to whoever draws them
// (no React re-render per frame); telemetry is exposed as a React store.
import { decode, encode } from "@msgpack/msgpack";
import { useSyncExternalStore } from "react";

export type Overlay = {
  w: number;
  h: number;
  blobs: [number, number, number, number][];
  track?: { x: number; y: number; r: number; confirmed: boolean };
  aim?: { x: number; y: number };
};

export type Telemetry = {
  time: number;
  mode: "IDLE" | "MANUAL" | "TRACK" | "ENGAGE";
  estop: boolean;
  link: {
    connected: boolean;
    synced: boolean;
    hash_mismatch: boolean;
    fw_version: number;
    sync_residual_ms: number | null;
    drift_ppm: number;
  };
  turret: null | {
    yaw_enc: number | null;
    yaw_step: number;
    yaw_ref: number;
    yaw_rate: number;
    pitch: number;
    status: number;
    faults: number;
  };
  track: null | {
    az: number;
    el: number;
    az_rate: number;
    el_rate: number;
    sigma: number;
    confirmed: boolean;
    age: number;
  };
  detections: number;
  pipeline: {
    fps: number;
    latency_ms_p50: number | null;
    latency_ms_p95: number | null;
    process_ms_p50: number | null;
    matched: number;
    unmatched: number;
    usb_delay_ms: number;
  };
  sim?: { speed: number; throws: number; track_coverage: number | null; bearing_err_deg_p50: number | null };
};

export type Hello = {
  version: string;
  source: string;
  camera: { w: number; h: number; hfov: number };
  limits: { yaw_min_deg: number; yaw_max_deg: number; pitch_min_deg: number; pitch_max_deg: number; max_rate_dps: number };
};

export type FrameMsg = { seq: number; jpeg: Uint8Array; ov: Overlay };

type State = { connected: boolean; hello: Hello | null; tel: Telemetry | null };

class Link {
  private ws: WebSocket | null = null;
  private state: State = { connected: false, hello: null, tel: null };
  private listeners = new Set<() => void>();
  private frameListeners = new Set<(f: FrameMsg) => void>();
  private retry = 500;

  constructor() {
    this.connect();
  }

  private connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retry = 500;
      this.set({ connected: true });
    };
    ws.onclose = () => {
      this.set({ connected: false });
      setTimeout(() => this.connect(), this.retry);
      this.retry = Math.min(this.retry * 2, 5000);
    };
    ws.onmessage = (ev) => {
      const msg = decode(new Uint8Array(ev.data as ArrayBuffer)) as { t: string } & Record<string, unknown>;
      if (msg.t === "frame") {
        const f = msg as unknown as FrameMsg;
        this.frameListeners.forEach((fn) => fn(f));
      } else if (msg.t === "tel") {
        this.set({ tel: msg as unknown as Telemetry });
      } else if (msg.t === "hello") {
        this.set({ hello: msg as unknown as Hello });
      }
    };
    this.ws = ws;
  }

  private set(patch: Partial<State>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((fn) => fn());
  }

  send(cmd: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(encode(cmd));
  }

  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  onFrame(fn: (f: FrameMsg) => void) {
    this.frameListeners.add(fn);
    return () => {
      this.frameListeners.delete(fn);
    };
  }

  get snapshot() {
    return this.state;
  }
}

export const link = new Link();

export function useLink(): State {
  return useSyncExternalStore(link.subscribe, () => link.snapshot);
}

export const commands = {
  mode: (mode: Telemetry["mode"]) => link.send({ t: "mode", mode }),
  jog: (yaw: number, pitch: number) => link.send({ t: "jog", yaw, pitch }),
  estop: (on: boolean) => link.send({ t: "estop", on }),
  videoFps: (fps: number) => link.send({ t: "video", fps }),
};
