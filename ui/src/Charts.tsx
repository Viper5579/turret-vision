// Rolling 10-second strip charts fed by telemetry (30 Hz).
import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { Telemetry } from "./link";

const WINDOW_S = 10;
const INK = "#8d9aab";
const GRID = "rgba(141,154,171,0.15)";

type Series = { label: string; color: string; dash?: number[] };

function Strip({ title, unit, series, data }: { title: string; unit: string; series: Series[]; data: number[][] }) {
  const el = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!el.current) return;
    const opts: uPlot.Options = {
      width: el.current.clientWidth,
      height: 130,
      legend: { show: true, live: false },
      cursor: { show: false },
      scales: {
        x: { time: false },
        // Pad flat or empty data so a steady value still draws as a visible line.
        y: { range: (_u, lo, hi) => (lo == null || hi == null ? [-1, 1] : lo === hi ? [lo - 1, hi + 1] : [lo, hi]) },
      },
      axes: [
        {
          stroke: INK,
          grid: { stroke: GRID },
          ticks: { stroke: GRID },
          size: 26,
          incrs: [1, 2, 5],
          values: (_u, v) => v.map((x) => `${x}s`),
        },
        { stroke: INK, grid: { stroke: GRID }, ticks: { stroke: GRID }, size: 44, label: unit, labelSize: 14 },
      ],
      series: [{}, ...series.map((s) => ({ label: s.label, stroke: s.color, width: 1.6, dash: s.dash, points: { show: false } }))],
    };
    plot.current = new uPlot(opts, [[], ...series.map(() => [])] as uPlot.AlignedData, el.current);
    const ro = new ResizeObserver(() => {
      if (el.current && plot.current) plot.current.setSize({ width: el.current.clientWidth, height: 130 });
    });
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
    };
    // series config is fixed per chart
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    plot.current?.setData(data as uPlot.AlignedData);
  }, [data]);

  return (
    <div className="strip">
      <h3>{title}</h3>
      <div ref={el} />
    </div>
  );
}

export function Charts({ tel }: { tel: Telemetry | null }) {
  const buf = useRef<number[][]>([[], [], [], [], [], [], []]);
  const lastT = useRef(-1);
  if (tel && tel.time !== lastT.current) {
    lastT.current = tel.time;
    const b = buf.current;
    const yaw = tel.turret ? (tel.turret.yaw_enc ?? tel.turret.yaw_step) : NaN;
    b[0].push(tel.time);
    b[1].push(tel.turret?.yaw_ref ?? NaN);
    b[2].push(yaw);
    b[3].push(tel.track?.confirmed ? tel.track.az : NaN);
    b[4].push(tel.track?.confirmed ? tel.track.az - yaw : NaN);
    b[5].push(tel.pipeline.latency_ms_p50 ?? NaN);
    b[6].push(tel.pipeline.process_ms_p50 ?? NaN);
    while (b[0].length && b[0][0] < tel.time - WINDOW_S) b.forEach((s) => s.shift());
  }
  const b = buf.current;
  const t0 = b[0].length ? b[0][b[0].length - 1] : 0;
  const x = b[0].map((t) => t - t0);
  return (
    <section className="charts" aria-label="Telemetry">
      <Strip
        title="Yaw"
        unit="deg"
        series={[
          { label: "commanded", color: "#e7ecf2", dash: [4, 4] },
          { label: "actual", color: "#7fb3ff" },
          { label: "target", color: "#ff4fb0" },
        ]}
        data={[x, b[1], b[2], b[3]]}
      />
      <Strip title="Pointing error" unit="deg" series={[{ label: "target minus heading", color: "#ff4fb0" }]} data={[x, b[4]]} />
      <Strip
        title="Timing"
        unit="ms"
        series={[
          { label: "exposure to command", color: "#f5c518" },
          { label: "vision compute", color: "#7fb3ff" },
        ]}
        data={[x, b[5], b[6]]}
      />
    </section>
  );
}
