// Top-down view of the turret: where it points, what the camera sees, its travel
// limits, and where the target is. Yaw is positive to the left (counter-clockwise).
import type { Telemetry } from "./link";

const S = 220;
const C = S / 2;
const R = 84;

// Screen angle for a yaw: 0 deg points up, positive yaw turns counter-clockwise.
function pt(yawDeg: number, r: number): [number, number] {
  const a = ((-yawDeg - 90) * Math.PI) / 180;
  return [C + r * Math.cos(a), C + r * Math.sin(a)];
}

function arc(from: number, to: number, r: number): string {
  const [x0, y0] = pt(from, r);
  const [x1, y1] = pt(to, r);
  const large = Math.abs(to - from) > 180 ? 1 : 0;
  const sweep = to > from ? 0 : 1; // increasing yaw runs counter-clockwise on screen
  return `M ${x0} ${y0} A ${r} ${r} 0 ${large} ${sweep} ${x1} ${y1}`;
}

export function TurretDial({ tel, hfov, limits }: { tel: Telemetry | null; hfov: number; limits: [number, number] }) {
  const yaw = tel?.turret ? (tel.turret.yaw_enc ?? tel.turret.yaw_step) : 0;
  const ref = tel?.turret?.yaw_ref ?? yaw;
  const track = tel?.track;
  const wedgeL = pt(yaw + hfov / 2, R);
  const wedgeR = pt(yaw - hfov / 2, R);
  const nose = pt(yaw, R - 10);
  const [rx, ry] = pt(ref, R + 9);
  return (
    <figure className="dial">
      <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label={`Turret heading ${yaw.toFixed(1)} degrees`}>
        <circle cx={C} cy={C} r={R} className="dial-ring" />
        <path d={arc(limits[0], limits[1], R + 4)} className="dial-travel" />
        {[-90, 0, 90, 180].map((a) => {
          const [x0, y0] = pt(a, R - 4);
          const [x1, y1] = pt(a, R + 4);
          return <line key={a} x1={x0} y1={y0} x2={x1} y2={y1} className="dial-tick" />;
        })}
        <path d={`M ${C} ${C} L ${wedgeL[0]} ${wedgeL[1]} ${arc(yaw + hfov / 2, yaw - hfov / 2, R).replace(/^M [^A]+/, "")} Z`} className="dial-fov" />
        <line x1={C} y1={C} x2={nose[0]} y2={nose[1]} className="dial-heading" />
        <circle cx={rx} cy={ry} r={3.5} className="dial-ref" />
        {track && (() => {
          const [tx, ty] = pt(track.az, R - 22);
          return (
            <g className={track.confirmed ? "dial-target" : "dial-target tentative"}>
              <circle cx={tx} cy={ty} r={7} />
              <circle cx={tx} cy={ty} r={2} className="dot" />
            </g>
          );
        })()}
        <circle cx={C} cy={C} r={5} className="dial-hub" />
        <line x1={wedgeR[0]} y1={wedgeR[1]} x2={C} y2={C} className="dial-fov-edge" />
        <line x1={wedgeL[0]} y1={wedgeL[1]} x2={C} y2={C} className="dial-fov-edge" />
      </svg>
      <figcaption>
        <span className="big num">{yaw.toFixed(1)}°</span>
        <span className="quiet">heading</span>
        {track && (
          <>
            <span className="big num target">{(track.az - yaw).toFixed(1)}°</span>
            <span className="quiet">to target</span>
          </>
        )}
      </figcaption>
    </figure>
  );
}
