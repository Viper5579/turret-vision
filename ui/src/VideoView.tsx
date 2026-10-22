// Live camera view. JPEGs are decoded off the main thread (createImageBitmap)
// and drawn to a canvas together with the overlay that belongs to that frame.
import { useEffect, useRef, useState } from "react";
import { link, type FrameMsg, type Overlay } from "./link";

const TARGET = "#ff4fb0";

function drawOverlay(ctx: CanvasRenderingContext2D, ov: Overlay, show: { blobs: boolean; track: boolean }) {
  ctx.lineWidth = 2;
  if (show.blobs) {
    ctx.strokeStyle = "rgba(231,236,242,0.75)";
    ctx.setLineDash([]);
    for (const [x0, y0, x1, y1] of ov.blobs) ctx.strokeRect(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4);
  }
  if (show.track && ov.track) {
    const { x, y, r, confirmed } = ov.track;
    ctx.strokeStyle = TARGET;
    ctx.setLineDash(confirmed ? [] : [6, 6]);
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(r, 14), 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    // reticle ticks
    const R = Math.max(r, 14);
    ctx.beginPath();
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      ctx.moveTo(x + dx * (R + 4), y + dy * (R + 4));
      ctx.lineTo(x + dx * (R + 16), y + dy * (R + 16));
    }
    ctx.stroke();
    if (ov.aim) {
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(ov.aim.x, ov.aim.y);
      ctx.stroke();
      ctx.fillStyle = TARGET;
      ctx.beginPath();
      ctx.arc(ov.aim.x, ov.aim.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

export function VideoView({ live }: { live: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [show, setShow] = useState({ blobs: true, track: true });
  const showRef = useRef(show);
  showRef.current = show;
  const [hasFrame, setHasFrame] = useState(false);

  useEffect(() => {
    let busy = false;
    let pending: FrameMsg | null = null;
    const draw = async (f: FrameMsg) => {
      busy = true;
      try {
        const bmp = await createImageBitmap(new Blob([f.jpeg], { type: "image/jpeg" }));
        const c = canvas.current;
        if (c) {
          if (c.width !== bmp.width || c.height !== bmp.height) {
            c.width = bmp.width;
            c.height = bmp.height;
          }
          const ctx = c.getContext("2d")!;
          ctx.drawImage(bmp, 0, 0);
          drawOverlay(ctx, f.ov, showRef.current);
          setHasFrame(true);
        }
        bmp.close();
      } catch {
        /* a corrupt JPEG is skipped; the next one replaces it */
      }
      busy = false;
      if (pending) {
        const p = pending;
        pending = null;
        void draw(p);
      }
    };
    return link.onFrame((f) => {
      if (busy) pending = f; // keep only the newest frame while decoding
      else void draw(f);
    });
  }, []);

  return (
    <section className="video" aria-label="Live camera">
      <canvas ref={canvas} width={1280} height={800} />
      {!hasFrame && (
        <p className="video-empty">{live ? "Waiting for the first frame" : "Connecting to the turret"}</p>
      )}
      <div className="video-toggles" role="group" aria-label="Overlays">
        <label>
          <input type="checkbox" checked={show.blobs} onChange={(e) => setShow({ ...show, blobs: e.target.checked })} />
          Motion
        </label>
        <label>
          <input type="checkbox" checked={show.track} onChange={(e) => setShow({ ...show, track: e.target.checked })} />
          Track
        </label>
      </div>
    </section>
  );
}
