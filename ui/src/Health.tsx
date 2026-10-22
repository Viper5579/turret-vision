// The `doctor` results, as served by /api/health, with the fix for each problem.
import { useEffect, useState } from "react";

type Check = { name: string; status: "pass" | "warn" | "fail" | "skip"; detail: string; fix: string };
type Health = { checks: Check[]; running: boolean; source: string; version: string };

const WORD = { pass: "Pass", warn: "Warning", fail: "Failed", skip: "Skipped" } as const;

export function HealthView() {
  const [h, setH] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    try {
      const r = await fetch("/api/health");
      setH(await r.json());
      setErr(null);
    } catch {
      setErr("Could not reach the turret. Is `tv up` running?");
    }
  };
  useEffect(() => {
    void load();
    const id = setInterval(load, 2000);
    return () => clearInterval(id);
  }, []);

  return (
    <section className="health">
      <header>
        <div>
          <h2>System check</h2>
          <p className="quiet">
            The checks that are safe while the turret runs. For the ESP32 link and compute benchmark, stop it and run <code>tv doctor</code>.
          </p>
        </div>
        <button onClick={() => fetch("/api/health/run", { method: "POST" }).then(load)} disabled={h?.running}>
          {h?.running ? "Checking" : "Run checks again"}
        </button>
      </header>
      {err && <p className="error">{err}</p>}
      <ul>
        {h?.checks.map((c) => (
          <li key={c.name} className={`check ${c.status}`}>
            <span className="check-status">{WORD[c.status]}</span>
            <span className="check-name">{c.name}</span>
            <span className="check-detail">{c.detail}</span>
            {(c.status === "warn" || c.status === "fail") && c.fix && <span className="check-fix">{c.fix}</span>}
          </li>
        ))}
        {h && !h.checks.length && <li className="quiet">Running the first check</li>}
      </ul>
      {h && (
        <p className="quiet">
          Software {h.version}, running against the {h.source === "sim" ? "simulator" : "hardware"}.
        </p>
      )}
    </section>
  );
}
