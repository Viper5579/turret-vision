"""Phase 3 exit criterion, enforced in CI: sim_target must drive the full
pipeline against MockLink and pass all its own criteria (exit code 0)."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_sim_target_passes_all_criteria():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "sim_target.py"),
                        "--frames", "300"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"sim_target failed:\n{r.stdout}\n{r.stderr}"
    assert "FAIL" not in r.stdout
    assert "zero wire errors" in r.stdout
