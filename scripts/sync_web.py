"""Build the dashboard and copy it into the package (src/membridge/_web) so `pip install` / `uvx` ship a working UI without Node.

Run after changing anything under web/:   uv run python scripts/sync_web.py
The bundle is committed (about 650 KB); CI fails if it is stale.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src" / "membridge" / "_web"


def main() -> None:
    npm = shutil.which("npm") or shutil.which("npm.cmd")
    if npm is None:
        sys.exit("npm not found: install Node 20+ to rebuild the dashboard")
    subprocess.run([npm, "run", "build"], cwd=ROOT / "web", check=True)
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(ROOT / "web" / "dist", DEST)
    print(f"copied dashboard bundle -> {DEST}")


if __name__ == "__main__":
    main()
