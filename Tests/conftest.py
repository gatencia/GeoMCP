# tests/conftest.py
import os
import sys
from pathlib import Path

# 1) project root (where pytest is running)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2) common subfolders we saw in your commands
# adjust these if your real code is somewhere else
CANDIDATES = [
    ROOT / "modules",
    ROOT / "GeoMCP",
    ROOT / "AA - GeoMCP",
]

for c in CANDIDATES:
    if c.exists() and str(c) not in sys.path:
        sys.path.insert(0, str(c))