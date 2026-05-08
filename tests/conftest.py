from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if os.environ.get("TCGD_TEST_INSTALLED") == "1":
    sys.path.append(str(SRC))
else:
    sys.path.insert(0, str(SRC))
