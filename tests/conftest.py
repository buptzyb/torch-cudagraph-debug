from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if os.environ.get("TCGD_TEST_INSTALLED") == "1":
    sys.path.append(str(SRC))
else:
    sys.path.insert(0, str(SRC))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail release validation when an environment-dependent test was skipped."""

    del exitstatus
    if os.environ.get("TCGD_FAIL_ON_SKIP") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = [] if reporter is None else reporter.stats.get("skipped", [])
    if skipped:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter.write_sep("=", f"TCGD_FAIL_ON_SKIP: {len(skipped)} skipped tests")
