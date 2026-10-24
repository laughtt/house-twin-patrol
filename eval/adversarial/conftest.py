import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import sim  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="also run the MuJoCo cases")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: MuJoCo and the long route cases (run with --slow)")
    os.environ.setdefault("ADV_SEEDS", "8" if config.getoption("--slow") else "4")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        for it in items:
            if "slow" in it.keywords:
                it.add_marker(pytest.mark.skip(reason="slow case: run with --slow"))


def pytest_terminal_summary(terminalreporter):
    if not sim.REPORT:
        return
    tr = terminalreporter
    tr.section("adversarial cases")
    for name, ok, detail in sim.REPORT:
        tr.write_line(f"{'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
    out = sim.ROOT / "out/adversarial"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps([{"case": n, "pass": ok, "detail": d} for n, ok, d in sim.REPORT], indent=1))

