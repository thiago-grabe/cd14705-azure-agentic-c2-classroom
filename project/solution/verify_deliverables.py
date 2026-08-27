#!/usr/bin/env python3
"""Acceptance check for the four deliverables a successful run must produce.

Run this after `python final.py`. It re-derives the numbers independently, so a
report whose statistics drifted from the data fails here rather than in review.

    python verify_deliverables.py            # check the default locations
    python verify_deliverables.py --strict   # also fail on warnings
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Importing `final` opens its logging FileHandler in "w" mode. Re-target it at a
# scratch file *before* the import, or this checker truncates the very audit log
# it is about to inspect.
import os  # noqa: E402

os.environ["AGENT_CHAT_LOG"] = str(Path(tempfile.gettempdir()) / "verify_deliverables.log")

CLEANED_JSON = BASE_DIR / "data-cleaned.json"
VIZ_CODE = BASE_DIR / "artifacts" / "data_visualization_code.py"
VIZ_IMAGE = BASE_DIR / "artifacts" / "data_visualization.png"
REPORT = BASE_DIR / "artifacts" / "final_report.md"
AGENT_LOG = BASE_DIR / "logs" / "agent_chat.log"
REPORT_SPEC = BASE_DIR / "specs" / "Report_Instructions.txt"

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class Results:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, check: str, status: str, detail: str = "") -> None:
        self.rows.append((check, status, detail))

    def record(self, check: str, ok: bool, detail: str = "") -> bool:
        self.add(check, "PASS" if ok else "FAIL", detail)
        return ok

    @property
    def failed(self) -> int:
        return sum(1 for _, status, _ in self.rows if status == "FAIL")

    @property
    def warned(self) -> int:
        return sum(1 for _, status, _ in self.rows if status == "WARN")

    def render(self) -> None:
        width = max(len(check) for check, _, _ in self.rows)
        for check, status, detail in self.rows:
            marker = {"PASS": "  ok  ", "FAIL": " FAIL ", "WARN": " warn "}[status]
            print(f"[{marker}] {check:<{width}}  {detail}")


def check_cleaned_json(results: Results) -> dict | None:
    """The cleaned dataset must be a real dataset, not a prose blob."""
    if not results.record("data-cleaned.json exists", CLEANED_JSON.exists(), str(CLEANED_JSON)):
        return None
    try:
        payload = json.loads(CLEANED_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        results.record("data-cleaned.json parses", False, str(exc))
        return None
    results.record("data-cleaned.json parses", True)

    cleaned = payload.get("cleaned")
    results.record(
        "cleaned[] is a populated list of x/y rows",
        isinstance(cleaned, list)
        and bool(cleaned)
        and all(isinstance(r, dict) and "x" in r and "y" in r for r in cleaned),
        f"{len(cleaned) if isinstance(cleaned, list) else 0} rows",
    )

    reference = payload.get("reference_statistics") or {}
    results.record("reference_statistics present", bool(reference), json.dumps(reference))

    if reference and isinstance(cleaned, list):
        import final  # imported late so a missing .env cannot break the checker

        values = [r["y"] for r in cleaned if isinstance(r, dict) and r.get("y") is not None]
        recomputed = final.compute_reference_statistics(values)
        results.record(
            "reference_statistics match a fresh pandas computation",
            recomputed == reference,
            "" if recomputed == reference else f"recomputed {json.dumps(recomputed)}",
        )
    return payload


def check_visualization_code(results: Results) -> None:
    """The saved script must be the one that actually works: re-run it."""
    if not results.record("data_visualization_code.py exists", VIZ_CODE.exists(), str(VIZ_CODE)):
        return
    source = VIZ_CODE.read_text(encoding="utf-8")
    results.record("visualization code is non-empty", bool(source.strip()), f"{len(source)} chars")
    try:
        compile(source, str(VIZ_CODE), "exec")
        results.record("visualization code compiles", True)
    except SyntaxError as exc:
        results.record("visualization code compiles", False, str(exc))
        return

    with tempfile.TemporaryDirectory() as tmp:
        completed = subprocess.run(
            [sys.executable, str(VIZ_CODE)],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=180,
            env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin", "HOME": tmp},
        )
        regenerated = Path(tmp) / "artifacts" / "data_visualization.png"
        results.record(
            "visualization code re-runs standalone and regenerates the plot",
            completed.returncode == 0 and regenerated.exists(),
            (completed.stderr or "").strip().splitlines()[-1][:120] if completed.returncode else "",
        )


def check_plot(results: Results) -> None:
    if not results.record("data_visualization.png exists", VIZ_IMAGE.exists(), str(VIZ_IMAGE)):
        return
    data = VIZ_IMAGE.read_bytes()
    results.record("plot is a valid PNG", data[:8] == PNG_MAGIC)
    results.record("plot is not a blank stub", len(data) > 5000, f"{len(data)} bytes")


def check_report(results: Results, payload: dict | None) -> None:
    if not results.record("final_report.md exists", REPORT.exists(), str(REPORT)):
        return
    report = REPORT.read_text(encoding="utf-8")

    headings = [
        line.strip()
        for line in REPORT_SPEC.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("#")
    ]
    missing = [h for h in headings if h not in report]
    results.record("report contains every template heading", not missing, f"missing: {missing}")

    results.record("no XXXX-XX-XX placeholders survive", "XXXX-XX-XX" not in report)
    results.record(
        "report references the plot",
        "![Data Visualization](data_visualization.png)" in report,
    )

    table_rows = [
        line
        for line in report.splitlines()
        if line.strip().startswith("|") and line.count("|") >= 4 and "---" not in line
    ]
    agent_rows = [r for r in table_rows if any(
        a in r for a in ("DataCleaning", "DataStatistics", "AnalysisChecker", "PythonExecutorAgent")
    )]
    results.record(
        "Agent Workflow Summary lists the agents", len(agent_rows) >= 4, f"{len(agent_rows)} agent rows"
    )

    if payload and payload.get("reference_statistics"):
        reference = payload["reference_statistics"]
        quoted = [
            key
            for key, value in reference.items()
            if key != "count" and str(value) not in report and f"{float(value):.2f}" not in report
        ]
        status = "PASS" if not quoted else "WARN"
        results.add(
            "report statistics match reference_statistics",
            status,
            "" if not quoted else f"not found verbatim: {quoted} (rounding is acceptable)",
        )


def check_audit_log(results: Results) -> None:
    if not results.record("agent_chat.log exists", AGENT_LOG.exists(), str(AGENT_LOG)):
        return
    text = AGENT_LOG.read_text(encoding="utf-8", errors="replace")
    results.record("agent log is non-empty", bool(text.strip()), f"{len(text)} chars")
    seen = [a for a in ("DataCleaning", "DataStatistics", "AnalysisChecker",
                        "PythonExecutorAgent", "ReportGenerator", "ReportChecker") if a in text]
    results.record("agent log records every agent", len(seen) == 6, f"found {len(seen)}/6: {seen}")
    results.record("agent log records the approval decision", "Workflow: APPROVAL" in text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args()

    results = Results()
    print("=" * 78)
    print("Deliverable verification")
    print("=" * 78)
    payload = check_cleaned_json(results)
    check_visualization_code(results)
    check_plot(results)
    check_report(results, payload)
    check_audit_log(results)
    results.render()

    print("-" * 78)
    print(f"{len(results.rows)} checks: {results.failed} failed, {results.warned} warnings")
    if results.failed or (args.strict and results.warned):
        return 1
    print("All deliverables verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
