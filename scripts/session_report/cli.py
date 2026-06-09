#!/usr/bin/env python3
"""세션 스냅샷 JSON → 리포트 생성 CLI.

예:
  python scripts/session_report/cli.py --input snapshot.json --out report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.session_report.generate import generate_session_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate session report from snapshot JSON")
    parser.add_argument("--input", required=True, help="Path to snapshot JSON")
    parser.add_argument("--out", help="Write markdown report to this path")
    parser.add_argument("--json-out", help="Write full report JSON to this path")
    args = parser.parse_args()

    snapshot = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = generate_session_report(snapshot)

    if args.out:
        Path(args.out).write_text(result["report_md"], encoding="utf-8")
        print(f"Wrote markdown: {args.out}")
    else:
        print(result["report_md"])

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result["report_json"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
