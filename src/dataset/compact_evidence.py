"""Keep only evidence memories for query-label construction.

The compact file is used to validate whether a candidate query can retrieve
the known correct memory. It is not passed to the gate model; final SFT export
contains messages only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compact(input_path: Path, output_path: Path) -> dict[str, int]:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = 0
    output: list[dict[str, Any]] = []
    for row in rows:
        evidence_ids = {str(value) for value in row.get("evidence_ids") or []}
        row = dict(row)
        row["memory_rows"] = [
            memory for memory in row.get("memory_rows") or [] if str(memory.get("id")) in evidence_ids
        ]
        if row.get("gate_target", {}).get("retrieve") and not row["memory_rows"]:
            continue
        output.append(row)
        kept += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    return {"total": len(rows), "kept": kept, "dropped": len(rows) - kept}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(compact(args.input, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
