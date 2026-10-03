"""Normalize six evidence subjects so the existing FTS5 query can retrieve them."""

from __future__ import annotations

import json
from pathlib import Path


INPUT = Path("data/gate_final_2855.full.jsonl")

SUBJECTS = {
    "rhelm_0033": "family vacation remote cabin constraints",
    "rhelm_0061": "retirement course launch project constraints",
    "rhelm_0141": "seminar standing presenting constraints",
    "personamem_v2_0353": "self documentary programs flight",
    "personamem_v2_0405": "self social security number ssn",
    "personamem_v2_0444": "self social security number ssn",
}


def main() -> None:
    rows = [
        json.loads(line)
        for line in INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    for row in rows:
        subject = SUBJECTS.get(row.get("id"))
        if subject is None:
            continue
        memory_rows = row.get("memory_rows") or []
        if not memory_rows:
            raise ValueError(f"{row['id']} has no memory_rows")
        for memory in memory_rows:
            memory["subject"] = subject
        quality = row.setdefault("quality", {})
        flags = quality.setdefault("repair_flags", [])
        flag = "memory_subject_normalized_for_fts5"
        if flag not in flags:
            flags.append(flag)
        changed += 1

    if changed != len(SUBJECTS):
        raise ValueError(f"expected to repair {len(SUBJECTS)} rows, changed {changed}")

    INPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"file": str(INPUT), "rows_repaired": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
