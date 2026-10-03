"""Select the best query candidate using the local retrieval validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.dataset.builder import validate_sample


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _recall(sample: dict[str, Any]) -> float:
    try:
        return float(validate_sample(sample)["validation"]["recall"])
    except ValueError:
        return 0.0


def select(baseline_path: Path, remote_path: Path, output_path: Path) -> dict[str, int]:
    baseline = {row["id"]: row for row in _read(baseline_path)}
    remote = {row["id"]: row for row in _read(remote_path)}
    selected: list[dict[str, Any]] = []
    remote_wins = 0
    baseline_wins = 0
    ties = 0
    for sample_id, base in baseline.items():
        candidate = remote.get(sample_id)
        if candidate is None:
            selected.append(base)
            baseline_wins += 1
            continue
        base_recall = _recall(base)
        remote_recall = _recall(candidate)
        # Prefer the deterministic query on ties: it is cheaper to reproduce
        # and does not add semantic synonyms that BM25 may not contain.
        if remote_recall > base_recall:
            chosen = candidate
            remote_wins += 1
            provider = "openrouter"
        elif base_recall > remote_recall:
            chosen = base
            baseline_wins += 1
            provider = "heuristic"
        else:
            chosen = base
            ties += 1
            provider = "heuristic_tie"
        chosen = dict(chosen)
        chosen["query_selection"] = {
            "provider": provider,
            "heuristic_recall": base_recall,
            "remote_recall": remote_recall,
        }
        selected.append(chosen)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    return {"total": len(selected), "remote_wins": remote_wins, "heuristic_wins": baseline_wins, "ties": ties}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("remote", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(select(args.baseline, args.remote, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
