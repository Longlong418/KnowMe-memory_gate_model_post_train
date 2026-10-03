"""Select a teacher query candidate by exact evidence-id hit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.dataset.builder import validate_sample


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_variants(query: str) -> list[str]:
    terms = query.split()
    if len(terms) <= 6:
        return [query]
    # Keep the API candidate's wording but enforce the gate contract. Trying
    # both ends preserves either the topic terms or the discriminative tail.
    variants = [" ".join(terms[:6]), " ".join(terms[-6:])]
    return list(dict.fromkeys(variants))


def select(input_path: Path, output_path: Path, rejected_path: Path) -> dict[str, int]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in _read(input_path):
        target = row.get("gate_target") or {}
        retrieve = bool(target.get("retrieve"))
        candidates = row.get("query_candidates") or ([] if retrieve else [""])
        attempts: list[dict[str, Any]] = []
        best: tuple[tuple[float, int, int], dict[str, Any]] | None = None
        candidate_index = 0
        for query in candidates:
            for normalized_query in _candidate_variants(str(query).strip()):
                index = candidate_index
                candidate_index += 1
                query = normalized_query
                candidate = dict(row)
                candidate["gate_target"] = {
                    "retrieve": retrieve,
                    "query": query if retrieve else "",
                    "reason": "需要查询历史记忆" if retrieve else "当前信息已足够",
                }
                try:
                    built = validate_sample(candidate)
                except ValueError as error:
                    attempts.append({"query": candidate["gate_target"]["query"], "error": str(error)})
                    continue
                validation = built["validation"]
                recall = float(validation["recall"])
                term_count = len(candidate["gate_target"]["query"].split()) if retrieve else 0
                attempts.append({
                    "query": candidate["gate_target"]["query"],
                    "recall": recall,
                    "retrieved_ids": validation["retrieved_ids"],
                })
                # Maximize evidence recall, then minimize query length, then keep
                # teacher ordering deterministic.
                score = (-recall, term_count, index)
                if best is None or score < best[0]:
                    best = (score, built)
        if best is None or not best[1]["validation"]["passed"]:
            rejected.append({"sample": row, "candidate_attempts": attempts})
            continue
        selected = best[1]
        selected["query_candidates"] = candidates
        selected["query_selection"] = {
            "criterion": "evidence_id_hit_then_shortest_query",
            "attempts": attempts,
            "selected_query": selected["gate_target"]["query"],
        }
        accepted.append(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted), encoding="utf-8")
    rejected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected), encoding="utf-8")
    return {"total": len(accepted) + len(rejected), "accepted": len(accepted), "rejected": len(rejected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rejected", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.input, args.output, args.rejected), ensure_ascii=False))


if __name__ == "__main__":
    main()
