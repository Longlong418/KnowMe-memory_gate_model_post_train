"""Evaluate gate predictions with classification and retrieval metrics.

Retrieval rank is reported under two scopes, because the production fixture
concatenates two independently ordered branches (``facts`` top-4, then
``episodes`` top-3):

* ``evidence_only`` uses the production concatenation order, i.e. the order the
  model actually receives.  This is where an episode landed behind four
  unrelated facts still reads as rank 5.
* ``retrieval.fact`` / ``retrieval.episode`` use branch-internal rank, counted
  from 1 inside each branch.  ``retrieval.branch`` aggregates them as the best
  branch-internal rank of the branch holding the evidence.

Branch-internal rank is the meaningful ranking-quality signal: the relative
order of the two branches is fixed by ``_retrieved_branches`` rather than by
relevance, and the BM25 scores of the two FTS5 tables are not comparable
(separate IDF/avgdl statistics).  Reward code should build on
``_retrieved_branches`` and ``_evidence_rank`` rather than re-concatenating.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.retrieval import EpisodeStore, FactStore, connect


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def _extract_json(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"<\|think\|>.*?<\|/think\|>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def parse_prediction(row: dict[str, Any]) -> dict[str, Any] | None:
    if "retrieve" in row:
        return row
    for key in ("prediction", "raw_output", "output", "response"):
        parsed = _extract_json(row.get(key))
        if parsed is not None:
            return parsed
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                parsed = _extract_json(message.get("content"))
                if parsed is not None:
                    return parsed
    return None


def output_is_valid(prediction: dict[str, Any] | None, language: str) -> bool:
    if not isinstance(prediction, dict):
        return False
    if not isinstance(prediction.get("retrieve"), bool):
        return False
    query = prediction.get("query")
    if not isinstance(query, str):
        return False
    if not isinstance(prediction.get("reason"), str):
        return False
    if prediction["retrieve"]:
        terms = query.split()
        if not 2 <= len(terms) <= 6:
            return False
        if language == "zh" and len(terms) > 1 and " " not in query:
            return False
    return not prediction["retrieve"] and query == "" or prediction["retrieve"]


def _retrieved_branches(
    memory_rows: list[dict[str, Any]], query: str
) -> tuple[list[str], list[str]]:
    """Return ``(facts[:4], episodes[:3])``, each in its own BM25 order."""
    database = connect()
    facts = FactStore(database)
    episodes = EpisodeStore(database)
    database_ids: dict[tuple[str, int], str] = {}
    try:
        for memory in memory_rows:
            kind = memory["kind"]
            if kind == "fact":
                database_id = facts.add(memory["subject"], memory["content"])
            elif kind == "episode":
                database_id = episodes.add(memory["happened_at"], memory["summary"])
            else:
                raise ValueError(f"unsupported memory kind: {kind}")
            database_ids[(kind, database_id)] = str(memory["id"])

        fact_ids = [database_ids[("fact", row["id"])] for row in facts.search(query, top_k=4)]
        episode_ids = [
            database_ids[("episode", row["id"])] for row in episodes.search(query, top_k=3)
        ]
        return fact_ids, episode_ids
    finally:
        database.close()


def _retrieved_ids(memory_rows: list[dict[str, Any]], query: str) -> list[str]:
    """Production concatenation order: ``facts[:4] + episodes[:3]``.

    Kept as the entry point for ``src/dataset/build_grpo_hard_haystack.py``,
    which needs the exact production order to decide whether a candidate query
    retrieves its evidence. Rank-scored callers -- ``scripts/grpo_gate_reward.py``
    and the ``retrieval.branch`` metrics -- use ``_retrieved_branches`` so episode
    evidence is not ranked behind the facts branch.
    """
    fact_ids, episode_ids = _retrieved_branches(memory_rows, query)
    return fact_ids + episode_ids


def _evidence_kinds(memory_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Map ``str(memory["id"])`` to its kind, so evidence can be routed to a branch."""
    return {
        str(memory["id"]): memory["kind"]
        for memory in memory_rows
        if isinstance(memory, dict) and memory.get("kind") in {"fact", "episode"}
    }


def _first_rank(retrieved: list[str], evidence_ids: set[str]) -> int | None:
    for index, memory_id in enumerate(retrieved, start=1):
        if memory_id in evidence_ids:
            return index
    return None


def _evidence_rank(
    fact_ids: list[str], episode_ids: list[str], evidence_ids: set[str]
) -> int | None:
    """Best branch-internal rank over the branches that hold the evidence.

    Each branch counts from 1, so episode evidence is not pushed behind the
    facts branch just because the two branches are concatenated for the model.
    """
    fact_rank = _first_rank(fact_ids, evidence_ids)
    if fact_rank is not None:
        return fact_rank
    return _first_rank(episode_ids, evidence_ids)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


# Each branch can only return as many rows as its production top-k, so the rank
# cutoffs differ per branch: a rank of 4 is reachable for facts, not episodes.
_BRANCH_CUTOFFS: dict[str, tuple[int, ...]] = {
    "fact": (1, 2, 3, 4),
    "episode": (1, 2, 3),
    "branch": (1, 2, 3, 4),
}


def _new_branch_bucket(name: str) -> dict[str, Any]:
    return {
        "positive_total": 0,
        "query_candidates": 0,
        "hits": 0,
        "rr_sum": 0.0,
        "rank_hits": {cutoff: 0 for cutoff in _BRANCH_CUTOFFS[name]},
    }


def _record_branch(bucket: dict[str, Any], rank: int | None) -> None:
    if rank is None:
        return
    bucket["hits"] += 1
    bucket["rr_sum"] += 1.0 / rank
    for cutoff in bucket["rank_hits"]:
        bucket["rank_hits"][cutoff] += int(rank <= cutoff)


def _branch_metrics(name: str, bucket: dict[str, Any]) -> dict[str, Any]:
    positive_total = bucket["positive_total"]
    query_candidates = bucket["query_candidates"]
    metrics: dict[str, Any] = {
        "rank_scope": "branch_internal",
        "positive_total": positive_total,
        "query_candidates": query_candidates,
        "hits": bucket["hits"],
        "hit_rate": _ratio(bucket["hits"], positive_total),
        "conditional_hit_rate": _ratio(bucket["hits"], query_candidates),
        "mrr": round(bucket["rr_sum"] / positive_total, 6) if positive_total else None,
        "conditional_mrr": (
            round(bucket["rr_sum"] / query_candidates, 6) if query_candidates else None
        ),
    }
    for cutoff in _BRANCH_CUTOFFS[name]:
        metrics[f"hit_at_{cutoff}"] = _ratio(bucket["rank_hits"][cutoff], positive_total)
    return metrics


def _evaluate_rows(
    rows: list[dict[str, Any]], retrieval_scope: str = "full_memory"
) -> dict[str, Any]:
    tp = tn = fp = fn = valid = format_valid = 0
    positive_total = positive_hits = query_candidates = query_hits = 0
    rank_hits = {1: 0, 3: 0, 5: 0}
    reciprocal_rank_sum = 0.0
    conditional_reciprocal_rank_sum = 0.0
    branches: dict[str, dict[str, Any]] = {
        name: _new_branch_bucket(name) for name in _BRANCH_CUTOFFS
    }
    unresolved_evidence_rows = 0
    records: list[dict[str, Any]] = []

    for row in rows:
        target = row["gate_target"]
        gold_retrieve = bool(target["retrieve"])
        prediction = parse_prediction(row.get("prediction_row", {}))
        predicted_retrieve = prediction.get("retrieve") if isinstance(prediction, dict) else None
        is_valid = isinstance(predicted_retrieve, bool)
        is_format_valid = output_is_valid(prediction, row.get("language", ""))
        valid += int(is_valid)
        format_valid += int(is_format_valid)

        if gold_retrieve:
            positive_total += 1
            if predicted_retrieve is True:
                tp += 1
            else:
                fn += 1
        elif predicted_retrieve is False:
            tn += 1
        else:
            fp += 1

        evidence_ids: set[str] = set()
        if gold_retrieve:
            # Which branch holds this row's evidence decides the branch
            # denominators; a row carrying both kinds counts towards both.
            evidence_ids = {str(value) for value in row.get("evidence_ids") or []}
            kinds = _evidence_kinds(row.get("memory_rows") or [])
            unresolved_evidence_rows += int(any(value not in kinds for value in evidence_ids))
            is_candidate = predicted_retrieve is True and is_format_valid
            for name in ("fact", "episode"):
                if any(kinds.get(value) == name for value in evidence_ids):
                    branches[name]["positive_total"] += 1
                    branches[name]["query_candidates"] += int(is_candidate)
            branches["branch"]["positive_total"] += 1
            branches["branch"]["query_candidates"] += int(is_candidate)

        hit = False
        rank = None
        if gold_retrieve and predicted_retrieve is True and is_format_valid:
            query_candidates += 1
            fact_ids, episode_ids = _retrieved_branches(
                row.get("memory_rows") or [], prediction["query"]
            )
            retrieved = fact_ids + episode_ids
            hit = bool(evidence_ids.intersection(retrieved))
            query_hits += int(hit)
            positive_hits += int(hit)
            rank = _first_rank(retrieved, evidence_ids)
            if rank is not None:
                reciprocal_rank_sum += 1.0 / rank
                conditional_reciprocal_rank_sum += 1.0 / rank
                for cutoff in rank_hits:
                    rank_hits[cutoff] += int(rank <= cutoff)
            _record_branch(branches["fact"], _first_rank(fact_ids, evidence_ids))
            _record_branch(branches["episode"], _first_rank(episode_ids, evidence_ids))
            _record_branch(
                branches["branch"], _evidence_rank(fact_ids, episode_ids, evidence_ids)
            )
        records.append({"row": row, "hit": hit, "rank": rank, "format_valid": is_format_valid})

    precision = _ratio(tp, tp + fp)
    positive_recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1_positive = _ratio(2 * tp, 2 * tp + fp + fn)
    f1_negative = _ratio(2 * tn, 2 * tn + fp + fn)
    macro_f1 = (
        round((f1_positive + f1_negative) / 2, 6)
        if f1_positive is not None and f1_negative is not None
        else None
    )
    accuracy = _ratio(tp + tn, len(rows))

    return {
        "rows": len(rows),
        "retrieval_scope": retrieval_scope,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "classification": {
            "accuracy": accuracy,
            "precision": precision,
            "positive_recall": positive_recall,
            "negative_specificity": specificity,
            "unnecessary_retrieval_rate": _ratio(fp, tn + fp),
            "f1_positive": f1_positive,
            "f1_negative": f1_negative,
            "macro_f1": macro_f1,
        },
        "output": {
            "json_valid_rate": _ratio(valid, len(rows)),
            "format_valid_rate": _ratio(format_valid, len(rows)),
        },
        "evidence_only": {
            "positive_total": positive_total,
            "hits": positive_hits,
            "hit_rate": _ratio(positive_hits, positive_total),
            "hit_at_1": _ratio(rank_hits[1], positive_total),
            "hit_at_3": _ratio(rank_hits[3], positive_total),
            "hit_at_5": _ratio(rank_hits[5], positive_total),
            "mrr": round(reciprocal_rank_sum / positive_total, 6) if positive_total else None,
            "conditional_mrr": (
                round(conditional_reciprocal_rank_sum / query_candidates, 6)
                if query_candidates else None
            ),
        },
        "retrieval": {
            "unresolved_evidence_rows": unresolved_evidence_rows,
            "fact": _branch_metrics("fact", branches["fact"]),
            "episode": _branch_metrics("episode", branches["episode"]),
            "branch": _branch_metrics("branch", branches["branch"]),
        },
        "end_to_end": {
            "positive_total": positive_total,
            "successful_positives": positive_hits,
            "memory_recall": _ratio(positive_hits, positive_total),
            "query_candidates": query_candidates,
            "query_hits": query_hits,
            "conditional_query_hit": _ratio(query_hits, query_candidates),
        },
        "_records": records,
    }


def _without_records(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "_records"}


def _json_column(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _apply_haystack(
    row: dict[str, Any], haystack_row: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(row)
    gate_target = dict(merged.get("gate_target") or {})
    if "gate_label" not in haystack_row:
        raise ValueError(f"haystack row missing gate_label: {haystack_row.get('id')}")
    gate_target["retrieve"] = bool(haystack_row["gate_label"])
    merged["gate_target"] = gate_target
    merged["memory_rows"] = _json_column(haystack_row.get("memory_rows_json"), [])
    merged["evidence_ids"] = _json_column(haystack_row.get("evidence_ids_json"), [])
    merged["hard_negative_ids"] = _json_column(
        haystack_row.get("hard_negative_ids_json"), []
    )
    merged["retrieval_scope"] = "hard_haystack"
    return merged


def evaluate(
    split_path: Path,
    full_path: Path,
    predictions_path: Path,
    haystack_path: Path | None = None,
) -> dict[str, Any]:
    split_rows = read_jsonl(split_path)
    full_by_id = {row["id"]: row for row in read_jsonl(full_path)}
    haystack_by_id: dict[str, dict[str, Any]] = {}
    if haystack_path is not None:
        haystack_by_id = {row["id"]: row for row in read_jsonl(haystack_path)}
    prediction_rows = read_jsonl(predictions_path)
    predictions: dict[str, dict[str, Any]] = {}
    for index, prediction in enumerate(prediction_rows):
        sample_id = prediction.get("id")
        if sample_id is None and index < len(split_rows):
            sample_id = split_rows[index]["id"]
        if sample_id is not None:
            predictions[str(sample_id)] = prediction
    joined: list[dict[str, Any]] = []
    for index, split_row in enumerate(split_rows):
        sample_id = split_row["id"]
        if sample_id not in full_by_id:
            raise ValueError(f"split id missing from full data: {sample_id}")
        row = dict(full_by_id[sample_id])
        if haystack_path is not None:
            haystack_row = haystack_by_id.get(sample_id)
            if haystack_row is None:
                raise ValueError(f"haystack id missing for split row: {sample_id}")
            row = _apply_haystack(row, haystack_row)
        prediction = predictions.get(sample_id)
        if prediction is None and index < len(prediction_rows):
            prediction = prediction_rows[index]
        row["prediction_row"] = prediction or {}
        joined.append(row)

    retrieval_scope = "hard_haystack" if haystack_path is not None else "full_memory"
    overall = _evaluate_rows(joined, retrieval_scope)
    slices: dict[str, dict[str, Any]] = {}
    for name, key_fn in (
        ("language", lambda row: row.get("language", "unknown")),
        ("source", lambda row: (row.get("source") or {}).get("dataset", "unknown")),
    ):
        slices[name] = {
            str(key): _without_records(
                _evaluate_rows([row for row in joined if key_fn(row) == key], retrieval_scope)
            )
            for key in sorted({key_fn(row) for row in joined})
        }

    result = _without_records(overall)
    result.update({
        "split": str(split_path),
        "predictions": str(predictions_path),
        "haystack": str(haystack_path) if haystack_path is not None else None,
        "slices": slices,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--haystack",
        type=Path,
        help="optional GRPO hard-haystack file containing hidden memory rows",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.split, args.full, args.predictions, args.haystack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
