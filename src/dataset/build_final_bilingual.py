"""Build paired hard negatives and a balanced bilingual gate SFT file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.dataset.builder import build_dataset
from src.dataset.generate_hard_negatives import generate as generate_negatives
from src.dataset.generate_query_candidates import generate as generate_candidates
from src.dataset.select_query_candidates import select as select_candidates
from src.dataset.translate_with_opencode import translate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = ("locomo_500", "longmemeval_500", "rhelm_158", "personamem_v2_500")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build(
    sources: list[str],
    output_dir: Path,
    translate_fraction: float,
    hard_negative_cache: Path,
    translation_cache: Path,
    limit: int | None = None,
    hard_negative_workers: int = 1,
) -> dict[str, int]:
    positives: list[dict[str, Any]] = []
    for name in sources:
        path = ROOT / "data/derived/final" / f"{name}.sft.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        positives.extend(read_jsonl(path))
    if limit is not None and limit > 0:
        positives = positives[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    positive_input = output_dir / "positive_all.jsonl"
    write_jsonl(positive_input, positives)

    negative_input = output_dir / "hard_negatives.jsonl"
    negative_summary = generate_negatives(
        positive_input,
        negative_input,
        hard_negative_cache,
        workers=hard_negative_workers,
    )
    negatives = read_jsonl(negative_input)
    negative_by_positive = {row["source"].get("paired_positive_id"): row for row in negatives}

    pair_ids = [row["id"] for row in positives if row["id"] in negative_by_positive]
    translate_count = int(len(pair_ids) * translate_fraction)
    translated_pair_ids = set(pair_ids[:translate_count])
    translate_rows = [
        row for row in positives + negatives
        if row.get("id") in translated_pair_ids
        or row.get("source", {}).get("paired_positive_id") in translated_pair_ids
    ]
    translation_input = output_dir / "translated_pairs_input.jsonl"
    translation_output = output_dir / "translated_pairs.jsonl"
    write_jsonl(translation_input, translate_rows)
    translation_summary = translate(
        translation_input,
        translation_output,
        translation_cache,
        id_suffix="_zh",
    )
    translated = read_jsonl(translation_output)
    translated_positive = [row for row in translated if row.get("gate_target", {}).get("retrieve")]
    translated_negative = [row for row in translated if not row.get("gate_target", {}).get("retrieve")]

    zh_positive_input = output_dir / "zh_positive_input.jsonl"
    zh_positive_candidates = output_dir / "zh_positive_candidates.jsonl"
    zh_positive_selected = output_dir / "zh_positive_selected.jsonl"
    zh_positive_rejected = output_dir / "zh_positive_query_rejected.jsonl"
    zh_positive_sft = output_dir / "zh_positive.sft.jsonl"
    zh_positive_sft_rejected = output_dir / "zh_positive.rejected.jsonl"
    write_jsonl(zh_positive_input, translated_positive)
    if translated_positive:
        generate_candidates(
            zh_positive_input,
            zh_positive_candidates,
            output_dir / "zh_query_candidate_cache.json",
        )
        select_candidates(zh_positive_candidates, zh_positive_selected, zh_positive_rejected)
        build_dataset(
            zh_positive_selected,
            zh_positive_sft,
            zh_positive_sft_rejected,
            min_recall=0.0,
        )
    else:
        write_jsonl(zh_positive_sft, [])

    translated_positive_ids = {row["id"].removesuffix("_zh") for row in read_jsonl(zh_positive_sft)}
    chinese_negative = [
        row
        for row in translated_negative
        if not (row.get("translation") or {}).get("fallback")
        and row.get("source", {}).get("paired_positive_id", "").removesuffix("_zh") in translated_positive_ids
    ]
    translated_negative_ids = {
        row.get("source", {}).get("paired_positive_id", "").removesuffix("_zh")
        for row in chinese_negative
    }
    effective_translated_pairs = translated_positive_ids & translated_negative_ids
    english_negative = [
        row for row in negatives
        if row.get("source", {}).get("paired_positive_id") not in effective_translated_pairs
    ]
    english_negative_input = output_dir / "english_negatives.jsonl"
    chinese_negative_input = output_dir / "chinese_negatives.jsonl"
    english_negative_sft = output_dir / "english_negatives.sft.jsonl"
    chinese_negative_sft = output_dir / "chinese_negatives.sft.jsonl"
    write_jsonl(english_negative_input, english_negative)
    write_jsonl(chinese_negative_input, chinese_negative)
    build_dataset(
        english_negative_input,
        english_negative_sft,
        output_dir / "english_negatives.rejected.jsonl",
        min_recall=0.0,
    )
    build_dataset(
        chinese_negative_input,
        chinese_negative_sft,
        output_dir / "chinese_negatives.rejected.jsonl",
        min_recall=0.0,
    )

    final_rows: list[dict[str, Any]] = []
    for row in positives:
        if row["id"] in effective_translated_pairs:
            continue
        final_rows.append(row)
    final_rows.extend(
        row
        for row in read_jsonl(zh_positive_sft)
        if row["id"].removesuffix("_zh") in effective_translated_pairs
    )
    final_rows.extend(read_jsonl(english_negative_sft))
    final_rows.extend(read_jsonl(chinese_negative_sft))

    final_path = output_dir / "gate_final.sft.jsonl"
    sidecar_path = output_dir / "gate_final.sidecar.jsonl"
    write_jsonl(final_path, final_rows)
    sidecar = []
    for row in final_rows:
        sidecar.append({key: row.get(key) for key in ("id", "language", "source", "user_message", "memory_rows", "evidence_ids", "gate_target", "validation")})
    write_jsonl(sidecar_path, sidecar)
    return {
        "positive_input": len(positives),
        "negative_generated": negative_summary["negative_generated"],
        "negative_failed": negative_summary["failed"],
        "translated_rows": translation_summary["translated"],
        "translation_fallback": translation_summary["fallback"],
        "final_rows": len(final_rows),
        "final_zh": sum(row.get("language") == "zh" for row in final_rows),
        "final_en": sum(row.get("language") == "en" for row in final_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/derived/final_bilingual")
    parser.add_argument("--translate-fraction", type=float, default=0.5)
    parser.add_argument("--hard-negative-cache", type=Path, default=ROOT / "data/derived/hard_negative_cache.json")
    parser.add_argument("--translation-cache", type=Path, default=ROOT / "data/derived/translation_cache.json")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N positive rows")
    parser.add_argument("--hard-negative-workers", type=int, default=1)
    args = parser.parse_args()
    if not 0.0 <= args.translate_fraction <= 1.0:
        parser.error("--translate-fraction must be between 0 and 1")
    print(json.dumps(build(args.sources, args.output_dir, args.translate_fraction, args.hard_negative_cache, args.translation_cache, args.limit or None, args.hard_negative_workers), ensure_ascii=False))


if __name__ == "__main__":
    main()
