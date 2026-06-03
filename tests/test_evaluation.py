"""Benchmark CSV evaluation core tests."""

from __future__ import annotations

from pathlib import Path

from g2p.evaluation import _BENCHMARKS, evaluate_benchmark_csv, evaluate_benchmark_rows
from g2p.phones import phones_to_text
from g2p.pipeline import transcribe_word


def test_eval_rows_reports_category_and_source_breakdowns() -> None:
    result = transcribe_word("ekmek", use_reference=True)
    artifacts = evaluate_benchmark_rows(
        [
            {
                "id": "row-1",
                "word": "ekmek",
                "category": "clean_core",
                "target_phones": phones_to_text(result.phonemes),
            }
        ],
        name="fixture",
        use_reference=True,
    )

    assert artifacts.report["rows_scored"] == 1
    assert artifacts.report["metrics"]["exact_match"] == 1.0
    # G2P final: REFERENCE_IPA minimal — ekmek artık kural kaynaklı.
    assert artifacts.report["source_breakdown"]["rules"]["rows"] == 1
    assert artifacts.report["category_breakdown"]["clean_core"]["rows"] == 1
    assert artifacts.report["exception_type_breakdown"]["none"]["rows"] == 1
    assert artifacts.report["phone_error_breakdown"]["by_target_phone"]
    assert artifacts.mismatches == []


def test_eval_csv_writes_json_and_mismatch_csv(tmp_path: Path) -> None:
    benchmark_csv = tmp_path / "benchmark.csv"
    benchmark_csv.write_text(
        "word,category,target_phones\nekmek,clean_core,æ k ˈ m æ k\n",
        encoding="utf-8",
    )

    artifacts = evaluate_benchmark_csv(
        benchmark_csv,
        name="fixture",
        output_dir=tmp_path / "reports",
        use_reference=False,
    )

    assert artifacts.report_path is not None
    assert artifacts.report_path.exists()
    assert artifacts.mismatches_path is not None
    assert artifacts.mismatches_path.exists()
    assert artifacts.phone_errors_path is not None
    assert artifacts.phone_errors_path.exists()
    assert artifacts.substitutions_path is not None
    assert artifacts.substitutions_path.exists()
    assert "reference_comparison" in artifacts.report


def test_eval_rows_reports_phone_level_confusion() -> None:
    artifacts = evaluate_benchmark_rows(
        [
            {
                "id": "row-1",
                "word": "memur",
                "category": "fixture",
                "target_phones": "m eː m u ɾ̞̊",
            }
        ],
        name="fixture",
        use_reference=False,
    )

    breakdown = artifacts.report["phone_error_breakdown"]
    assert breakdown["by_target_phone"]["eː"]["substitutions"] == 1
    assert breakdown["by_target_phone"]["eː"]["per"] == 1.0
    assert {"target": "eː", "predicted": "e", "count": 1} in breakdown[
        "top_substitutions"
    ]


def test_builtin_benchmark_paths_match_committed_data_dir() -> None:
    assert _BENCHMARKS["blind_75"] == Path("data/g2p/blind_75.csv")
    assert _BENCHMARKS["mfa"] == Path("data/g2p/mfa.csv")
    assert _BENCHMARKS["wiktionary"] == Path("data/g2p/wiktionary.csv")
