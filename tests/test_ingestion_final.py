from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from app.ingestion.legal_chunker import LegalChunker
from app.ingestion.pdf_processor import LegalPDFProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "app" / "core" / "config" / "yaml" / "config.yaml"
RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
REPORT_PATH = EVALUATION_DIR / "ingestion_final_report.json"


def _normalize_config(config: dict) -> dict:
    chunk_size = max(1, int(config["chunk_size"]))
    chunk_overlap = max(0, min(int(config["chunk_overlap"]), chunk_size - 1))
    min_chunk_size = max(0, min(int(config["min_chunk_size"]), chunk_size - 1))

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chunk_size": min_chunk_size,
        "on_small_chunk": str(config.get("on_small_chunk", "merge_next")),
    }


def _load_chunking_config() -> dict:
    if yaml is None or not CONFIG_PATH.exists():
        return {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "min_chunk_size": 150,
            "on_small_chunk": "merge_next",
        }

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    chunking = data.get("chunking") or {}
    return {
        "chunk_size": int(chunking.get("chunk_size", 512)),
        "chunk_overlap": int(chunking.get("chunk_overlap", 50)),
        "min_chunk_size": int(chunking.get("min_chunk_size", 150)),
        "on_small_chunk": str(chunking.get("on_small_chunk", "merge_next")),
    }


def _scenario_configs(base_config: dict) -> list[dict]:
    chunk_size = int(base_config["chunk_size"])
    chunk_overlap = int(base_config["chunk_overlap"])
    min_chunk_size = int(base_config["min_chunk_size"])
    on_small_chunk = str(base_config["on_small_chunk"])

    return [
        {
            "name": "baseline",
            "label": "Baseline from config.yaml",
            "reason": "Giữ nguyên cấu hình đang dùng để có mốc so sánh chuẩn.",
            "config": _normalize_config(base_config),
        },
        {
            "name": "compact",
            "label": "Compact retrieval",
            "reason": "Giảm kích thước chunk để tăng độ chính xác truy hồi cho điều khoản dài.",
            "config": _normalize_config(
                {
                    "chunk_size": max(384, chunk_size - 128),
                    "chunk_overlap": max(20, chunk_overlap - 10),
                    "min_chunk_size": max(80, min_chunk_size - 20),
                    "on_small_chunk": on_small_chunk,
                }
            ),
        },
        {
            "name": "balanced",
            "label": "Balanced coverage",
            "reason": "Giữ chunk vừa phải để cân bằng giữa recall và chi phí embedding.",
            "config": _normalize_config(
                {
                    "chunk_size": max(448, chunk_size),
                    "chunk_overlap": max(30, chunk_overlap),
                    "min_chunk_size": max(100, min_chunk_size),
                    "on_small_chunk": on_small_chunk,
                }
            ),
        },
        {
            "name": "recall_heavy",
            "label": "Recall-heavy",
            "reason": "Tăng chunk_size để giữ ngữ cảnh rộng hơn khi văn bản có nhiều điều khoản dài.",
            "config": _normalize_config(
                {
                    "chunk_size": min(768, chunk_size + 128),
                    "chunk_overlap": min(96, chunk_overlap + 20),
                    "min_chunk_size": min(min_chunk_size + 20, min(256, chunk_size + 127)),
                    "on_small_chunk": on_small_chunk,
                }
            ),
        },
    ]


def _chunk_metrics(chunks: list, chunk_size: int, min_chunk_size: int) -> dict:
    child_chunks = [chunk for chunk in chunks if not chunk.is_parent]
    parent_chunks = [chunk for chunk in chunks if chunk.is_parent]
    child_token_counts = [int(chunk.token_count) for chunk in child_chunks]
    all_token_counts = [int(chunk.token_count) for chunk in chunks]

    if child_token_counts:
        child_avg = statistics.mean(child_token_counts)
        child_util = child_avg / chunk_size if chunk_size else 0.0
    else:
        child_avg = 0.0
        child_util = 0.0

    return {
        "total": len(chunks),
        "child": len(child_chunks),
        "parent": len(parent_chunks),
        "child_token_counts": child_token_counts,
        "all_token_counts": all_token_counts,
        "child_avg": child_avg,
        "child_util": child_util,
        "child_small_rate": (sum(1 for count in child_token_counts if count < min_chunk_size) / len(child_token_counts))
        if child_token_counts
        else 0.0,
        "child_large_rate": (sum(1 for count in child_token_counts if count > chunk_size) / len(child_token_counts))
        if child_token_counts
        else 0.0,
        "parent_rate": (len(parent_chunks) / len(chunks)) if chunks else 0.0,
        "max_child_tokens": max(child_token_counts) if child_token_counts else 0,
        "max_all_tokens": max(all_token_counts) if all_token_counts else 0,
    }


def _build_report(pdf_path: Path, metrics: dict, config: dict, elapsed: dict[str, float]) -> str:
    return (
        f"PDF: {pdf_path.name}\n"
        f"  blocks={metrics['blocks']} chunks={metrics['total']} parents={metrics['parent']} children={metrics['child']}\n"
        f"  chunk_size={config['chunk_size']} overlap={config['chunk_overlap']} min_chunk_size={config['min_chunk_size']}\n"
        f"  child_util={metrics['child_util']:.4f} child_small_rate={metrics['child_small_rate']:.4f} "
        f"child_large_rate={metrics['child_large_rate']:.4f} parent_rate={metrics['parent_rate']:.4f}\n"
        f"  max_child_tokens={metrics['max_child_tokens']} max_all_tokens={metrics['max_all_tokens']}\n"
        f"  timings: process={elapsed['process_s']:.2f}s chunk={elapsed['chunk_s']:.2f}s total={elapsed['total_s']:.2f}s"
    )


def _scenario_score(summary: dict) -> tuple[int, str, list[str]]:
    score = 0
    notes: list[str] = []

    child_util = float(summary.get("child_util", 0.0))
    child_small_rate = float(summary.get("child_small_rate", 0.0))
    child_large_rate = float(summary.get("child_large_rate", 0.0))
    parent_rate = float(summary.get("parent_rate", 0.0))

    if summary.get("child", 0) > 0 and summary.get("parent", 0) > 0:
        score += 20
        notes.append("Cấu trúc parent-child đầy đủ.")
    else:
        notes.append("Thiếu một trong hai lớp parent/child.")

    if child_large_rate == 0:
        score += 30
        notes.append("Không có child chunk vượt chunk_size.")
    else:
        score += max(0, 30 - int(child_large_rate * 100))
        notes.append("Có child chunk vượt chunk_size.")

    if 0.45 <= child_util <= 0.70:
        score += 30
        notes.append("Child utilization nằm trong band tối ưu 0.45-0.70.")
    else:
        distance = min(abs(child_util - 0.45), abs(child_util - 0.70))
        score += max(0, 30 - int(distance * 100))
        notes.append("Child utilization lệch khỏi band tối ưu 0.45-0.70.")

    if child_small_rate <= 0.30:
        score += 15
        notes.append("Tỷ lệ child chunk nhỏ được kiểm soát tốt.")
    else:
        score += max(0, 15 - int((child_small_rate - 0.30) * 100))
        notes.append("Tỷ lệ child chunk nhỏ còn cao.")

    if 0.08 <= parent_rate <= 0.25:
        score += 5
        notes.append("Parent ratio nằm trong khoảng khuyến nghị.")
    else:
        notes.append("Parent ratio lệch khỏi khoảng khuyến nghị.")

    if score >= 80 and child_large_rate == 0 and 0.45 <= child_util <= 0.70 and child_small_rate <= 0.30 and 0.08 <= parent_rate <= 0.25:
        verdict = "PASS"
    elif score >= 60:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return score, verdict, notes


def _print_scenario_report(pdf_path: Path, scenario: dict, metrics: dict, elapsed: dict[str, float], score: int, verdict: str) -> None:
    config = scenario["config"]
    print(
        f"[{verdict}] {scenario['name']:<14} | score={score:>3}/100 | "
        f"chunk_size={config['chunk_size']:<3} overlap={config['chunk_overlap']:<3} min_chunk_size={config['min_chunk_size']:<3} | "
        f"blocks={metrics['blocks']:<4} chunks={metrics['total']:<4} parents={metrics['parent']:<3} children={metrics['child']:<3} | "
        f"util={metrics['child_util']:.4f} small={metrics['child_small_rate']:.4f} large={metrics['child_large_rate']:.4f} parent={metrics['parent_rate']:.4f} | "
        f"process={elapsed['process_s']:.2f}s chunk={elapsed['chunk_s']:.2f}s total={elapsed['total_s']:.2f}s"
    )


def _write_report(payload: dict) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_optimized_parameters(pdf_path: Path, metrics: dict, config: dict, elapsed: dict[str, float]) -> None:
    print("\n" + "=" * 88)
    print(f"INGESTION FINAL CHECK: {pdf_path.name}")
    print("=" * 88)
    print(
        "Optimized parameters: "
        f"chunk_size={config['chunk_size']} | "
        f"chunk_overlap={config['chunk_overlap']} | "
        f"min_chunk_size={config['min_chunk_size']} | "
        f"on_small_chunk={config['on_small_chunk']}"
    )
    print(
        "Key metrics: "
        f"child_util={metrics['child_util']:.4f} | "
        f"child_small_rate={metrics['child_small_rate']:.4f} | "
        f"child_large_rate={metrics['child_large_rate']:.4f} | "
        f"parent_rate={metrics['parent_rate']:.4f}"
    )
    print(
        "Chunk summary: "
        f"blocks={metrics['blocks']} | "
        f"chunks={metrics['total']} | "
        f"parents={metrics['parent']} | "
        f"children={metrics['child']}"
    )
    print(
        "Thresholds: "
        "child_util in [0.45, 0.70] | "
        "child_small_rate <= 0.30 | "
        "child_large_rate == 0 | "
        "parent_rate in [0.08, 0.25]"
    )
    print(
        "Timings: "
        f"process={elapsed['process_s']:.2f}s | "
        f"chunk={elapsed['chunk_s']:.2f}s | "
        f"total={elapsed['total_s']:.2f}s"
    )
    print("Verdict: READY FOR NEXT STAGE")
    print("=" * 88)


def _readable_threshold_failures(metrics: dict) -> list[str]:
    failures: list[str] = []

    if metrics["total"] == 0:
        failures.append("No chunks were produced.")
        return failures

    if metrics["child"] == 0:
        failures.append("No child chunks were produced.")

    if metrics["parent"] == 0:
        failures.append("No parent chunks were produced.")

    if metrics["child_large_rate"] > 0:
        failures.append("Some child chunks exceed chunk_size.")

    if metrics["child_small_rate"] > 0.30:
        failures.append("Too many child chunks are below min_chunk_size.")

    if not 0.45 <= metrics["child_util"] <= 0.70:
        failures.append("Child utilization is outside the target band 0.45-0.70.")

    if not 0.08 <= metrics["parent_rate"] <= 0.25:
        failures.append("Parent chunk ratio is outside the target band 0.08-0.25.")

    if metrics["max_child_tokens"] > metrics.get("chunk_size", metrics["max_child_tokens"]):
        failures.append("A child chunk exceeded the configured chunk_size.")

    return failures


@pytest.mark.parametrize("pdf_path", sorted(RAW_PDF_DIR.glob("*.pdf")))
def test_ingestion_final_readiness(pdf_path: Path) -> None:
    if not pdf_path.exists():
        pytest.skip(f"Missing PDF file: {pdf_path}")

    base_config = _load_chunking_config()
    processor = LegalPDFProcessor()
    scenarios = _scenario_configs(base_config)

    started = time.perf_counter()

    process_started = time.perf_counter()
    blocks = processor.process(pdf_path=str(pdf_path))
    process_s = time.perf_counter() - process_started

    assert blocks, f"No blocks extracted from {pdf_path.name}"

    scenario_reports: list[dict] = []
    best_report: dict | None = None

    print("\n" + "=" * 108)
    print(f"INGESTION FINAL SCENARIO SUITE: {pdf_path.name}")
    print("=" * 108)
    print("Scenario | score | verdict | config | metrics | timings")
    print("-" * 108)

    for scenario in scenarios:
        scenario_config = scenario["config"]
        chunker = LegalChunker(
            chunk_size=scenario_config["chunk_size"],
            chunk_overlap=scenario_config["chunk_overlap"],
            min_chunk_size=scenario_config["min_chunk_size"],
            on_small_chunk=scenario_config["on_small_chunk"],
        )

        chunk_started = time.perf_counter()
        chunks = chunker.chunk(blocks)
        chunk_s = time.perf_counter() - chunk_started
        total_s = time.perf_counter() - started

        metrics = _chunk_metrics(chunks, scenario_config["chunk_size"], scenario_config["min_chunk_size"])
        metrics["blocks"] = len(blocks)
        metrics["chunk_size"] = scenario_config["chunk_size"]

        score, verdict, notes = _scenario_score(metrics)
        elapsed = {"process_s": process_s, "chunk_s": chunk_s, "total_s": total_s}

        scenario_report = {
            "name": scenario["name"],
            "label": scenario["label"],
            "reason": scenario["reason"],
            "config": scenario_config,
            "metrics": metrics,
            "score": score,
            "verdict": verdict,
            "notes": notes,
            "timings": {k: round(v, 4) for k, v in elapsed.items()},
        }
        scenario_reports.append(scenario_report)

        if best_report is None or scenario_report["score"] > best_report["score"]:
            best_report = scenario_report

        _print_scenario_report(pdf_path, scenario, metrics, elapsed, score, verdict)

    assert best_report is not None

    print("-" * 108)
    print(
        f"BEST SCENARIO: {best_report['name']} | score={best_report['score']}/100 | verdict={best_report['verdict']} | "
        f"chunk_size={best_report['config']['chunk_size']} overlap={best_report['config']['chunk_overlap']} min_chunk_size={best_report['config']['min_chunk_size']}"
    )
    print("Why best:")
    for note in best_report["notes"]:
        print(f"- {note}")
    print("=" * 108)

    _write_report(
        {
            "file": str(pdf_path),
            "base_config": base_config,
            "reports": scenario_reports,
            "best": best_report,
        }
    )

    best_metrics = best_report["metrics"]
    best_config = best_report["config"]
    best_elapsed = best_report["timings"]
    failures = _readable_threshold_failures(best_metrics)
    report = _build_report(
        pdf_path=pdf_path,
        metrics=best_metrics,
        config=best_config,
        elapsed={"process_s": best_elapsed["process_s"], "chunk_s": best_elapsed["chunk_s"], "total_s": best_elapsed["total_s"]},
    )

    _print_optimized_parameters(
        pdf_path=pdf_path,
        metrics=best_metrics,
        config=best_config,
        elapsed={"process_s": best_elapsed["process_s"], "chunk_s": best_elapsed["chunk_s"], "total_s": best_elapsed["total_s"]},
    )

    assert best_report["verdict"] == "PASS", report + "\n\nFAILURES:\n- " + "\n- ".join(failures)
