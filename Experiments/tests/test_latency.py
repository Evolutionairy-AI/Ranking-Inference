"""Tests for latency benchmarking timer utilities."""
import json
import time
from pathlib import Path

import pytest
from exp07_latency.src.timer import TimingResult, time_operation, system_info
from exp07_latency.src.charts import plot_pareto_frontier, generate_summary_table
from exp07_latency.src.flops import estimate_flops

SAMPLE_RI_RESULTS = {
    "per_bin": {
        "0-50": {
            "full": [{"median_us": 5072.1, "median_ms": 5.072, "p5_ns": 4833800, "p95_ns": 6316900}],
            "gap_only": [{"median_us": 54.2, "median_ms": 0.054, "p5_ns": 52900, "p95_ns": 66400}],
        },
        "50-100": {
            "full": [{"median_us": 11377.0, "median_ms": 11.377, "p5_ns": 10913400, "p95_ns": 13587700}],
            "gap_only": [{"median_us": 158.7, "median_ms": 0.159, "p5_ns": 148600, "p95_ns": 204400}],
        },
    },
    "setup": {
        "rank_table_load": {"median_ms": 157.2},
        "grounding_scores": {"median_ms": 642.9},
    },
    "n_examples": 160,
}

SAMPLE_BASELINE_RESULTS = {
    "forward_pass": {
        "0-50": {"median_ms": 1600.0},
        "50-100": {"median_ms": 1700.0},
    },
    "semantic_entropy": {
        "k=2": {"per_bin": {"0-50": {"total_ms": 3250.0}, "50-100": {"total_ms": 3450.0}}, "auc_roc": 0.75},
        "k=5": {"per_bin": {"0-50": {"total_ms": 8050.0}, "50-100": {"total_ms": 8550.0}}, "auc_roc": 0.75},
        "k=10": {"per_bin": {"0-50": {"total_ms": 18300.0}, "50-100": {"total_ms": 18800.0}}, "auc_roc": 0.75},
    },
    "selfcheckgpt": {
        "N=2": {"per_bin": {"0-50": {"total_ms": 3240.0}, "50-100": {"total_ms": 3440.0}}, "auc_roc": 0.75},
        "N=5": {"per_bin": {"0-50": {"total_ms": 8040.0}, "50-100": {"total_ms": 8540.0}}, "auc_roc": 0.75},
        "N=10": {"per_bin": {"0-50": {"total_ms": 16080.0}, "50-100": {"total_ms": 16580.0}}, "auc_roc": 0.75},
    },
}


def test_timing_result_fields():
    tr = TimingResult(
        operation="test_op",
        n_runs=10,
        median_ns=1000,
        p5_ns=800,
        p95_ns=1200,
        total_ns=10000,
    )
    assert tr.operation == "test_op"
    assert tr.median_ns == 1000
    assert tr.median_ms == pytest.approx(0.001, abs=1e-9)


def test_time_operation_measures_something():
    def slow_op():
        time.sleep(0.001)

    result = time_operation("sleep_1ms", slow_op, n_runs=5, n_warmup=2)
    assert result.operation == "sleep_1ms"
    assert result.n_runs == 5
    assert result.median_ns > 500_000
    assert result.p5_ns <= result.median_ns <= result.p95_ns


def test_time_operation_with_args():
    def add(a, b):
        return a + b

    result = time_operation("add", add, args=(1, 2), n_runs=10, n_warmup=2)
    assert result.median_ns > 0


def test_system_info_returns_dict():
    info = system_info()
    assert "cpu" in info
    assert "python_version" in info
    assert "os" in info
    assert "ram_gb" in info


from shared.utils.entity_extraction import (
    compute_entity_gaps,
    extract_entities,
    EntitySpan,
)


def test_compute_entity_gaps_accepts_preextracted_entities():
    """compute_entity_gaps should skip NER when entities are provided."""
    import numpy as np
    text = "Barack Obama visited France."
    entities = [
        EntitySpan(text="Barack Obama", entity_type="PERSON", char_start=0, char_end=12),
        EntitySpan(text="France", entity_type="GPE", char_start=21, char_end=27),
    ]
    from unittest.mock import MagicMock
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode = MagicMock(side_effect=lambda ids: "x" * len(ids))
    mock_rank_table = MagicMock()
    mock_rank_table.vocab_size = 10
    mock_rank_table.rank_to_freq = np.array([0, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10], dtype=np.float64)
    mock_rank_table.token_to_rank = {i: i + 1 for i in range(5)}
    mock_rank_table.get_rank = MagicMock(return_value=1000)

    try:
        result = compute_entity_gaps(
            text=text,
            token_ids=[1, 2, 3, 4, 5],
            logprobs=[-1.0, -2.0, -3.0, -1.5, -0.5],
            tokenizer=mock_tokenizer,
            rank_table=mock_rank_table,
            entities=entities,
        )
    except TypeError as e:
        if "entities" in str(e):
            raise AssertionError("compute_entity_gaps does not accept 'entities' parameter") from e
        raise


def test_estimate_flops_returns_all_methods():
    result = estimate_flops(n_tokens=100)
    assert "ri_gap_only" in result
    assert "ri_full" in result
    assert "se_k2" in result
    assert "se_k5" in result
    assert "se_k10" in result
    assert "scgpt_n2" in result
    assert "scgpt_n5" in result
    assert "scgpt_n10" in result


def test_estimate_flops_ri_much_cheaper():
    result = estimate_flops(n_tokens=100)
    assert result["ri_gap_only"]["flops"] < result["se_k5"]["flops"]
    assert result["ri_gap_only"]["extra_forward_passes"] == 0
    assert result["se_k5"]["extra_forward_passes"] == 5


def test_estimate_flops_scales_with_tokens():
    r100 = estimate_flops(n_tokens=100)
    r200 = estimate_flops(n_tokens=200)
    assert r200["se_k5"]["flops"] > r100["se_k5"]["flops"]
    assert r200["ri_gap_only"]["flops"] > r100["ri_gap_only"]["flops"]


def test_estimate_flops_markdown_table():
    result = estimate_flops(n_tokens=100)
    table = result["markdown_table"]
    assert "| Method" in table
    assert "RI (gap-only)" in table
    assert "SE (k=5)" in table
    assert "Forward Passes" in table


def test_summary_table_contains_flops_section():
    table = generate_summary_table(SAMPLE_RI_RESULTS, SAMPLE_BASELINE_RESULTS)
    assert "## FLOPs Comparison" in table
    assert "Forward Passes" in table
    assert "RI (gap-only)" in table
    assert "SE (k=5)" in table


def test_pareto_chart_generates(tmp_path):
    out = tmp_path / "pareto.png"
    plot_pareto_frontier(SAMPLE_RI_RESULTS, SAMPLE_BASELINE_RESULTS, out)
    assert out.exists()
    assert out.stat().st_size > 1000
