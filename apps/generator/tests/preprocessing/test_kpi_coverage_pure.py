"""Unit tests for the pure-Python / numpy parts of kpi_coverage — no Spark required."""

import numpy as np

from genpm.preprocessing.logic.kpi_coverage import _greedy_order, find_elbow, suggest_threshold

# ---------------------------------------------------------------------------
# find_elbow
# ---------------------------------------------------------------------------


def _make_curve(counts: list[int]) -> list[dict]:
    return [{"step": i + 1, "joint_windows": c} for i, c in enumerate(counts)]


def test_find_elbow_cliff_at_step_two():
    # Coverage accelerates downward at step 2 (d2 most negative there → elbow at index 1).
    # [1000, 800, 100, 90, 85]: d1=[-200,-700,-10,-5], d2=[-500,690,5] → argmin=0 → elbow=1
    curve = _make_curve([1000, 800, 100, 90, 85])
    elbow = find_elbow(curve)
    assert elbow == 1


def test_find_elbow_returns_valid_index():
    curve = _make_curve([500, 400, 300, 100, 95])
    elbow = find_elbow(curve)
    assert 0 <= elbow < len(curve)


def test_find_elbow_monotone_curve_does_not_raise():
    # Perfectly linear drop — no true elbow, but should not raise
    curve = _make_curve([100, 80, 60, 40, 20])
    elbow = find_elbow(curve)
    assert isinstance(elbow, int)


# ---------------------------------------------------------------------------
# suggest_threshold
# ---------------------------------------------------------------------------


def test_suggest_threshold_returns_value_before_elbow():
    curve = _make_curve([1000, 900, 200, 190])
    elbow_idx = 2
    threshold = suggest_threshold(curve, elbow_idx)
    assert threshold == 900  # step before elbow (index 1)


def test_suggest_threshold_at_index_zero_returns_first():
    curve = _make_curve([500, 300, 100])
    threshold = suggest_threshold(curve, 0)
    assert threshold == 500


# ---------------------------------------------------------------------------
# _greedy_order
# ---------------------------------------------------------------------------


def _bool_matrix(rows: list[list[int]], n_cols: int) -> np.ndarray:
    M = np.zeros((len(rows), n_cols), dtype=bool)
    for i, cols in enumerate(rows):
        M[i, cols] = True
    return M


def test_greedy_order_seed_is_highest_coverage_kpi():
    # KPI 0: 3 windows, KPI 1: 5 windows, KPI 2: 1 window
    M = _bool_matrix([[0, 1, 2], [0, 1, 2, 3, 4], [4]], n_cols=5)
    order, counts = _greedy_order(M, forced_rows=[])
    assert order[0] == 1, "seed should be the KPI with most windows (row 1)"


def test_greedy_order_output_length_equals_n_kpis():
    M = _bool_matrix([[0, 1], [1, 2], [2, 3]], n_cols=4)
    order, counts = _greedy_order(M, forced_rows=[])
    assert len(order) == 3
    assert len(counts) == 3


def test_greedy_order_counts_non_increasing():
    M = _bool_matrix([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], n_cols=6)
    _, counts = _greedy_order(M, forced_rows=[])
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))


def test_greedy_order_forced_rows_appear_first():
    M = _bool_matrix([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], n_cols=6)
    order, _ = _greedy_order(M, forced_rows=[2])
    assert order[0] == 2, "forced row must be seeded first"


def test_greedy_order_all_rows_appear_exactly_once():
    M = _bool_matrix([[0, 1], [2, 3], [0, 2]], n_cols=4)
    order, _ = _greedy_order(M, forced_rows=[])
    assert sorted(order) == [0, 1, 2]


def test_greedy_order_single_kpi():
    M = _bool_matrix([[0, 1, 2]], n_cols=3)
    order, counts = _greedy_order(M, forced_rows=[])
    assert order == [0]
    assert counts == [3]
