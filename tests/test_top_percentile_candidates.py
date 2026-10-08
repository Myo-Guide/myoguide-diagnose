import numpy as np
import pytest
from types import SimpleNamespace
from mgdiagnose.training import get_top_percentile_candidates


def _search(scores):
    """Fake fitted search with one halving iteration."""
    return SimpleNamespace(cv_results_={
        'iter': [0] * len(scores),
        'mean_test_score': scores,
        'params': [{'i': i} for i in range(len(scores))],
    })


def test_failed_candidates_are_ignored():
    # A NaN (failed fit) must not sink the whole selection
    params, scores = get_top_percentile_candidates(_search([0.1, np.nan, 0.9, 0.5]), percentile=90)
    assert params == [{'i': 2}] and scores == [0.9]


def test_all_failed_raises():
    with pytest.raises(ValueError, match='all 2 fits failed'):
        get_top_percentile_candidates(_search([np.nan, np.nan]))


def test_uses_last_iteration_only():
    s = _search([0.1, 0.9])
    s.cv_results_['iter'] = [0, 1]
    assert get_top_percentile_candidates(s, percentile=0)[1] == [0.9]
