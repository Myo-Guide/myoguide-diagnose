"""Model card building. Needs mgcard[metrics]; skipped without it.

The end-to-end check is scripts/card_20260313181917.py against the real artifacts;
this covers the two places a silent misalignment could get in.
"""
import numpy as np
import pandas as pd
import pytest

mgcard = pytest.importorskip("mgcard")
from mgdiagnose.export import card  # noqa: E402


@pytest.fixture
def run():
    rng = np.random.default_rng(0)
    n = 400
    groups = np.arange(n) // 2
    y = rng.integers(0, 4, n)
    X = pd.DataFrame({"a": rng.normal(size=n)})
    test_idx = card.regenerate_outer_test_indices(X, y, groups, 5, 420)
    probs = [np.eye(4)[y[i]] * 0.6 + 0.1 for i in test_idx]
    eval_results = {"trues": [y[i] for i in test_idx], "probs": probs,
                    "preds": [p.argmax(1) for p in probs]}
    df = pd.DataFrame({"sex": rng.choice(["M", "F"], n)}, index=np.arange(n) * 3)
    return df, y, eval_results, test_idx


def test_pooled_rows_follow_the_folds(run):
    df, y, eval_results, test_idx = run
    rows = card.pooled_test_rows(df, y, eval_results, test_idx)
    assert (rows.index == df.index[np.concatenate(test_idx)]).all()


def test_pooled_rows_refuse_a_different_split(run):
    df, y, eval_results, _ = run
    other = card.regenerate_outer_test_indices(pd.DataFrame({"a": y}), y, np.arange(len(y)) // 2, 5, 1)
    with pytest.raises(ValueError, match="do not reproduce"):
        card.pooled_test_rows(df, y, eval_results, other)


def test_metrics_validate(run):
    df, y, eval_results, test_idx = run
    rows = card.pooled_test_rows(df, y, eval_results, test_idx)

    class LE:
        classes_ = np.array(["A", "B", "C", "D"])

    metrics = card.build_metrics(
        eval_results, LE(), model_id="diagnosis-test-xgb-v1", evaluation_id="eval-test",
        protocol={"type": "cross_validation", "split_unit": "patient",
                  "evaluated_artifact": "fold ensembles"},
        dataset={"n_patients": 200}, subgroups={"sex": rows["sex"].to_numpy()},
    )
    mgcard.validate_metrics(metrics)
    assert metrics["overall"]["n_test"] == len(y)
    assert metrics["confidence"]["n_folds"] == 5
