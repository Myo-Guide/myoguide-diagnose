import math
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV
from tqdm.auto import tqdm

import mgdiagnose.pipeline as pipeline
from mgdiagnose.evaluation.evaluation import balanced_accuracy_scorer, balanced_accuracy_no_warn

'''This module provides the nested cross-validation training and evaluation loops,
ensuring all experiments are run with the same methodology.
'''


def compute_min_resources(X, y, groups, max_k_neighbors=10,
                           cv_outer_splits=10, cv_inner_splits=5, seed=None):
    '''Compute the minimum resource budget for HalvingRandomSearchCV such that
    SMOTENC never encounters a class with fewer samples than it needs in any inner fold.

    The constraint is that for the rarest class in any outer training fold:

        inner_train_class_count >= max_k_neighbors + 1

    Solving for min_resources (assuming stratified subsampling throughout):

        min_resources >= ceil(
            (max_k_neighbors + 1) * cv_inner_splits / (cv_inner_splits - 1)
            / min_class_freq
        )

    Parameters
    ----------
    X : pd.DataFrame
    y : np.ndarray
    groups : pd.Series
    max_k_neighbors : int
        Upper bound on oversampler__k_neighbors in the search space. Default 10.
    cv_outer_splits : int
    cv_inner_splits : int
    seed : int, optional

    Returns
    -------
    int
        The minimum resource value to pass to HalvingRandomSearchCV.
    '''
    cv_tmp = StratifiedGroupKFold(n_splits=cv_outer_splits, shuffle=True, random_state=seed)
    worst = 0
    for train_idx, _ in cv_tmp.split(X, y, groups):
        y_fold = y[train_idx]
        n_train = len(train_idx)
        _, counts = np.unique(y_fold, return_counts=True)
        min_class_freq = counts.min() / n_train
        required = math.ceil(
            (max_k_neighbors + 1) * cv_inner_splits / (cv_inner_splits - 1) / min_class_freq
        )
        worst = max(worst, required)
    return worst


def _halving_schedule(n_candidates, factor, n_inner_splits):
    '''Return a list of (iteration, n_candidates, n_fits) tuples for logging.'''
    schedule, n, i = [], n_candidates, 0
    while n >= 1:
        schedule.append((i, n, n * n_inner_splits))
        n = n // factor
        i += 1
    return schedule


def get_top_percentile_candidates(search, percentile=90):
    '''Return the parameter dicts of candidates in the final halving iteration
    whose inner-CV score is at or above `percentile`.

    Parameters
    ----------
    search : HalvingRandomSearchCV
        A fitted search object.
    percentile : float
        Score percentile threshold (0-100). Default 90.

    Returns
    -------
    tuple[list[dict], list[float]]
        Parameter dicts and corresponding mean inner-CV scores for the top candidates.
    '''
    results = pd.DataFrame(search.cv_results_)
    last_iter = results['iter'].max()
    last_iter_results = results[results['iter'] == last_iter].copy()
    threshold = np.percentile(last_iter_results['mean_test_score'], percentile)
    top = last_iter_results[last_iter_results['mean_test_score'] >= threshold]
    return top['params'].tolist(), top['mean_test_score'].tolist()


def retrain_top_candidates(top_params, base_pipeline, X_train, y_train, verbose=True):
    '''Clone and refit `base_pipeline` for each parameter dict in `top_params`.

    Parameters
    ----------
    top_params : list[dict]
        List of hyperparameter dicts, e.g. from get_top_percentile_candidates.
    base_pipeline : Pipeline
        Unfitted base pipeline to clone for each candidate.
    X_train : pd.DataFrame
    y_train : np.ndarray
    verbose : bool
        Print progress via tqdm.write. Default True.

    Returns
    -------
    list
        List of fitted pipelines.
    '''
    top_pipelines = []
    for i, params in enumerate(top_params):
        if verbose:
            tqdm.write(f'    Retraining {i + 1}/{len(top_params)}')
        p = clone(base_pipeline)
        p.set_params(**params)
        p.fit(X_train, y_train)
        top_pipelines.append(p)
    return top_pipelines


def run_nested_cv(X, y, groups, base_pipeline, param_distributions,
                  cv_outer_splits=10, cv_inner_splits=5,
                  halving_n_candidates=100, halving_factor=3,
                  halving_resource='n_samples', halving_min_resources=None,
                  max_k_neighbors=10, ensemble_percentile=90,
                  n_jobs=-1, seed=None, verbose=True):
    '''Run nested cross-validation with HalvingRandomSearchCV as the inner search.

    For each outer fold:
      1. Run HalvingRandomSearchCV on the training set.
      2. Select the top `ensemble_percentile`th-percentile candidates from the
         final halving iteration.
      3. Retrain each top candidate on the full outer training fold.
      4. Evaluate the ensemble on the held-out outer test fold.

    Parameters
    ----------
    X : pd.DataFrame
    y : np.ndarray
    groups : pd.Series
        Patient identifiers for StratifiedGroupKFold (prevents data leakage).
    base_pipeline : Pipeline
        Unfitted pipeline as returned by make_pipeline.
    param_distributions : dict
        Search space for HalvingRandomSearchCV.
    cv_outer_splits : int
    cv_inner_splits : int
    halving_n_candidates : int
        Number of candidates in the first halving iteration.
    halving_factor : int
        Reduction factor per halving iteration.
    halving_resource : str
    halving_min_resources : int, optional
        If None, computed automatically via compute_min_resources.
    max_k_neighbors : int
        Only used when halving_min_resources is None; passed to compute_min_resources.
    ensemble_percentile : float
        Percentile threshold for selecting ensemble members (0-100).
    n_jobs : int
    seed : int, optional
    verbose : bool

    Returns
    -------
    list[dict]
        One dict per outer fold with keys:
        split, train_idx, test_idx, top_params, top_inner_cv_scores, outer_balanced_acc.
    '''
    if halving_min_resources is None:
        halving_min_resources = compute_min_resources(
            X, y, groups,
            max_k_neighbors=max_k_neighbors,
            cv_outer_splits=cv_outer_splits,
            cv_inner_splits=cv_inner_splits,
            seed=seed,
        )
        if verbose:
            tqdm.write(f'Computed HALVING_MIN_RESOURCES = {halving_min_resources}')

    cv_outer = StratifiedGroupKFold(n_splits=cv_outer_splits, shuffle=True, random_state=seed)
    cv_inner = StratifiedKFold(n_splits=cv_inner_splits, shuffle=True, random_state=seed)

    run_results = []
    outer_scores = []

    outer_bar = tqdm(
        cv_outer.split(X, y, groups),
        total=cv_outer_splits,
        desc='Outer folds',
        disable=not verbose,
    )

    for split, (train_index, test_index) in enumerate(outer_bar):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if verbose:
            schedule = _halving_schedule(halving_n_candidates, halving_factor, cv_inner_splits)
            total_fits = sum(fits for _, _, fits in schedule)
            schedule_str = '  |  '.join(
                f'iter {i}: {n} candidates ({fits} fits)' for i, n, fits in schedule
            )
            tqdm.write(f'\n--- Outer split {split + 1}/{cv_outer_splits} '
                       f'(n_train={len(train_index)}, n_test={len(test_index)}) ---')
            tqdm.write(f'  Halving schedule: {schedule_str}  |  total: {total_fits} fits')

        search = HalvingRandomSearchCV(
            estimator=clone(base_pipeline),
            param_distributions=param_distributions,
            n_candidates=halving_n_candidates,
            factor=halving_factor,
            resource=halving_resource,
            min_resources=halving_min_resources,
            scoring=balanced_accuracy_scorer,
            cv=cv_inner,
            refit=False,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=0,
        )
        search.fit(X_train, y_train)

        top_params, top_scores = get_top_percentile_candidates(search, percentile=ensemble_percentile)

        if verbose:
            tqdm.write(
                f'  Ensemble size (top {ensemble_percentile}th pct): {len(top_params)} candidates  '
                f'inner CV scores: min={min(top_scores):.4f} max={max(top_scores):.4f}'
            )

        top_pipelines = retrain_top_candidates(top_params, base_pipeline, X_train, y_train, verbose=verbose)

        y_pred = pipeline.ensemble_predict(top_pipelines, X_test)
        test_score = balanced_accuracy_no_warn(y_test, y_pred)

        if verbose:
            tqdm.write(f'  outer_split_balanced_acc: {test_score:.4f}')

        outer_scores.append(test_score)
        run_results.append({
            'split': split,
            'train_idx': train_index.tolist(),
            'test_idx': test_index.tolist(),
            'top_params': top_params,
            'top_inner_cv_scores': top_scores,
            'outer_balanced_acc': test_score,
        })

    if verbose:
        tqdm.write(
            f'\nFinal Nested CV Balanced Accuracy: '
            f'{np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}'
        )

    return run_results


def evaluate_nested_cv(run_results, X, y, base_pipeline, le, sex=True, classifier='xgboost', verbose=True):
    '''Retrain ensembles from nested-CV results and compute predictions, probabilities,
    SHAP values, and preprocessed feature matrices for each outer split.

    Parameters
    ----------
    run_results : list[dict]
        Output of run_nested_cv.
    X : pd.DataFrame
    y : np.ndarray
    base_pipeline : Pipeline
        Unfitted base pipeline (same as used in run_nested_cv).
    le : LabelEncoder
    sex : bool
    classifier : str
        Name of the classifier step in the pipeline (e.g. 'xgboost').
    verbose : bool

    Returns
    -------
    dict
        Keys: trues, preds, probs, shaps, interactions, X_test_processed.
        Each value is a list with one entry per outer fold.
    '''
    trues, preds, probs, shaps, interactions, X_processed_splits = [], [], [], [], [], []

    for result in tqdm(run_results, desc='Outer splits', disable=not verbose):
        train_index = np.array(result['train_idx'])
        test_index  = np.array(result['test_idx'])
        top_params  = result['top_params']

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        top_pipelines = retrain_top_candidates(top_params, base_pipeline, X_train, y_train, verbose=verbose)

        y_pred       = pipeline.ensemble_predict(top_pipelines, X_test)
        y_prob       = pipeline.ensemble_predict_proba(top_pipelines, X_test)
        shap_vals, interact_vals = pipeline.ensemble_shap_values(top_pipelines, X_test, y_test, sex=sex)
        X_test_proc  = pipeline.ensemble_preprocess_X(top_pipelines, X_test, sex=sex)

        trues.append(y_test)
        preds.append(y_pred)
        probs.append(y_prob)
        shaps.append(shap_vals)
        interactions.append(interact_vals)
        X_processed_splits.append(X_test_proc)

    return {
        'trues':           trues,
        'preds':           preds,
        'probs':           probs,
        'shaps':           shaps,
        'interactions':    interactions,
        'X_test_processed': X_processed_splits,
    }
