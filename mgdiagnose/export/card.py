'''Model cards for exported bundles, in the mgcard format.

The card lives BESIDE the bundle, never inside it. A bundle cannot carry the hash
of its own file, and an mgcard object pickled into it would be pickled by
reference, making the endpoint depend on mgcard just to unpickle. So export_model()
and save_model() are unchanged; carding is a separate step over a saved file:

    metrics  = build_metrics(eval_results, le, ...)
    manifest = build_manifest(bundle, identity, governance, ...)
    save_model_card(card_dir, bundle_path, manifest, metrics, card_md)

Requires mgcard[metrics] (numpy + pandas only), imported lazily so the rest of
mgdiagnose does not depend on it.
'''
import copy
import os
import pickle
import re
import shutil

import numpy as np

FORMAT = 'myoguide.ModelBundle/1'
DEMOGRAPHICS = ('patient__sex', 'age')


def _mgcard():
    try:
        import mgcard
        import mgcard.dataset
        import mgcard.metrics_classification
    except ImportError as exc:
        raise ImportError('Model cards need mgcard: pip install "mgcard[metrics]"') from exc
    return mgcard


# ── Which rows were scored ──────────────────────────────────────────────────────

def regenerate_outer_test_indices(X, y, groups, n_splits, seed):
    '''The outer test indices run_nested_cv used, for eval_results saved without
    them. Valid only because the outer splitter is seeded; pooled_test_rows()
    proves it against the stored labels before anything is built on it.'''
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [test for _, test in cv.split(X, y, groups)]


def pooled_test_rows(df, y, eval_results, test_indices):
    '''Rows of `df` (the frame prepare_data() was called on) in the order
    eval_results pools them, so per-patient attributes line up with predictions.

    test_indices: [r['test_idx'] for r in run_results], or
    regenerate_outer_test_indices(...) for an older run.'''
    if len(test_indices) != len(eval_results['trues']):
        raise ValueError('one test index array per outer fold expected')
    for i, (idx, trues) in enumerate(zip(test_indices, eval_results['trues'])):
        if len(idx) != len(trues) or not np.array_equal(np.asarray(y)[idx], trues):
            raise ValueError(f'outer fold {i}: test indices do not reproduce the stored labels')
    return df.iloc[np.concatenate(test_indices)]


# ── metrics.json ────────────────────────────────────────────────────────────────

def build_metrics(eval_results, le, *, model_id, evaluation_id, protocol, dataset,
                  subgroups=None, caveats=(), is_primary=True, level=0.95):
    '''metrics.json from the nested-CV predictions. Every record appears in exactly
    one outer test fold, so pooling gives one legitimate confusion matrix and the
    folds give the interval. subgroups: {name: array aligned with the pooled rows}.'''
    mgcard = _mgcard()
    trues = eval_results['trues']
    folds = np.concatenate([np.full(len(t), i) for i, t in enumerate(trues)])
    block = mgcard.metrics_classification.classification_metrics(
        np.concatenate(trues), np.concatenate(eval_results['probs']), le.classes_, folds,
        y_pred=np.concatenate(eval_results['preds']), subgroups=subgroups, level=level,
    )
    return {
        'schema_version': mgcard.SCHEMA_VERSION,
        'evaluation_id': evaluation_id,
        'model_id': model_id,
        'generated_at': mgcard.utc_now(),
        'is_primary': is_primary,
        'task': 'multiclass_classification',
        'protocol': protocol,
        'dataset': dataset,
        'caveats': [*caveats, *([block['confidence']['interval']] if 'interval' in block['confidence'] else [])],
        **block,
    }


def build_dataset_card(rows, config, csv_path, *, name, availability, site_col=None, **extra):
    '''The `dataset` block. `rows` are the records actually used, with their RAW
    datapull values (sex as M/F/U, not the 0/1 the pipeline maps it to).'''
    mgcard = _mgcard()
    return mgcard.dataset.dataset_card(
        rows, patient_col='patient__id', name=name, availability=availability,
        rows='records', sex_col='patient__sex', age_col='age', class_col=config['label_col'],
        site_col=site_col if site_col in rows else None,
        scales_used=sorted(map(str, rows['scale'].dropna().unique())) if 'scale' in rows else None,
        source=os.path.basename(csv_path), source_sha256=mgcard.sha256_file(csv_path),
        **extra,
    )


# ── manifest.json ───────────────────────────────────────────────────────────────

def build_manifest(bundle, identity, governance, *, evaluation_ids=None, no_metrics_reason=None,
                   deployment=None, provenance=None):
    '''manifest.json for a bundle. identity / governance follow the mgcard schema;
    deployment carries container_image / hf_repo / hardware. Everything that can be
    read off the bundle is read off it.'''
    mgcard = _mgcard()
    vocab = mgcard.load_muscles()
    features = list(bundle._feature_names)
    runtime = dict(bundle.runtime)
    manifest = {
        'schema_version': mgcard.SCHEMA_VERSION,
        'identity': identity,
        'artifact': {'format': FORMAT, 'self_contained': True},
        'runtime': {'python': runtime.pop('python'), 'packages': runtime, **(deployment or {})},
        'capability': {
            'task': 'multiclass_classification',
            'classes': [{'code': str(c)} for c in bundle.le.classes_],
            'features': {
                'muscles': [f for f in features if f in vocab],
                'demographics': [f for f in features if f in DEMOGRAPHICS],
                'derived': [f for f in features if f not in vocab and f not in DEMOGRAPHICS],
                'missing_handling': 'Standardised per ensemble member, then KNN-imputed '
                                    'against the training set; any feature may be absent.',
            },
            'scales': list(mgcard.validate.SCALES),
            'explanations': {'shap': hasattr(bundle, 'explain'), 'shap_output_space': 'margin'},
        },
        'evaluation': ({'metrics_file': 'metrics.json', 'evaluation_ids': list(evaluation_ids)}
                       if evaluation_ids else {'metrics_file': None, 'reason': no_metrics_reason}),
        'governance': governance,
    }
    if provenance:
        manifest['provenance'] = provenance
    return manifest


# ── Writing ─────────────────────────────────────────────────────────────────────

def save_model_card(card_dir, bundle_path, manifest, metrics, card_md):
    '''Copy the saved bundle into card_dir/weights/ and write the card around it.
    The bundle file is copied byte for byte: the card describes exactly that file.'''
    mgcard = _mgcard()
    weights = os.path.join(card_dir, 'weights')
    os.makedirs(weights, exist_ok=True)
    shutil.copy2(bundle_path, os.path.join(weights, os.path.basename(bundle_path)))
    return mgcard.write_card(card_dir, manifest, metrics, card_md)


def card_reexport(old_card_dir, new_bundle_path, new_card_dir, check_df):
    '''Card a re-exported bundle (reexport_model) as the next artifact revision of
    the SAME model_id — allowed only if it predicts identically.

    A re-export can change the embedded process.py source and class definition, so
    "same model" is a claim to prove, not assume. Both bundles score `check_df`
    (raw rows, as the endpoint receives them); any difference refuses the revision,
    because a change in predictions is a new model_id. Metrics carry over unchanged:
    identical predictions have identical metrics.'''
    mgcard = _mgcard()
    manifest, metrics, card_md = mgcard.read_card(old_card_dir)
    old_file = next(f['path'] for f in manifest['artifact']['files'] if f['path'].endswith('.pkl'))
    with open(os.path.join(old_card_dir, old_file), 'rb') as f:
        old = pickle.load(f)
    with open(new_bundle_path, 'rb') as f:
        new = pickle.load(f)

    p_old, p_new = old.predict_proba(check_df.copy()), new.predict_proba(check_df.copy())
    if p_old.shape != p_new.shape or not np.array_equal(p_old, p_new):
        diff = np.abs(p_old - p_new).max() if p_old.shape == p_new.shape else 'shape'
        raise ValueError(f'predictions differ (max |Δp| = {diff}): this is a new model_id, '
                         'not a new revision of ' + manifest['identity']['model_id'])

    new_manifest = copy.deepcopy(manifest)
    ident = new_manifest['identity']
    old_rev = ident['artifact_revision']
    ident['artifact_revision'] = 'r' + str(int(re.fullmatch(r'r(\d+)', old_rev).group(1)) + 1)
    runtime = dict(new.runtime)
    new_manifest['runtime'].update(python=runtime.pop('python'), packages=runtime)
    new_manifest['provenance'] = {
        **manifest.get('provenance', {}),
        'reexported_from': {'artifact_revision': old_rev,
                            'bundle_sha256': manifest['artifact']['bundle_sha256']},
        'reexported_at': mgcard.utc_now(),
        'equivalence': {'check': 'identical predict_proba (np.array_equal)',
                        'n_rows': int(len(check_df))},
    }
    return save_model_card(new_card_dir, new_bundle_path, new_manifest, metrics, card_md)
