'''Card the deployed diagnosis model 20260313181917 from artifacts already on disk.
No retraining. Re-runnable: it regenerates the card from the same inputs.

    python scripts/card_20260313181917.py --bundle <deployed .pkl> --out <card dir>

Inputs: the deployed bundle (its config is authoritative for how the data was
read), the nested-CV eval_results, and the datapull CSV that config names.
eval_results was saved without its outer test indices, so they are regenerated
from the seeded splitter and checked fold by fold against the stored labels.
'''
import argparse
import os
import pickle

import numpy as np
import pandas as pd

import mgdiagnose as mgd
from mgdiagnose.export import card

TAG = '20260313181917'
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, '..', '..'))   # MYO-Guide_diagnose/code

# Nested-CV constants, from notebooks/new/nested_cv.ipynb cell 7 for this run.
SEED, OUTER, INNER = 420, 10, 5

# TODO(confirm before publishing): identity is permanent once in the registry.
IDENTITY = {
    'model_id': 'diagnosis-thigh-xgb-v3',
    'family': 'diagnosis-thigh-xgb',
    'display_name': 'Meryon 1.0',
    'tool': 'diagnosis',
    'version': '3',
    'artifact_revision': 'r1',
    'released_at': '2026-03-13',
    'supersedes': None,
    'citation': 'Verdú-Díaz et al., J Cachexia Sarcopenia Muscle, 2025',
    'doi': None,
}
GOVERNANCE = {
    'intended_use': 'Research use. Decision support for clinicians reading muscle MRI; '
                    'not a diagnostic device.',
    'status': 'published',
    'regulatory': 'Not CE-marked; not FDA-cleared.',
}
DEPLOYMENT = {
    'container_image': 'myoguide-diagnosis-container:1.7',
    'hf_repo': 'Myo-Guide/myoguide-diagnose',
}
EVALUATION_ID = f'eval-{TAG}-nested-cv'

CARD_MD = '''# {display_name} ({model_id})

## Intended use
Research use only. Ranks {n_classes} genetically defined neuromuscular disease groups by
probability from muscle MRI fat-replacement scores, age and sex, to support (never replace)
a clinician's differential diagnosis. Not a diagnostic device.

## Inputs
Per-muscle scores for {n_muscles} pelvis, thigh and lower-leg muscles on any of five scales
(0-4, 1-4, 0-5, 2a/2b, fat fraction), left and right, plus age and sex. Any muscle may be
missing; missing values are imputed from the training set.

## Training data
{n_records} records from {n_patients} patients with a confirmed genetic diagnosis
(datapull {datapull}). Centre is on file for {with_centre} patients, from {centres} centres.

## Evaluation
Nested cross-validation: {outer} outer folds, grouped by patient so no patient is in both
training and test, {inner} inner folds for hyperparameter search. Each figure on the model
page is shown with its n and a 95% interval across outer folds.

**The figures measure the training pipeline, not the deployed ensemble.** The deployed model
is the 40 best hyperparameter candidates from all outer folds refit on every record; no
held-out or external test set exists for it.

## Limitations
- Trained only on genetically confirmed cases; performance on undiagnosed or
  non-neuromuscular patients is not measured.
- Classes with few records have wide intervals; read the n beside every figure.
- Probabilities are an ensemble average and are not calibrated frequencies. In
  cross-validation the top-ranked probability was {calibration}
  (expected calibration error {ece:.2f}); see the calibration figure.
'''


def age_band(age):
    return np.array([None if not np.isfinite(a) else '<18' if a < 18 else '18-39' if a < 40
                     else '40-59' if a < 60 else '60+' for a in age], dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', required=True, help='the deployed .pkl')
    ap.add_argument('--eval-results', default=os.path.join(
        CODE, 'trained_models', TAG, f'{TAG}_eval_results.joblib'))
    ap.add_argument('--data-dir', default=os.path.join(CODE, '..', 'data'))
    ap.add_argument('--out', required=True, help='card directory to write')
    args = ap.parse_args()

    from joblib import load
    with open(args.bundle, 'rb') as f:
        bundle = pickle.load(f)
    # The deployed bundle is in inference mode, which skips the training-only row
    # filters; switch it off to rebuild the training frame.
    config = {k: v for k, v in bundle.config.items() if k != 'inference_mode'}
    csv_path = os.path.join(args.data_dir, f"{config['datapull_date']}.csv")

    df = mgd.read_csv(args.data_dir, dict(config))
    X, y, le, groups = mgd.prepare_data(df, dict(config))
    if list(le.classes_) != list(bundle.le.classes_):
        raise SystemExit('class list differs from the bundle: wrong datapull or config')

    eval_results = load(args.eval_results)
    test_idx = card.regenerate_outer_test_indices(X, y, groups, OUTER, SEED)
    pooled = card.pooled_test_rows(df, y, eval_results, test_idx)

    # Raw datapull values for the same records (process_data keeps the row index).
    raw = pd.read_csv(csv_path, low_memory=False)
    # 'U' is the datapull's "unknown"; the pipeline maps it to missing, and so does the card.
    raw['patient__sex'] = raw['patient__sex'].replace('U', np.nan)
    raw_used = raw.loc[df.index]
    raw_pooled = raw.loc[pooled.index]

    dataset = card.build_dataset_card(
        raw_used, config, csv_path, name=f'MYO-Guide cohort, datapull {config["datapull_date"]}',
        availability='Not publicly available.', site_col='patient__site',
        external_validation=False,
    )
    protocol = {
        'type': 'cross_validation', 'folds': OUTER, 'repeats': 1, 'seed': SEED,
        'inner_folds': INNER, 'split_unit': 'patient',
        'stratified_by': [config['label_col']],
        'model_selection': 'Halving random search (100 candidates, factor 3) per outer fold; '
                           'ensemble of the candidates above the 90th percentile of inner-CV score.',
        'evaluated_artifact': "Per-outer-fold ensembles, each refit on that fold's training "
                              'records and scored on its held-out records.',
        'notes': 'The deployed ensemble is the union of every fold\'s top candidates (40 members) '
                 'refit on all records, so these figures measure the pipeline, not those members.',
    }
    metrics = card.build_metrics(
        eval_results, le, model_id=IDENTITY['model_id'], evaluation_id=EVALUATION_ID,
        protocol=protocol, dataset=dataset,
        subgroups={
            'sex': raw_pooled['patient__sex'].fillna('unknown').to_numpy(),
            'age_band': age_band(raw_pooled['age'].to_numpy(float)),
            'scale': raw_pooled['scale'].to_numpy(),
        },
        caveats=['Cohort is genetically confirmed cases; performance on undiagnosed patients '
                 'is not measured.',
                 'Every figure is shown with its n; figures from few records are imprecise.'],
    )
    manifest = card.build_manifest(
        bundle, IDENTITY, GOVERNANCE, evaluation_ids=[EVALUATION_ID],
        deployment=DEPLOYMENT, provenance={'source_id': TAG},
    )
    bins = [b for b in metrics['curves']['calibration'] if b['n']]
    gap = sum((b['observed'] - b['predicted']) * b['n'] for b in bins) / sum(b['n'] for b in bins)
    card_md = CARD_MD.format(
        calibration=('usually lower than how often it was right, i.e. under-confident' if gap > 0
                     else 'usually higher than how often it was right, i.e. over-confident'),
        ece=metrics['overall']['calibration']['ece']['value'],
        n_classes=len(le.classes_), n_muscles=len(manifest['capability']['features']['muscles']),
        n_records=dataset['n_records'], n_patients=dataset['n_patients'],
        datapull=config['datapull_date'],
        with_centre=dataset['n_patients'] - dataset['patients_without_centre'],
        centres=dataset['centres'], outer=OUTER, inner=INNER, **IDENTITY,
    )
    written = card.save_model_card(args.out, args.bundle, manifest, metrics, card_md)
    o = metrics['overall']
    print(f"{written['identity']['model_id']} {written['identity']['artifact_revision']} "
          f"bundle_sha256={written['artifact']['bundle_sha256'][:12]}…")
    print(f"n={o['n_test']}  top1={o['top1_accuracy']['value']:.3f} {o['top1_accuracy']['ci']}  "
          f"top3={o['top3_accuracy']['value']:.3f}  bal_acc={o['balanced_accuracy']['value']:.3f}  "
          f"ECE={o['calibration']['ece']['value']:.3f}")


if __name__ == '__main__':
    main()
