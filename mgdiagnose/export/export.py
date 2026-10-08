import sys
import inspect
import pickle
import warnings
import importlib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

import mgdiagnose.process.process as _process_file

'''Utilities for exporting a trained ensemble to a self-contained ModelBundle
that can be deployed without the mgdiagnose package.
'''


class ModelBundle:
    '''Self-contained deployment bundle produced by export_model().

    All model artifacts and the preprocessing logic are embedded in a single
    object.  The only runtime dependencies are:
        numpy, pandas, scikit-learn, xgboost

    mgdiagnose is NOT required at load time.

    Attributes
    ----------
    config : dict
        Training config (private keys starting with "_" are stripped).
    le : LabelEncoder
        Fitted label encoder; use ``bundle.classes_`` for the class names.
    '''

    def __init__(self, config, le, ensemble, sex, feature_names, process_source, runtime):
        self.config = config
        self.le = le
        self.runtime = runtime                  # dict of package versions at export time
        self._ensemble = ensemble               # list[dict], each has scaler_mean/scale arrays
        self._sex = sex
        self._feature_names = list(feature_names)
        self._process_source = process_source   # captured source of process.py
        self._process_ns = None                 # lazy namespace

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        '''Preprocess raw data and return class probability estimates.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data in the same format as training data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        '''
        X = self._prepare_X(df)
        probs = [
            member['classifier'].predict_proba(self._member_input(X, member))
            for member in self._ensemble
        ]
        return np.mean(probs, axis=0)

    def shap_values(self, df: pd.DataFrame) -> tuple:
        '''Return TreeSHAP values explaining the ensemble's prediction.

        The contributions are computed on **each member's own transformed
        input** — the same array that member's classifier scores in
        predict_proba (standardised, KNN-imputed, sex-snapped). Explaining the
        un-imputed feature matrix instead attributes the prediction to a point
        the model never scored, which silently decouples the explanation from
        the ranking.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data in the same format as training data.

        Returns
        -------
        shaps : np.ndarray of shape (n_samples, n_features, n_classes)
        base_values : np.ndarray of shape (n_classes,)

        Notes
        -----
        ``base_values[c] + shaps[i, :, c].sum()`` is the ensemble-mean **raw
        margin** (log-odds) for class ``c``, not a probability. It is not
        directly comparable to ``predict_proba``, which averages probabilities
        rather than margins; the two agree on ranking but not on scale.
        '''
        out = self.explain(df, show_shap=True)
        return out['shaps'], out['base_values']

    def explain(self, df: pd.DataFrame, show_shap: bool = True) -> dict:
        '''Predict and (optionally) explain in a single pass over the ensemble.

        Probabilities and SHAP contributions are derived from the same
        per-member arrays, so the explanation cannot drift from the prediction.
        This is also cheaper than calling predict_proba() and shap_values()
        separately, which would transform the input through every member twice.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data in the same format as training data.
        show_shap : bool
            Compute SHAP contributions. Roughly triples the cost.

        Returns
        -------
        dict with keys:
            'probs'         (n_samples, n_classes)
            'feature_names' list[str]
            'X'             (n_samples, n_features) preprocessed, pre-imputation
            'shaps'         (n_samples, n_features, n_classes)  — if show_shap
            'base_values'   (n_classes,)                        — if show_shap
        '''
        X = self._prepare_X(df)
        feature_names = list(self._feature_names)

        dmatrix = None
        if show_shap:
            import xgboost as xgb  # only needed on the SHAP path
            dmatrix = xgb.DMatrix

        probs = []
        contribs_sum = None
        for member in self._ensemble:
            X_member = self._member_input(X, member)
            probs.append(member['classifier'].predict_proba(X_member))
            if show_shap:
                # (n_samples, n_classes, n_features + 1); last column is the base value
                contribs = member['classifier'].get_booster().predict(
                    dmatrix(X_member, feature_names=feature_names),
                    pred_contribs=True,
                )
                contribs_sum = contribs if contribs_sum is None else contribs_sum + contribs

        out = {
            'probs': np.mean(probs, axis=0),
            'feature_names': feature_names,
            'X': X,
        }
        if show_shap:
            contribs_mean = contribs_sum / len(self._ensemble)
            out['shaps'] = np.moveaxis(contribs_mean[:, :, :-1], 1, -1)
            out['base_values'] = contribs_mean[0, :, -1]
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        '''Preprocess raw data and return the most likely class labels.

        Returns
        -------
        np.ndarray of decoded string class labels.
        '''
        return self.le.inverse_transform(
            np.argmax(self.predict_proba(df), axis=1)
        )

    @property
    def classes_(self):
        return self.le.classes_

    def check_runtime(self, strict=False) -> dict:
        '''Compare current package versions against those recorded at export time.

        Parameters
        ----------
        strict : bool
            If True, raise RuntimeError on any version mismatch.
            If False (default), emit warnings instead.

        Returns
        -------
        dict
            Mapping of package name → {'exported': str, 'current': str, 'match': bool}.
        '''
        results = {}
        for pkg, exported_ver in self.runtime.items():
            if pkg == 'python':
                current_ver = '{}.{}.{}'.format(*sys.version_info[:3])
            else:
                try:
                    current_ver = importlib.import_module(pkg).__version__
                except (ImportError, AttributeError):
                    current_ver = 'not installed'
            match = (current_ver == exported_ver)
            results[pkg] = {'exported': exported_ver, 'current': current_ver, 'match': match}
            if not match:
                msg = (
                    f"Version mismatch for '{pkg}': "
                    f"exported with {exported_ver}, current is {current_ver}."
                )
                if strict:
                    raise RuntimeError(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return results

    def runtime_summary(self) -> str:
        '''Return a formatted string of exported package versions.'''
        lines = ['Runtime at export time:']
        for pkg, ver in self.runtime.items():
            lines.append(f'  {pkg:<20} {ver}')
        return '\n'.join(lines)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_process_ns(self):
        if self._process_ns is None:
            ns = {}
            exec(self._process_source, ns)  # noqa: S102
            self._process_ns = ns
        return self._process_ns

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        ns = self._get_process_ns()
        return ns['process_data'](df, self.config)

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df_proc = self._preprocess(df)
        missing = [c for c in self._feature_names if c not in df_proc.columns]
        if missing:
            # Silently dropping columns would shift every downstream feature by
            # one position, misaligning scaler/imputer parameters and any
            # feature-name mapping built by enumerate(self._feature_names).
            raise ValueError(
                f'Preprocessed data is missing {len(missing)} expected feature '
                f'column(s): {missing}'
            )
        return df_proc[list(self._feature_names)].to_numpy()

    def _member_input(self, X: np.ndarray, member: dict) -> np.ndarray:
        '''Transform the preprocessed feature matrix into one member's input space.

        Scale first (NaN-preserving), then impute, then snap sex to its trained
        two-point range. This is the single definition of what an ensemble
        member actually sees; predict_proba() and explain() both go through it
        so a prediction and its explanation can never be computed on different
        arrays.
        '''
        X_member = (X - member['scaler_mean']) / member['scaler_scale']
        X_member = member['imputer'].transform(X_member)
        if member['sex_params']:
            sp = member['sex_params']
            idx = self._feature_names.index('patient__sex')
            X_member = X_member.copy()
            X_member[:, idx] = np.where(
                X_member[:, idx] <= sp['split_val'],
                sp['range'][0],
                sp['range'][1],
            )
        return X_member


# ── Runtime capture ──────────────────────────────────────────────────────────

_CRITICAL_PACKAGES = [
    'numpy',
    'pandas',
    'sklearn',       # scikit-learn exposes __version__ as sklearn.__version__
    'xgboost',
    'imblearn',      # imbalanced-learn
    'scipy',
    'cloudpickle',
]

def _capture_runtime() -> dict:
    '''Capture current Python and package versions.'''
    runtime = {'python': '{}.{}.{}'.format(*sys.version_info[:3])}
    for pkg in _CRITICAL_PACKAGES:
        try:
            runtime[pkg] = importlib.import_module(pkg).__version__
        except (ImportError, AttributeError):
            pass  # omit packages that are not installed
    return runtime


# ── Conversion helpers ────────────────────────────────────────────────────────

def _convert_imputer(mgd_imputer) -> KNNImputer:
    '''Convert a fitted PandasKNNImputer to a standard sklearn KNNImputer.'''
    std = KNNImputer(
        n_neighbors=mgd_imputer.n_neighbors,
        weights=mgd_imputer.weights,
        keep_empty_features=mgd_imputer.keep_empty_features,
    )
    # Copy all fitted attributes; skip PandasKNNImputer-specific ones
    skip = {'columns', 'n_neighbors', 'weights', 'keep_empty_features'}
    for key, val in mgd_imputer.__dict__.items():
        if key not in skip:
            setattr(std, key, val.copy() if hasattr(val, 'copy') else val)
    # Remove feature_names_in_ so the imputer accepts plain numpy arrays
    if hasattr(std, 'feature_names_in_'):
        del std.feature_names_in_
    return std


def _convert_standard_scaler(pandas_std_scaler) -> dict:
    '''Extract the fitted parameters of a PandasStandardScaler as plain numpy arrays.

    Storing raw arrays rather than a sklearn object means the bundle has no
    mgdiagnose dependency and sidesteps sklearn's NaN validation on transform.
    '''
    return {
        'mean': pandas_std_scaler.mean_.copy(),
        'scale': pandas_std_scaler.scale_.copy(),
    }


# ── Public functions ──────────────────────────────────────────────────────────

def export_model(ensemble_pipelines, config, le, feature_names, sex=True) -> ModelBundle:
    '''Build a self-contained ModelBundle from a trained ensemble.

    Parameters
    ----------
    ensemble_pipelines : list
        Fitted mgdiagnose Pipeline objects (e.g. from retrain_top_candidates).
    config : dict
        Training config after process_data() has run, so
        ``config['_fitted_scaler']`` is populated.
    le : LabelEncoder
        Fitted label encoder from prepare_data().
    feature_names : list or pd.Index
        Column names of the feature matrix X (i.e. X.columns from prepare_data).
    sex : bool
        Whether a sex_rounder step is present in the pipelines.

    Returns
    -------
    ModelBundle
    '''
    # ── Inference-only ensemble ───────────────────────────────────────────────
    ensemble_members = []
    for pipe in ensemble_pipelines:
        scaler_params = _convert_standard_scaler(pipe['scaler'])
        std_imputer = _convert_imputer(pipe['imputer'])
        sex_params = None
        if sex:
            sr = pipe['sex_rounder']
            sex_params = {'split_val': sr.split_val, 'range': sr.range}
        _, classifier = pipe.steps[-1]   # classifier is always the last step
        ensemble_members.append({
            'scaler_mean':  scaler_params['mean'],
            'scaler_scale': scaler_params['scale'],
            'imputer':      std_imputer,
            'sex_params':   sex_params,
            'classifier':   classifier,
        })

    # ── Embed process.py source ───────────────────────────────────────────────
    process_source = inspect.getsource(_process_file)

    # ── Capture runtime versions ──────────────────────────────────────────────
    runtime = _capture_runtime()

    # ── Strip private/runtime keys from config ────────────────────────────────
    export_config = {k: v for k, v in config.items() if not k.startswith('_')}

    return ModelBundle(
        config=export_config,
        le=le,
        ensemble=ensemble_members,
        sex=sex,
        feature_names=list(feature_names),
        process_source=process_source,
        runtime=runtime,
    )


def save_model(bundle: ModelBundle, path: str) -> None:
    '''Save a ModelBundle so it can be loaded without mgdiagnose.

    Uses cloudpickle (required in the training environment) to serialise the
    ModelBundle class definition inline.  The resulting file can be loaded
    with standard pickle — no mgdiagnose needed at deployment.

    Parameters
    ----------
    bundle : ModelBundle
    path : str
        Destination file path, e.g. ``'model_v1.pkl'``.
    '''
    try:
        import cloudpickle
    except ImportError as exc:
        raise ImportError(
            'cloudpickle is required for save_model(). '
            'Install it with: pip install cloudpickle'
        ) from exc
    import sys
    # Embed the ModelBundle class definition inline rather than storing a
    # module reference.  Without this, loading the bundle on a machine that
    # does not have mgdiagnose installed raises ModuleNotFoundError.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    with open(path, 'wb') as f:
        cloudpickle.dump(bundle, f)


def load_model(path: str) -> ModelBundle:
    '''Load a ModelBundle saved with save_model().

    No mgdiagnose import is required.

    Parameters
    ----------
    path : str

    Returns
    -------
    ModelBundle
    '''
    with open(path, 'rb') as f:
        return pickle.load(f)


def reexport_model(old_path: str, new_path: str, inference_mode: bool = False,
                   refresh_class: bool = True) -> ModelBundle:
    '''Re-save a bundle, optionally switching it to inference mode.

    Use this to migrate an existing bundle so that mgdiagnose is no longer
    required at load time — without retraining.  When ``inference_mode=True``
    the bundle's config is updated and its embedded process.py source is
    refreshed from the currently installed mgdiagnose package, so that the
    training-only steps (filter_status, select_labels, remove_unscored) are
    skipped during inference.

    Requires mgdiagnose to be installed in the current environment (needed to
    unpickle the old bundle), but the resulting file at ``new_path`` can be
    loaded with ``load_model()`` anywhere, without mgdiagnose.

    Parameters
    ----------
    old_path : str
        Path to the bundle saved with plain pickle.
    new_path : str
        Destination path for the cloudpickle bundle.
    inference_mode : bool
        If True, set ``config['inference_mode'] = True`` in the bundle and
        refresh the embedded process.py source from the current installation.
        Default is False (behaviour identical to the original reexport_model).
    refresh_class : bool
        Rebind the loaded bundle to the ModelBundle class defined in *this*
        module before re-saving, so the new file carries the current methods.
        Without this the old class definition — which cloudpickle embedded by
        value in the source file — is simply round-tripped, and fixes made to
        ModelBundle never reach a re-exported bundle. Default True.

    Returns
    -------
    ModelBundle
        The loaded (and re-saved) bundle.
    '''
    with open(old_path, 'rb') as f:
        bundle = pickle.load(f)

    if refresh_class and type(bundle) is not ModelBundle:
        required = ('config', 'le', 'runtime', '_ensemble', '_sex',
                    '_feature_names', '_process_source')
        missing = [a for a in required if not hasattr(bundle, a)]
        if missing:
            raise TypeError(
                'Cannot rebind this bundle to the current ModelBundle: it is '
                f'missing {missing}. Re-export from the trained ensemble with '
                'export_model() instead, or pass refresh_class=False.'
            )
        member_keys = ('scaler_mean', 'scaler_scale', 'imputer', 'sex_params', 'classifier')
        missing_keys = [k for k in member_keys if k not in bundle._ensemble[0]]
        if missing_keys:
            raise TypeError(
                'Cannot rebind this bundle to the current ModelBundle: ensemble '
                f'members are missing {missing_keys}. Pass refresh_class=False.'
            )
        bundle.__class__ = ModelBundle

    if inference_mode:
        bundle.config['inference_mode'] = True
        bundle._process_source = inspect.getsource(_process_file)
        bundle._process_ns = None   # invalidate the cached exec namespace

    save_model(bundle, new_path)
    return bundle
