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
        probs = []
        for member in self._ensemble:
            # Scale first (NaN-preserving) then impute
            X_scaled = (X - member['scaler_mean']) / member['scaler_scale']
            X_imp = member['imputer'].transform(X_scaled)
            if member['sex_params']:
                sp = member['sex_params']
                idx = self._feature_names.index('patient__sex')
                X_imp = X_imp.copy()
                X_imp[:, idx] = np.where(
                    X_imp[:, idx] <= sp['split_val'],
                    sp['range'][0],
                    sp['range'][1],
                )
            probs.append(member['classifier'].predict_proba(X_imp))
        return np.mean(probs, axis=0)

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
        return df_proc[[c for c in self._feature_names if c in df_proc.columns]].to_numpy()


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


def reexport_model(old_path: str, new_path: str) -> ModelBundle:
    '''Re-save a bundle that was exported with plain pickle using cloudpickle.

    Use this to migrate an existing bundle so that mgdiagnose is no longer
    required at load time — without retraining.

    Requires mgdiagnose to be installed in the current environment (needed to
    unpickle the old bundle), but the resulting file at ``new_path`` can be
    loaded with ``load_model()`` anywhere, without mgdiagnose.

    Parameters
    ----------
    old_path : str
        Path to the bundle saved with plain pickle.
    new_path : str
        Destination path for the cloudpickle bundle.

    Returns
    -------
    ModelBundle
        The loaded (and re-saved) bundle.
    '''
    with open(old_path, 'rb') as f:
        bundle = pickle.load(f)
    save_model(bundle, new_path)
    return bundle
