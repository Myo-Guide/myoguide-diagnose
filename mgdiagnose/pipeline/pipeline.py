import gc
import shap
import numbers
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.base import clone
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
from imblearn.pipeline import Pipeline
from imblearn.utils import check_target_type
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTENC, SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling._smote.base import BaseOverSampler

class PandasMinMaxScaler(MinMaxScaler):
    def __init__(self, columns, feature_range=None):
        self.columns = columns
        super().__init__(feature_range = feature_range)

    def fit(self, X, y=None):
        return super().fit(X[self.columns], y)

    def transform(self, X, y=None):
        _X = super().transform(X[self.columns])
        X[self.columns] = _X
        return X

class PandasKNNImputer(KNNImputer):
    def __init__(self, columns, keep_empty_features=False, n_neighbors=5, weights='uniform'):
        self.columns = columns

        super().__init__(
            keep_empty_features = keep_empty_features,
            n_neighbors = n_neighbors,
            weights = weights
        )

    def fit(self, X, y=None):
        return super().fit(X, y)

    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X), columns=self.columns)

class RoundSexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, split_val, range):
        self.split_val = split_val
        self.range = range

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _sex = np.where(_X['patient__sex'] <= self.split_val, self.range[0], self.range[1])
        _X['patient__sex'] = _sex.copy()
        return _X.copy()

class ScoreNoiseTransformer(BaseOverSampler):
    """Base class for the different SMOTE algorithms."""

    _parameter_constraints: dict = {
        **BaseOverSampler._parameter_constraints,
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        column_indexes,
        sampling_strategy="auto",
        random_state=None,
        n_jobs=None,
        sigma=5,
        repetitions=1
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.sigma = sigma
        self.repetitions = repetitions
        self.column_indexes = column_indexes

    def _fit_resample(self, X, y):
        n, m = X.shape
        _X = np.repeat(X, self.repetitions, axis=0)

        noise_shape = (n*(self.repetitions-1), len(self.column_indexes))
        noise = np.random.normal(0, self.sigma, size=noise_shape)

        _X[n:, self.column_indexes] += noise
        np.clip(_X[:, self.column_indexes], 0, 100, out=_X[:, self.column_indexes])

        gc.collect()

        return _X, np.repeat(y, self.repetitions, axis=0) 
    
    def _check_X_y(self, X, y, accept_sparse=None):
        if accept_sparse is None:
            accept_sparse = ["csr", "csc"]
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        y = self._validate_data('no_validation', y, reset=True)
        return X, y, binarize_y

class ScaleMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cols
    ):
        self.cols = cols

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        _X = X.copy()

        _scores = _X[self.cols].to_numpy()
        _scores_result = _scores.copy()

        # Prepare overall patient mean
        _mean = np.nanmean(_scores, axis=1)

        for c in range(_scores.shape[1]):
            _subset = np.delete(_scores, obj=c, axis=1)
            _leave_one_out_mean = np.nanmean(_subset, axis=1)
            _scores_result[:, c] = _scores[:, c] - _leave_one_out_mean[:]

        _X[self.cols] = _scores_result.copy()
        _X['mean'] = _mean.copy()

        return _X.copy()

class ZScoreTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to apply per-sample (row-wise) z-score standardization across specified columns.

    Parameters
    ----------
    cols : List[str]
        List of column names to standardize.
    """

    def __init__(self, cols) -> None:
        """
        Initialize the transformer with the columns to standardize.

        Parameters
        ----------
        cols : List[str]
            Column names on which to perform z-score standardization.
        """
        self.cols = cols

    def fit(self, X: pd.DataFrame, y = None) -> "ZScoreTransformer":
        """
        Fit does nothing for this transformer (no learning required).

        Parameters
        ----------
        X : pandas.DataFrame
            Input data.
        y : optional
            Ignored, present for API consistency.

        Returns
        -------
        self : ZScoreTransformer
            Fitted transformer (self).
        """
        return self

    def transform(self, X: pd.DataFrame, y = None) -> pd.DataFrame:
        """
        Apply row-wise z-score standardization across self.cols.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data containing columns to standardize.
        y : optional
            Ignored, present for API consistency.

        Returns
        -------
        pandas.DataFrame
            Copy of X with specified columns replaced by their z-score values,
            and two additional columns added:
            - 'z_mean': the per-row mean of original values.
            - 'z_std': the per-row standard deviation of original values.
        """
        _X = X.copy()
        # Extract values
        scores = _X[self.cols].to_numpy()
        # Compute per-row mean and standard deviation
        means = np.nanmean(scores, axis=1)
        stds = np.nanstd(scores, axis=1)
        # Avoid division by zero for rows with zero variance
        stds_adj = np.where(stds == 0, 1, stds)
        # Standardize each row
        scores_z = (scores - means[:, None]) / stds_adj[:, None]
        # Assign back to DataFrame
        _X[self.cols] = scores_z
        _X['mean'] = means
        _X['std'] = stds
        return _X



class AugmentSMOTE(SMOTE):
    def __init__(
            self,
            augment_factor = 1,
            random_state=None,
            k_neighbors=5,
            n_jobs=None,
        ):

        self.augment_factor = augment_factor

        super().__init__(
            sampling_strategy=self.resample_augment,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

    def resample_augment(self, y):
        unique, counts = np.unique(y, return_counts=True)
        target_samples = int(counts.max() * self.augment_factor)
        return {u: target_samples for u in unique}

    def fit_resample(self, X, y):
        return super().fit_resample(X, y)

class AugmentSMOTENC(SMOTENC):
    def __init__(
            self,
            categorical_features,
            augment_factor = 1,
            categorical_encoder=None,
            random_state=None,
            k_neighbors=5,
            n_jobs=None,
        ):

        self.augment_factor = augment_factor

        super().__init__(
            categorical_features=categorical_features,
            categorical_encoder=categorical_encoder,
            sampling_strategy=self.resample_augment,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

    def resample_augment(self, y):
        unique, counts = np.unique(y, return_counts=True)
        target_samples = int(counts.max() * self.augment_factor)
        return {u: target_samples for u in unique}

    def fit_resample(self, X, y):
        return super().fit_resample(X, y)

# Backward-compatible alias for models pickled before the typo was fixed
AgumentSMOTENC = AugmentSMOTENC

def make_pipeline(classifier_step, X, config, le, sex:bool=True):
    feature_range=(-100, 100)

    imputer = PandasKNNImputer(columns=X.columns, keep_empty_features=True)
    sex_rounder = RoundSexTransformer(split_val=0, range=feature_range)
    if sex:
        oversampler = AugmentSMOTENC(categorical_features=['patient__sex'], random_state=config['seed'])
    else:
        oversampler = AugmentSMOTE(random_state=config['seed'])

    if classifier_step == 'xgboost': 
        classifier = XGBClassifier(
            objective="multi:softmax", 
            num_class=len(le.classes_), 
            booster="gbtree", 
            seed=config['seed'], 
            verbosity=0
        )
    elif classifier_step == 'svc': classifier = SVC(probability=True)
    elif classifier_step == 'knn': classifier = KNeighborsClassifier()
    elif classifier_step == 'randomforest': classifier = RandomForestClassifier()
    else: raise Exception(f'Unexpected classifier: {classifier_step}')

    steps = []
    steps.append(('imputer', imputer))
    if sex: steps.append(('sex_rounder', sex_rounder))
    steps.append(('oversampler', oversampler))
    steps.append((classifier_step, classifier))
    return Pipeline(steps)

def get_top_percentile_trials(study, percentile=None, return_idx=False):
    completed_trial_values = [trial.value for trial in study.trials if trial.value is not None]
    if not completed_trial_values:
            raise ValueError("There are no completed trials with values.")

    value_threshold = np.percentile(completed_trial_values, percentile)
    qualifying_trials = [trial for trial in study.trials if trial.value is not None and trial.value >= value_threshold]

    return qualifying_trials

def retrain_top_pipelines(top_trials, base_pipeline, X_train, y_train):
    top_pipelines = []

    for i, trial in enumerate(top_trials):
        print(f'{i+1}/{len(top_trials)}')

        # Clone the base pipeline
        cloned_pipeline = clone(base_pipeline)
        # Set hyperparameters from the trial
        cloned_pipeline.set_params(**trial.params)
        # Re-train the pipeline on the full training data
        cloned_pipeline.fit(X_train, y_train)
        # Store the re-trained pipeline
        top_pipelines.append(cloned_pipeline)

    return top_pipelines

def ensemble_predict_proba(top_pipelines, X, use_margins=False):
    if use_margins:
        prob_predictions = ensemble_predict_margins(top_pipelines, X)
    else:
        prob_predictions = np.array([pipeline.predict_proba(X) for pipeline in top_pipelines])
    
    avg_prob_predictions = np.mean(prob_predictions, axis=0)

    if use_margins:
        avg_prob_predictions = softmax(avg_prob_predictions)
    
    return avg_prob_predictions

def ensemble_predict(top_pipelines, X, use_margins=False):
    final_predictions = np.argmax(ensemble_predict_proba(top_pipelines, X, use_margins), axis=1)
    return final_predictions

def ensemble_predict_margins(top_pipelines, X):
    margin_outputs = np.array([pipeline.predict(X, output_margin=True) for pipeline in top_pipelines])
    return margin_outputs

def ensemble_shap_values(top_pipelines, X_test, y_test, sex:bool=True):
    n = len(top_pipelines)
    shap_values_mean = None
    interaction_values_mean = None

    for i, pipeline in enumerate(top_pipelines):
        print(f'######### Shaps {i+1}/{n} #########')
        X_test_processed = X_test.copy()
        X_test_processed = pipeline['imputer'].transform(X_test_processed)
        if sex: X_test_processed = pipeline['sex_rounder'].transform(X_test_processed)

        explainer = shap.TreeExplainer(pipeline['xgboost'])
        shap_values = np.array(explainer.shap_values(X_test_processed, y=y_test)) / n
        interaction_values = np.array(explainer.shap_interaction_values(X_test_processed, y=y_test)) / n

        if shap_values_mean is None:
            shap_values_mean = shap_values
            interaction_values_mean = interaction_values
        else:
            shap_values_mean += shap_values
            interaction_values_mean += interaction_values

    return shap_values_mean, interaction_values_mean

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)