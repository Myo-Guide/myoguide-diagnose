import os
import warnings
import numpy as np
import pandas as pd
from joblib import load
from typing import Iterable
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
'''This module provides functions for processing and reshaping the raw data into the desired format
'''

def _check_required_config_keys(config: dict) -> None:
    required = ['datapull_date', 'muscles', 'non_muscle_columns', 'label_col']
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f'Config is missing required keys: {missing}')


def read_csv(data_dir:str, config:dict, process:bool=True) -> pd.DataFrame:
    '''Loads the data from a csv file. Requires the config file to specify the parameters `datapull_date`, `muscles`, `non_muscle_columns` and `label_col`. See :ref:`configuration_file` for more details.

    Parameters
    ----------
    data_dir
        Path to the data directory
    config
        Configuration dictionary
    process
        If True, data is processed according to the operations defined in `data_operations` in the config. By default True.

    Returns
    -------
        DataFrame with the loaded data
    '''
    _check_required_config_keys(config)
    muscles = config['muscles']
    load_cols = [
        *config['non_muscle_columns'],
        config['label_col'],
        *muscles,
        *[f'{m}_l' for m in muscles if not m == "tongue"],
        *[f'{m}_r' for m in muscles if not m == "tongue"],
    ]
    df = pd.read_csv(os.path.join(data_dir, f'{config["datapull_date"]}.csv'), usecols=load_cols)

    if process:
        df = process_data(df, config)
    return df

def read_excel(data_dir:str, config:dict, process:bool=True) -> pd.DataFrame:
    '''Loads the data from an excel file. Requires the config file to specify the parameters `datapull_date`, `muscles`, `non_muscle_columns` and `label_col`. See :ref:`configuration_file` for more details.

    Parameters
    ----------
    data_dir
        Path to the data directory
    config
        Configuration dictionary
    process
        If True, data is processed according to the operations defined in `data_operations` in the config. By default True.

    Returns
    -------
        DataFrame with the loaded data
    '''
    _check_required_config_keys(config)
    muscles = config['muscles']
    load_cols = [
        *config['non_muscle_columns'],
        config['label_col'],
        *muscles,
        *[f'{m}_l' for m in muscles if not m == "tongue"],
        *[f'{m}_r' for m in muscles if not m == "tongue"],
    ]
    df = pd.read_excel(os.path.join(data_dir, f'{config["datapull_date"]}.xlsx'), usecols=load_cols)

    if process:
        df = process_data(df, config)
    return df

def merge(df:pd.DataFrame, input:Iterable[str], output:str) -> pd.DataFrame:
    '''Merges multiple columns into a grouped column by averaging each input column.

    Parameters
    ----------
    df
        Input dataframe
    input
        Iterable with the names of the columns to merge
    output
        Name of the target column

    Returns
    -------
        Dataframe with merged columns
    '''

    if output not in df.columns:
        raise Exception(f'Column {output} not found in dataframe')
    for c in input:
        if c not in df.columns:
            raise Exception(f'Column {c} not found in dataframe')
    
    _df = df.copy()
    for s in ['', '_l', '_r']:
        _input = [f'{c}{s}' for c in input]
        _output = f'{output}{s}'

        input_merge = _df[_input].to_numpy()
        target = _df[_output].to_numpy()
        with warnings.catch_warnings():
            # Warning issue: https://stackoverflow.com/q/29688168/7376038
            warnings.simplefilter("ignore", category=RuntimeWarning)
            merge_mean = np.nanmean(input_merge, axis=1)
        merged = np.where(pd.isna(target), merge_mean, target)

        _df[_output] = merged
        for c in _input: _df = _df.drop(c, axis=1).copy()

    return _df.copy() 

def expand(df:pd.DataFrame, input:str, output:Iterable[str]) -> pd.DataFrame:
    '''Expands a grouped column into multiple columns by copying the value
    into each output column

    Parameters
    ----------
    df
        Input DataFrame
    input
        Grouped column to be expanded
    output
        Iterator of column names where the values should be expanded

    Returns
    -------
        DataFrame with the expanded column
    '''

    if input not in df.columns:
        raise Exception(f'Column {input} not found in dataframe')
    for c in output:
        if c not in df.columns:
            raise Exception(f'Column {c} not found in dataframe')
        
    _df = df.copy()
    for s in ['', '_l', '_r']:
        _output = [f'{c}{s}' for c in output]
        _input = f'{input}{s}'

        expand = df[_input].to_numpy()
        target = df[_output].to_numpy()
        expanded = np.apply_along_axis(lambda c: np.where(pd.isna(c), expand, c), 0, target)

        _df[_output] = expanded
        _df = _df.drop(_input, axis=1).copy()

    return _df.copy()

def _config_operation(df:pd.DataFrame, operation:dict, label_col:str) -> callable:
    '''Applies a single data operation to the input dataframe

    Parameters
    ----------
    df
        Input DataFrame
    operation
        Operation dict

    Returns
    -------
        DataFrame after applying the operation
    '''
    _required_keys = {
        'merge':             ['input', 'output'],
        'expand':            ['input', 'output'],
        'combine_labels':    ['input', 'output'],
        'map_column_values': ['column', 'map'],
    }
    op_type = operation.get('type')
    if op_type not in _required_keys:
        raise ValueError(f'Unknown data operation type: {op_type!r}')
    missing = [k for k in _required_keys[op_type] if k not in operation]
    if missing:
        raise ValueError(f'Operation {op_type!r} is missing required keys: {missing}')

    if op_type == 'merge':
        return merge(df, operation['input'], operation['output'])
    elif op_type == 'expand':
        return expand(df, operation['input'], operation['output'])
    elif op_type == 'combine_labels':
        return combine_labels(df, operation['input'], operation['output'], label_col=label_col)
    elif op_type == 'map_column_values':
        return map_column_values(df, operation['column'], operation['map'])

def _data_operations(df:pd.DataFrame, config:dict) -> pd.DataFrame:
    '''Applies all data operations to the input dataframe

    Parameters
    ----------
    df
        Input DataFrame
    config
        Configuration dict

    Returns
    -------
        DataFrame after applying all the data operations
    '''
    if config['data_operations'] == None: return df
    for op in config['data_operations']: df = _config_operation(df, op, config['label_col'])
    return df

def _leave_one_out_mean(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    _df = df.copy()
    scores = _df[cols].to_numpy()
    result = scores.copy()
    mean = np.nanmean(scores, axis=1)
    for c in range(scores.shape[1]):
        subset = np.delete(scores, obj=c, axis=1)
        result[:, c] = scores[:, c] - np.nanmean(subset, axis=1)
    _df[cols] = result
    _df['mean'] = mean
    return _df

def _zscore(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    _df = df.copy()
    scores = _df[cols].to_numpy()
    means = np.nanmean(scores, axis=1)
    stds = np.nanstd(scores, axis=1)
    stds_safe = np.where(stds == 0, 1, stds)
    _df[cols] = (scores - means[:, None]) / stds_safe[:, None]
    _df['mean'] = means
    _df['std'] = stds
    return _df

def _apply_minmax(df: pd.DataFrame, scaler: MinMaxScaler, columns: list) -> pd.DataFrame:
    _df = df.copy()
    X = _df[columns].to_numpy().astype(float)
    # Apply directly via fitted parameters to preserve NaN values
    # (avoids sklearn's NaN validation check)
    lo, hi = scaler.feature_range
    X_scaled = X * scaler.scale_ + scaler.min_
    _df[columns] = X_scaled
    return _df

def process_data(df:pd.DataFrame, config:dict) -> pd.DataFrame:
    '''Apply all preprocessing steps specified in the config file

    Parameters
    ----------
    df
        Input DataFrame
    config
        Config dict

    Returns
    -------
        Processed DataFrame
    '''
    df = _data_operations(df, config)
    config['_muscle_columns_processed'] = list(filter(lambda c: c in config['muscles'], list(df.columns)))

    if config['scale_scores'] is not None:
        # Rescale all muscle scores to a common range before asymmetry is computed,
        # so that quantitative (0-100) and semiquantitative (0-4) scores produce
        # comparable asymmetry values. bilateral_to_mean has not run yet at this
        # point, so _l/_r columns are always present.
        df = scale_scores(
            df, scale_col='scale',
            cols=config['_muscle_columns_processed'],
            t_min=config['scale_scores'][0], t_max=config['scale_scores'][1],
            bilateral_scores=True,
        )

    if config['asymmetry']: df = asymmetry(df, cols=config['_muscle_columns_processed'])
    if config['bilateral_to_mean']: df = bilateral_to_mean(df, cols=config['_muscle_columns_processed'])

    if config['remove_unscored']:
        if isinstance(config['remove_unscored'], float):
            df = remove_unscored(df, cols=config['_muscle_columns_processed'], thresh=config['remove_unscored'])
        elif isinstance(config['remove_unscored'], bool):
            df = remove_unscored(df, cols=config['_muscle_columns_processed'])
        else:
            raise Exception('Unexpected type')

    if config['filter_status']: df = filter_status(df, include=config['filter_status'])

    if config['target_diseases']: df = select_labels(df, target_col=config['label_col'], labels=config['target_diseases'])

    if config['scale_mean'] == "leave-one-out":
        df = _leave_one_out_mean(df, config['_muscle_columns_processed'])
    elif config['scale_mean'] == "z-score":
        df = _zscore(df, config['_muscle_columns_processed'])

    if config['scale_min_max']:
        if config.get('scale_min_max_path', False):
            _scaler_data = load(config['scale_min_max_path'])
            # Support both the new (scaler, columns) tuple format and legacy objects
            if isinstance(_scaler_data, tuple):
                _scaler, _cols = _scaler_data
            else:
                _scaler = _scaler_data
                _cols = list(getattr(_scaler_data, 'columns',
                                     getattr(_scaler_data, 'feature_names_in_', None)))
        else:
            feature_range = (-100, 100)
            _cols = [c for c in df.columns
                     if c not in config.get('non_train_cols', [])
                     and c != config.get('label_col')]
            _scaler = MinMaxScaler(feature_range=feature_range)
            _scaler.fit(df[_cols].to_numpy().astype(float))
        df = _apply_minmax(df, _scaler, _cols)
        config['_fitted_scaler'] = (_scaler, _cols)

    return df

def prepare_data(df:pd.DataFrame, config:dict) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder, np.ndarray]:
    '''Selects the trainable columns and separates the features and labels. Labels are encoded using scikit-learn LabelEncoder.

    Parameters
    ----------
    df
        Input DataFrame
    config
        Configuration dictionary

    Returns
    -------
        Data features, data labels and LabelEncoder
    '''
    X = df.loc[:, ~df.columns.isin([*config['non_train_cols'], config['label_col']])]
    y = df[config['label_col']].astype('category')
    groups = df['patient__id'].astype('category')
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le, groups

def asymmetry(df:pd.DataFrame, cols:Iterable[str]=None, exclude:Iterable[str]=None) -> pd.DataFrame:
    '''Calculate the mean and std muscle asymmetry for each patient and add them as columns.

    Parameters
    ----------
    df
        Input DataFrame
    cols, optional
        Columns to use as to calculate the asymmetry. If provided, `excluded` must be none. By debault None
    exclude, optional
        Columns to exclude from the asymmetry calculation. If provided, `cols` must be none. By debault None

    Returns
    -------
        Dataframe with extra asymmetry columns.
    '''

    _cols = _validate_cols_exclude(df, cols, exclude)
    _df = df.copy()

    cols_l = [f'{c}_l' for c in _cols if not c == "tongue"]
    cols_r = [f'{c}_r' for c in _cols if not c == "tongue"]

    m_l = _df[cols_l].to_numpy()
    m_r = _df[cols_r].to_numpy()

    diff = np.abs(np.subtract(m_l, m_r))
    # Hide `RuntimeWarning`
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(diff, axis=1)
        std = np.nanstd(diff, axis=1)


    _df['asymm_mean'] = mean
    _df['asymm_std'] = std

    return _df.copy()

def bilateral_to_mean(df:pd.DataFrame, cols:Iterable[str]=None, exclude:Iterable[str]=None) -> pd.DataFrame:
    '''Joins the left-right scores into a mean score

    Parameters
    ----------
    df
        Input DataFrame
    cols, optional
        Columns to use. If provided, `excluded` must be none. By debault None
    exclude, optional
        Columns to exclude from the calculation. If provided, `cols` must be none. By debault None

    Returns
    -------
        DataFrame with left-right mean scores
    '''

    _cols = _validate_cols_exclude(df, cols, exclude)
    _df = df.copy()

    for c in _cols:
        if not c == "tongue":
            stereo = _df[[f'{c}_l', f'{c}_r']].to_numpy()
            mono = _df[c].to_numpy()
            with warnings.catch_warnings():
                # Warning issue: https://stackoverflow.com/q/29688168/7376038
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stereo_mean = np.nanmean(stereo, axis=1)
            mono_joined = np.where(pd.isna(mono), stereo_mean, mono)

            _df[c] = mono_joined
            _df = _df.drop(f'{c}_l', axis=1).copy()
            _df = _df.drop(f'{c}_r', axis=1).copy()

    return _df.copy()

def remove_unscored(df:pd.DataFrame, cols:Iterable[str]=None, exclude:Iterable[str]=None, thresh:float=None) -> pd.DataFrame:
    '''Remove rows where all the specified columns are nan

    Parameters
    ----------
    df
        Input DataFrame
    cols, optional
        Columns to use. If provided, `excluded` must be none. By debault None
    exclude, optional
        Columns to exclude. If provided, `cols` must be none. By debault None
    thresh, optional
        Threshold of missing values to remove a row (between 0. and 1.).
        If None, only rows where all values are missing will be removed.
        By default None.

    Returns
    -------
        DataFrame with selected rows removed
    '''

    _cols = _validate_cols_exclude(df, cols, exclude)
    _df = df.copy()

    for c in _cols:
        if c not in df.columns:
            raise Exception(f'Column {c} not found in dataframe')

    if thresh is None:
        _df = _df.dropna(axis=0, how='all', subset=_cols)
    else:
        _thresh = round(len(_cols) * thresh)
        _df = _df.dropna(axis=0, subset=_cols, thresh=_thresh)

    return _df.copy()

def _validate_cols_exclude(df:pd.DataFrame, cols:Iterable[str]=None, exclude:Iterable[str]=None) -> Iterable[str]:
    '''Validates the `cols` and `exclude` parameters.

    Parameters
    ----------
    df
        Input DataFrame
    cols, optional
        Columns to use as to calculate the asymmetry. If provided, `excluded` must be none. By debault None
    exclude, optional
        Columns to exclude from the asymmetry calculation. If provided, `cols` must be none. By debault None

    Returns
    -------
        List of validated selected columns
    '''
    if cols is not None and exclude is not None:
        raise Exception('The parameters `cols` and `exclude` are exclusive, please choose only one')
    if cols is None and exclude is None:
        raise Exception('Please provide either `cols` or `exclude`')

    _df = df.copy()

    if exclude is None: _cols = cols.copy()
    else:
        _cols = list(filter(lambda c: c not in exclude, list(_df.columns)))

    for c in _cols:
        if c not in df.columns:
            raise Exception(f'Column {c} not found in dataframe')

    return _cols.copy()

def map_column_values(df:pd.DataFrame, column:str, map:dict) -> pd.DataFrame:
    '''Maps the values of a DataFrame column using a dictionary

    Parameters
    ----------
    df
        Input DataFrame
    column
        Column name to map values
    map
        Dictionary defining how the values should be mapped

    Returns
    -------
        DataFrame after processing
    '''
    df[column] = df[column].map(map)
    return df

def combine_labels(df:pd.DataFrame, input:Iterable[str], output:str, label_col:str) -> pd.DataFrame:
    '''Combines labels into a common label

    Parameters
    ----------
    df
        Input DataFrame
    input
        Iterable of labels to be combined
    output
        Name of the common label to create
    label_col
        Name of the label column

    Returns
    -------
        DataFrame with combined labels
    '''

    _df = df.copy()
    _df[label_col] = _df[label_col].map({l: output for l in input}).fillna(_df[label_col]).copy()
    return _df.copy()

def scale_scores(df:pd.DataFrame, scale_col:str, t_min, t_max, cols:Iterable[str]=None, exclude:Iterable[str]=None, bilateral_scores:bool = False) -> pd.DataFrame:
    _cols = _validate_cols_exclude(df, cols, exclude)
    _df = df.copy()

    if bilateral_scores:
        _cols_copy = _cols.copy()
        _cols = []
        for c in _cols_copy:
            if not c == "tongue":
                _cols.append(f'{c}_l')
                _cols.append(f'{c}_r')
            else:
                _cols.append(c)

    cols_scale = _df[_cols].to_numpy()
    scale = _df[scale_col].to_numpy()

    '''scaled = (((float(value) - o_min) * (t_max - t_min)) / (o_max - o_min)) + t_min

        value: value to scale
        o_min: min value of the original scale
        o_max: max value of the original scale
        t_min: min value of the target scale
        t_max: max value of the target scale

        scaled = (((float(value) - o_min) * (t_max - t_min)) / (o_max - o_min)) + t_min
    '''
    
    def _scale_omin(x):
        # returns o_min
        if x == '0-4':
            return 0
        elif x == '0-3':
            return 0
        elif x == '1-4':
            return 1
        elif x == '0-5':
            return 0
        elif x == '2a2b':
            return 0
        elif x == 'FF':
            return 0
        
    def _scale_denominator(x):
        # returns o_max - o_min
        if x == '0-4':
            return 4
        elif x == '0-3':
            return 3
        elif x == '1-4':
            return 3
        elif x == '0-5':
            return 5
        elif x == '2a2b':
            return 4
        elif x == 'FF':
            return 100

    scale_omin = np.array(list(map(_scale_omin, scale)))
    scale_denominator = np.array(list(map(_scale_denominator, scale)))

    cols_scale = cols_scale - scale_omin[:, None]
    cols_scale = np.multiply(cols_scale, (t_max - t_min))
    cols_scale = np.divide(cols_scale, scale_denominator[:, None])
    cols_scale = cols_scale + t_min
    
    _df[_cols] = cols_scale.copy()

    return _df.copy()

def select_labels(df:pd.DataFrame, target_col:str, labels:Iterable[str]) -> pd.DataFrame:
    _df = df.copy()

    found_targets = list(_df[target_col].unique())
    for l in labels:
        if l not in found_targets:
            raise Exception(f'Label `{l}` not found in DataFrame')

    _df = _df.loc[_df[target_col].isin(labels)].copy()
    return _df.copy()

def scale_median(df, cols) -> pd.DataFrame:
    _df = df.copy()

    _scores = _df[cols].to_numpy()
    _scores_result = _scores.copy()
    _mean = np.nanmean(_scores, axis=1)

    for c in range(_scores.shape[1]):
        _subset = np.delete(_scores, obj=c, axis=1)
        _leave_one_out_mean = np.nanmean(_subset, axis=1)
        _scores_result[:, c] = _scores[:, c] - _leave_one_out_mean[:]

    _df[cols] = _scores_result.copy()
    _df['mean'] = _mean.copy()

    return _df.copy()

def filter_status(df, include=[]) -> pd.DataFrame:
    _df = df.loc[df['score_status'].isin(include)].copy()
    return _df.copy()