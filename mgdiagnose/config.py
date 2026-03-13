import yaml


# Keys that must be present in every config file
_REQUIRED_KEYS = [
    'datapull_date',
    'muscles',
    'non_muscle_columns',
    'label_col',
    'data_operations',
    'scale_scores',
    'asymmetry',
    'bilateral_to_mean',
    'remove_unscored',
    'filter_status',
    'target_diseases',
    'scale_mean',
    'scale_min_max',
]

_VALID_SCALE_MEAN = {'leave-one-out', 'z-score', 'none'}
_VALID_OPERATION_TYPES = {'merge', 'expand', 'combine_labels', 'map_column_values'}
# Keys required by each data operation type
_OPERATION_REQUIRED_KEYS = {
    'merge':             ['input', 'output'],
    'expand':            ['input', 'output'],
    'combine_labels':    ['input', 'output'],
    'map_column_values': ['column', 'map'],
}


def _err(msg):
    raise ValueError(f'[config] {msg}')


def _validate_list_of_str(value, key):
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        _err(f'`{key}` must be a list of strings, got: {value!r}')


def _validate_data_operations(ops):
    if ops is None:
        return
    if not isinstance(ops, list):
        _err('`data_operations` must be a list or null')
    for i, op in enumerate(ops):
        if not isinstance(op, dict):
            _err(f'`data_operations[{i}]` must be a dict, got: {op!r}')
        op_type = op.get('type')
        if op_type not in _VALID_OPERATION_TYPES:
            _err(
                f'`data_operations[{i}].type` must be one of '
                f'{sorted(_VALID_OPERATION_TYPES)}, got: {op_type!r}'
            )
        missing = [k for k in _OPERATION_REQUIRED_KEYS[op_type] if k not in op]
        if missing:
            _err(f'`data_operations[{i}]` (type={op_type!r}) is missing keys: {missing}')

        if op_type == 'merge':
            if not isinstance(op['input'], list):
                _err(f'`data_operations[{i}].input` must be a list for type "merge"')
            if not isinstance(op['output'], str):
                _err(f'`data_operations[{i}].output` must be a string for type "merge"')
        elif op_type == 'expand':
            if not isinstance(op['input'], str):
                _err(f'`data_operations[{i}].input` must be a string for type "expand"')
            if not isinstance(op['output'], list):
                _err(f'`data_operations[{i}].output` must be a list for type "expand"')
        elif op_type == 'combine_labels':
            if not isinstance(op['input'], list):
                _err(f'`data_operations[{i}].input` must be a list for type "combine_labels"')
            if not isinstance(op['output'], str):
                _err(f'`data_operations[{i}].output` must be a string for type "combine_labels"')
        elif op_type == 'map_column_values':
            if not isinstance(op['column'], str):
                _err(f'`data_operations[{i}].column` must be a string for type "map_column_values"')
            if not isinstance(op['map'], dict):
                _err(f'`data_operations[{i}].map` must be a dict for type "map_column_values"')


def validate_config(config: dict) -> None:
    '''Validates the configuration dictionary. Raises ``ValueError`` for any
    missing required key or unexpected value type / choice.

    Parameters
    ----------
    config
        Configuration dictionary as returned by :func:`load_config`.
    '''
    # --- required keys present -----------------------------------------------
    missing = [k for k in _REQUIRED_KEYS if k not in config]
    if missing:
        _err(f'Config is missing required keys: {missing}')

    # --- muscles --------------------------------------------------------------
    _validate_list_of_str(config['muscles'], 'muscles')

    # --- non_muscle_columns ---------------------------------------------------
    _validate_list_of_str(config['non_muscle_columns'], 'non_muscle_columns')

    # --- label_col ------------------------------------------------------------
    if not isinstance(config['label_col'], str):
        _err(f'`label_col` must be a string, got: {config["label_col"]!r}')

    # --- non_train_cols (optional) -------------------------------------------
    if 'non_train_cols' in config and config['non_train_cols'] is not None:
        _validate_list_of_str(config['non_train_cols'], 'non_train_cols')

    # --- data_operations ------------------------------------------------------
    _validate_data_operations(config['data_operations'])

    # --- scale_scores ---------------------------------------------------------
    ss = config['scale_scores']
    if ss is not None:
        if (
            not isinstance(ss, (list, tuple))
            or len(ss) != 2
            or not all(isinstance(v, (int, float)) for v in ss)
        ):
            _err('`scale_scores` must be null or a 2-element numeric list [t_min, t_max]')
        if ss[0] >= ss[1]:
            _err(f'`scale_scores` t_min must be strictly less than t_max, got: {list(ss)}')

    # --- asymmetry ------------------------------------------------------------
    if not isinstance(config['asymmetry'], bool):
        _err(f'`asymmetry` must be a boolean, got: {config["asymmetry"]!r}')

    # --- bilateral_to_mean ----------------------------------------------------
    if not isinstance(config['bilateral_to_mean'], bool):
        _err(f'`bilateral_to_mean` must be a boolean, got: {config["bilateral_to_mean"]!r}')

    # --- remove_unscored ------------------------------------------------------
    ru = config['remove_unscored']
    if ru is not None:
        if isinstance(ru, float):
            if not (0.0 < ru <= 1.0):
                _err(f'`remove_unscored` float must be in (0, 1], got: {ru!r}')
        elif not isinstance(ru, bool):
            _err(f'`remove_unscored` must be a boolean, a float in (0, 1], or null, got: {ru!r}')

    # --- filter_status --------------------------------------------------------
    fs = config['filter_status']
    if fs:
        _validate_list_of_str(fs, 'filter_status')

    # --- target_diseases ------------------------------------------------------
    td = config['target_diseases']
    if td:
        _validate_list_of_str(td, 'target_diseases')

    # --- scale_mean -----------------------------------------------------------
    sm = config['scale_mean']
    if sm is not None and sm not in _VALID_SCALE_MEAN:
        _err(
            f'`scale_mean` must be null or one of {sorted(_VALID_SCALE_MEAN)}, '
            f'got: {sm!r}'
        )

    # --- scale_min_max --------------------------------------------------------
    if not isinstance(config['scale_min_max'], bool):
        _err(f'`scale_min_max` must be a boolean, got: {config["scale_min_max"]!r}')

    # --- scale_min_max_path (optional) ----------------------------------------
    if config.get('scale_min_max_path') is not None:
        if not isinstance(config['scale_min_max_path'], str):
            _err(
                f'`scale_min_max_path` must be a string, '
                f'got: {config["scale_min_max_path"]!r}'
            )


def load_config(config_file):
    '''Loads and validates the configuration yaml file.
    See :ref:`configuration_file` for more details.

    Parameters
    ----------
    config_file
        Path to the configuration yaml file

    Returns
    -------
        Dictionary with the configuration parameters.

    Raises
    ------
    ValueError
        If any config parameter is missing, has an unexpected type, or an
        invalid value.
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config
