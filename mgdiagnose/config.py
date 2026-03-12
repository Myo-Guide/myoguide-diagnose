import yaml

def load_config(config_file):
    '''Loads the configuration yaml file. See :ref:`configuration_file` for more details.

    Parameters
    ----------
    config_file
        Path to the configuration yaml file

    Returns
    -------
        dictionary with the configuration parameters.
    '''
    with open(config_file, 'r') as f: config = yaml.safe_load(f)
    return config