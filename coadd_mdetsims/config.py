import yaml

from .defaults import METADETECT_CONFIG


def load_config(config_path):
    """Load a config file and return it.

    Parameters
    ----------
    config_path : str, optional
        The path to the config file.

    Returns
    -------
    sim_config : dict
        A dictionary of the sim config options.
    run_config : dict
        A dictionary of the run config options.
    shear_meas_config : dict
        A dictionary of the shear measurement options.
    swap12 : bool
        If True, swap the role of the 1- and 2-axes in the shear measurement.
    cut_interp : bool
        If True, cut objects with too much interpolation.
    """

    with open(config_path, 'r') as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    swap12 = config.pop('swap12', False)
    cut_interp = config.pop('cut_interp', False)

    run_config = {
        'n_patches_per_job': 200,
        'n_patches': 10_000_000,
        'n_jobs_per_script': 500}
    run_config.update(config.get('run', {}))

    shear_meas_config = config.get('shear_meas', {})
    shear_meas_config.update(METADETECT_CONFIG)

    return config['sim'], run_config, shear_meas_config, swap12, cut_interp
