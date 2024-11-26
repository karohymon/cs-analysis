"""This module implements a way to load default and local configs written in json

"""
import os
import sys
import json
import collections.abc


def json_loads(s):
    return json.loads(s)


def json_load(f):
    if isinstance(f, str):
        with open(f, 'r') as f:
            return json_loads(f.read())
    return json_loads(f.read())


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_local_config(path):
    """Takes the default config path and checks if a local config exists.
    """
    local_path = path.replace('.json', '.local.json')
    if os.path.exists(local_path):
        return [local_path]
    else:
        return []


def get_json_config(default=None, optional=None, overwrite=True):
    """Opens a default json file and updates it with
    the contend from optional json files.

    !!!If the same keys are present, the value from the latest file will be used!!!

    Parameters
    ----------
    config : str or file
        Json file with the default configuration.
    optional : list, optional
        A list containing additional path/files with content that
        should be added to the default configuration. If none is given it looks
        or a local config file in the same directory as the default one.
    overwrite : bool, optional
        Whether the default config should be overwritten or updated with the
        local one. (default=True, overwrite)

    """
    if default is not None:
        config = json_load(default)
    else:
        sys.exit('No default config given.')

    if optional is None:
        if hasattr(default, 'name'):
            path = default.name
        else:
            path = default
        optional = get_local_config(path)

    for localfile in optional:
        local_config = json_load(localfile)
        if overwrite:
            config = local_config
        else:
            config = update(config, local_config)

    return config
