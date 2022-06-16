import pandas as pd
from collections.abc import MutableMapping


# ref: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
# dep use pd.json_normalize(dict, sep = '.')

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)