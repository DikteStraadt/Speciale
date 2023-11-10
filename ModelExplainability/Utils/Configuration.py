import itertools
import json

def get_configurations():

    path = 'config.json'

    with open(path) as f:

        config = json.load(f)

        keys = list(config.keys())
        values = [config[key] if isinstance(config[key], list) else [config[key]] for key in keys]
        combinations = list(itertools.product(*values))

        return [dict(zip(keys, combo)) for combo in combinations]
