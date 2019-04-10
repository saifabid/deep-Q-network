import ruamel.yaml as yaml


def get_config(config_file):
    stream = open(config_file, "r")
    # yaml = yaml(typ='safe')  # default, if not specfied, is 'rt' (round-trip)
    config = yaml.safe_load(stream)

    return config
