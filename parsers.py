import numpy as np


def parse_deltaml_distance(path):
    return {'meta': {'sd': path.split('_')[-3].split('/')[-1], 'k0': path.split('_')[-2]}, 'data': np.loadtxt(path)}
