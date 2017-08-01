import numpy as np


def parse_deltaml_distance(path):
    meta = path.split('/')[-1].split('_')
    return {'meta': {'sd': meta[0], 'k0': meta[1], 'q': meta[2]},
            'data': np.loadtxt(path)}


def selective_parser(path, mws=(), r=(), tol=0.2):
    """
    Allows you to create a subset based on distances and magnitudes
    :param path: str: whole path to file (inc' file name)
    :param mws: tup: moment magnitudes to subset by (ints/floats)
    :param r: tup: distances to subset by ()
    :param tol: float: decimal percentage of tolerance around selected distance
    :return:
    """
    data = np.loadtxt(path)
    start = np.zeros((1, 3))
    if len(r) == 0:
        for mw in mws:
            try:
                start = np.vstack((start, data[np.where((data[:, 0] == mw))]))
            except IndexError:
                print('problem with: {}'.format(path.split('/')[-1]))
                pass
    elif len(mws) == 0:
        for d in r:
            try:
                start = np.vstack((start, data[np.where(np.abs((data[:, 1] - d)/d) <= tol)]))
            except IndexError:
                print('problem with: {}'.format(path.split('/')[-1]))
                pass
    else:
        for d in r:
            for mw in mws:
                try:
                    start = np.vstack((start, data[np.where((data[:, 0] == mw) & (np.abs((data[:, 1] - d)/d) <= tol))]))
                except IndexError:
                    print('problem with: {}'.format(path.split('/')[-1]))
                    pass
    data = start[1:][start[1:][:, 0].argsort()]
    meta = path.split('/')[-1].split('_')

    return {'meta': {'sd': meta[0], 'k0': meta[1], 'q': meta[2]},
            'data': data}


def parse_for_pandas(path):
    meta = path.split('/')[-1].split('_')
    data = np.loadtxt(path)
    return {'distance': data[:, 1], 'mw': data[:, 0], 'deltaML': data[:, 2],
            'k0': np.array([float(meta[1]) for _ in data]), 'sd': np.array([float(meta[0]) for _ in data]),
            'q': np.array([float(meta[2]) for _ in data])}


def parse_ampl(path):
    meta = path.split('/')[-1].split('.')[0].split('_')[-1]
    data = np.loadtxt(path, skiprows=1)
    return {'f0': meta, 'data': data}
