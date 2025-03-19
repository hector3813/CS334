import pandas as pd

def load_data(fname, d=1, weighted=False):
    """
    Loads the data in file specified by `fname`. The file specified should be a csv with n rows and (d+1) columns,
    with the first column being label/output

    Returns X: an nxd array, where n is the number of examples and d is the dimensionality.
            y: an nx1 array, where n is the number of examples
    """
    data = pd.read_csv(fname).values
    y = data[:, 0]
    if weighted:
        X = data[:, 1:-1]
        weights = data[:, -1]
        return X, y, weights
    else:
        X = data[:, 1:d+1]
        return X, y
