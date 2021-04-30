import numpy as np

def load_data(file_name):
    '''

    Args:
        file_name:  filename
        num_features: amount of features stored in file.
    '''
    assert file_name.endswith(".csv")

    data = np.genfromtxt(file_name, delimiter=',', skip_header = 1)

    indices = data[:, 0]
    data = data[:, 1:]

    return data
