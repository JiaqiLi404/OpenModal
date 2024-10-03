import scipy.io as sio


def load_mat(path):
    return sio.loadmat(path)
