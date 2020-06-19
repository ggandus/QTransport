import numpy as np
import multiprocessing as mp

def tonumpyarray(mp_arr, shape, dtype):
    """Convert shared multiprocessing array to numpy array.

    no data copying
    """
    return np.frombuffer(mp_arr, dtype=dtype).reshape(shape)
