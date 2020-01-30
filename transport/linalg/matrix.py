import numpy as np

def get_toeplitz(rows, cols=None):
    # https://stackoverflow.com/questions/36464191/toeplitz-matrix-of-toeplitz-matrix
    '''This function constructs the toepliz matrix from the
    input block rows. If cols is None, the matrix is Hermitian.'''
    if cols is None:
        cols  = rows
    else:
        assert len(cols) == len(rows) - 1
        cols.insert(0, rows[0])
    dtype = rows[0].dtype
    m, n  = rows[0].shape
    n_r   = len(rows)
    mat = np.zeros((n_r, m, n_r, n), dtype=dtype)
    rows_it  = iter(rows)
    cols_it  = iter(cols)
    for j, entry in enumerate(rows_it):
        col_mat = next(cols_it)
        for i in range(n_r-j):
            mat[i,:,i+j,:] = entry
            mat[j+i,:,i,:] = col_mat
    mat.shape = (n_r*m, n_r*n)
    return mat

def test_toepliz():
    ## test
    rows = [np.arange(4).reshape(2,2),
            np.arange(4,8).reshape(2,2),
            np.arange(8,12).reshape(2,2)]
    cols = [np.arange(12,16).reshape(2,2),
            np.arange(16,20).reshape(2,2)]
    mat  = get_toeplitz(np.array(rows), cols)

    assert np.allclose(mat, array([[ 0,  1,  4,  5,  8,  9],
                                   [ 2,  3,  6,  7, 10, 11],
                                   [12, 13,  0,  1,  4,  5],
                                   [14, 15,  2,  3,  6,  7],
                                   [16, 17, 12, 13,  0,  1],
                                   [18, 19, 14, 15,  2,  3]]))
