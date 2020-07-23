import numpy as np
from libc.stdlib cimport abort, malloc, free, rand

import recursive_sgf as rsg

def py_init_vecs(h_ii, s_ii, h_ij, s_ij,
            py_v_00, py_v_01, py_v_10, py_v_11,
            energy, eta=1e-5, bias=0.):

    z = energy - bias + eta * 1.j

    py_v_00[:] = z * s_ii.T.conj() - h_ii.T.conj()
    py_v_11[:] = py_v_00.copy()
    py_v_10[:] = z * s_ij - h_ij
    py_v_01[:] = z * s_ij.T.conj() - h_ij.T.conj()


def test_init_vecs():

    m = 3
    nkpts = 2
    a = np.random.random((nkpts,m,m)).astype(complex)

    cdef complex *v_00
    cdef complex *v_01
    cdef complex *v_10
    cdef complex *v_11

    cdef int sz = m*m*sizeof(double)
    v_00 = <complex*>malloc(sz)
    v_01 = <complex*>malloc(sz)
    v_10 = <complex*>malloc(sz)
    v_11 = <complex*>malloc(sz)

    py_v_00 = np.zeros((m,m), complex)
    py_v_10 = np.zeros((m,m), complex)
    py_v_01 = np.zeros((m,m), complex)
    py_v_11 = np.zeros((m,m), complex)

    args_common = a[0]*4
    args_py = py_v_00, py_v_01, py_v_10, py_v_11
    args_pyx = v_00, v_01, v_10, v_11

    py_init_vecs(*args_common,*args_py,0.)
    rsg.init_vecs(*args_common,*args_py,0.)
