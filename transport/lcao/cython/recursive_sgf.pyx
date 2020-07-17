# cython: infer_types=True
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
from scipy import linalg as la
from cython.parallel import prange, parallel
from scipy.linalg.cython_lapack cimport zgetrf, zgetrs, zgetri, zlange, zlacpy
from scipy.linalg.cython_blas cimport zgemm, zaxpy


cdef double conv = 1e-8


cdef inline complex conj(complex a) nogil:
    return a.real - a.imag*1j


cdef inline void mat_add(complex alpha, complex[::1,:] a, complex[::1,:] b) nogil:

    cdef int m = a.shape[0], i, j
    for j in prange(m, schedule='static'):
        for i in range(m):
            b[j,i] = alpha * a[j,i] + b[j,i]


cdef inline void init_vecs(complex[:,:] h_ii, complex[:,:] s_ii,
                           complex[:,:] h_ij, complex[:,:] s_ij,
                           complex[::1,:] v_00, complex[::1,:] v_01,
                           complex[::1,:] v_10, complex[::1,:] v_11,
                           double energy, double eta=1e-5, double bias=0.) nogil:

    cdef int m = h_ii.shape[0], i, j
    cdef complex z
    z = energy - bias + eta * 1.j

    for j in prange(m, schedule='static'):
        for i in range(m):
            v_00[j,i] = z * conj(s_ii[i,j]) - conj(h_ii[i,j])
            v_11[j,i] = v_00[j,i]
            v_01[j,i] = z * conj(s_ij[i,j]) - conj(h_ij[i,j])

    for j in prange(m, schedule='static'):
        for i in range(m):
            v_10[j,i] = z * s_ij[j,i] - h_ij[j,i]


cdef void get_Ginv(complex[::1,:] v_00, complex[::1,:] v_01,
                   complex[::1,:] v_10, complex[::1,:] v_11,
                   complex[::1,:] a, complex[::1,:] b,
                   complex[::1,:] lu, complex[::1,:] v_01_dot_b, int[:] ipiv) nogil:
    # """The inverse of the retarded surface Green function"""

    cdef int m = v_00.shape[0], i, j, info, inc = 1
    cdef int m2 = m*m
    cdef char trans = b'N', norm = b'M', UPLO = b'A'
    cdef complex alpha = 1.+0.j, beta = -1.+0.j, gamma = 0.+0.j

    cdef double delta = conv + 1
    cdef double work[0]

    ######### DEBUG ##########
    # py_v_00 = np.asarray(v_00).copy()
    # py_v_01 = np.asarray(v_01).copy()
    # py_v_10 = np.asarray(v_10).copy()
    # py_v_11 = np.asarray(v_11).copy()
    ###########################

    while delta > conv:
    # for i in range(3):

        ######### DEBUG ##########
        # py_lu, py_piv = la.lu_factor(py_v_11)
        # py_a = la.lu_solve((py_lu, py_piv), py_v_01)
        # py_b = la.lu_solve((py_lu, py_piv), py_v_10)
        # py_v_01_dot_b = np.dot(py_v_01, py_b)
        # py_v_00 -= py_v_01_dot_b
        # py_v_11 -= np.dot(py_v_10, py_a)
        # py_v_11 -= py_v_01_dot_b
        # py_v_01 = -np.dot(py_v_01, py_a)
        # py_v_10 = -np.dot(py_v_10, py_b)
        # py_delta = abs(py_v_01).max()
        ###########################

        zlacpy(&UPLO,&m,&m,&v_11[0,0],&m,&lu[0,0],&m)
        # assert np.allclose(np.asarray(v_11), np.asarray(lu)), 'copy v_11'
        # assert np.allclose(np.asarray(v_11), py_v_11), 'copy v_11'
        zgetrf(&m,&m,&lu[0,0],&m,&ipiv[0],&info)
        # assert np.allclose(np.asarray(lu), py_lu), 'lu'
        zlacpy(&UPLO,&m,&m,&v_01[0,0],&m,&a[0,0],&m)
        # assert np.allclose(np.asarray(a), py_v_01), 'comp a v_01'
        zgetrs(&trans,&m,&m,&lu[0,0],&m,&ipiv[0],&a[0,0],&m,&info);
        # assert np.allclose(np.asarray(a), py_a), 'a'
        zlacpy(&UPLO,&m,&m,&v_10[0,0],&m,&b[0,0],&m)
        zgetrs(&trans,&m,&m,&lu[0,0],&m,&ipiv[0],&b[0,0],&m,&info);
        # assert np.allclose(np.asarray(b), py_b) ,'b'
        zgemm(&trans,&trans,&m,&m,&m,&alpha,&v_01[0,0],&m,&b[0,0],
              &m,&gamma,&v_01_dot_b[0,0],&m)
        # assert np.allclose(np.asarray(v_01_dot_b), py_v_01_dot_b), 'v_01_dot_b'
        # mat_add(beta, v_01_dot_b, v_00)
        zaxpy(&m2,&beta,&v_01_dot_b[0,0],&inc,&v_00[0,0],&inc)
        # assert np.allclose(np.asarray(v_00), py_v_00), 'v_00'
        zgemm(&trans,&trans,&m,&m,&m,&beta,&v_10[0,0],&m,&a[0,0],
              &m,&alpha,&v_11[0,0],&m)
        # assert not np.allclose(np.asarray(v_11), py_v_11), 'v_11'
        # mat_add(beta, v_01_dot_b, v_11)
        zaxpy(&m2,&beta,&v_01_dot_b[0,0],&inc,&v_11[0,0],&inc)
        # assert np.allclose(np.asarray(v_11), py_v_11), 'v_11'
        zlacpy(&UPLO,&m,&m,&v_01[0,0],&m,&lu[0,0],&m)
        zgemm(&trans,&trans,&m,&m,&m,&beta,&lu[0,0],&m,&a[0,0],
              &m,&gamma,&v_01[0,0],&m)
        # assert np.allclose(np.asarray(v_01), py_v_01), 'v_01'
        zlacpy(&UPLO,&m,&m,&v_10[0,0],&m,&lu[0,0],&m)
        zgemm(&trans,&trans,&m,&m,&m,&beta,&lu[0,0],&m,&b[0,0],
              &m,&gamma,&v_10[0,0],&m)
        # assert np.allclose(np.asarray(v_10), py_v_10), 'v_10'
        delta = zlange(&norm,&m,&m,&v_01[0,0],&m,&work[0])

cdef inline void inv(complex[::1,:] a, complex[::1] work, int[:] ipiv) nogil:

    cdef int m = a.shape[0], info, lwork = work.shape[0]

    zgetrf(&m,&m,&a[0,0],&m,&ipiv[0],&info)
    zgetri(&m,&a[0,0],&m,&ipiv[0],&work[0],&lwork,&info)


def get_G(G_kMM,H_kii,S_kii,H_kij,S_kij,energy,eta=1e-5,bias=0.):

    nkpts = H_kii.shape[0]
    m = G_kMM.shape[1]
    n = G_kMM.shape[2]
    lwork = 32000

    # v_xx = z * OP(h_xx) - OP(s_xx)
    # OP :: .T or .H
    v_00 = np.zeros((m,n), complex, order='F')
    v_11 = np.zeros((m,n), complex, order='F')
    v_10 = np.zeros((m,n), complex, order='F')
    v_01 = np.zeros((m,n), complex, order='F')

    # Loop internal vars.
    a = np.zeros((m,n), complex, order='F')
    b = np.zeros((m,n), complex, order='F')
    lu = np.zeros((m,n), complex, order='F')
    v_01_dot_b = np.zeros((m,n), complex, order='F')

    # Lapack internal vars.
    ipiv = np.zeros(m, np.int32)
    work = np.zeros(lwork, complex)

    # Green's function at transverse k-points
    for i in range(nkpts):
        init_vecs(H_kii[i],S_kii[i],H_kij[i],S_kij[i],
                  v_00,v_01,v_10,v_11,energy)
        get_Ginv(v_00,v_01,v_10,v_11,
                 a,b,lu,v_01_dot_b,ipiv)
        inv(v_00,work,ipiv)
        G_kMM[i] = v_00#la.inv(v_00)#, overwrite_a=True, check_finite=False)
