import numpy as np
from functools import singledispatch
#
from .recursive import multiply, get_diagonal
from transport.greenfunction import RecursiveGF, GreenFunction
from transport.tools import dagger
# NeighborList
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist

def _orbital_current_gf(A_mm, H_mm, index_i, index_j,
                        A_dag_mm, H_dag_mm):
    ni = len(index_i)
    nj = len(index_j)
    J_ij = np.zeros((ni,nj))
    for i,j in np.ndindex(ni,nj):
        ii = index_i[i]
        jj = index_j[j]
        J_ij[i,j] = np.imag(H_dag_mm[jj,ii]*A_mm[ii,jj] -
                            H_mm[ii,jj]*A_dag_mm[jj,ii])
    return J_ij

def _orbital_current_rgf(A_xqmm, H_xqmm, index_i, index_j):
    A_qii, A_qij, A_qji = A_xqmm
    H_qii, H_qij, H_qji = H_xqmm
    sizes = np.cumsum([H_mm.shape[0] for H_mm in H_qii])
    # Indices of atoms in recursive partition
    # np.searchsorted(..,side='right') because
    # if indices_i[0] == sizes[q=0], then atom i
    # goes to next partition q=1
    #
    qi = np.searchsorted(sizes, index_i, side='right')
    qj = np.searchsorted(sizes, index_j, side='right')

    # Check
    assert len(set(qi))==1
    assert len(set(qj))==1
    # If OK, all orbitals belong to same partition
    qi = qi[0]
    qj = qj[0]

    M_q = np.insert(sizes,0,0)[:-1]
    index_i = np.asarray(index_i) - M_q[qi]
    index_j = np.asarray(index_j) - M_q[qj]

    if qi==qj:
        A_mm = A_qii[qi]
        H_mm = H_qii[qi]
        A_dag_mm = A_mm
        H_dag_mm = H_mm
    elif qj==qi+1:
        A_mm = A_qij[qi]
        H_mm = H_qij[qi]
        A_dag_mm = A_qji[qi]
        H_dag_mm = H_qji[qi]
    elif qi==qj+1:
        A_mm = A_qji[qj]
        H_mm = H_qji[qj]
        A_dag_mm = A_qij[qj]
        H_dag_mm = H_qij[qj]

    return _orbital_current_gf(A_mm, H_mm, index_i, index_j,
                               A_dag_mm, H_dag_mm)

def orbital_current(A, H, index_i, index_j):
    if isinstance(A, (list,tuple)):
        return _orbital_current_rgf(A, H, index_i, index_j)
    else:
        return _orbital_current_gf(A, H, index_i, index_j, A, H)

def bond_current(A, H, bfs_ai, nlists):
    J_aa = [[None for _ in nlist] for nlist in nlists]
    for a0, nlist in enumerate(nlists):
        index_i = bfs_ai[a0]
        for j, a1 in enumerate(nlist):
            index_j = bfs_ai[a1]
            J_aa[a0][j] = orbital_current(A, H,
                                          index_i,
                                          index_j).sum()
    return J_aa

def quiver_plot(atoms, J_aa, nlists):

    X = []
    U = []
    W = []
    for a in atoms:
        nlist = nlists[a.index]
        pos = np.tile(a.position, (len(nlist), 1))
        dist = (atoms[nlist].positions - a.position) #\
                #/ atoms.get_distances(a.index, nlist)[:,None]
        X.extend(pos.tolist())
        U.extend(dist.tolist())
        W.extend(J_aa[a.index])

    X = np.asarray(X)
    U = np.asarray(U)
    W = np.asarray(W)
    Wneg = W<0
    X[Wneg] += U[Wneg]/2
    U[Wneg] *= -1
    return X, U, np.abs(W)
