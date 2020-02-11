import numpy as np
# from transport.tools import tri2full
from gpaw.utilities.tools import tri2full
from ase.units import Hartree

def symm_reduce(bzk_kc):
    '''This function reduces inversion symmetry along 1st dimension.'''
    ibzk_kc = []
    bzk2ibzk_k = []
    for bzk_index, bzk_k in enumerate(bzk_kc):
        try:
            if bzk_k[np.nonzero(bzk_k)[0][0]] > 0:
                ibzk_kc.append(bzk_k)
                bzk2ibzk_k.append(bzk_index)
            else:
                continue
        # zero case
        except IndexError:
            ibzk_kc.append(bzk_k)
            bzk2ibzk_k.append(bzk_index)
    return np.array(ibzk_kc), np.array(bzk2ibzk_k)

def fourier_sum(A_kMM, k_kc, R_c):
    '''This function evaluates fourier sum'''
    shape = A_kMM.shape
    A_kx = A_kMM.reshape(shape[0], -1)
    phase_k  = np.exp(2.j * np.pi * np.dot(k_kc, R_c))
    A_x = np.sum(phase_k[:, None] * A_kx, axis=0)
    A_MM = A_x.reshape(shape[1:])
    return A_MM


def h_and_s(calc):
    """Return LCAO Hamiltonian and overlap matrix in fourier-space."""
    # Extract Bloch Hamiltonian and overlap matrix
    H_kMM = []
    S_kMM = []

    h = calc.hamiltonian
    wfs = calc.wfs
    kpt_u = wfs.kpt_u

    for kpt in kpt_u:
        H_MM = wfs.eigensolver.calculate_hamiltonian_matrix(h, wfs, kpt)
        S_MM = wfs.S_qMM[kpt.q]
        #XXX Converting to full matrices here
        tri2full(H_MM)
        tri2full(S_MM)
        H_kMM.append(H_MM)# * Hartree)
        S_kMM.append(S_MM)

    # Convert to arrays
    H_kMM = np.array(H_kMM)
    S_kMM = np.array(S_kMM)

    return H_kMM, S_kMM
