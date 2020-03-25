"""Simple tight-binding add-on for GPAWs LCAO module."""

from math import pi

import numpy as np
import scipy.linalg as sla

import ase.units as units

from gpaw.utilities.tools import tri2full
from .tk_lcao import h_and_s
from transport.tk_gpaw import subdiagonalize_atoms
from transport.tools import subdiagonalize, rotate_matrix, order_diagonal, cutcoupling
from transport.coupledhamiltonian import CoupledHamiltonian

class TightBinding(CoupledHamiltonian):
    """Simple class for tight-binding calculations."""

    def __init__(self, atoms, calc):
        """Init with ``Atoms`` and a converged LCAO calculation."""

        # Initialize gpaw calculator
        if calc.wfs.S_qMM is None:
            calc.wfs.set_positions(calc.spos_ac)

        # Store and extract useful attributes from the calculator
        self.atoms = atoms
        self.calc = calc
        self.kd = calc.wfs.kd
        wfs = calc.wfs
        kd = wfs.kd

        # Matrix size
        self.nao = wfs.setups.nao
        # K-point info
        self.gamma = kd.gamma
        if self.gamma:
            self.Nk_c = (1, 1, 1)
        else:
            self.Nk_c = tuple(kd.N_c)

        self.ibzk_kc = kd.ibzk_kc
        self.ibzk_qc = kd.ibzk_qc
        self.bzk_kc = kd.bzk_kc

        # Symmetry
        self.symmetry = kd.symmetry
        if self.symmetry.point_group:
            raise NotImplementedError("Only time-reversal symmetry supported.")

        # Lattice vectors and number of repetitions
        self.R_cN = None
        self.N_c = None

        # Init with default number of real-space cells
        self.set_num_cells()

    def set_num_cells(self, N_c=None):
        """Set number of real-space cells to use.

        Parameters
        ----------
        N_c: tuple or ndarray
            Number of unit cells in each direction of the basis vectors.

        """

        if N_c is None:
            self.N_c = tuple(self.Nk_c)
        else:
            self.N_c = tuple(N_c)

        if np.any(np.asarray(self.Nk_c) < np.asarray(self.N_c)):
            print("WARNING: insufficient k-point sampling.")

        # Lattice vectors
        R_cN = np.indices(self.N_c).reshape(3, -1)
        N_c = np.array(self.N_c)[:, np.newaxis]
        R_cN += N_c // 2
        R_cN %= N_c
        R_cN -= N_c // 2
        self.R_cN = R_cN

    def lattice_vectors(self):
        """Return real-space lattice vectors."""

        return self.R_cN

    def bloch_to_real_space(self, A_qxMM, R_c=None):
        """Transform quantity from Bloch to real-space representation.

        Parameters
        ----------
        A_qxMM: ndarray
            Bloch representation of matrix. May be parallelized over k-points.
        R_cN: ndarray
            Cell vectors for which the real-space matrices will be calculated
            and returned.

        """

        # Include all cells per default
        if R_c is None:
            R_Nc = self.R_cN.transpose()
        else:
            R_Nc = [R_c,]

        # Real-space quantities
        A_NxMM = []
        # Reshape input array
        shape = A_qxMM.shape
        A_qx = A_qxMM.reshape(shape[0], -1)

        # Fourier transform to real-space
        for R_c in R_Nc:
            # Evaluate fourier sum
            phase_q = np.exp(2.j * pi * np.dot(self.ibzk_qc, R_c))
            A_x = np.sum(phase_q[:, np.newaxis] * A_qx, axis=0)
            self.kd.comm.sum(A_x)
            A_xMM = A_x.reshape(shape[1:])

            # Time-reversal symmetry
            if not len(self.ibzk_kc) == len(self.bzk_kc):
                # Broadcast Gamma component
                gamma = np.where(np.sum(np.abs(self.ibzk_kc), axis=1) == 0.0)[0]
                rank, myu = self.kd.get_rank_and_index(0, gamma)
                #
                if self.kd.comm.rank == rank[0]:
                    A0_xMM = A_qxMM[myu[0]]
                else:
                    A0_xMM = np.zeros_like(A_xMM)
                #
                self.kd.comm.broadcast(A0_xMM, rank[0])

                # Add conjugate and subtract double counted Gamma component
                A_xMM += A_xMM.conj() - A0_xMM

            A_xMM /= np.prod(self.Nk_c)

            try:
                assert np.all(np.abs(A_xMM.imag) < 1e-10)
            except AssertionError:
                raise ValueError("MAX Im(A_MM): % .2e" % np.amax(np.abs(A_xMM.imag)))

            A_NxMM.append(A_xMM.real)

        return np.array(A_NxMM)

    def set_h_and_s(self, H_kMM=None, S_kMM=None):
        """Return LCAO Hamiltonian and overlap matrix in real-space."""

        # # Extract Bloch Hamiltonian and overlap matrix
        # H_kMM = []
        # S_kMM = []
        #
        # h = self.calc.hamiltonian
        # wfs = self.calc.wfs
        # kpt_u = wfs.kpt_u
        #
        # for kpt in kpt_u:
        #     H_MM = wfs.eigensolver.calculate_hamiltonian_matrix(h, wfs, kpt)
        #     S_MM = wfs.S_qMM[kpt.q]
        #     #XXX Converting to full matrices here
        #     tri2full(H_MM)
        #     tri2full(S_MM)
        #     H_kMM.append(H_MM * units.Hartree) # convert to eV
        #     S_kMM.append(S_MM)
        #
        # # Convert to arrays
        # H_kMM = np.array(H_kMM)
        # S_kMM = np.array(S_kMM)

        if H_kMM is None:
            H_kMM, S_kMM = h_and_s(self.calc)
            # Convert in Hartree units
            H_kMM *= units.Hartree
            # Align Fermi Level
            H_kMM -= self.calc.get_fermi_level() * S_kMM

        self.H_NMM = self.bloch_to_real_space(H_kMM)
        self.S_NMM = self.bloch_to_real_space(S_kMM)

        # return self.H_NMM, self.S_NMM

    def band_structure(self, path_kc, blochstates=False):
        """Calculate dispersion along a path in the Brillouin zone.

        Parameters
        ----------
        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        blochstates: bool
            Return LCAO expansion coefficients when True (default: False).

        """

        # Real-space matrices
        if not hasattr(self, 'H_NMM'):
            self.set_h_and_s()
        # self.H_NMM, self.S_NMM = self.h_and_s()

        assert self.H_NMM is not None
        assert self.S_NMM is not None

        # Lattice vectors
        R_cN = self.R_cN

        # Lists for eigenvalues and eigenvectors along path
        eps_kn = []
        psi_kn = []

        for k_c in path_kc:
            # Evaluate fourier sum
            phase_N = np.exp(-2.j * pi * np.dot(k_c, R_cN))
            H_MM = np.sum(phase_N[:, np.newaxis, np.newaxis] * self.H_NMM,
                          axis=0)
            S_MM = np.sum(phase_N[:, np.newaxis, np.newaxis] * self.S_NMM,
                          axis=0)

            if blochstates:
                eps_n, c_Mn = sla.eigh(H_MM, S_MM)
                # Sort eigenmodes according to increasing eigenvalues
                c_nM = c_Mn[:, eps_n.argsort()].transpose()
                psi_kn.append(c_nM)
            else:
                eps_n = sla.eigvalsh(H_MM, S_MM)

            # Sort eigenvalues in increasing order
            eps_n.sort()
            eps_kn.append(eps_n)

        # Convert to eV
        eps_kn = np.array(eps_kn) #* units.Hartree

        if blochstates:
            return eps_kn, np.array(psi_kn)

        return eps_kn

# Aliases for compatibility with CoupledHamiltonian

    @property
    def H(self):
        return self.H_NMM[0]
    @property
    def S(self):
        return self.S_NMM[0]
    @H.setter
    def H(self, H):
        self.H_NMM[0] = H
    @S.setter
    def S(self, S):
        self.S_NMM[0] = S

    def apply_rotation(self, c_mm):
        for h_mm, s_mm in zip(self.H_NMM, self.S_NMM):
            h_mm[:] = rotate_matrix(h_mm, c_mm)
            s_mm[:] = rotate_matrix(s_mm, c_mm)

    def take_bfs(self, bfs, apply=False):
        bfs = np.array(bfs)
        nbf = len(bfs)
        M = self.H_NMM.shape[-1]
        N = np.prod(self.N_c)
        # Rotation matrix
        c_mm = np.eye(M).take(bfs,1)
        #
        H_NMM = np.empty((N,nbf,nbf),dtype=self.H_NMM.dtype)
        S_NMM = np.empty((N,nbf,nbf),dtype=self.S_NMM.dtype)
        for R, (h_mm, s_mm) in enumerate(zip(self.H_NMM, self.S_NMM)):
            H_NMM[R] = rotate_matrix(h_mm, c_mm)
            S_NMM[R] = rotate_matrix(s_mm, c_mm)

        if apply:
            self.H_NMM = H_NMM
            self.S_NMM = S_NMM
            return

        return H_NMM, S_NMM, c_mm
