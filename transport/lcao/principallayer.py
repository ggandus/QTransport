import numpy as np
from scipy import linalg as la
from collections import namedtuple
# from gpaw.symmetry import Symmetry
from ase.dft.kpoints import monkhorst_pack
from .tklcao import *
from transport.tkgpaw import get_bf_centers
from transport.tools import rotate_matrix, dagger
from transport.selfenergy import LeadSelfEnergy
from transport.block import get_toeplitz

class PrincipalLayer:

    def __init__(self, calc, direction='x'):
        self.calc = calc
        self.direction = direction

        kd = self.calc.wfs.kd
        self.ibzk_kc = kd.ibzk_kc
        self.Nk_c = kd.N_c
        self.offset_c = kd.offset_c

        self.update(direction)


    def get_directions(self):
        # Define transport and transverse directions
        p_dir = 'xyz'.index(self.direction)
        t_dirs = np.delete([0, 1, 2], p_dir)
        return p_dir, t_dirs

    def initialize(self, H_kMM, S_kMM, direction='x'):

        # self.H_kMM, self.S_kMM = h_and_s(self.calc)

        # Transport direction
        self.update(direction)
        p_dir, t_dirs = self.get_directions()

        R_c = [0,0,0]
        self.H_kii = self.bloch_to_real_space_p(H_kMM, R_c)
        self.S_kii = self.bloch_to_real_space_p(S_kMM, R_c)

        R_c[p_dir] = 1
        self.H_kij = self.bloch_to_real_space_p(H_kMM, R_c)
        self.S_kij = self.bloch_to_real_space_p(S_kMM, R_c)

    @property
    def H_kji(self):
        return self.H_kij.swapaxes(1,2).conj()
    @property
    def S_kji(self):
        return self.S_kij.swapaxes(1,2).conj()

    def set_num_cells(self):

        t_dirs = self.get_directions()[1]

        # Lattice vectors
        R_cN = np.indices(self.Nk_c[t_dirs]).reshape(2, -1)
        N_c = np.array(self.Nk_c[t_dirs])[:, np.newaxis]
        R_cN += N_c // 2
        R_cN %= N_c
        R_cN -= N_c // 2
        self.R_cN = R_cN

        # Informations to map transverse directions in real-space matrix
        matrix = namedtuple('matrix',['index_rows','index_cols'])

        # Set indices of rows and columns in real-space matrix
        matrix.index_rows = symm_reduce(self.R_cN.T)[1]
        indices = np.arange(np.prod(self.Nk_c[t_dirs]))
        matrix.index_cols = np.setdiff1d(indices, matrix.index_rows)[::-1]

        # Store matrix informations
        self.matrix = matrix

    def update(self, direction='x'):

        # Define transport and transverse directions
        p_dir, t_dirs = self.get_directions()

        # K-points in the transport direction
        offset_p_c = np.zeros((3,))
        offset_p_c[p_dir] = self.offset_c[p_dir]
        bzk_p_kc = monkhorst_pack((self.Nk_c[p_dir], 1, 1)) + offset_p_c

        # K-points in the transverse directions
        offset_t_c = np.zeros((3,))
        offset_t_c[:len(t_dirs)] = self.offset_c[t_dirs]
        bzk_t_kc = monkhorst_pack(tuple(self.Nk_c[t_dirs]) + (1, )) + offset_t_c

        # Time-reversal symmetry
        ibzk_p_kc, bzk2ibzk_p_k = symm_reduce(bzk_p_kc)
        ibzk_t_kc = symm_reduce(bzk_t_kc)[0]

        # Take dimensions
        self.bzk_p_k   = bzk_p_kc[:, 0]
        # self.ibzk_t_kc = ibzk_t_kc[:, :2]
        self.ibzk_t_kc = bzk_t_kc[:, :2]
        self.bzk_t_kc = bzk_t_kc[:, :2]

        # Update number of cells
        self.set_num_cells()

    def bloch_to_real_space_p(self, A_kMM, R_c):

        # Output matrix evaluated at (each>?1) transverse k-point.
        A_NMM = []

        # Parameters
        shape = A_kMM.shape

        # Transport and transverse k-points
        p_dir, t_dirs = self.get_directions()

        # For each transverse k-point, Fourier transform in trasport direction
        for j, kt_c in enumerate(self.ibzk_t_kc):
            # Transport k-point that is fourier transformed in transport direction.
            k_kc  = np.zeros((len(self.bzk_p_k), 3))
            k_kc[:, p_dir]  = self.bzk_p_k
            # Fix tranverse directions.
            k_kc[:, t_dirs] = kt_c
            # Matrices at transverse k-point
            A_xMM = []
            for k_c in k_kc:
                try:
                    residue_k = np.linalg.norm(k_c-self.ibzk_kc, axis=1)
                    kc2ibzk = np.where(np.abs(residue_k) < 1e-7)[0][0]
                    A_xMM.append(A_kMM[kc2ibzk])
                except IndexError:
                    # e.g. k_c = (-0.4,0.4) -> -k_c = (0.4,-0.4)
                    # A_kMM[k_c] = A_kMM[-k_c].conj()
                    # Since we require A_kMM[k_c], we can take A_kMM[-k_c].conj(),
                    # which is equivalente and present in lcao calculation.
                    try:
                        residue_k = np.linalg.norm((-k_c)-self.ibzk_kc, axis=1)
                        kc2ibzk = np.where(np.abs(residue_k) < 1e-7)[0][0]
                        A_xMM.append(A_kMM[kc2ibzk].conj())
                    except IndexError:
                        raise IndexError('k-point {} not found'.format(k_c))
            A_MM  = fourier_sum(np.array(A_xMM), k_kc, R_c)

            A_MM /= self.Nk_c[p_dir]

            A_NMM.append(A_MM)

        return np.array(A_NMM)

    def bloch_to_real_space_t(self, A_kMM, R_Nc=None):

        # Output matrix evaluated at (each>?1) transverse k-point.
        A_NMM = []

        # Parameters
        shape = A_kMM.shape
        if R_Nc is None:
            R_Nc = self.R_cN.T

        # Transport and transverse k-points
        p_dir, t_dirs = self.get_directions()

        # Detect gamma point
        gamma_point = True
        try:
            gamma = np.where(np.linalg.norm(self.ibzk_kc, axis=1) < 1e-7)[0][0]
        except IndexError:
            gamma_point = False

        # For each real space point, Fourier transform in trasverse direction
        for i, R_c in enumerate(R_Nc):
            # Evaualte fourier sum in tranverse direction
            A_MM = fourier_sum(A_kMM, self.ibzk_t_kc, R_c)

            if len(self.ibzk_t_kc) < len(self.bzk_t_kc):
                # Add conjugate and subtract double counted Gamma (in transport component)
                if gamma_point:
                    A0_MM = A_kMM[gamma]
                    A_MM += A_MM.conj() - A0_MM
                else:
                    A_MM += A_MM.conj()

            A_MM /= np.prod(self.Nk_c[t_dirs])

            A_NMM.append(A_MM)

        return np.array(A_NMM)

    def remove_pbc(self, A_kMM, eps=-1e-3):

        # atoms of principal layer
        atoms = self.calc.atoms
        atoms.set_calculator(self.calc)

        # Transport direction
        p_dir = self.get_directions()[0]

        L = atoms.cell[p_dir, p_dir]

        centers_ic = get_bf_centers(atoms)
        cutoff = L - eps
        # Coordinates of central unit cell i (along transport)
        centers_p_i = centers_ic[:, p_dir]
        # Coordinates of neighbooring unit cell j
        centers_p_j = centers_p_i + L
        # Distance between j atoms and i atoms
        dist_p_ji = np.abs(centers_p_j[:, None] - centers_p_i[None, :])
        # Mask j atoms farther than L
        mask_ji = (dist_p_ji > cutoff).astype(A_kMM.dtype)

        A_kMM *= mask_ji[None, :]

    def bloch_to_real_space_block(self, A_kMM):

        # Indices of cell vectors in matrix
        index_rows = self.matrix.index_rows
        index_cols = self.matrix.index_cols

        # Fourier transform in transverse directions
        A_NMM = self.bloch_to_real_space_t(A_kMM)

        # The new dimension (x) equals M \times the number of rows
        rows = A_NMM.take(index_rows, axis=0)
        cols = A_NMM.take(index_cols, axis=0)
        A_xx = get_toeplitz(rows=rows, cols=cols)

        return A_xx

class PrincipalSelfEnergy(PrincipalLayer):

    def __init__(self, calc, direction='x'):

        super().__init__(calc, direction)

    def initialize(self, H_kMM, S_kMM, direction='x'):

        super().initialize(H_kMM, S_kMM, direction)

        self.remove_pbc(self.H_kij)
        self.remove_pbc(self.S_kij)

        # self.lowdin_rotation()

        self.selfenergies = [LeadSelfEnergy((h_ii,s_ii),
                                            (h_ij,s_ij),
                                            (h_ij,s_ij))
                             for h_ii, s_ii, h_ij, s_ij in zip(self.H_kii,
                                                               self.S_kii,
                                                               self.H_kij,
                                                               self.S_kij)]


        self.H_Nii = self.bloch_to_real_space_t(self.H_kii)
        self.S_Nii = self.bloch_to_real_space_t(self.S_kii)
        self.H_Nij = self.bloch_to_real_space_t(self.H_kij)
        self.S_Nij = self.bloch_to_real_space_t(self.S_kij)

        self.h_ii = self.bloch_to_real_space_block(self.H_kii)
        self.s_ii = self.bloch_to_real_space_block(self.S_kii)
        self.h_ij = self.bloch_to_real_space_block(self.H_kij)
        self.s_ij = self.bloch_to_real_space_block(self.S_kij)



    def lowdin_rotation(self, apply=True):
        # Lowding rotation at each k-point

        # number of k-point(s) and basis functions
        nkt, nbf = self.H_kii.shape[:2]

        # Indices of H_kii and H_kij subblocks in H_kmm
        index_kii = np.ix_(range(nkt),range(nbf),range(nbf))
        index_kij = np.ix_(range(nkt),range(nbf),range(nbf,2*nbf))

        # Construct bigger matrices
        H_kmm = np.block([[self.H_kii,self.H_kij],
                          [self.H_kji,self.H_kii]])
        S_kmm = np.block([[self.S_kii,self.S_kij],
                          [self.S_kji,self.S_kii]])

        # Lowdin transform at every k-point
        for h_mm, s_mm in zip(H_kmm, S_kmm):
            eig, rot_mm = np.linalg.eigh(s_mm)
            eig = np.abs(eig)
            rot_mm = np.dot(rot_mm / np.sqrt(eig), dagger(rot_mm))
            if apply:
                self.uptodate = False
                h_mm[:] = rotate_matrix(h_mm, rot_mm)  # rotate C region
                s_mm[:] = rotate_matrix(s_mm, rot_mm)

        # Update
        self.H_kii[:] = H_kmm[index_kii]
        self.S_kii[:] = S_kmm[index_kii]
        self.H_kij[:] = H_kmm[index_kij]
        self.S_kij[:] = S_kmm[index_kij]

    @property
    def H(self):
        return self.h_ii
    @property
    def S(self):
        return self.s_ii

    def dos(self, energy):
        """Total density of states -1/pi Im(Tr(GS))"""
        if not hasattr(self, 'S'):
            return -self.retarded(energy).imag.trace() / np.pi
        else:
            # S = self.get_toeplitz(self.S_kii)
            G = self.retarded(energy)
            S = self.S
            return -G.dot(S).imag.trace() / np.pi

    def pdos(self, energy):
        """Projected density of states -1/pi Im(SGS/S)"""
        if not hasattr(self, 'S'):
            return -self.retarded(energy).imag.diagonal() / np.pi
        else:
            # S = self.get_toeplitz(self.S_kii)
            G = self.retarded(energy)
            S = self.S
            SGS = np.dot(S, G.dot(S))
            return -(SGS.diagonal() / S.diagonal()).imag / np.pi

    def retarded(self, energy):

        # Green's functions at thanverse k-points
        G_kMM = []

        # Compute self-energies at transverse k-points
        for selfenergy in self.selfenergies:
            G_kMM.append(la.inv(selfenergy.get_Ginv(energy)))
        G_kMM = np.array(G_kMM)

        # Compute quantities in realspace
        return self.bloch_to_real_space_block(G_kMM)
