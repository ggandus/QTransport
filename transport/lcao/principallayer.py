import numpy as np
from gpaw.symmetry import Symmetry
from ase.dft.kpoints import monkhorst_pack
from .tklcao import *

class PrincipalLayer:

    def __init__(self, calc, direction='x'):
        self.calc = calc
        self.direction = direction
        self.update(direction)

    def get_directions(self):
        # Define transport and transverse directions
        p_dir = 'xyz'.index(self.direction)
        t_dirs = np.delete([0, 1, 2], p_dir)
        return p_dir, t_dirs

    def initialize(self, H_kMM, S_kMM, direction):

        # self.H_kMM, self.S_kMM = h_and_s(self.calc)

        # Transport direction
        self.update(direction)
        p_dir, t_dirs = self.get_directions()

        R_c = [0,0,0]
        self.H_kii = self.bloch_to_real_space_p(H_kMM, R_c)
        # self.S_kii = self.bloch_to_real_space_p(S_kMM, R_c)

        # R_c[p_dir] = 1
        # self.H_kij = self.bloch_to_real_space_p(H_kMM, R_c)
        # self.S_kij = self.bloch_to_real_space_p(S_kMM, R_c)
        #
        # self.H_kji = self.H_kij.swapaxes(1,2).conj()
        # self.S_kji = self.S_kij.swapaxes(1,2).conj()

        # self.sym = Symmetry(calc.atoms.numbers, np.array(calc.wfs.gd.cell_cv))
        # self.tb  = TightBinding.__init__(atoms, calc)

    def update(self, direction='x'):

        # Set irreducible k-points alogn transport and transverse dirs.
        kd = self.calc.wfs.kd
        Nk_c = kd.N_c
        offset_c = kd.offset_c

        # Define transport and transverse directions
        p_dir, t_dirs = self.get_directions()

        # K-points in the transport direction
        offset_p_c = np.zeros((3,))
        offset_p_c[p_dir] = offset_c[p_dir]
        bzk_p_kc = monkhorst_pack((Nk_c[p_dir], 1, 1)) + offset_p_c

        # K-points in the transverse directions
        offset_t_c = np.zeros((3,))
        offset_t_c[:len(t_dirs)] = offset_c[t_dirs]
        bzk_t_kc = monkhorst_pack(tuple(Nk_c[t_dirs]) + (1, )) + offset_t_c

        # Time-reversal symmetry
        ibzk_p_kc, bzk2ibzk_p_k = symm_reduce(bzk_p_kc)
        ibzk_t_kc = symm_reduce(bzk_t_kc)[0]

        # # Detect gamma point in transport direction and store index
        # self.gamma_point = True
        # try:
        #     self.gamma_index = np.where(np.linalg.norm(ibzk_p_kc, axis=1) < 1e-7)[0][0]
        # except IndexError:
        #     self.gamma_point = False

        # Take dimensions and store bkz_p_k mapping in ibkz_p_k
        self.bzk_p_k   = bzk_p_kc[:, 0]
        # self.bzk2ibzk_p_k = bzk2ibzk_p_k
        # self.ibzk_p_k  = ibzk_p_kc[:, 0]
        self.ibzk_t_kc = ibzk_t_kc[:, :2]

    def bloch_to_real_space_p(self, A_kMM, R_c):

        # Output matrix evaluated at (each>?1) transverse k-point.
        A_NMM = []

        # Parameters
        shape = A_kMM.shape
        kd = self.calc.wfs.kd
        ibzk_kc = kd.ibzk_kc
        Nk_c = kd.N_c
        offset_c = kd.offset_c

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
                    residue_k = np.linalg.norm(k_c-ibzk_kc, axis=1)
                    kc2ibzk = np.where(np.abs(residue_k) < 1e-7)[0][0]
                    A_xMM.append(A_kMM[kc2ibzk])
                except IndexError:
                    # e.g. k_c = (-0.4,0.4) -> -k_c = (0.4,-0.4)
                    # A_kMM[k_c] = A_kMM[-k_c].conj()
                    # Since we require A_kMM[k_c], we can take A_kMM[-k_c].conj(),
                    # which is equivalente and present in lcao calculation.
                    try:
                        residue_k = np.linalg.norm((-k_c)-ibzk_kc, axis=1)
                        kc2ibzk = np.where(np.abs(residue_k) < 1e-7)[0][0]
                        A_xMM.append(A_kMM[kc2ibzk].conj())
                    except IndexError:
                        raise IndexError('k-point {} not found'.format(k_c))
            A_MM  = fourier_sum(np.array(A_xMM), k_kc, R_c)

            A_MM /= Nk_c[p_dir]

            A_NMM.append(A_MM)

        return np.array(A_NMM)

    def bloch_to_real_space_t(self, A_kMM, R_Nc):

        # Output matrix evaluated at (each>?1) transverse k-point.
        A_NMM = []

        # Parameters
        shape = A_kMM.shape
        kd = self.calc.wfs.kd
        ibzk_kc = kd.ibzk_kc
        Nk_c = kd.N_c

        # Transport and transverse k-points
        p_dir, t_dirs = self.get_directions()
        ibzk_t_kc = self.ibzk_t_kc

        # Detect gamma point in transport direction and store index
        gamma_point = True
        try:
            gamma = np.where(np.linalg.norm(ibzk_t_kc, axis=1) < 1e-7)[0][0]
        except IndexError:
            gamma_point = False

        # For each real space point, Fourier transform in trasverse direction
        for i, R_c in enumerate(R_Nc):
            # Evaualte fourier sum in tranverse direction
            A_MM = fourier_sum(A_kMM, ibzk_t_kc, R_c)

            # Add conjugate and subtract double counted Gamma (in transport component)
            if gamma_point:
                A0_MM = A_kMM[gamma]
                A_MM += A_MM.conj() - A0_MM
            else:
                A_MM += A_MM.conj()

            A_MM /= np.prod(Nk_c[t_dirs])

            A_NMM.append(A_MM)

        return np.array(A_NMM)
