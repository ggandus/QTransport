import numpy as np
from functools import lru_cache
from .coupledhamiltonian import CoupledHamiltonian

#Recursive GreenFunction helpers
from .solvers.tridiagonal   import tridiagonalize, cutoff
from .solvers.recursive import get_mat_lists, recursive_gf


class GreenFunction(CoupledHamiltonian):
    """Equilibrium retarded Green function."""

    def __init__(self, H, S=None, selfenergies=[], eta=1e-4):
        self.H = H
        self.S = S
        self.selfenergies = selfenergies
        self.eta = eta
        self.energy = None
        self.Ginv = np.empty(H.shape, complex)

        self.parameters = {'eta': 1e-5,
                           'align_bf': None}
        self.initialized = False

    def set(self, **kwargs):
        for key in kwargs:
            if key in ['eta','align_bf']:
                self.initialized = False
                break
            elif key not in self.parameters:
                raise KeyError('%r not a vaild keyword' % key)

        self.parameters.update(kwargs)

    def initialize(self):#, calc, pl1, pl2=None, cutoff=cutoff, align_bf=None):
        '''
            calc: scattering calculator
            pl1: # of bfs left principal layer
            pl2: # of bfs right principal layer (default to pl1)
            cutoff: cutoff for block neighbor interaction
        '''

        if self.initialized:
            return

        p = self.parameters
        align_bf = p['align_bf']

        #Set eta
        self.eta = p['eta']

        #Align first basis funciton for Left lead
        if align_bf is not None:
            self.align_bf(align_bf)

        self.initialized = True


    # @lru_cache(maxsize=None)
    def retarded(self, energy, inverse=False):
        """Get retarded Green function at specified energy.

        If 'inverse' is True, the inverse Green function is returned (faster).
        """
        if energy != self.energy:
            self.energy = energy
            z = energy + self.eta * 1.j

            if self.S is None:
                self.Ginv[:] = 0.0
                self.Ginv.flat[:: len(self.S) + 1] = z
            else:
                self.Ginv[:] = z
                self.Ginv *= self.S
            self.Ginv -= self.H

            for selfenergy in self.selfenergies:
                self.Ginv -= selfenergy.retarded(energy)

        if inverse:
            return self.Ginv
        else:
            return np.linalg.inv(self.Ginv)

    def calculate(self, energy, sigma):
        """XXX is this really needed"""
        ginv = energy * self.S - self.H - sigma
        return np.linalg.inv(ginv)

    def apply_retarded(self, energy, X):
        """Apply retarded Green function to X.

        Returns the matrix product G^r(e) . X
        """
        return np.linalg.solve(self.retarded(energy, inverse=True), X)

    def dos(self, energy):
        """Total density of states -1/pi Im(Tr(GS))"""
        if self.S is None:
            return -self.retarded(energy).imag.trace() / np.pi
        else:
            GS = self.apply_retarded(energy, self.S)
            return -GS.imag.trace() / np.pi

    def pdos(self, energy):
        """Projected density of states -1/pi Im(SGS/S)"""
        if self.S is None:
            return -self.retarded(energy).imag.diagonal() / np.pi
        else:
            S = self.S
            SGS = np.dot(S, self.apply_retarded(energy, S))
            return -(SGS.diagonal() / S.diagonal()).imag / np.pi

    def take_bfs(self, bfs, apply):
        h_pp, s_pp, c_mm = CoupledHamiltonian.take_bfs(self, bfs, apply)
        if apply:
            self.Ginv = np.empty(self.H.shape, complex)
        return h_pp, s_pp, c_mm


class RecursiveGF(CoupledHamiltonian):

    def __init__(self, H, S=None, selfenergies=[], eta=1e-4):

        super().__init__(self, H, S, selfenergies)

        self.eta = eta
        self.energy = None
        self.g1N = None

        self.parameters = {'eta': eta,
                           'calc': None,
                           'pl1': None,
                           'pl2': None,
                           'cutoff': cutoff,
                           'align_bf': None}

        self.initialized = False

    def set(self, **kwargs):
        for key in kwargs:
            if key in ['eta','calc', 'pl1', 'pl2',
                       'cutoff', 'align_bf']:
                self.initialized = False
                break
            elif key not in self.parameters:
                raise KeyError('%r not a vaild keyword' % key)

        self.parameters.update(kwargs)

    def initialize(self):#, calc, pl1, pl2=None, cutoff=cutoff, align_bf=None):
        '''
            calc: scattering calculator
            pl1: # of bfs left principal layer
            pl2: # of bfs right principal layer (default to pl1)
            cutoff: cutoff for block neighbor interaction
        '''
        if self.initialized:
            return
            
        p = self.parameters()
        align_bf = p['align_bf']
        calc = p['calc']
        pl1 = p['pl1']
        pl1 = p['pl2']

        #Set eta
        self.eta = p['eta']

        #Align first basis funciton for Left lead
        if align_bf is not None:
            self.align_bf(align_bf)

        #Construct block tridiagonal lists
        self.hs_list_ii, self.hs_list_ij = tridiagonalize(
                                           calc, self.H, self.S,
                                           pl1, pl2, cutoff)

        #Note h_im[:,:pl1]=h_ij for Left and h_im[:,-pl2:]=h_ij for Right
        for pl, selfenergy in zip((pl1,pl2),selfenergies[:2]):
            dtype = selfenergy.sigma_mm.dtype
            selfenergy.h_im = selfenergy.h_ij
            selfenergy.s_im = selfenergy.s_ij
            selfenergy.sigma_mm = np.empty((pl,pl), dtype=dtype)

        self.initialized = True

    def get_g1N(self, energy):
        #
        if energy != self.energy:
            self.energy = energy
            z = energy + self.eta * 1.j

            sigma_L = self.selfenergies[0].retarded(energy)
            sigma_R = self.selfenergies[1].retarded(energy)

            mat_lists = get_mat_lists(z, self.hs_list_ii, self.hs_list_ij,
                                      sigma_L, sigma_R)

            self.g1N = recursive_gf(mat_list_ii, mat_list_ij, mat_list_ji)

        return self.g1N
