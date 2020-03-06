import numpy as np
from .tools import subdiagonalize, rotate_matrix, dagger, \
                   order_diagonal, cutcoupling, get_subspace


from .tk_gpaw import subdiagonalize_atoms, get_bfs_indices, \
                     extract_orthogonal_subspaces, flatten

class CoupledHamiltonian:

    def __init__(self, H, S, selfenergies):
        self.H = H
        self.S = S
        self.selfenergies = selfenergies

    def align_bf(self, align_bf):
        h_mm = self.H
        s_mm = self.S
        h1_ii = self.selfenergies[0].h_ii
        if align_bf is not None:
            diff = ((h_mm[align_bf, align_bf] - h1_ii[align_bf, align_bf]) /
                    s_mm[align_bf, align_bf])
            # print('# Aligning scat. H to left lead H. diff=', diff)
            h_mm -= diff * s_mm

    def apply_rotation(self, c_mm):
          h_mm = self.H
          s_mm = self.S
          h_mm[:] = rotate_matrix(h_mm, c_mm)
          s_mm[:] = rotate_matrix(s_mm, c_mm)
          # Rotate coupling between lead and central region
          for alpha, sigma in enumerate(self.selfenergies):
              sigma.h_im[:] = np.dot(sigma.h_im, c_mm)
              sigma.s_im[:] = np.dot(sigma.s_im, c_mm)

    def diagonalize(self, apply=False):
        nbf = len(self.H)
        return self.subdiagonalize_bfs(range(nbf), apply)

    def subdiagonalize_bfs(self, bfs, apply=False):
      bfs = np.array(bfs)
      h_mm = self.H
      s_mm = self.S
      ht_mm, st_mm, c_mm, e_m = subdiagonalize(h_mm, s_mm, bfs)
      if apply:
          self.apply_rotation(c_mm)
          return

      c_mm = np.take(c_mm, bfs, axis=0)
      c_mm = np.take(c_mm, bfs, axis=1)
      return ht_mm, st_mm, e_m.real, c_mm

    def cutcoupling_bfs(self, bfs, apply=False):
      bfs = np.array(bfs)
      h_pp = self.H.copy()
      s_pp = self.S.copy()
      cutcoupling(h_pp, s_pp, bfs)
      if apply:
          self.H = h_pp
          self.S = s_pp
          for alpha, sigma in enumerate(self.selfenergies):
              for m in bfs:
                  sigma.h_im[:, m] = 0.0
                  sigma.s_im[:, m] = 0.0
      return h_pp, s_pp

    def take_bfs(self, bfs, apply=False):
        nbf = len(self.H)
        c_mm = np.eye(nbf).take(bfs,1)
        h_pp = rotate_matrix(self.H, c_mm)
        s_pp = rotate_matrix(self.S, c_mm)
        if apply:
            self.H = h_pp
            self.S = s_pp
            for alpha, sigma in enumerate(self.selfenergies):
                sigma.h_im = np.dot(sigma.h_im, c_mm)
                sigma.s_im = np.dot(sigma.s_im, c_mm)
                sigma.sigma_mm = np.empty(self.H.shape, complex)
        return h_pp, s_pp, c_mm

    def lowdin_rotation(self, apply=False):
      h_mm = self.H
      s_mm = self.S
      eig, rot_mm = np.linalg.eigh(s_mm)
      eig = np.abs(eig)
      rot_mm = np.dot(rot_mm / np.sqrt(eig), dagger(rot_mm))
      if apply:
          self.apply_rotation(rot_mm)
          return

      return rot_mm

    def order_diagonal(self, apply=False):
      h_mm = self.H
      s_mm = self.S
      ht_mm, st_mm, c_mm = order_diagonal(h_mm, s_mm)
      if apply:
          self.apply_rotation(c_mm)
          return

      return ht_mm, st_mm, c_mm

    def take_bfs_activespace(self, calc, a, key=lambda x: abs(x)<np.inf, cutoff=np.inf):
        '''
            key     := Apply condition to eigenvalues of subdiagonalized atoms.
            cutoff  := Cutoff for effective embedding to incudle in active space,
                       form subdiagonalized [a] orbitals.
        '''
        h_mm = self.H
        s_mm = self.S
        nbf = h_mm.shape[-1]

        c_MM, e_aj = subdiagonalize_atoms(calc, h_mm, s_mm, a)
        bfs_imp = get_bfs_indices(calc, a)
        # In [a] but not in active space
        bfs_not_m = [bfs_imp[i] for i,
                                    eigval in enumerate(flatten(e_aj))
                     if not key(eigval)] #Take complementary
        # Active space
        bfs_m = list(np.setdiff1d(bfs_imp,bfs_not_m))
        # o := orbitals other atoms not in [a]
        bfs_m_and_o_i = list(np.setdiff1d(range(nbf),bfs_not_m))
        nbfs_m_and_o = len(bfs_m_and_o_i)

        #Apply subdiagonalization
        self.apply_rotation(c_MM)

        if cutoff<np.inf:
            # h_imp = get_subspace(self.H, bfs_imp)
            # s_imp = get_subspace(self.S, bfs_imp)
            #Effective active space := activespace and effective embedding
            bfs_eff_i = []
            bfs_eff_imp = []
            for bfm in bfs_m: # for each bfm (bfm := basis of active)
                row_m_imp = abs(self.H[bfm,bfs_imp])
                #Index of couplings in [a]
                coupling  = np.where(row_m_imp>cutoff)[0]
                bfs_eff_imp = np.union1d(bfs_eff_imp, coupling)
                #Effective active embedding for bfm
                bfs_emb_i = np.arange(self.H.shape[0])[bfs_imp][coupling]
                #Effective activespace
                bfs_eff_i = np.union1d(bfs_eff_i, bfs_emb_i)
            #Unify with others
            bfs_eff_i = bfs_eff_i.tolist()
            bfs_m_and_o_i = np.union1d(bfs_eff_i, bfs_m_and_o_i).tolist()
            #Modify activespace with effective active space
            h_imp = get_subspace(self.H, bfs_eff_i)
            s_imp = get_subspace(self.S, bfs_eff_i)
        else:
            h_imp = get_subspace(self.H, bfs_m)
            s_imp = get_subspace(self.S, bfs_m)


        #Take (effective) activespace
        self.take_bfs(bfs_m_and_o_i, apply=True)

        return h_imp, s_imp

    def get_activespace(self, calc, a, key=lambda x: abs(x)<np.inf, orthogonal=True, inverse=False):
        '''
            key     := Apply condition to eigenvalues of subdiagonalized atoms.
            inverse := Get embedding instead. Put activespace in selfenergy
        '''
        from .internalselfenergy import InternalSelfEnergy

        h_mm = self.H
        s_mm = self.S
        nbf = h_mm.shape[-1]

        c_MM, e_aj = subdiagonalize_atoms(calc, h_mm, s_mm, a)
        bfs_imp = get_bfs_indices(calc, a)
        # Active space
        bfs_m = [bfs_imp[i] for i,
                                eigval in enumerate(flatten(e_aj))
                 if key(eigval)]
        # Embedding
        bfs_i = list(np.setdiff1d(range(nbf),bfs_m))
        nbf_i = len(bfs_i)

        hs_mm, hs_ii, hs_im =  extract_orthogonal_subspaces(h_mm,
                                                            s_mm,
                                                            bfs_m.copy(), # Do not modify
                                                            bfs_i.copy(), # Do not modify
                                                            c_MM,
                                                            orthogonal)

        # Reduce h_im, s_im
        for selfenergy in self.selfenergies:
            try:
                sigma_mm = np.empty((nbf_i,nbf_i), complex)
                h_im = selfenergy.h_im.take(bfs_i, axis=1)
                s_im = selfenergy.s_im.take(bfs_i, axis=1)
            except IndexError:
                print('selfenergies already ok!')
            else:
                selfenergy.h_im = h_im
                selfenergy.s_im = s_im
                selfenergy.sigma_mm = sigma_mm

        if inverse: #Take embedding instead. Put activespace in embedding
            #Activespace selfenergy
            hs_mi = tuple(hs_im[i].T.conj() for i in range(2))
            selfenergy = InternalSelfEnergy(hs_mm, hs_mi)

            if hasattr(self, 'Ginv'):
                self.Ginv = np.empty(hs_ii[0].shape, complex)

            self.__init__(hs_ii[0], hs_ii[1], self.selfenergies+[selfenergy])

            return e_aj

        #Embedding selfenergy
        selfenergy = InternalSelfEnergy(hs_ii, hs_im,
                                        selfenergies=self.selfenergies)

        if hasattr(self, 'Ginv'):
            self.Ginv = np.empty(hs_mm[0].shape, complex)

        self.__init__(hs_mm[0], hs_mm[1], [selfenergy])

        return e_aj
