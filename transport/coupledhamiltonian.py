import numpy as np
from .tools import subdiagonalize, rotate_matrix, dagger, order_diagonal, cutcoupling

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
