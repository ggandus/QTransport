#!/usr/bin/env python3

from utilities import *
from ase.io import read

if __name__=='__main__':
    filename = get_filename('*.xyz')
    atoms = read(filename)
    n_uc = int(input('Number of unit cell atoms : '))
    n_rep = int(input('Number of unit cell repetitions : '))
    extend_atoms(atoms, n_uc, n_rep)
    filename = get_basename(filename) + '_ext.xyz'
    atoms.write(filename)
