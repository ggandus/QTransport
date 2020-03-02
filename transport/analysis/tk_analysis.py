import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist
from ase.io import write

def get_external_internal(atoms, symbols=None):

    # Atoms to include in external and internal indices
    if symbols is None:
        symbols = list(set(atoms.symbols))
        try:
            symbols.pop(symbols.index('H'))
        except ValueError as e:
            print(e)
    #
    symbols = list(symbols)

    # Define list of neighbors
    cov_radii = [covalent_radii[a.number] for a in atoms]
    nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
    nl.update(atoms)

    external_i = []
    internal_i = []
    for a in atoms:
        if a.symbol not in symbols:
            continue
        nlist = nl.get_neighbors(a.index)[0]
        n_is_H = [True if atoms[n0].symbol == 'H' else False for n0 in nlist]
        if any(n_is_H):
            external_i.append(a.index)
        else:
            internal_i.append(a.index)

    return external_i, internal_i


def plot_mol_wavefunctions(calc, atoms_i, ao_j, v_jj, e_j, spin=0):

    atoms = calc.atoms
    nao = calc.wfs.setups.nao
    nk = len(calc.wfs.kd.ibzk_kc)          #number of kpoints

    for ii in ao_j:
        p1 = v_jj[:,ii]
        n1,cc = 0,0
        psi = np.zeros([nk,nao])  #initialize psi matrix
        for i in range(len(atoms)):
            if i in atoms_i:                   #if i is atom list
                no = calc.wfs.setups[i].nao #get bfs on atom i
                n2 = n1 + no                     #max wfs in psi
                psi[0,n1:n2] = p1[cc:cc+no]      #add coefficients of molecular subspace to list
                cc += no                         #set start for next loop
            n1 += calc.wfs.setups[i].nao    #min wfs in psi (for next step)
        psi = psi.reshape(1,-1)                  #reshape psi to get a vector
        psi_g = calc.wfs.gd.zeros(nk, dtype=calc.wfs.dtype) #initialize
        ss = psi_g.shape                         #get dimensions of psi_g
        psi_g = psi_g.reshape(1, -1)             #reshape psi_g to get a vector
        calc.wfs.basis_functions.lcao_to_grid(psi, psi_g, q=0)
        psi_g = psi_g.reshape(ss)                #resreo original shape

        # write output
        write('orb_%1.4f_spin_%i.cube' %(e_j[ii].real,spin), atoms, data=psi_g[0])
        print('Cube files generated')
        print('.....done!')
