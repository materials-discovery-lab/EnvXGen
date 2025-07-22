import copy
from ase import Atoms
from ase.io import read
import numpy as np

def get_atomic_positions_by_type(environment, atom_type):
    """ searches for all positions of atoms of a given type in a given environment 
        and gives it as a list of atomic coordinates"""
    
    all_positions = environment.cell.positions
    all_atoms = environment.cell.get_chemical_symbols()

    selected_positions = []
    for i, atom in enumerate(all_atoms):
        if atom == atom_type:
            selected_positions.append(all_positions[i])

    return selected_positions


def sort_atoms_by_type(cell, sort_order):
    """
    Sorts the types of atoms and their positions in ase Atoms object according to a given atomic sequence.
    
    :param atoms: Atoms object.
           sort_order: list of atoms in order that we want to sort Atoms object.
    :return: new ase Atoms object with sorted atoms.
    """

    symbols = cell.get_chemical_symbols()
    positions = cell.get_positions()

    index_order = {atom: index for index, atom in enumerate(sort_order)}
    sorted_indices = sorted(range(len(symbols)), key=lambda i: index_order.get(symbols[i]))
    sorted_symbols = [symbols[i] for i in sorted_indices]
    sorted_positions = positions[sorted_indices]
    sorted_cell = Atoms(symbols=sorted_symbols, positions=sorted_positions, cell=cell.get_cell(), pbc=cell.get_pbc())
    
    return sorted_cell


def get_initial_indices(atoms_init, atoms_final):
    """
    Determines the indices of atoms that were originally present in POSCAR_init.

    :param poscar_init_path: path to the initial POSCAR file
    :param poscar_final_path: path to the final POSCAR file
    :return: a list of indices of atoms from POSCAR_init in the final structure
    """

    #atoms_init = read(poscar_init_path, format="vasp")
    #atoms_final = read(poscar_final_path, format="vasp")

    symbols_init = atoms_init.get_chemical_symbols()
    symbols_final = atoms_final.get_chemical_symbols()

    count_init = {}
    for symbol in symbols_init:
        count_init[symbol] = count_init.get(symbol, 0) + 1

    initial_indices = []
    count_tracker = {symbol: 0 for symbol in count_init}

    for i, symbol in enumerate(symbols_final):
        if symbol in count_init and count_tracker[symbol] < count_init[symbol]:
            initial_indices.append(i)
            count_tracker[symbol] += 1

    return initial_indices


class PlacedAtoms:
    """ atoms that need to be placed in the cell during cell generation"""

    def __init__(self, atom_types, n_atoms, cov_radii):
        self.atom_types = atom_types
        self.n_atoms = n_atoms
        self.cov_radii = cov_radii

    def __deepcopy__(self, memo):

        new_instance = type(self)(
            copy.deepcopy(self.atom_types, memo),  
            copy.deepcopy(self.n_atoms, memo), 
            copy.deepcopy(self.cov_radii, memo) 
        )
        
        return new_instance


class PlacedMolecules:
    """ molecules that need to be placed in the cell during cell generation"""

    def __init__(self, molecule_types, n_molecules, cov_radii, vdw_radii):
        self.molecule_types = molecule_types
        self.n_molecules = n_molecules
        self.cov_radii = cov_radii
        self.vdw_radii = vdw_radii

    def get_avg_distances(self):
        avg_distances = []
        for mol in self.molecule_types:
            print(mol.get_all_distances())

    def get_GCOM(self, i):
        """ Calculates the Geometric Center of Molecule геометрический центр молекулы в формате ASE.

        :param i: index of molecule type 
        :return: Coordinates Geometric Center of Molecule in NumPy
        """
        positions = self.molecule_types[i].get_positions()
        geometric_center = np.mean(positions, axis=0)

        return geometric_center
    
    def get_max_distance_to_GCOM(self, i, gcom):
        """ Calculate max distance from atoms to GCOM """

        positions = self.molecule_types[i].get_positions()
        distances = np.linalg.norm(positions - gcom, axis=1)
        max_distance = np.max(distances)

        return max_distance

    def __deepcopy__(self, memo):

        new_instance = type(self)(
            copy.deepcopy(self.molecule_types, memo),  
            copy.deepcopy(self.n_molecules, memo),            
            copy.deepcopy(self.cov_radii, memo),
            copy.deepcopy(self.vdw_radii, memo) 
        )
        
        return new_instance