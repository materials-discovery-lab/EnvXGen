import numpy as np
from ase.build.supercells import make_supercell
from .structural_elements import sort_atoms_by_type
import copy

def get_extended_cell(selected_cell):
    """ Creates a supercell for the selected cell with current Environment.
        The size of the expanded supercell is 3x3x3,
        the origin is the same as the origin in the selected cell"""
    
    lattice_vectors = selected_cell.cell[:]
    shift_a = np.linalg.norm(lattice_vectors[0])
    shift_b = np.linalg.norm(lattice_vectors[1])
    shift_c = np.linalg.norm(lattice_vectors[2])

    supercell_matrix = [[3, 0, 0],
                        [0, 3, 0],
                        [0, 0, 3]]

    extended_cell = make_supercell(prim=selected_cell, P=supercell_matrix)
    rearranged_positions = []

    for position in extended_cell.get_positions():
        rearranged_positions.append(position - [shift_a, shift_b, shift_c])

    extended_cell.set_positions(rearranged_positions)

    return extended_cell


class Environment:
    """ current atomic environment in the cell"""
    
    def __init__(self, cell, cov_radii, vdw_radii):
        self.cell = cell
        self.cov_radii = cov_radii
        self.vdw_radii = vdw_radii
        self.atom_types = list(dict.fromkeys(cell.get_chemical_symbols()))

    def add_atom(self, potential_point, atom_type, cov_radii):

        self.cell.append(atom_type)
        self.cell.positions[-1] = potential_point

        if not atom_type in self.atom_types:
            self.atom_types.append(atom_type)
            self.cov_radii.update({atom_type:cov_radii})


    def add_molecule(self, molecule, cov_radii, vdw_radii):
        
        for i in range(len(molecule)):
            self.cell.append(molecule[i].symbol)
            self.cell.positions[-1] = molecule[i].position
        
        for atom_type in molecule.get_chemical_symbols():
            if not atom_type in self.atom_types:
                self.atom_types.append(atom_type)

        self.cov_radii.update(cov_radii)
        self.vdw_radii.update(vdw_radii)

    
    def sort_atoms(self):
        self.cell = sort_atoms_by_type(self.cell, self.atom_types)

    def __deepcopy__(self, memo):

        new_instance = type(self)(
            copy.deepcopy(self.cell, memo),
            copy.deepcopy(self.cov_radii, memo),
            copy.deepcopy(self.vdw_radii, memo)
        )
        new_instance.atom_types = copy.deepcopy(self.atom_types, memo)

        return new_instance