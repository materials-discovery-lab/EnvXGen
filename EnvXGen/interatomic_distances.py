import pandas as pd

def distance_ij_cov(atoms_a, atoms_b):

    """ calculates interatomic distances based on two dictionaries containing atom types 
    and covalent or van der Waals radii 
    for all pairs of atom types: i from atoms_a, j from atoms_b"""

    ij = {}
    ij.setdefault('Atom_type_1', [])
    ij.setdefault('Atom_type_2', [])
    ij.setdefault('Distance', [])

    for i in list(atoms_a.keys()):
        r_i = atoms_a[i]
        for j in list(atoms_b.keys()):
            r_j = atoms_b[j]

            ij['Atom_type_1'].append(i)
            ij['Atom_type_2'].append(j)
            ij['Distance'].append(round((r_i + r_j), 3))

    df_ij= pd.DataFrame(ij)
    return df_ij


def distance_ij_vdw(atoms_a, atoms_b):
    """ calculates interatomic distances based on two dictionaries containing 
    atom types and covalent or van der Waals radii 
    for all pairs of atom types: i from atoms_a, j from atoms_b"""

    ij = {}
    ij.setdefault('Atom_type_1', [])
    ij.setdefault('Atom_type_2', [])
    ij.setdefault('Distance', [])

    for i in list(atoms_a.keys()):
        r_i = atoms_a[i]
        for j in list(atoms_b.keys()):
            r_j = atoms_b[j]

            if i == j:
                ij['Atom_type_1'].append(i)
                ij['Atom_type_2'].append(j)
                ij['Distance'].append(round((r_i + r_j), 3))

    df_ij= pd.DataFrame(ij)
    return df_ij