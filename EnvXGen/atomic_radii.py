import pandas as pd
import numpy as np
import logging


logger = logging.getLogger()


def get_default_cov_radii(atom_types):
    """ takes a list of atom types as input, produces a dictionary:
    key - atom, value - covalent radius from the database"""

    database = pd.read_csv('RSGFE_generator/data/default_radii.csv')

    cov_radii = []
    for i in atom_types:
        cov_radius = float(database.loc[database.chem_element == i].covalent_radius)
        cov_radii.append(cov_radius)

    return dict(zip(atom_types, cov_radii))

def get_default_vdw_radii(atom_types):
    """ takes a list of atom types as input, produces a dictionary:
    key - atom, value - Van der Waalts radius from the database"""

    database = pd.read_csv('RSGFE_generator/data/default_radii.csv')

    vdw_radii = {}

    for atom in atom_types:
        vdw_radius = database.loc[database.chem_element == atom, 'vdW_radius'].values
        
        if len(vdw_radius) > 0 and not pd.isna(vdw_radius[0]):
            vdw_radii[atom] = float(vdw_radius[0])
        else:
            logger.warning(f"WARNING! Van der Waals radius for '{atom}' not found in the database."
                             "The value has been replaced with the Alvarez Van der Waals radii."
                             "Please note: using default radii of different types is not really correct." 
                             "We recommend to set your own radii or use only Alvarez radii.")
            vdw_radii[atom] = get_default_Alvarez_vdw_radii([atom])[atom]

    return vdw_radii


def get_default_Alvarez_vdw_radii(atom_types):
    """ takes a list of atom types as input, produces a dictionary:
    key - atom, value - Van der Waalts radius from the database
    from Alvarez article"""

    database = pd.read_csv('RSGFE_generator/data/default_radii.csv')

    vdw_radii = {}

    for atom in atom_types:
        vdw_radius = database.loc[database.chem_element == atom, 'vdW_radius_Alvarez'].values
        
        if len(vdw_radius) > 0 and not pd.isna(vdw_radius[0]):
            vdw_radii[atom] = float(vdw_radius[0])
        else:
            logger.warning(f"WARNING! Alvarez Van der Waals radius for '{atom}' not found in the database."
                             "The value has been replaced with the default one."
                             "Please note: using default radii of different types is not really correct." 
                             "We recommend to set your own radii or use only default radii.")
            vdw_radii[atom] = get_default_vdw_radii([atom])[atom]

    return vdw_radii