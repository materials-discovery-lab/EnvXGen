import pandas as pd
import numpy as np
import sys
import yaml
import logging

from .atomic_radii import get_default_cov_radii, get_default_vdw_radii, get_default_Alvarez_vdw_radii
from ase.io import read as read_vasp
from ase.io import write as write_vasp
from ase.data import g2
from ase.data.s22 import s22, s22x5, s26, create_s22_system
from ase.build import molecule
from RSGFE_generator.data import pubchem
from RSGFE_generator.interatomic_distances import distance_ij_cov, distance_ij_vdw

logger = logging.getLogger()


def read_cfg_file(file_path):
    """
    Reads a YAML configuration file and loads its contents into a list of dictionaries.

    Parameters:
    - file_path (str): The path to the YAML configuration file. The file should contain structured data 
                       in YAML format, which may include multiple documents separated by '---'.

    Returns:
    - list: A list of dictionaries where each dictionary represents a document in the YAML file. 
            Each document contains configuration settings. 
            First document contains main information about generation, calculation and postprocessing stages. 
            Second document contains information about initial and placed environment.
    """

    with open(file_path, 'r') as file:
        data = list(yaml.safe_load_all(file))
        
    return data


def merge_atom_radii_dicts(dict_list):

    """
    Merges multiple dictionaries containing atomic radii into a single dictionary.

    This function iterates through a list of dictionaries, where each dictionary contains 
    atomic types as keys and their corresponding radii as values. If an atomic type appears in 
    multiple dictionaries, the function retains the maximum radius value if the new value is not 
    'default'. If the existing value is 'default', it will be replaced by the new value if it is 
    not 'default'.

    Parameters:
    - dict_list (list): A list of dictionaries where each dictionary contains atomic types as keys 
                        and their respective radii as values. The values can be of type float or 
                        the string 'default'.

    Returns:
    - dict: A single dictionary containing merged atomic types and their maximum radii.
    """

    merged_dict = {}
    
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = value
            elif value != 'default':
                merged_dict[key] = max(merged_dict[key], value) if isinstance(merged_dict[key], (float, int)) else value

    return merged_dict


def get_initial_environment(loaded_data):
    """Processes the initial environment data from a YAML configuration file and an atomic structure file. 
       The function ensures that only those atom types present in both the YAML configuration 
       and the atomic types from Initial Environment file are included in the output.

        Parameters:
        loaded_data (dict): A dictionary containing data loaded from cfg.yaml
        
        Returns:
        initial_environment_dict (dict): A dictionary summarizing the initial environment with the following structure:
        'Initial Environment': {
            'Filename' (str): The name of the input atomic structure file,
            'atom_types' (list): List of valid atom types found in both the YAML data and the atomic structure file,
            'covalent_radii' (dict): { 
                'atom_type' (float | None): Mapping of valid single atom types to their covalent radii
                      ...
                      },
                'vdw_radii': { 
                      atom_type (float | None): Mapping of valid molecule atom types to their Van der Waals radii
                      ...
                      }
                },
            }
                  """

    filename = loaded_data['Initial Environment']['Filename']
    fileformat = loaded_data['Initial Environment']['Format']

    single_atom_data = loaded_data['Initial Environment'].get('Atom types from single atoms')
    molecule_atom_data = loaded_data['Initial Environment'].get('Atom types from molecules')

    if single_atom_data is not None:
        single_atom_types = set(single_atom_data.keys())
    else:
        single_atom_types = set()

    if molecule_atom_data is not None:
        molecule_atom_types = set(molecule_atom_data.keys())
        default_value = molecule_atom_data['default values']
    else:
        molecule_atom_types = set()

    all_defined_atoms = single_atom_types.union(molecule_atom_types)

    try:
        atoms = read_vasp(filename)  
        file_atom_types = set(atoms.get_chemical_symbols())  
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        file_atom_types = set()

    valid_atoms = all_defined_atoms.intersection(file_atom_types)

    cov_radii = {}
    for atom in valid_atoms.intersection(single_atom_types):
        properties = single_atom_data[atom]
        if properties['covalent radius'] == 'default':
            cov_radii[atom] = get_default_cov_radii([atom]).get(atom, None)
        else:
            cov_radii[atom] = properties['covalent radius']

    vdw_radii = {}
    for atom in valid_atoms.intersection(molecule_atom_types):
        properties = molecule_atom_data[atom]
        cov_radius = get_default_cov_radii([atom]).get(atom, None)
#        default_value = properties['default values']
        if default_value == 'default':
            #vdw_radii[atom] = get_default_vdw_radii([atom])[atom]
            vdw_radii[atom] = get_default_vdw_radii([atom]).get(atom, cov_radius)
        elif default_value == 'Alvarez':
            #vdw_radii[atom] = get_default_Alvarez_vdw_radii([atom])[atom]
            vdw_radii[atom] = get_default_Alvarez_vdw_radii([atom]).get(atom, cov_radius)

    initial_environment_dict = {
        'Initial Environment': {
            'Filename': filename,
            'Fileformat': fileformat,
            'atom_types': list(valid_atoms),
            'covalent_radii': cov_radii,
            'vdw_radii': vdw_radii
        }
    }

    return initial_environment_dict



def read_molecule_from_database(mol_type, db_name):
    """
    Reads a molecule from the specified database and returns it as an ASE Atoms object.

    Parameters:
    mol_type (str): The name of the molecule from ASE or Pubchem database, or PubChem CID.
    db_name (str): The name of the database to search ('ase', 'pubchem_name', 'pubchem_cid').

    Returns:
    ase.Atoms: An ASE Atoms object representing the specified molecule.

    Raises:
    ValueError: If an unsupported database name, molecule or PubChem CID is provided.
    """

    if db_name == 'ase_name':
        ase_data_g2 = g2.molecule_names
        ase_data_s22_s26 = s22 + s26 + s22x5

        try:
            if mol_type in ase_data_g2:
                mol = molecule(mol_type)
            elif mol_type in ase_data_s22_s26:
                mol = create_s22_system(mol_type)
        except ValueError:
            raise ValueError(f"Warning! The ASE database does not contain this molecule: {mol_type}. Please, use the PubChem database.")
            sys.exit()
        
    elif db_name == 'pubchem_name':
        try:
            mol = pubchem.pubchem_atoms_search(name=mol_type)
        except ValueError:
            raise ValueError(f"Warning! PubChem does not contain this molecule: {mol_type}. "
                             "Please, use PubChem with pubchem_cid or ASE database.")
            sys.exit()
        
    elif db_name == 'pubchem_cid':
        try:
            mol = pubchem.pubchem_atoms_search(cid=int(mol_type))
        except ValueError:
            raise ValueError(f"Warning! PubChem does not contain this CID: {mol_type}."
                             "Please, use PubChem with pubchem_name or ASE database.")
            sys.exit()

    else:
        raise ValueError(f"Unsupported database name: {db_name}. Use 'ase', 'pubchem_name', or 'pubchem_cid'.")

    return mol


def get_placed_environment(loaded_data, mode):

    placed_environment_dict = {
        "Placed Environment": {
            "File": [],
            "atom_types": [],
            "n_atoms": [],
            "covalent_radii_atoms": {},
            "molecule_types": [],
            "molecules": [],
            "n_molecules": [],
            "covalent_radii_molecules": {},
            "vdW_radii_molecules": {}
        }
    }

    try:
        single_atoms_data = loaded_data['Placed structural elements']['Single atoms']
    except:
        single_atoms_data = None
        #single_atoms_data = []
    try:
        molecules_from_file = loaded_data['Placed structural elements']['Molecules from file']
    except:
        #molecules_from_file = []
        molecules_from_file = None
    try:
        molecules_from_databases = loaded_data['Placed structural elements']['Molecules from databases']
    except:
        #molecules_from_databases = []
        molecules_from_databases = None

    single_atoms = {}

    if (mode == 'atoms') or (mode == 'cluster'):
        for atom_type, properties in single_atoms_data.items():
            chemical_symbol = properties["chemical symbol"]
            single_atoms[atom_type] = {
                "chemical_symbol": chemical_symbol,
                "number_of_atoms": properties["number of atoms"],
                "covalent_radius": properties["covalent radius"] if properties["covalent radius"] != "default" 
                                                                else get_default_cov_radii([chemical_symbol])[str(chemical_symbol)]
                }

    molecules = {}
    molecule_filenames = []

    if mode == 'molecules':

        if isinstance(molecules_from_file, dict):
            for molecule_type, properties in molecules_from_file.items():

                filename = properties['Filename']
                molecule_filenames.append(filename)
                default_type = properties["default values"]
                molecule_structure = read_vasp(filename)
                atom_types_from_file = list(set(molecule_structure.get_chemical_symbols()))

                for radii_type in ['covalent radii', 'Van der Waals radii']:
                    required_atom_types = set(properties[radii_type].keys())

                    excess_atom_types = required_atom_types - set(atom_types_from_file)

                    if excess_atom_types:
                        logger.warning(f"Warning! The following atoms are not found in the {filename}: {excess_atom_types}.")
                        for excess_atom_type in excess_atom_types:
                                properties[radii_type].pop(excess_atom_type, None)

                    missing_atom_types = set(atom_types_from_file) - required_atom_types

                    for missing_atom_type in missing_atom_types:
                        logger.info(f"Adding missing atom type: {missing_atom_type}. Loading default values for {radii_type}.")
                        if radii_type == 'covalent radii':
                            properties['covalent radii'][missing_atom_type] = get_default_cov_radii([missing_atom_type])[missing_atom_type]
                        elif radii_type == 'Van der Waals radii':
                            properties['Van der Waals radii'][missing_atom_type] = get_default_vdw_radii([missing_atom_type])[missing_atom_type]

                for covalent_radii_info in properties['covalent radii'].items():
                    if str(covalent_radii_info[1]) == 'default':
                        properties['covalent radii'][covalent_radii_info[0]] = get_default_cov_radii([covalent_radii_info[0]])[covalent_radii_info[0]]
                    else:
                        properties['covalent radii'][covalent_radii_info[0]] = float(covalent_radii_info[1])

                for vdW_radii_info in properties['Van der Waals radii'].items():
                    if vdW_radii_info[1] == 'default':
                        if default_type == 'Alvarez':
                            properties['Van der Waals radii'][vdW_radii_info[0]] = get_default_Alvarez_vdw_radii([vdW_radii_info[0]])[vdW_radii_info[0]]
                        else:
                            properties['Van der Waals radii'][vdW_radii_info[0]] = get_default_vdw_radii([vdW_radii_info[0]])[vdW_radii_info[0]]
                    else:
                        properties['Van der Waals radii'][vdW_radii_info[0]] = vdW_radii_info[1]
                        
                molecules[molecule_type] = {
                    "molecule_name": properties["name"],
                    "molecule": molecule_structure,
                    "number_of_molecules": properties["number of molecules"],
                    "covalent_radii": properties["covalent radii"],
                    "vdW_radii": properties["Van der Waals radii"]
                    }

        if isinstance(molecules_from_databases, dict):        
            for molecule_type, properties in molecules_from_databases.items():
                if properties['ase_name']:
                    database = 'ase_name'
                    molecule_name = str(properties['ase_name'])
                elif properties['pubchem_name']:
                    database = 'pubchem_name'
                    molecule_name = str(properties['pubchem_name'])
                elif properties['pubchem_cid']:
                    database = 'pubchem_name'
                    molecule_name = str(properties['pubchem_name'])
                else:
                    logger.warning('Warning! Problems with database')
                molecule_structure = read_molecule_from_database(molecule_name, database)

                default_type = properties["default values"]

                atom_types_from_molecule_structure = list(set(molecule_structure.get_chemical_symbols()))

                for radii_type in ['covalent radii', 'Van der Waals radii']:
                    required_atom_types = set(properties[radii_type].keys())

                    excess_atom_types = required_atom_types - set(atom_types_from_molecule_structure)

                    if excess_atom_types:
                        logger.warning(f"Warning! The following atoms are not found in the {filename}: {excess_atom_types}.")
                        for excess_atom_type in excess_atom_types:
                                properties[radii_type].pop(excess_atom_type, None)

                    missing_atom_types = set(atom_types_from_molecule_structure) - required_atom_types

                    for missing_atom_type in missing_atom_types:
                        logger.info(f"Adding missing atom type: {missing_atom_type}. Loading default values for {radii_type}.")
                        if radii_type == 'covalent radii':
                            properties['covalent radii'][missing_atom_type] = get_default_cov_radii([missing_atom_type])[missing_atom_type]
                        elif radii_type == 'Van der Waals radii':
                            properties['Van der Waals radii'][missing_atom_type] = get_default_vdw_radii([missing_atom_type])[missing_atom_type]

                for covalent_radii_info in properties['covalent radii'].items():
                    if str(covalent_radii_info[1]) == 'default':
                        properties['covalent radii'][covalent_radii_info[0]] = get_default_cov_radii([covalent_radii_info[0]])[covalent_radii_info[0]]
                    else:
                        properties['covalent radii'][covalent_radii_info[0]] = float(covalent_radii_info[1])


                for vdW_radii_info in properties['Van der Waals radii'].items():
                    if vdW_radii_info[1] == 'default':
                        if default_type == 'Alvarez':
                            properties['Van der Waals radii'][vdW_radii_info[0]] = get_default_Alvarez_vdw_radii([vdW_radii_info[0]])[vdW_radii_info[0]]
                        else:
                            properties['Van der Waals radii'][vdW_radii_info[0]] = get_default_vdw_radii([vdW_radii_info[0]])[vdW_radii_info[0]]
                    else:
                        properties['Van der Waals radii'][vdW_radii_info[0]] = vdW_radii_info[1]
                        
                molecules[molecule_type] = {
                    "molecule_name": properties["name"],
                    "molecule": molecule_structure,
                    "number_of_molecules": properties["number of molecules"],
                    "covalent_radii": properties["covalent radii"],
                    "vdW_radii": properties["Van der Waals radii"]
                    }

    placed_environment_dict = {
        "Placed Environment": {
            "File": molecule_filenames,
            #"Format": file_format,
            "atom_types": list(single_atoms.keys()),
            "n_atoms": [single_atoms[atom]['number_of_atoms'] for atom in single_atoms.keys()],
            "covalent_radii_atoms": {atom: single_atoms[atom]['covalent_radius'] for atom in single_atoms.keys()},
            "molecule_types": list(molecules.keys()),
            "molecules": [molecules[molecule]['molecule'] for molecule in molecules.keys()],
            "n_molecules": [molecules[molecule]['number_of_molecules'] for molecule in molecules.keys()],
            "covalent_radii_molecules": merge_atom_radii_dicts(molecules[molecule]['covalent_radii'] for molecule in molecules.keys()),
            "vdW_radii_molecules": merge_atom_radii_dicts(molecules[molecule]['vdW_radii'] for molecule in molecules.keys())
            }
        }
    
    return(placed_environment_dict)


def calculate_n_bins(initial_cell, n_bins_dict, mode, initial_environment_cfg, placed_environment_cfg):

    if any(value == 'default' for value in n_bins_dict.values()):

        if mode == 'atoms' or mode == 'cluster':

            initial_environment_cov_radii = initial_environment_cfg["Initial Environment"]['covalent_radii']
            placed_environment_cov_radii = placed_environment_cfg['Placed Environment']['covalent_radii_atoms']

            ij_cov_distances = distance_ij_cov(initial_environment_cov_radii, 
                                               placed_environment_cov_radii)
            jj_cov_distances = distance_ij_cov(placed_environment_cov_radii, 
                                               placed_environment_cov_radii)
            
            default_distance = min(ij_cov_distances.Distance.to_list() + jj_cov_distances.Distance.to_list())

        elif mode == 'molecules':

            initial_environment_cov_radii = initial_environment_cfg["Initial Environment"]['covalent_radii']
            placed_molecules_cov_radii = placed_environment_cfg['Placed Environment']['covalent_radii_placed_molecules']
            initial_environment_vdw_radii = initial_environment_cfg["Initial Environment"]['vdw_radii']
            placed_molecules_vdw_radii =  placed_environment_cfg['Placed Environment']['vdW_radii_placed_molecules']

            ij_cov_distances = distance_ij_cov(initial_environment_cov_radii, placed_molecules_cov_radii)
            ij_vdw_distances = distance_ij_vdw(initial_environment_vdw_radii, placed_molecules_vdw_radii)
            jj_vdw_distances = distance_ij_vdw(placed_molecules_vdw_radii, placed_molecules_vdw_radii)

            default_distance = min(ij_cov_distances.Distance.to_list() + ij_vdw_distances.Distance.to_list() + jj_vdw_distances.Distance.to_list())

        if n_bins_dict['n_bins_x'] == 'default':
            n_bins_x = int(np.linalg.norm(initial_cell.cell[0]) // default_distance)
            n_bins_dict['n_bins_x'] = n_bins_x

        if n_bins_dict['n_bins_y'] == 'default':
            n_bins_y = int(np.linalg.norm(initial_cell.cell[0]) // default_distance)
            n_bins_dict['n_bins_y'] = n_bins_y
            
        if n_bins_dict['n_bins_z'] == 'default':
            n_bins_z = int(np.linalg.norm(initial_cell.cell[0]) // default_distance)
            n_bins_dict['n_bins_z'] = n_bins_z

    return n_bins_dict


def sort_atom_types_by_radii(configs):
    """ Updates configurations by sorting the list of added atoms 
        according to decreasing covalent radius"""
    
    sorted_atoms = sorted(
        zip(configs['Placed Environment']['atom_types'], configs['Placed Environment']['n_atoms']),
        key=lambda x: configs['Placed Environment']['covalent_radii_atoms'].get(x[0], float('-inf')),
        reverse=True
        )
    
    sorted_atom_types, sorted_n_atoms = zip(*sorted_atoms)

    configs['Placed Environment']['atom_types'] = list(sorted_atom_types)
    configs['Placed Environment']['n_atoms'] = list(sorted_n_atoms)

    return configs












def get_placed_environment_old(loaded_data):

    placed_environment_dict = {}

    # If placed structural data in file:
    try:
        filename = loaded_data['Placed structural elements']['Filename']
        fileformat = loaded_data['Placed structural elements']['Format']
        placed_environment_dict.update('Filename', filename)
        placed_environment_dict.update('Fileformat', fileformat)

        single_atom_from_file_data = loaded_data['Placed structural elements'].get('Single atoms from file')
        molecule_atom_from_file_data = loaded_data['Placed structural elements'].get('Molecules from file')

        if ((single_atom_from_file_data is not None) and (molecule_atom_from_file_data is not None)):
            all_atom_types_from_file = list(single_atom_from_file_data.keys()) + list(molecule_atom_from_file_data.keys())
        elif ((single_atom_from_file_data is not None) and (molecule_atom_from_file_data is None)):
            all_atom_types_from_file = list(single_atom_from_file_data.keys())
        elif ((single_atom_from_file_data is None) and (molecule_atom_from_file_data is not None)):
            all_atom_from_file_types = list(molecule_atom_from_file_data.keys())
        else:
            logger.warning('Warning! Placed Environment is empty.')
        
        cov_radii_from_file = {}

        for atom, properties in single_atom_from_file_data.items():
            if properties['covalent radius'] == 'default':
                cov_radii_from_file[atom] = get_default_cov_radii([atom]).get(atom, None)
            else:
                cov_radii_from_file[atom] = properties['covalent radius']

        for atom in molecule_atom_from_file_data.keys():
            if atom not in cov_radii_from_file:
                cov_radii_from_file[atom] = get_default_cov_radii([atom]).get(atom, None)

        vdw_radii_from_file = {}

        for atom, properties in molecule_atom_from_file_data.items():
            default_value = properties['default values']
            if default_value == 'default':
                vdw_radii_from_file[atom] = get_default_vdw_radii([atom])[atom]
            elif default_value == 'Alvarez':
                vdw_radii_from_file[atom] = get_default_Alvarez_vdw_radii([atom])[atom]

    except:
        logger.info('Placed environment does not have a file')

    
    #If placed structural data from database


    single_atom_data = loaded_data['Initial Environment'].get('Atom types from single atoms')
    molecule_atom_data = loaded_data['Initial Environment'].get('Atom types from molecules')

    if ((single_atom_data is not None) and (molecule_atom_data is not None)):
        all_atom_types = list(single_atom_data.keys()) + list(molecule_atom_data.keys())
    elif ((single_atom_data is not None) and (molecule_atom_data is None)):
        all_atom_types = list(single_atom_data.keys())
    elif ((single_atom_data is None) and (molecule_atom_data is not None)):
        all_atom_types = list(molecule_atom_data.keys())
    else:
        logger.error('Warning! Initial Environment is empty.')

    cov_radii_all = {}

    for atom, properties in single_atom_data.items():
        if properties['covalent radius'] == 'default':
            cov_radii_all[atom] = get_default_cov_radii([atom]).get(atom, None)
        else:
            cov_radii_all[atom] = properties['covalent radius']

    for atom in molecule_atom_data.keys():
        if atom not in cov_radii_all:
            cov_radii_all[atom] = get_default_cov_radii([atom]).get(atom, None)

    vdw_radii = {}

    for atom, properties in molecule_atom_data.items():
        default_value = properties['default values']
        if default_value == 'default':
            vdw_radii[atom] = get_default_vdw_radii([atom])[atom]
        elif default_value == 'Alvarez':
            vdw_radii[atom] = get_default_Alvarez_vdw_radii([atom])[atom]

    initial_environment_dict = {
        'Initial Environment': {
            'Filename': filename,
            'Fileformat': fileformat,
            'atom_types': all_atom_types,
            'covalent_radii': cov_radii_all,
            'vdw_radii': vdw_radii
            }
    }

    return initial_environment_dict
