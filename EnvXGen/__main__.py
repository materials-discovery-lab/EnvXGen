from EnvXGen.interatomic_distances import distance_ij_cov, distance_ij_vdw
from .atomic_radii import get_default_cov_radii, get_default_vdw_radii, get_default_Alvarez_vdw_radii
from .environment import Environment
from .potential_points import PotentialPoints, Check_Convex_Hull, rotate_and_translate_molecule, check_rotation
from .structural_elements import PlacedAtoms, PlacedMolecules, get_initial_indices
from .prepare_calcfolds import prepare_relaxation_files
from .get_cfg_file import read_cfg_file, merge_atom_radii_dicts, get_initial_environment, read_molecule_from_database, get_placed_environment, calculate_n_bins, sort_atom_types_by_radii
from .potential_points import get_neighbors, get_potential_points_by_index, select_distant_points, filter_potential_points
from .calculation_params import get_header, get_calculation_params
from .relaxation_scripts import update_incar_files

import sys
import os
import shutil
import ase
from ase import Atoms
from ase.io import read as read_vasp
from ase.io import write as write_vasp
from ase.constraints import FixAtoms

import numpy as np
import pandas as pd
import random
import copy
import pickle
from tqdm import tqdm, trange
from datetime import datetime
from time import sleep
from itertools import chain

import logging
from logging.handlers import TimedRotatingFileHandler

from .run_relaxation import run_vasp, run_lammps


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#console_handler = logging.StreamHandler()
#console_handler.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#console_handler.setFormatter(formatter)
#logger.addHandler(console_handler)

#handler = TimedRotatingFileHandler(
#    'log', 
#    when='M',
#    interval=5
#)

handler = logging.FileHandler('log')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():

    yaml_file_name = 'cfg.yaml'
    #atom_types_for_convex = 'all'
    #upper_multiplier = 1.2
    #down_multiplier = 0.8

    loaded_data = read_cfg_file(yaml_file_name)

    relaxation_mode = loaded_data[0]['Relaxation mode']

    if ((relaxation_mode == 'vasp') or (relaxation_mode == 'vasp_fixed') or (relaxation_mode == 'vasp_fixed_and_unfixed')):
        try:
            vasp_steps = loaded_data[0]['VASP steps']
        except:
            logger.error(f"The number of VASP relaxation steps is not specified")
            sys.exit(1)

    if relaxation_mode == 'lammps':
        try:
            force_field_file_name = loaded_data[0]['Force-field filename']
        except:
            logger.error(f"Force-field for lammps relaxation is not specified")
            sys.exit(1)

    try:
        pressure = int(loaded_data[0]['Pressure (GPa)'])
    except:
        pressure=None

    try:
        descriptors = loaded_data[0]['Crystal structure descriptors']
    except:
        pass

    try:
        reducer = loaded_data[0]['Reducer']
    except:
        pass

    HEADER = loaded_data[0]['Calculation settings']
    n_parallel_calcs = loaded_data[0]['Number of parallel calculations']

    crystal_system = loaded_data[0]['Crystal system']
    n_crystals = loaded_data[0]['Number of generated structures']
    mode = loaded_data[0]['Mode']

    initial_environment_cfg = get_initial_environment(loaded_data[1])
    initial_cell = read_vasp(initial_environment_cfg['Initial Environment']['Filename'])
    initial_atom_types = initial_environment_cfg['Initial Environment']['atom_types']
    n_fixed_atoms = initial_cell.get_global_number_of_atoms()

    convex_checker = Check_Convex_Hull(initial_cell)

    placed_environment_cfg = get_placed_environment(loaded_data[1], mode)
    #placed_environment_cfg = sort_atom_types_by_radii(placed_environment_cfg)

    n_bins_loaded = loaded_data[0]['Number of bins along each lattice vector']
    n_bins = calculate_n_bins(initial_cell, n_bins_loaded, mode, initial_environment_cfg, placed_environment_cfg)

    initial_potential_points = PotentialPoints()
    initial_potential_points.get_initial_points(initial_cell.cell, n_bins)
    initial_environment = Environment(initial_cell, 
                                      initial_environment_cfg['Initial Environment']['covalent_radii'], 
                                      initial_environment_cfg['Initial Environment']['vdw_radii'])
    

    logger.info(f"Crystals generation")

    if not os.path.exists('result'):
        os.makedirs('result')

    startTime = datetime.now()

    crystals = []

    max_generation_attempts = 10
    max_rotation_attempts = 10
    generation_attempts = 0
    num_crystal = 0

    with tqdm(total=n_crystals) as pbar:
        #while num_crystal < n_crystals:

            if (mode == 'cluster'):
                logger.info(f"{'This mode is under development'}")

            else:

                while num_crystal < n_crystals:

                    if (mode == 'atoms'):

                        placed_environment_cfg = sort_atom_types_by_radii(placed_environment_cfg)

                        placed_atoms = PlacedAtoms(placed_environment_cfg['Placed Environment']['atom_types'], 
                                                placed_environment_cfg['Placed Environment']['n_atoms'], 
                                                placed_environment_cfg['Placed Environment']['covalent_radii_atoms'])

                        pbar.set_description(f"Crystal {num_crystal + 1}")

                        initial_potential_points = PotentialPoints()
                        initial_potential_points.get_initial_points(initial_cell.cell, n_bins)
                        initial_environment = Environment(initial_cell, 
                                                        initial_environment_cfg['Initial Environment']['covalent_radii'], 
                                                        initial_environment_cfg['Initial Environment']['vdw_radii'])

                        current_environment = copy.deepcopy(initial_environment)
                        current_placed_atoms = copy.deepcopy(placed_atoms)
                        added_atom_types = current_placed_atoms.atom_types

                        generation_attempts = 0

                        logger.info(f"{current_placed_atoms.atom_types}")
                        logger.info(f"{current_placed_atoms.n_atoms}")

                        for i in range(len(added_atom_types)):

                            attempt = 0
                            potential_points_err = 0

                            while current_placed_atoms.n_atoms[i] > 0:

                                #logger.info(f"{current_placed_atoms.atom_types}")
                                #logger.info(f"{current_placed_atoms.n_atoms}")

                                try:
                                    current_potential_points = PotentialPoints.get_points_for_adding_atom(initial_potential_points,
                                                                                                        current_environment,
                                                                                                        current_placed_atoms.cov_radii[added_atom_types[i]])
                                    
                                    #print(type(current_potential_points))
                                except ValueError:
                                    logger.error(f"Fail in crystal generation")
                                    potential_points_err += 1

                                    if num_crystal == 0:
                                        logger.error(f"Unable to create list of potential points")
                                        logger.error(f"Increase the number of bins manually (current n_bins_a, n_bins_b, n_bins_c: )")
                                        logger.error(f"or decrease the r_cov/r_vdw manually")
                                        sys.exit()
                                    
                                if not current_potential_points:
                                    logger.error(f"No available points for adding atom")
                                    attempt += 1
                                    break         

                                while attempt < max_generation_attempts:
                                    coordinates_index = random.choice(range(len(current_potential_points)))
                                    coordinate = current_potential_points[coordinates_index]      

                                    try:
                                        current_environment.add_atom(coordinate, added_atom_types[i], current_placed_atoms.cov_radii[added_atom_types[i]])
                                        current_placed_atoms.n_atoms[i] -= 1
                                        break

                                    except ValueError:
                                        attempt += 1
                                        del current_potential_points[coordinates_index]

                                if attempt == max_generation_attempts:
                                    tqdm.write(f'Fail in crystal generation: failed to add {added_atom_types[i]} atom after {max_generation_attempts} attempts')
                                    #print(f"Failed to add {added_atom_types[i]} after {max_generation_attempts} attempts.")
                                    break    

                                if any(current_placed_atoms.n_atoms):
                                    generation_attempts += 1
                                
                                if generation_attempts >= max_generation_attempts:
                                    continue

                        pbar.set_postfix_str(f"Сrystal {num_crystal+1} generated", refresh=True)

                        ####### ATTENTION!!! Maybe sort atoms after making db

                        current_environment.sort_atoms()
                        crystals.append(current_environment.cell) 
                        num_crystal += 1                        

                    #elif (mode == 'molecules'):
                    elif mode == 'molecules':
                    
                        pbar.set_description(f"Crystal {num_crystal + 1}")

                        initial_potential_points = PotentialPoints()
                        initial_potential_points.get_initial_points(initial_cell.cell, n_bins)
                        initial_environment = Environment(initial_cell, 
                                                        initial_environment_cfg['Initial Environment']['covalent_radii'], 
                                                        initial_environment_cfg['Initial Environment']['vdw_radii'])
                        
                        #logger.info(f"Placed env: {placed_environment_cfg['Placed Environment']}")
                        
                        placed_molecules = PlacedMolecules(placed_environment_cfg['Placed Environment']['molecules'], 
                                                        placed_environment_cfg['Placed Environment']['n_molecules'], 
                                                        placed_environment_cfg['Placed Environment']['covalent_radii_molecules'], 
                                                        placed_environment_cfg['Placed Environment']['vdW_radii_molecules'])

                        current_environment = copy.deepcopy(initial_environment)

                        current_placed_molecules = copy.deepcopy(placed_molecules)
                        added_molecule_types = current_placed_molecules.molecule_types
                        #logger.info(f"added_molecule_types: {added_molecule_types}")

                        for i in range(len(added_molecule_types)):
                            attempt = 0
                            potential_points_err = 0

                            #cov_radii_mol = {key: crystal_configs['covalent_radii_mol'][key] for key in added_molecule_types[i].get_chemical_symbols() if key in crystal_configs['covalent_radii_mol']}
                            #vdw_radii_mol = {key: crystal_configs['vdW_radii'][key] for key in added_molecule_types[i].get_chemical_symbols() if key in crystal_configs['vdW_radii']}

                            cov_radii_mol = {key: placed_environment_cfg['Placed Environment']['covalent_radii_molecules'][key] 
                                            for key in added_molecule_types[i].get_chemical_symbols() 
                                            if key in placed_environment_cfg['Placed Environment']['covalent_radii_molecules']}
                            vdw_radii_mol = {key: placed_environment_cfg['Placed Environment']['vdW_radii_molecules'][key] 
                                            for key in added_molecule_types[i].get_chemical_symbols() 
                                            if key in placed_environment_cfg['Placed Environment']['vdW_radii_molecules']}

                            min_cov_radius = min(cov_radii_mol.values())
                            max_vdw_radius = max(vdw_radii_mol.values())

                            geometric_center = PlacedMolecules.get_GCOM(current_placed_molecules, i)
                            r_cut_neighbours = max_vdw_radius + PlacedMolecules.get_max_distance_to_GCOM(current_placed_molecules, i, geometric_center)

                            # Attempt to add molecules until all are placed or we reach the maximum attempts
                            while current_placed_molecules.n_molecules[i] > 0 and generation_attempts < max_generation_attempts:
                                pbar.set_postfix_str(f'Attempt {attempt}')

                                try:
                                    current_potential_points = PotentialPoints.get_points_for_adding_molecule(initial_potential_points,
                                                                                                            current_environment,
                                                                                                            min_cov_radius)
                                except ValueError:
                                    pbar.set_postfix_str('Fail in crystal generation')
                                    potential_points_err += 1
                                    if num_crystal == 0:
                                        sys.exit()

                                if not current_potential_points:  # Check for available points
                                    pbar.set_postfix_str('No available points for adding molecule')
                                    break

                                # Limit the number of coordinates to check
                                coordinate_attempts = 0

                                while coordinate_attempts < min(len(current_potential_points), 10):
                                    coordinates_index = random.choice(range(len(current_potential_points)))
                                    coordinate = current_potential_points[coordinates_index]

                                    rotation_attempt = 0
                                    success = False

                                    while rotation_attempt < max_rotation_attempts:
                                        rotated_molecule = rotate_and_translate_molecule(coordinate, added_molecule_types[i], geometric_center)

                                        try:
                                            #logger.info(f"current_environment: {current_environment.cov_radii}")
                                            check_rotation(current_environment, rotated_molecule, cov_radii_mol, vdw_radii_mol)
                                            current_environment.add_molecule(rotated_molecule, cov_radii_mol, vdw_radii_mol)  
                                            current_placed_molecules.n_molecules[i] -= 1
                                            success = True
                                            #logger.info(f"Молекула {added_molecule_types[i]} успешно добавлена")
                                            break  # Exit the rotation attempt loop on success

                                        except ValueError:
                                            rotation_attempt += 1
                                            pbar.set_postfix_str(f'Coordinate attempt: {coordinate_attempts}, Rotation attempt {rotation_attempt}')
                                            #print(f'Coordinate attempt: {coordinate_attempts}, Rotation attempt {rotation_attempt}')
                                    if success:
                                        #print(f'Molecule {rotated_molecule} is sucessfully added')
                                        break  # Exit the coordinate attempt loop on success

                                    # If rotation attempts failed, remove this coordinate and try another one
                                    del current_potential_points[coordinates_index]
                                    coordinate_attempts += 1
                                

                            if coordinate_attempts == min(len(current_potential_points), 10):
                                tqdm.write(f'Fail in generation: failed generate {added_molecule_types[i]} after checking all coordinates')
                                break

                        if current_placed_molecules.n_molecules[i] == 0:  # If all molecules were successfully placed
                            pbar.set_postfix_str(f"Crystal {num_crystal + 1} generated", refresh=True)
                            current_environment.sort_atoms()
                            crystals.append(current_environment.cell)  # Add successfully generated crystal to the list

                            num_crystal += 1
                            pbar.update(1)


    indexed_crystals = [
        {'ID': index + 1, 'generated_structure': crystal} 
        for index, crystal in enumerate(crystals)
    ]

    with open(f'result/POSCARS_{crystal_system}_generated.pkl', 'wb') as f:
        pickle.dump(indexed_crystals, f)


    logger.info(f"Time elasped: ', {datetime.now() - startTime}")
    logger.info(f"Crystal structures sucesfully generated!")

    ##################################################
    # RELAXATION part

    header = get_header(HEADER)
    database_filename = f'result/POSCARS_{crystal_system}_generated.pkl'

    os.makedirs('results')
    shutil.copy(database_filename, 'results')

    if relaxation_mode == 'vasp_fixed_and_unfixed':
        relaxation_params_fixed = get_calculation_params(relaxation_mode='vasp_fixed',
                                                         vasp_steps=vasp_steps)
        if pressure:
            update_incar_files(incars_directory=relaxation_params_fixed['incars_dir'], 
                               crystal_system=crystal_system, 
                               pstress_value=pressure)
            

        database_filename = f'result/POSCARS_{crystal_system}_generated.pkl'

        with open(database_filename, 'rb') as database:
            data_generated = pickle.load(database)

        poscar_final_template = data_generated[0]['generated_structure']
        fix_indices = get_initial_indices(initial_cell, poscar_final_template)
        #print(f"fix_indices = {fix_indices}")
        #fix_indices = list(range(0, n_fixed_atoms))
        constraint = FixAtoms(indices=fix_indices)

        for structure in data_generated:
            structure['generated_structure'].set_constraint(constraint)

        with open(f'result/POSCARS_{crystal_system}_generated_fixed.pkl', 'wb') as f:
            pickle.dump(data_generated, f)
        
        fixed_database_filename = f'result/POSCARS_{crystal_system}_generated_fixed.pkl'
        
        success = run_vasp(database_filename=fixed_database_filename, 
                           relaxation_params=relaxation_params_fixed, 
                           header=header, 
                           n_parallel_calcs=n_parallel_calcs)

        shutil.move("result", "result_vasp_fixed")

        vasp_fixed_database_filename = f'result_vasp_fixed/relaxation_results_all.pkl'
        with open(vasp_fixed_database_filename, 'rb') as vasp_fixed_database:
            data = pickle.load(vasp_fixed_database)

        df_results = pd.DataFrame(data)
        df_filtered = df_results.dropna(subset=['relaxed_structure'])
        new_df_filtered = df_filtered[['ID', 'relaxed_structure']].rename(columns={'relaxed_structure': 'generated_structure'})
        new_df_filtered.reset_index(drop=True, inplace=True)
        list_of_dicts = new_df_filtered.to_dict(orient='records')

        os.makedirs('result')

        with open(f'result/POSCARS_{crystal_system}_vasp_fixed.pkl', 'wb') as f:
            pickle.dump(list_of_dicts, f)

        vasp_fixed_relaxed_database_filename = f'result/POSCARS_{crystal_system}_vasp_fixed.pkl'
        with open(vasp_fixed_relaxed_database_filename, 'rb') as vasp_fixed_relaxed_database:
            data = pickle.load(vasp_fixed_relaxed_database)
        
        for structure in data:
            atoms_copy = Atoms(
                symbols=structure['generated_structure'].symbols,
                pbc=structure['generated_structure'].pbc,
                cell=structure['generated_structure'].cell,
                positions=structure['generated_structure'].get_positions()
            )
            structure['generated_structure'] = atoms_copy

        with open(f'result/POSCARS_{crystal_system}_vasp_fixed_relaxed.pkl', 'wb') as f:
            pickle.dump(data, f)

        database_filename = f'result/POSCARS_{crystal_system}_vasp_fixed_relaxed.pkl'

        relaxation_params = get_calculation_params(relaxation_mode='vasp',
                                                   vasp_steps=vasp_steps)
        if pressure:
            update_incar_files(incars_directory=relaxation_params['incars_dir'], 
                               crystal_system=crystal_system, 
                               pstress_value=pressure)
        
        success = run_vasp(database_filename=database_filename, 
                           relaxation_params=relaxation_params, 
                           header=header, 
                           n_parallel_calcs=n_parallel_calcs)
        
        shutil.move("result", "result_vasp_unfixed")
        shutil.move("result_vasp_fixed", 'results/result_vasp_fixed')
        shutil.move("result_vasp_unfixed", 'results/result_vasp_unfixed')
        shutil.copy("results/result_vasp_unfixed/relaxation_results_all.pkl", "results/relaxation_results_summarized.pkl")
        shutil.copy("results/result_vasp_unfixed/relaxation_results_all.csv", "results/relaxation_results_summarized.csv")
        shutil.copy("EnvXGen/Postprocessing.ipynb", "results/Postprocessing.ipynb")

    elif relaxation_mode == 'vasp':

        relaxation_params = get_calculation_params(relaxation_mode=relaxation_mode,
                                                   vasp_steps=vasp_steps)
        if pressure:
            update_incar_files(incars_directory=relaxation_params['incars_dir'], 
                            crystal_system=crystal_system, 
                            pstress_value=pressure)
        
        success = run_vasp(database_filename=database_filename, 
                           relaxation_params=relaxation_params, 
                           header=header, 
                           n_parallel_calcs=n_parallel_calcs)
        
        shutil.move("result", "result_vasp_unfixed")
        shutil.move("result_vasp_unfixed", 'results/result_vasp_unfixed')
        shutil.copy("results/result_vasp_unfixed/relaxation_results_all.pkl", "results/relaxation_results_summarized.pkl")
        shutil.copy("results/result_vasp_unfixed/relaxation_results_all.csv", "results/relaxation_results_summarized.csv")
        shutil.copy("EnvXGen/Postprocessing.ipynb", "results/Postprocessing.ipynb")

    #elif relaxation_mode == 'vasp_fixed':
    #    relaxation_params = get_calculation_params(relaxation_mode=relaxation_mode,
    #                                               vasp_steps=vasp_steps)
    #    if pressure:
    #        update_incar_files(incars_directory=relaxation_params['incars_dir'], 
    #                        crystal_system=crystal_system, 
    #                        pstress_value=pressure)
    #    
    #    success = run_vasp(database_filename=database_filename, 
    #                       relaxation_params=relaxation_params, 
    #                       header=header, 
    #                       n_parallel_calcs=n_parallel_calcs)
        

    elif relaxation_mode == 'lammps':
        relaxation_params = get_calculation_params(relaxation_mode=relaxation_mode,
                                                   force_field_file_name=force_field_file_name)
        
        success = run_lammps(database_filename=database_filename, 
            relaxation_params=relaxation_params, 
            header=header, 
            n_parallel_calcs=n_parallel_calcs)
        
        shutil.move("result", "result_lammps")
        shutil.move("result_lammps", 'results/result_lammps')
        shutil.copy("results/results/result_lammps/relaxation_results_all.pkl", "results/relaxation_results_summarized.pkl")
        shutil.copy("results/results/result_lammps/relaxation_results_all.csv", "results/relaxation_results_summarized.csv")
        shutil.copy("EnvXGen/Postprocessing.ipynb", "results/Postprocessing.ipynb")
    #return 0

if __name__ == "__main__":
    main()