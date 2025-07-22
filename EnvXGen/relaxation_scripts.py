
from ase.io import read, write
from time import sleep, perf_counter
from concurrent.futures import ThreadPoolExecutor
from .prepare_calcfolds import get_atom_style_from_lammps
import pickle5 as pkl
import spglib as sp
import pandas as pd
import numpy as np

import os
import re
import sys
import shutil
import subprocess
import logging
import time


logger = logging.getLogger()

def concatenate_potcar_files(dat):

    first_structure = dat[0]['generated_structure']
    atom_types = first_structure.get_chemical_symbols()  
    unique_atom_types = sorted(set(atom_types), key=atom_types.index)

    potcar_files = []
    
    for atom_type in unique_atom_types:
        potcar_file = f'inputs/vasp/potcars/POTCAR_{atom_type}'
        if os.path.exists(potcar_file):
            potcar_files.append(potcar_file)
        else:
            logger.error(f"File {potcar_file} does not exist.")

    concatenated_potcar_path = os.path.join('inputs/vasp/potcars', 'POTCAR')
    
    if potcar_files:
        try:
            command = f"cat {' '.join(potcar_files)} > {concatenated_potcar_path}"
            subprocess.run(command, shell=True) 
            logger.info(f"Concatenated POTCAR files into {concatenated_potcar_path}.")
        except subprocess.CalledProcessError as e: 
            logger.error(f"Error during concatenation: {e}")
    else:
        logger.warning("No POTCAR files to concatenate.")


def update_incar_files(incars_directory, crystal_system, pstress_value):

    command = f'find "{incars_directory}" -maxdepth 1 -type f -name "INCAR_*" | wc -l'
    process = subprocess.run(command, shell=True, text=True, capture_output=True)
    n_incars = int(process.stdout.strip())
        
    for i in range(1, n_incars): 
        incar_filename = os.path.join(incars_directory, f'INCAR_{i}')

        if os.path.exists(incar_filename):
            with open(incar_filename, 'r') as incar_file:
                content = incar_file.readlines()

            for j in range(len(content)):
                if content[j].startswith("SYSTEM"):
                    content[j] = f"SYSTEM = {crystal_system}\n"
                elif content[j].startswith("PSTRESS"):
                    content[j] = f"PSTRESS = {int(pstress_value)}\n"

            with open(incar_filename, 'w') as incar_file:
                incar_file.writelines(content)

def write_lammps_coo(calc_fold, system):
    atom_style = get_atom_style_from_lammps()
    filename = f'{calc_fold}/coo.data'
    write(filename, system, format='lammps-data', atom_style=atom_style)


def get_vasp_energy(calc_fold):

    outcar_file = os.path.join(calc_fold, 'OUTCAR')

    if not os.path.exists(outcar_file):
        logger.error(f"OUTCAR file not found in directory: {calc_fold}")
        return None
    
    try:
        command = f"grep 'free  energy   TOTEN' {outcar_file} | tail -n 1 | sed 's/.*= //; s/ eV//'"
        last_free_energy = subprocess.check_output(command, shell=True).decode('utf-8').strip()

        if last_free_energy:
            return float(last_free_energy)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {e}")
        return None
            
    except Exception as e:
        logger.warning("ERROR: failed to extract energy value from OUTCAR file")
        return None
    
        
def get_vasp_initial_energy(calc_fold):

    outcar_file = os.path.join(calc_fold, 'OUTCAR')

    if not os.path.exists(outcar_file):
        logger.error(f"OUTCAR file not found in directory: {calc_fold}")
        return None
    
    try:
        command = f"grep -m 1 'TOTEN' {outcar_file} | sed 's/.*= //; s/ eV//'"
        initial_free_energy = subprocess.check_output(command, shell=True).decode('utf-8').strip()

        if initial_free_energy:
            return float(initial_free_energy)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {e}")
        return None
            
    except Exception as e:
        logger.warning("ERROR: failed to extract energy value from OUTCAR file")
        return None


def get_lammps_initial_energy(calc_fold):
    
    filename="lammps.out"

    try:
        with open(filename, 'r') as f:
            content = f.read()
        match = re.search(r"Energy initial,\s*next-to-last,\s*final =\s*(-?\d+\.\d+)", content)

        if match:
            initial_energy = float(match.group(1))
            return initial_energy
        else:
            logger.warning("ERROR: failed to extract energy value from OUTCAR file")
            return None
    
    except Exception as e:
        logger.warning("ERROR: failed to extract energy value from lammps.out file")
        return None


def get_symmetry_info(structure, symprec):

    lattice = structure.get_cell()
    scaled_positions = structure.get_scaled_positions()
    numbers = structure.get_atomic_numbers()
    cell = (lattice, scaled_positions, numbers)

    symmetry_info = sp.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=-1)

    return {'SG': symmetry_info['number'], 'international_symbol': symmetry_info['international']}


def extract_lammps_relaxed_structure(calc_fold):

    try:
        poscar_dir = os.path.join(calc_fold, "POSCAR")
        initial_structure = read(poscar_dir, format="vasp")
        atom_types_initial = initial_structure.get_chemical_symbols()

        relaxed_structure_dir = os.path.join(calc_fold, "out_thermo.atom")
        relaxed_structure = read(relaxed_structure_dir, format="lammps-dump-text", index=-1)
        atom_types_final = relaxed_structure.get_chemical_symbols()
        
        if len(atom_types_initial) != len(atom_types_final):
            logger.error(f"Warning! Number of atoms do not match ({len(atom_types_initial)} in POSCAR and {len(atom_types_final)} in out_thermo.atom)")

        relaxed_structure.set_chemical_symbols(atom_types_initial)
        
        write("POSCAR_relaxed", relaxed_structure, format="vasp")
    
    except Exception as e:
        logger.error(f"Extracting structure error: {e}")



class Calculator():
    
    def __init__(self, params, header, n_parallel_calcs, epoch):
        self.np = n_parallel_calcs
        self.params = params
        self.header = header
        self.epoch = epoch
        self.base_dir = '.'
        self.relaxation_dir = self.params['relaxationDirectory']
        self.structure_ids = []

        if not os.path.exists(self.relaxation_dir):
            os.makedirs(self.relaxation_dir)

    def test_run_single_calculation_lammps(self, structure, structure_id, index):
        calc_fold = os.path.join(self.relaxation_dir, f'CalcFold_{index + 1}')
        logger.info(f'Starting calculation for configuration with structure ID {structure_id} in CalcFold_{index + 1}')
        os.makedirs(calc_fold, exist_ok=True)

        # Copy POSCAR of generated structure to the CalcFolder
        try:
            write(f'{calc_fold}/POSCAR', structure, format='vasp')
            logger.debug('POSCAR written successfully')
            shutil.copy(os.path.join(calc_fold, 'POSCAR'), os.path.join(calc_fold, 'POSCAR_generated'))
        except Exception as e:
            logger.error(f'Cannot write POSCAR: {e}')
            return
        
        # Create coo.data file from POSCAR for lammps calculation
        write_lammps_coo(calc_fold, structure)

        # Copy lammps_in to the CalcFolder
        lammps_in_file = self.params['lammps_in']

        if os.path.exists(lammps_in_file):
            shutil.copy(lammps_in_file, calc_fold)
        else:
            logger.error(f"lammps.in file not found: {lammps_in_file}")
            logger.error("Terminating process due to missing lammps.in file.")
            sys.exit(1)

        # Copy force_field to the CalcFolder
        force_field_file = self.params['force_field']

        if os.path.exists(force_field_file):
            shutil.copy(force_field_file, calc_fold)
        else:
            logger.error(f"Force field file not found: {force_field_file}")
            logger.error("Terminating process due to missing Force field file.")
            sys.exit(1)
        
        # Write submittion script to the CalcFolder
        script_path = os.path.join(calc_fold, 'lammpsrun.sh')
        with open(script_path, 'w') as f:
            f.write(f'{self.header} \n#SBATCH -o output_lammps \n#SBATCH -e errors_lammps \n#SBATCH -J b{self.epoch}_cf{index + 1} \n')
            #f.write(f'#SBATCH -J e{self.epoch}_cf{index + 1} \n')
            f.write(self.params['commandExecutable'])
        
        warnings = []

        process = subprocess.Popen(['sbatch', 'lammpsrun.sh'], cwd=calc_fold, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            output = stdout.decode('utf-8')
            job_id_string = re.search(r'Submitted batch job (\d+)', output)
                
            if job_id_string:
                job_id = job_id_string.group(1)
            else:
                logger.error("Could not extract Job ID from sbatch output.")
                return
                
            logger.info(f'Submitted job with ID: {job_id} for configuration with structure ID {structure_id}')

            errors_lammps_path = os.path.join(calc_fold, 'errors_lammps')

            while True:
                result = subprocess.run(['squeue', '--job', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    logger.error(f"Error getting job info: {result.stderr.decode('utf-8')}")
                    break
                
                if not str(job_id) in str(result.stdout).split():
                    if os.path.exists(errors_lammps_path) and os.stat(errors_lammps_path).st_size > 0:
                        logger.info(f"Job {job_id} completed with Warnings")
                        with open(errors_lammps_path, 'r') as error_file:
                            error_content = error_file.read()
                            warnings.append(error_content)
                    else:
                        logger.info(f"Job {job_id} completed.")
                        warnings.append(None)
                
                    try:
                        initial_energy = get_lammps_initial_energy(calc_fold)
                        initial_energy_dir = os.path.join(calc_fold, 'generated_structure_energy')
                        with open(initial_energy_dir , 'w') as output_file:
                            output_file.write(f"Generated structure energy, eV: {initial_energy}\n")
                    except:
                        logger.error(f"Error initial energy for structure ID {structure_id}")

                    break

                logger.info(f"Job {job_id} is still running...")
                sleep(60)

            
    def test_run_single_calculation_vasp(self, structure, structure_id, index):

        calc_fold = os.path.join(self.relaxation_dir, f'CalcFold_{index + 1}')
        logger.info(f'Starting calculation for configuration with structure ID {structure_id} in CalcFold_{index + 1}')
        os.makedirs(calc_fold, exist_ok=True)

        # Copy POSCAR of generated structure to the CalcFolder
        try:
            write(f'{calc_fold}/POSCAR', structure, format='vasp')
            logger.debug('POSCAR written successfully')
            shutil.copy(os.path.join(calc_fold, 'POSCAR'), os.path.join(calc_fold, 'POSCAR_generated'))
        except Exception as e:
            logger.error(f'Cannot write POSCAR: {e}')
            return
        
        # Copy INCARS to the CalcFolder
        incars_dir = self.params['incars_dir']
        relaxation_steps = self.params['steps']

        for step in range(1, relaxation_steps + 1):
            incar_file = os.path.join(incars_dir, f'INCAR_{step}')
            if os.path.exists(incar_file):
                shutil.copy(incar_file, calc_fold)
            else:
                logger.error(f"INCAR file not found: {incar_file}")
                logger.error("Terminating process due to missing INCAR file.")
                sys.exit(1)

        # Copy POTCAR to the CalcFolder
        potcar_file = self.params['potcar']
        if os.path.exists(potcar_file):
            shutil.copy(potcar_file, calc_fold)
        else:
            logger.error(f"POTCAR file not found: {potcar_file}")
            logger.error("Terminating process due to missing POTCAR file.")
            sys.exit(1)

        # Write submittion script to the CalcFolder
        script_path = os.path.join(calc_fold, 'vasprun.sh')
        with open(script_path, 'w') as f:
            f.write(f'{self.header} \n#SBATCH -o output_vasp \n#SBATCH -e errors_vasp \n#SBATCH -J b{self.epoch}_cf{index + 1} \n')
            #f.write(f'#SBATCH -J e{self.epoch}_cf{index + 1} \n')
            f.write(self.params['commandExecutable'])

        subprocess.run(['chmod', '+x', script_path])

        logger.info(f'Relaxation steps: {relaxation_steps}')

        warnings = []

        for step in range(1, relaxation_steps + 1):

            if os.path.exists(os.path.join(calc_fold, 'CONTCAR')):
                os.rename(os.path.join(calc_fold, 'CONTCAR'), os.path.join(calc_fold, 'POSCAR'))

            for filename in os.listdir(calc_fold):
                if filename not in ['POSCAR', 'POSCAR_generated', 'POTCAR', 'vasprun.sh', 'generated_structure_energy'] and not filename.startswith('INCAR'):
                    os.remove(os.path.join(calc_fold, filename))

            incar_file = os.path.join(calc_fold, f'INCAR_{step}')
            logger.info(f"Calculating step {step} for {calc_fold}")
            
            if not os.path.exists(incar_file):
                logger.error(f"File {incar_file} not found.")
                continue

            shutil.copy(incar_file, os.path.join(calc_fold, 'INCAR'))

            process = subprocess.Popen(['sbatch', 'vasprun.sh'], cwd=calc_fold, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                output = stdout.decode('utf-8')
                job_id_string = re.search(r'Submitted batch job (\d+)', output)
                
                if job_id_string:
                    job_id = job_id_string.group(1)
                else:
                    logger.error("Could not extract Job ID from sbatch output.")
                    return
                
                logger.info(f'Submitted job with ID: {job_id} for configuration with structure ID {structure_id}, step {step}')

                errors_vasp_path = os.path.join(calc_fold, 'errors_vasp')

                while True:
                    result = subprocess.run(['squeue', '--job', job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode != 0:
                        logger.error(f"Error getting job info: {result.stderr.decode('utf-8')}")
                        break

                    if not str(job_id) in str(result.stdout).split():
                        if os.path.exists(errors_vasp_path) and os.stat(errors_vasp_path).st_size > 0:
                            logger.info(f"Job {job_id} completed with Warnings")
                            with open(errors_vasp_path, 'r') as error_file:
                                error_content = error_file.read()
                                warnings.append(error_content)
                        else:
                            logger.info(f"Job {job_id} completed.")
                            warnings.append(None)

                        if step == 1:
                            try:
                                initial_energy = get_vasp_initial_energy(calc_fold)
                                initial_energy_dir = os.path.join(calc_fold, 'generated_structure_energy')
                                with open(initial_energy_dir , 'w') as output_file:
                                    output_file.write(f"Generated structure energy, eV: {initial_energy}\n")
                            except:
                                logger.error(f"Error initial energy for structure ID {structure_id}")
                        break
                    
                    logger.info(f"Job {job_id} is still running...")
                    sleep(60)
        
        optimization_steps = list(range(1, relaxation_steps + 1))
        df_warnings = pd.DataFrame({
            'optimization_step': optimization_steps,
            'warnings': warnings
        })

        df_warnings.to_csv(os.path.join(calc_fold, 'warnings.csv'), index=False)
                    
        return True    


    def run_vasp_calculations_in_parallel(self, configurations):
        with ThreadPoolExecutor(max_workers=self.np) as executor:
            futures = [executor.submit(self.test_run_single_calculation_vasp, config['generated_structure'], config['ID'], i) for i, config in enumerate(configurations)]
            self.structure_ids = [config['ID'] for config in configurations]
            results = [future.result() for future in futures]
            return all(results)
    
    def run_lammps_calculations_in_parallel(self, configurations):
        with ThreadPoolExecutor(max_workers=self.np) as executor:
            futures = [executor.submit(self.test_run_single_calculation_lammps, config['generated_structure'], config['ID'], i) for i, config in enumerate(configurations)]
            self.structure_ids = [config['ID'] for config in configurations]
            results = [future.result() for future in futures]
            return all(results)

    def gather_results(self):
        result_dir = os.path.join(self.base_dir, 'result')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        result_list = []

        for i in range(1, self.np + 1):

            calc_fold = os.path.join(self.relaxation_dir, f'CalcFold_{i}')
            
            poscar_generated_path = os.path.join(calc_fold, 'POSCAR_generated')
            contcar_path = os.path.join(calc_fold, 'CONTCAR')
            warnings_path = os.path.join(calc_fold, 'warnings.csv')

            df_warnings = pd.read_csv(warnings_path)
            df_warnings = df_warnings.dropna()
            warnings = []
            for row in df_warnings.iloc():
                warnings.append(f"step {row['optimization_step']}: {row['warnings']}")

            poscar_atoms = read(poscar_generated_path) 
            poscar_symmetry_info = get_symmetry_info(poscar_atoms, symprec=0.1)
            poscar_sg = poscar_symmetry_info['SG']
            poscar_international_symbol = poscar_symmetry_info['international_symbol']
            poscar_volume = float(poscar_atoms.cell.volume)

            if warnings:

                energy = np.nan
                initial_energy = np.nan
                relaxed_atoms = np.nan
                relaxed_symmetry_info = np.nan
                relaxed_sg = np.nan
                relaxed_international_symbol = np.nan
                relaxed_volume = np.nan

            else:

                energy = get_vasp_energy(calc_fold)

                initial_energy_dir = os.path.join(calc_fold, 'generated_structure_energy')
                with open(initial_energy_dir, 'r') as file:
                    content = file.read()
                initial_energy = float(content.split(':')[-1].strip())

                try:
                    relaxed_atoms = read(contcar_path)
                    relaxed_symmetry_info = get_symmetry_info(relaxed_atoms, symprec=0.1)
                    relaxed_sg = relaxed_symmetry_info['SG']
                    relaxed_international_symbol = relaxed_symmetry_info['international_symbol']
                    relaxed_volume = float(relaxed_atoms.cell.volume)
                except Exception as e:
                    logger.error(f"Error reading {contcar_path}: {e}")
                    error_file_path = os.path.join(calc_fold, 'output_vasp')

                    try:
                        check_error = subprocess.run(['grep', '-q', 'BAD TERMINATION', error_file_path], check=False)
                        if check_error.returncode == 0:
                            logger.error(f"BAD TERMINATION VASP error")
                            return True
                    except FileNotFoundError:
                        logger.error(f"Can't identify error, file not found: {error_file_path}")
                        return False
                    
            result_dict = {
                'ID': f'ID-{self.structure_ids[i - 1]}',
                'batch': self.epoch,
                'CalcFold': i,
                'generated_structure_energy': initial_energy,
                'generated_structure_volume': poscar_volume,
                'generated_structure_SG': poscar_sg,
                'generated_structure_symbol': poscar_international_symbol,
                'generated_structure': poscar_atoms,
                'relaxed_structure_energy': energy,
                'relaxed_structure_volume': relaxed_volume,
                'relaxed_structure_SG': relaxed_sg,
                'relaxed_structure_symbol': relaxed_international_symbol,
                'relaxed_structure': relaxed_atoms,
                'warnings': warnings if warnings else np.nan,
                }

            result_list.append(result_dict)


            #if os.path.exists(os.path.join(calc_fold, 'errors_vasp')) and os.stat(os.path.join(calc_fold, 'errors_vasp')).st_size > 0:
            #    logger.warning(f"Errors occurred during relaxation for configuration {i}")

            #    poscar_atoms = read(poscar_generated_path) 
            #    poscar_symmetry_info = get_symmetry_info(poscar_atoms, symprec=0.1)
            #    poscar_sg = poscar_symmetry_info['SG']
            #    poscar_international_symbol = poscar_symmetry_info['international_symbol']
            #    poscar_volume = float(poscar_atoms.cell.volume)

            #    result_dict = {
            #            'ID': self.structure_ids[i - 1],
            #            'epoch': self.epoch,
            #            'CalcFold': i,
            #            'generated_structure_energy': np.nan,
            #            'generated_structure_volume': poscar_volume,
            #            'generated_structure_SG': poscar_sg,
            #            'generated_structure_symbol': poscar_international_symbol,
            #            'generated_structure': poscar_atoms,
            #            'relaxed_structure_energy': np.nan,
            #            'relaxed_structure_volume': np.nan,
            #            'relaxed_structure_SG': np.nan,
            #            'relaxed_structure_symbol': np.nan,
            #            'relaxed_structure': np.nan,
            #            'warnings': warnings,
            #        }
                
            #    result_list.append(result_dict)
                #continue

#            if os.path.exists(contcar_path) and os.stat(contcar_path).st_size > 0:

#                energy = get_vasp_energy(calc_fold)

#                initial_energy_dir = os.path.join(calc_fold, 'generated_structure_energy')
#                with open(initial_energy_dir, 'r') as file:
#                    content = file.read()
#                initial_energy = float(content.split(':')[-1].strip())

#                try:
#                    contcar_atoms = read(contcar_path)
#                    contcar_symmetry_info = get_symmetry_info(contcar_atoms, symprec=0.1)
#                    contcar_sg = contcar_symmetry_info['SG']
#                    contcar_international_symbol = contcar_symmetry_info['international_symbol']
#                    contcar_volume = float(contcar_atoms.cell.volume)

#                    poscar_atoms = read(poscar_generated_path) 
#                    poscar_symmetry_info = get_symmetry_info(poscar_atoms, symprec=0.1)
#                    poscar_sg = poscar_symmetry_info['SG']
#                    poscar_international_symbol = poscar_symmetry_info['international_symbol']
#                    poscar_volume = float(poscar_atoms.cell.volume)

#                    result_dict = {
#                        'ID': self.structure_ids[i - 1],
#                        'epoch': self.epoch,
#                        'CalcFold': i,
#                        'generated_structure_energy': initial_energy,
#                        'generated_structure_volume': poscar_volume,
#                        'generated_structure_SG': poscar_sg,
#                        'generated_structure_symbol': poscar_international_symbol,
#                        'generated_structure': poscar_atoms,
#                        'relaxed_structure_energy': energy,
#                        'relaxed_structure_volume': contcar_volume,
#                        'relaxed_structure_SG': contcar_sg,
#                        'relaxed_structure_symbol': contcar_international_symbol,
#                        'relaxed_structure': contcar_atoms,
#                        'warnings': warnings if warnings else np.nan,
#                    }

#                    result_list.append(result_dict)

#                except Exception as e:
#                    logger.error(f"Error reading {contcar_path}: {e}")
#                    error_file_path = os.path.join(calc_fold, 'output_vasp')

#                    try:
#                        check_error = subprocess.run(['grep', '-q', 'BAD TERMINATION', error_file_path], check=False)
#                        if check_error.returncode == 0:
#                            logger.error(f"BAD TERMINATION VASP error")
#                            return True
#                    except FileNotFoundError:
#                        logger.error(f"Can't identify error, file not found: {error_file_path}")
#                        return False
#                
#                for filename in os.listdir(calc_fold):
#                    if filename not in ['POTCAR', 'vasprun.sh'] and not filename.startswith('INCAR'):
#                        os.remove(os.path.join(calc_fold, filename))

        if result_list:
            result_file_path = os.path.join(result_dir, f'result_batch_{self.epoch}.pkl')
            with open(result_file_path, 'wb') as outfile:
                pkl.dump(result_list, outfile)
            logger.info(f'Successfully gathered results for batch {self.epoch}.')
        else:
            logger.warning(f'No valid results for batch {self.epoch}.')