from .relaxation_scripts import Calculator, concatenate_potcar_files, write_lammps_coo

import os
import shutil
import logging
import pickle5 as pkl
import pandas as pd

from ase import Atoms


logger = logging.getLogger()


class System(Atoms):

    _newID = 0
    _isBad = False


    def __init__(self, **kwargs):
        super(System, self).__init__(**kwargs)
        self.ID = self.getNewID()
        self.setNewID(self.ID + 1)
        self.howCome = None

    @classmethod
    def fromAtoms(cls, atoms):
        cell = atoms.get_cell()
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_scaled_positions()
        system = cls(symbols=symbols, scaled_positions=positions, cell=cell, pbc=True)
        if atoms._calc is not None:
            system._calc = atoms._calc
            if 'stress' in atoms._calc.results:
                system._calc.results['stress'] *= -system.get_volume()
        if atoms.constraints:
            system.set_constraint(atoms.constraints)

        return system
    
    @classmethod
    def getNewID(cls):
        return cls._newID    
    
    @classmethod
    def setNewID(cls, ID):
        cls._newID = ID


def run_vasp(database_filename, relaxation_params, header, n_parallel_calcs):

    with open(database_filename, 'rb') as database:
        data = pkl.load(database)

    #if relaxation_params.relaxation_mode == 'vasp' or relaxation_params.relaxation_mode == 'vasp_fixed':
    concatenate_potcar_files(data)

    for i in range(len(data)):
        sys = System()
        atoms = data[i]['generated_structure']
        data[i]['generated_structure'] = sys.fromAtoms(atoms=atoms)
        #data[i].ID = data[i]['ID']

    total_structures = len(data)

    num_epochs = (total_structures + n_parallel_calcs - 1) // n_parallel_calcs
    logger.info(f"Total num batches: {num_epochs}.")

    calculator = Calculator(params=relaxation_params, header=header, n_parallel_calcs=n_parallel_calcs, epoch=0)

    for epoch in range(num_epochs):
        calculator.epoch = epoch

        for i in range(1, n_parallel_calcs + 1):
            calc_fold_path = os.path.join(calculator.relaxation_dir, f'CalcFold_{i}')
            if os.path.exists(calc_fold_path):
                shutil.rmtree(calc_fold_path)

        start_index = epoch * n_parallel_calcs
        end_index = min(start_index + n_parallel_calcs, total_structures)  
        
        if start_index >= total_structures:
            logger.warning(f"No structures available for batch {epoch}.")
            break

        current_data_subset = data[start_index:end_index]  
        calculator.np = len(current_data_subset)
        logger.info(f'Batch {epoch}: Processing structures from index {start_index} to {end_index - 1}.')

        try:
            success = calculator.run_vasp_calculations_in_parallel(current_data_subset)
            if success:
                calculator.gather_results()
            else:
                logger.warning(f"Error in gather results")
        except Exception as e:
            logger.error(f"Error during calculations: {e}")

    
    logger.info("Preparing to gather relaxation results.")
    relaxation_results = []

    #for filename in sorted(os.listdir(f"{calculator.relaxation_dir}/result")):
    for filename in sorted(os.listdir(f"result")):
        if filename.startswith('result_batch_') and filename.endswith('.pkl'):
            file_path = os.path.join(f"result", filename)
            epoch = filename.split('_')[2].split('.')[0]
            try:
                with open(file_path, 'rb') as f:
                    data = pkl.load(f)
                    relaxation_results.append(data)
            except Exception as e:
                logger.error(f"Error in loading results for batch {epoch}.")
    
    flat_list_of_dicts = [d for sublist in relaxation_results for d in sublist]
    df_all_results = pd.DataFrame(flat_list_of_dicts)
    df_sorted = df_all_results.sort_values(by='relaxed_structure_energy', ascending=True)

    df_sorted.to_pickle('result/relaxation_results_all.pkl')
    df_sorted = df_sorted.drop(columns=['generated_structure', 'relaxed_structure'])
    df_sorted.to_csv('result/relaxation_results_all.csv', index=False)

    #with open(os.path.join(calculator.relaxation_dir, f'result/relaxation_results_all.pkl'), 'wb') as f:
    #with open(os.path.join(f'result/relaxation_results_all.pkl'), 'wb') as f:
    #    pkl.dump(flat_list_of_dicts, f)

    logger.info(f"Relaxation done. Results saved")


##############

def run_lammps(database_filename, relaxation_params, header, n_parallel_calcs):

    with open(database_filename, 'rb') as database:
        data = pkl.load(database)

    for i in range(len(data)):
        sys = System()
        atoms = data[i]['generated_structure']
        data[i]['generated_structure'] = sys.fromAtoms(atoms=atoms)
        #data[i].ID = data[i]['ID']

    total_structures = len(data)

    num_epochs = (total_structures + n_parallel_calcs - 1) // n_parallel_calcs
    logger.info(f"Total num batches: {num_epochs}.")

    calculator = Calculator(params=relaxation_params, header=header, n_parallel_calcs=n_parallel_calcs, epoch=0)

    for epoch in range(num_epochs):
        calculator.epoch = epoch

        for i in range(1, n_parallel_calcs + 1):
            calc_fold_path = os.path.join(calculator.relaxation_dir, f'CalcFold_{i}')
            if os.path.exists(calc_fold_path):
                shutil.rmtree(calc_fold_path)

        start_index = epoch * n_parallel_calcs
        end_index = min(start_index + n_parallel_calcs, total_structures)  
        
        if start_index >= total_structures:
            logger.warning(f"No structures available for batch {epoch}.")
            break

        current_data_subset = data[start_index:end_index]  
        calculator.np = len(current_data_subset)
        logger.info(f'Batch {epoch}: Processing structures from index {start_index} to {end_index - 1}.')

        try:
            success = calculator.run_lammps_calculations_in_parallel(current_data_subset)
            if success:
                calculator.gather_results()
            else:
                logger.warning(f"Error in gather results")
        except Exception as e:
            logger.error(f"Error during calculations: {e}")

    
    logger.info("Preparing to gather relaxation results.")
    relaxation_results = []
