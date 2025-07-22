def get_header(HEADER_dict):

    sbatch_lines = []
    
    for key, value in HEADER_dict.items():
        sbatch_lines.append(f"{key}={value}")
    
    header_sbatch_string = "\n".join(f"#{line}" for line in sbatch_lines)
    header_sbatch_string = f"#!/bin/sh \n{header_sbatch_string}"
    
    return header_sbatch_string


def get_calculation_params(relaxation_mode, force_field_file_name=None, vasp_steps=None):

    if relaxation_mode == 'lammps':
        force_field_file_name = force_field_file_name
        relaxation_params = {
            'type' : 'lammps',
            'lammps_in': 'inputs/lammps/lammps.in',
            'force_field': 'inputs/lammps/' + force_field_file_name,
            'relaxationDirectory' : 'relaxation_lammps',
            'commandExecutable': '\nmodule purge\nmodule load mpi/impi-5.0.3 lammps/lammps-201822\n\nmpirun lammps < lammps.in'
            }
    elif relaxation_mode == 'vasp':
        vasp_steps = vasp_steps
        relaxation_params = {
            'type' : 'vasp',
            'steps': vasp_steps,
            'incars_dir': 'inputs/vasp/incars_unfixed',
            'potcar': 'inputs/vasp/potcars/POTCAR',
            'relaxationDirectory' : 'relaxation_vasp',
            'commandExecutable': '\nmodule purge\nmodule load compilers/intel-2020 vasp/6.1.0\n\nmpirun vasp_std'
        }
    elif relaxation_mode == 'vasp_fixed':
        vasp_steps = vasp_steps
        relaxation_params = {
            'type' : 'vasp_fixed',
            'steps': vasp_steps,
            'incars_dir': 'inputs/vasp/incars_fixed',
            'potcar': 'inputs/vasp/potcars/POTCAR',
            'relaxationDirectory' : 'relaxation_vasp_fixed',
            'commandExecutable': '\nmodule purge\nmodule load compilers/intel-2020 vasp/6.1.0\n\nmpirun vasp_std'
        }
    
    return relaxation_params