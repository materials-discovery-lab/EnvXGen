import os
import glob
import logging

logger = logging.getLogger()

def get_max_encut(potcar_path):

    encuts = []

    with open(potcar_path, 'r') as potcar_file:
            for line in potcar_file:
                if line.startswith("   ENMAX"):
                    value = float(line.split()[2][:-1])
                    encuts.append(value)

    max_encut = max(encuts)
    return int(max_encut)


def update_incar_files(dir, crystal_system, pstress_value, max_encut):
    """
    Updates INCAR files for the given relaxation type.
    :param n_structures: number of structures to generate
    :param pressure: pressure in GPa
    :param crystal_system: crystal system name
    :param relaxation_mode: relaxation type (vasp_unfixed, vasp or lammps)
    """
        
    for i in range(1, 6): 
        incar_filename = os.path.join(dir, f'INCAR_{i}')

        if os.path.exists(incar_filename):
            with open(incar_filename, 'r') as incar_file:
                content = incar_file.readlines()

            # Обновляем значения SYSTEM, PSTRESS и ENCUT
            for j in range(len(content)):
                if content[j].startswith("SYSTEM"):
                    content[j] = f"SYSTEM = {crystal_system}\n"
                elif content[j].startswith("PSTRESS"):
                    content[j] = f"PSTRESS = {int(pstress_value)}\n"
                elif content[j].startswith("ENCUT"):
                    content[j] = f"ENCUT = {int(max_encut)}\n"

            with open(incar_filename, 'w') as incar_file:
                incar_file.writelines(content)
        

def prepare_relaxation_files(n_structures, pressure, crystal_system, relaxation_mode):

    directories = ['relaxation/vasp_fixed', 'relaxation/vasp_unfixed']

    if relaxation_mode == 'vasp':
        directories = ['relaxation/vasp_fixed', 'relaxation/vasp_unfixed']
        potcar_path = os.path.join(directories[0], 'POTCAR')
        max_encut = get_max_encut(potcar_path)
        pstress_value = pressure * 10
        update_incar_files(directories[0], crystal_system, pstress_value, max_encut)
        update_incar_files(directories[1], crystal_system, pstress_value, max_encut)

    elif relaxation_mode == 'vasp_unfixed':
        directories = ['relaxation/vasp_unfixed']
        potcar_path = os.path.join(directories[0], 'POTCAR')
        max_encut = get_max_encut(potcar_path)
        pstress_value = pressure * 10
        update_incar_files(directories[0], crystal_system, pstress_value, max_encut)


    elif relaxation_mode == 'lammps':
        directories = ['relaxation/lammps']

    else:
        logger.error(f"This stype of optimization does not holded")



def get_atom_style_from_lammps():
    """Reads the atom_style from lammps.in if available."""

    filename = 'inputs/lammps/lammps.in'
    
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('atom_style'):
                return line.split()[1].strip()
    
    return 'charge' # Default atom style