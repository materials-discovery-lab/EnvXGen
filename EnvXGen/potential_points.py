import numpy as np
import sys
from .interatomic_distances import distance_ij_cov
from .atomic_radii import get_default_cov_radii
from .environment import get_extended_cell, Environment
from .structural_elements import get_atomic_positions_by_type
import itertools as IT
import scipy.spatial as spatial
import scipy.spatial.distance as dist
from ase import Atoms
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, Delaunay
import random
import logging


logger = logging.getLogger()

def get_neighbors(positions_a, positions_b, cutoff):
    """ 
    Builds the KDTree using the *larger* positions array.

    1) Using list of potential point as positions_a
    and list of atomic positions from current environment as positions_b
    we can find potential points that do not satisfy the minimum 
    interatomic distances for adding an atom or molecule of a given type.
        
    2) Using extended added atoms or molecules as positions_a
    and list of atomic positions from current environment as positions_b
    we can check that the added structural element at a given potential 
    point satisfies the conditions of minimum interatomic distances.
    """

    tree = spatial.cKDTree(positions_a)
    groups = tree.query_ball_point(positions_b, cutoff)
    indices = np.unique(IT.chain.from_iterable(groups))

    return indices

def filter_potential_points(potential_points, indexes_to_drop):
    """ 
    Filters list of current potential points from points that do not satisfy 
    the minimum interatomic distances
    """
    
    #potential_points_list = list(map(list, potential_points))
    arr = np.array(potential_points, dtype='float')
    filtered_points = list(np.delete(arr, indexes_to_drop, 0))

    return filtered_points

def get_potential_points_by_index(potential_points, indexes_to_get):
    """ 
    Filters list of current potential points from points that do not satisfy 
    the minimum interatomic distances
    """
    
    #potential_points_list = list(map(list, potential_points))
    arr = np.array(potential_points, dtype='float')
    filtered_points = arr[indexes_to_get]

    return filtered_points
        

class PotentialPoints:
    """ 
    List of potential coordinates
    """


    def __init__(self, coordinates=[], n_bins=[]):
        self.coordinates = coordinates
        self.n_bins = n_bins
        

    def get_initial_points(self, lattice_vectors, n_bins):


        ijk0 = np.array([lattice_vectors[0] / (n_bins['n_bins_x'] + 1), 
                         lattice_vectors[1] / (n_bins['n_bins_y'] + 1),
                         lattice_vectors[2] / (n_bins['n_bins_z'] + 1)])
        
        i_cut = np.array([ijk0[0]])
        j_cut = np.array([ijk0[1]])
        k_cut = np.array([ijk0[2]])
        
        for i in range(0, n_bins['n_bins_x'] + 1):
            i_cut = np.vstack((i_cut, ijk0[0] * i))
            new_i_cut = np.delete(i_cut, 0, 0)
        for i in range(0, n_bins['n_bins_y'] + 1):
            j_cut = np.vstack([j_cut, ijk0[1] * i])
            new_j_cut = np.delete(j_cut, 0, 0)
        for i in range(0, n_bins['n_bins_z'] + 1):
            k_cut = np.vstack([k_cut, ijk0[2] * i])
            new_k_cut = np.delete(k_cut, 0, 0)

        points = []
        for i in new_i_cut:
            for j in new_j_cut:
                for k in new_k_cut:
                    points.append(i + j + k)

        points_list = list(map(list, points))

        self.coordinates = points_list
        self.n_bins = [n_bins['n_bins_x'], n_bins['n_bins_y'], n_bins['n_bins_z']]


    def get_points_for_adding_atom(self, current_environment, cov_radius):
        """ 
        Removes from the list of potential coordinates those that do not satisfy 
        the minimum interatomic distance between the Сurrent Environment 
        and the added atom of a given type
        """
    
        current_available_points = self.coordinates
        extended_cell = get_extended_cell(current_environment.cell)
        extended_cell_cov_radii = current_environment.cov_radii
        extended_cell_vdw_radii = current_environment.vdw_radii
        extended_environment = Environment(extended_cell, extended_cell_cov_radii, extended_cell_vdw_radii)

        for atom_type_from_environment in extended_environment.atom_types:

            atomic_positions = get_atomic_positions_by_type(extended_environment, atom_type_from_environment)
        ### ATTENTION TO THIS FORMULA!!! ###
            cutoff = (cov_radius + extended_environment.cov_radii[atom_type_from_environment]) / 2
            bad_indexes = list(set(get_neighbors(current_available_points, atomic_positions, cutoff)[0]))
            points_filtered = filter_potential_points(current_available_points, bad_indexes)
            #print(len(points_filtered))
            #if len(points_filtered) == 0:
                #print(f'Not enough potential points to add a structural element')
                #print(f'Increase the number of bins manually (current n_bins_a, n_bins_b, n_bins_c: {self.n_bins[0], self.n_bins[1], self.n_bins[2]})')
                #print(f'or decrease the r_cov/r_vdw manually')
                #raise ValueError(f"Potential points does not require interatomic distances")
            current_available_points = points_filtered
        
        return current_available_points
    
    def get_points_for_adding_atom_cluster(self, current_environment, cov_radius, down_multiplier):
        """ 
        Removes from the list of potential coordinates those that do not satisfy 
        the minimum interatomic distance between the Сurrent Environment 
        and the added atom of a given type
        """
    
        current_available_points = self.coordinates
        extended_cell = get_extended_cell(current_environment.cell)
        extended_cell_cov_radii = current_environment.cov_radii
        extended_cell_vdw_radii = current_environment.vdw_radii
        extended_environment = Environment(extended_cell, extended_cell_cov_radii, extended_cell_vdw_radii)


        for atom_type_from_environment in extended_environment.atom_types:

            atomic_positions = get_atomic_positions_by_type(extended_environment, atom_type_from_environment)
        ### ATTENTION TO THIS FORMULA!!! ###
            cutoff = down_multiplier * (cov_radius + extended_environment.cov_radii[atom_type_from_environment])
            bad_indexes = list(set(get_neighbors(current_available_points, atomic_positions, cutoff)[0]))
            points_filtered = filter_potential_points(current_available_points, bad_indexes)
            #print(len(points_filtered))
            #if len(points_filtered) == 0:
                #print(f'Not enough potential points to add a structural element')
                #print(f'Increase the number of bins manually (current n_bins_a, n_bins_b, n_bins_c: {self.n_bins[0], self.n_bins[1], self.n_bins[2]})')
                #print(f'or decrease the r_cov/r_vdw manually')
                #raise ValueError(f"Potential points does not require interatomic distances")
            current_available_points = points_filtered
        
        return current_available_points

    

    def check_point_for_adding_atom(self, point_for_adding_atom, current_environment, atom_type, cov_radius, vdw_radius):
        """ 
        Add atom of specific type in a potential point from list of potential 
        coordinates for current environment and atoms of this type. 
        Coordinate is chosen randomly and passed as an index
        """
    
        added_environment_ase = Atoms(symbols=atom_type,
                                positions=[point_for_adding_atom],
                                cell=current_environment.cell.cell[:],
                                pbc=True)

        added_environment = Environment(added_environment_ase, [cov_radius], [vdw_radius])

        extended_added_cell = get_extended_cell(added_environment.cell)
        current_environment_atom_types = current_environment.atom_types
        #current_environment_cov_radii = get_default_cov_radii(current_environment_atom_types)

        for atom_type_from_environment in current_environment.atom_types:

        #    print(f"Checking {atom_type_from_environment}")

            atomic_positions = get_atomic_positions_by_type(current_environment, atom_type_from_environment)
        ### ATTENTION TO THIS FORMULA!!! ###
            cutoff = (cov_radius + current_environment.cov_radii[atom_type_from_environment])

            # test_result -- list of indexes of atoms that causes errors
            test_result = list(set(get_neighbors(extended_added_cell.get_positions(), atomic_positions, cutoff)[0])) 

            if test_result:
                raise ValueError(f"Potential point does not require interatomic distances")
        
        return point_for_adding_atom
            


    def get_points_for_adding_molecule(self, current_environment, cov_radius):
        """ 
        Removes from the list of potential coordinates those that do not satisfy 
        the minimum interatomic distance between the Сurrent Environment 
        and the added molecule of a given type
        """
        
        current_available_points = self.coordinates
        extended_cell = get_extended_cell(current_environment.cell)
        #extended_cell_atom_types = list(set(extended_cell.get_chemical_symbols()))

        extended_cell_cov_radii = current_environment.cov_radii
        extended_cell_vdw_radii = current_environment.vdw_radii
        extended_environment = Environment(extended_cell, extended_cell_cov_radii, extended_cell_vdw_radii)

        for atom_type_from_environment in extended_environment.atom_types:

            atomic_positions = get_atomic_positions_by_type(extended_environment, atom_type_from_environment)
        ### ATTENTION TO THIS FORMULA!!! ###
            logger.info(f"extended_environment: {extended_environment}")
            #cutoff = cov_radius + extended_environment.cov_radii[atom_type_from_environment]
            cutoff = (cov_radius + extended_environment.vdw_radii[atom_type_from_environment]) / 2
            bad_indexes = list(set(get_neighbors(current_available_points, atomic_positions, cutoff)[0]))
            points_filtered = filter_potential_points(current_available_points, bad_indexes)
            if len(points_filtered) == 0:
                raise ValueError(f"Potential points does not require interatomic distances")

            current_available_points = points_filtered
        
        return current_available_points
    

def select_distant_points(potential_points, num_points, min_distance):
    """
    Selects random points from a list of coordinates in 3D space,
    that are the minimum distance from each other,
    using a KD tree to optimize the search.

    :param points: List of coordinates (list of tuples), e.g. [(x1, y1, z1), (x2, y2, z2), ...]
    :param num_points: Number of points to select
    :param min_distance: Minimum distance between selected points
    :return: List of selected points
    """

    if num_points > len(potential_points):
        raise ValueError("The number of requested points is greater than the number of points available in the list.")

    selected_points = []  

    tree = KDTree(potential_points)

    while len(selected_points) < num_points:
        candidate = random.choice(potential_points)

        if len(selected_points) == 0 or tree.query(candidate)[0] >= min_distance:
            selected_points.append(candidate)
            tree = KDTree(selected_points)

    return selected_points    
    

def random_rotation_matrix():
    """ 
    Generates a random rotation matrix in 3D 
    """

    theta = np.random.uniform(0, 2 * np.pi) 
    phi = np.random.uniform(0, 2 * np.pi)
    psi = np.random.uniform(0, 2 * np.pi)

    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(psi), -np.sin(psi)],
                   [0, np.sin(psi), np.cos(psi)]])

    R = Rz @ Ry @ Rx
    return R


def rotate_and_translate_molecule(point_for_adding_molecule, molecule_type, geometric_center):
    """
    Rotates molecule and translates it to chosen point

    :param molecule: объект класса Atoms из библиотеки ASE
    :param point: координаты точки вращения (x, y, z)
    :return: новая молекула с измененными координатами
    """

    R = random_rotation_matrix()
    positions = molecule_type.get_positions()

    shifted_positions = positions - geometric_center
    rotated_positions = shifted_positions @ R.T
    new_positions = rotated_positions + geometric_center

    translation_vector = point_for_adding_molecule - np.mean(new_positions, axis=0)
    final_positions = new_positions + translation_vector

    rotated_molecule = molecule_type.copy()
    rotated_molecule.set_positions(final_positions)

    return rotated_molecule


def check_rotation(current_environment, molecule_type, cov_radii_mol, vdw_radii_mol):
    """ 
    Add atom of specific type in a potential point from list of potential 
    coordinates for current environment and atoms of this type. 
    Coordinate is chosen randomly and passed as an index
    """
        
    added_environment_ase = Atoms(symbols=molecule_type.get_chemical_symbols(),
                            positions=molecule_type.get_positions(),
                            cell=current_environment.cell.cell[:],
                            pbc=True)

    added_environment = Environment(added_environment_ase, cov_radii_mol, vdw_radii_mol)

    extended_added_cell = get_extended_cell(added_environment.cell)
    current_environment_atom_types = current_environment.atom_types
    extended_added_environment = Environment(extended_added_cell, cov_radii_mol, vdw_radii_mol)


    for atom_type_from_current_environment in current_environment.atom_types:
        atomic_positions_current_env = get_atomic_positions_by_type(current_environment, atom_type_from_current_environment)
        #cutoff_current = current_environment.vdw_radii[atom_type_from_current_environment]

        if current_environment.vdw_radii:
            cutoff_current = current_environment.vdw_radii[atom_type_from_current_environment]
        else:
            cutoff_current = current_environment.cov_radii[atom_type_from_current_environment]

        for atom_type_from_extended_added_environment in extended_added_environment.atom_types:
            atomic_position_extended_env = get_atomic_positions_by_type(extended_added_environment, atom_type_from_extended_added_environment)
            cutoff_added = extended_added_environment.vdw_radii[atom_type_from_extended_added_environment]

            cutoff = (cutoff_current + cutoff_added) / 2
            test_result = list(set(get_neighbors(atomic_position_extended_env, atomic_positions_current_env, cutoff)[0]))

            if test_result:
                raise ValueError(f"Rotation does not require interatomic distances")
            
    return 0



def get_points_for_adding_atom_old(self, current_environment, atom_type, cov_radius):
    """ 
    Removes from the list of potential coordinates those that do not satisfy 
    the minimum interatomic distance between the Сurrent Environment 
    and the added atom of a given type  -- old slow version
    """

        #print(ij_cov_distances)
        #print(self.coordinates)
        #print(current_environment.cell.get_positions())

    dimensions = 3
    box = [current_environment.cell.cell[:][0][0], 
           current_environment.cell.cell[:][1][1], 
           current_environment.cell.cell[:][2][2]]

    current_available_points = self.coordinates

    for atom_type_from_environment in current_environment.atom_types:
        atomic_positions = get_atomic_positions_by_type(current_environment, atom_type_from_environment)
        pair_distances = np.empty((len(current_available_points), len(atomic_positions)))
        for i, point in enumerate(current_available_points):
            for j, atomic_position in enumerate(atomic_positions):
                dist_nd_sq = 0
                for dim in range(dimensions):
                    dist_1d = abs(atomic_position[dim] - point[dim])
                    if dist_1d > (box[dim] / 2):  # check if d_ij > box_size / 2
                        dist_1d = box[dim] - dist_1d
                    dist_nd_sq += dist_1d ** 2
                    pair_distances[i, j] = dist_nd_sq

        pair_distances = np.sqrt(pair_distances)
        r_cut = cov_radius + current_environment.cov_radii[atom_type_from_environment]

            #print(atom_type_from_environment)
            #print(r_cut)

        bad_indexes = []

        for index, i in enumerate(pair_distances):
            for j in i:
                if j < r_cut:
                    bad_indexes.append(index)
        bad_indexes = set(bad_indexes)

        points_filtered = []
        for i in range(len(current_available_points)):
            if i not in bad_indexes:
                points_filtered.append(current_available_points[i])

        if len(points_filtered) == len(current_available_points) - len(bad_indexes):
            logger.info('Test of potential coordinates sucessfully done')

        current_available_points = points_filtered

    return(current_available_points)


class Check_Convex_Hull:
    def __init__(self, initial_cell):
        self.atoms = initial_cell
        self.symbols = np.array((self.atoms.get_chemical_symbols()))
        self.hull = []
        self.points_hull = []
        self.hull_delaunay = None

    def create_ConvexHull(self, atom_type='all'):

        if atom_type == 'all':
            self.points_hull = self.atoms.get_positions()
        else:
            self.points_hull = self.atoms.get_positions()[self.symbols == atom_type]

        self.hull = ConvexHull(self.points_hull)

    def check_in_hull(self, points):

        self.hull_delaunay = Delaunay(self.points_hull[self.hull.vertices])

        indexes_inside_hull = []

        for i, point_to_check in enumerate(points):
            if self.hull_delaunay.find_simplex(point_to_check) > 0:
                indexes_inside_hull.append(i)

        return indexes_inside_hull