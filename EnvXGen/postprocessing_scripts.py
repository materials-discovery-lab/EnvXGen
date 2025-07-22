import pickle as pkl
import pandas as pd
from matminer.featurizers.structure import RadialDistributionFunction
from pymatgen.io.ase import AseAtomsAdaptor
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ase.io import read
from tqdm import tqdm
import sys
import os

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath("../EnvXGen"))
#% run modules.ipynb

from structural_elements import get_initial_indices
import umap


# ---------------------------------------------------------------------------------------
# Portions of this code are adapted from the ALIGNN (Atomistic LIne Graph Neural Network)
# project: https://github.com/usnistgov/alignn
#
# Specifically, this code reuses and adapts functionality related to:
#   - Computation of ALIGNN descriptors
#
# ALIGNN is distributed under the Apache License 2.0.
# If you use this code or ALIGNN in your work, please cite:
#
#   Choudhary, K., & DeCost, B. (2021).
#   "Atomistic Line Graph Neural Network for improved materials property predictions"
#   npj Computational Materials 7, 185.
#   DOI: https://doi.org/10.1038/s41524-021-00650-1
#
# The original code has been modified for use in this project.
# ---------------------------------------------------------------------------------------


def calculate_rdf(ase_structure, cutoff, bin_size):
    """
    Computes the Radial Distribution Function (RDF) for a given structure.

    :param ase_structure: Structure in ASE (ase.Atoms) format
    :return: Array of RDF values
    """
    pymatgen_structure = AseAtomsAdaptor.get_structure(ase_structure)
    rdf_values = RadialDistributionFunction(cutoff, bin_size).featurize(pymatgen_structure)
    
    return rdf_values


def calculate_alignn(ase_structure, model, cutoff=8, device="cpu"):
    symbols = ase_structure.get_chemical_symbols()
    coords = ase_structure.get_positions()
    cell = ase_structure.get_cell()

    jarvis_atoms = JarvisAtoms(
        elements=symbols,
        coords=coords.tolist(),
        lattice_mat=cell.tolist(),
    )

    g, lg = Graph.atom_dgl_multigraph(jarvis_atoms, cutoff=float(cutoff))
    h = model([g.to(device), lg.to(device)])

    return h

def calculate_alignn_batch(structures_batch, model, cutoff=8, device="cpu"):
    """
    Optimized batch calculation of ALIGNN descriptors.
    
    :param structures_batch: List of ASE structures
    :param model: ALIGNN model
    :param cutoff: Cutoff distance
    :param device: Computing device
    :return: List of descriptor arrays
    """

    model.eval()
    descriptors = []
    
    with torch.no_grad():
        for ase_structure in structures_batch:
            try:
                symbols = ase_structure.get_chemical_symbols()
                coords = ase_structure.get_positions()
                cell = ase_structure.get_cell()

                jarvis_atoms = JarvisAtoms(
                    elements=symbols,
                    coords=coords.tolist(),
                    lattice_mat=cell.tolist(),
                )

                g, lg = Graph.atom_dgl_multigraph(jarvis_atoms, cutoff=float(cutoff))
                h = model([g.to(device), lg.to(device)])
                descriptors.append(h.detach().cpu().numpy()[0])
            except Exception as e:
                print(f"Error calculating ALIGNN descriptor: {e}")
                descriptors.append(None)
    
    return descriptors


def calculate_single_rdf(args):
    """
    Helper function for parallel RDF calculation.
    """

    ase_structure, cutoff, bin_size = args
    return calculate_rdf(ase_structure, cutoff, bin_size)


def find_optimal_clusters(X, min_clusters=2, max_clusters=None):
    """
    Function to determine the optimal number of clusters using the silhouette score.
    It also generates a plot showing the average silhouette score for different cluster counts.

    :param X: Data for clustering (feature vector or DataFrame)
    :param min_clusters: Minimum number of clusters to evaluate
    :param max_clusters: Maximum number of clusters to evaluate (if not specified, defaults to len(X) - 1)
    :return: Optimal number of clusters and a generated plot
    """

    if max_clusters is None:
        max_clusters = len(X) - 1
    
    silhouette_scores = []
    optimal_n_clusters = min_clusters

    for n_clusters in range(min_clusters, max_clusters + 1):

        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_n_clusters = np.argmax(silhouette_scores) + min_clusters  

    plt.figure(figsize=(8, 5))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', color='b', label='Silhouette Score')
    plt.axvline(x=optimal_n_clusters, color='r', linestyle='--', label=f'Optimal n_clusters = {optimal_n_clusters}')
    plt.title("Silhouette Analysis for Optimal Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Average Silhouette Score")
    plt.xticks(range(min_clusters, max_clusters + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_n_clusters


def reduce_dimensionality_pca(preprocessed_data):
    """
    Reduces the dimensionality of the given data using PCA and retains the ID column.
    
    :param preprocessed_data: Input DataFrame with an 'ID' column and high-dimensional features.
    :return: DataFrame with 'ID', 'PCA_2D_x', 'PCA_2D_y', 'PCA_3D_x', 'PCA_3D_y', 'PCA_3D_z'.
    """

    ids = preprocessed_data["ID"]
    features = preprocessed_data.drop(columns=["ID"])

    pca_2d_reducer = PCA(n_components=2, svd_solver='full')
    pca_2d = pca_2d_reducer.fit_transform(features)

    pca_3d_reducer = PCA(n_components=3, svd_solver='full')
    pca_3d = pca_3d_reducer.fit_transform(features)

    pca_results = pd.DataFrame({
        "ID": ids,
        "PCA_2D_x": pca_2d[:, 0],
        "PCA_2D_y": pca_2d[:, 1],
        "PCA_3D_x": pca_3d[:, 0],
        "PCA_3D_y": pca_3d[:, 1],
        "PCA_3D_z": pca_3d[:, 2]
    })

    return pca_results


def reduce_dimensionality_umap(preprocessed_data, 
                               n_neighbors=10):
    """
    Reduces the dimensionality of the given data using UMAP and retains the ID column.

    :param preprocessed_data: Input DataFrame with an 'ID' column and high-dimensional features.
    :param n_neighbors: The number of neighbors considered in UMAP.
    :return: DataFrame with 'ID', 'UMAP_2D_x', 'UMAP_2D_y', 'UMAP_3D_x', 'UMAP_3D_y', 'UMAP_3D_z'.
    """

    ids = preprocessed_data["ID"]
    features = preprocessed_data.drop(columns=["ID"])

    umap_2d_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.0,
        n_components=2,
        metric='euclidean'
    )
    umap_2d = umap_2d_reducer.fit_transform(features)

    umap_3d_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.0,
        n_components=3,
        metric='euclidean'
    )
    umap_3d = umap_3d_reducer.fit_transform(features)

    umap_results = pd.DataFrame({
        "ID": ids,
        "UMAP_2D_x": umap_2d[:, 0],
        "UMAP_2D_y": umap_2d[:, 1],
        "UMAP_3D_x": umap_3d[:, 0],
        "UMAP_3D_y": umap_3d[:, 1],
        "UMAP_3D_z": umap_3d[:, 2]
    })

    return umap_results


def calculate_descriptors(descriptor_algorithm, 
                         poscar_init_path='../POSCAR_init', 
                         database_filename='relaxation_results_summarized.pkl',
                         device='auto',
                         batch_size=32,
                         n_jobs=-1):
    """
    Optimized descriptor calculation with device selection and parallelization.

    :param descriptor_algorithm: Name of the descriptor calculation algorithm ('RDF' or 'ALIGNN')
    :param poscar_init_path: Path to the POSCAR_init file
    :param database_filename: Path to the file containing the results with saved structures
    :param device: Computing device ('cpu', 'cuda', 'auto')
    :param batch_size: Batch size for ALIGNN calculations
    :param n_jobs: Number of parallel jobs (-1 for all cores)
    """
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    print(f"Using {n_jobs} parallel processes")

    poscar_init = read(poscar_init_path, format="vasp")

    with open(database_filename, 'rb') as database:
        data = pkl.load(database)
    df_results = pd.DataFrame(data)

    descriptors_folder = "descriptors"
    subfolder = f'{descriptor_algorithm}'
    
    os.makedirs(descriptors_folder, exist_ok=True)
    os.makedirs(os.path.join(descriptors_folder, subfolder), exist_ok=True)

    init_indices = get_initial_indices(poscar_init, df_results['relaxed_structure'][0])

    if not init_indices:
        print("Warning! Please check the path to the POSCAR_init file. It may be incorrect.")
        return

    model = None
    rdf_cutoff = 10
    rdf_bin_size = 0.065

    if descriptor_algorithm == "ALIGNN":
        print("Loading ALIGNN model...")
        
        target_dir = os.path.join("..", "EnvXGen", "models")
        os.makedirs(target_dir, exist_ok=True)

        model_path = "../EnvXGen/models/mp_e_form_alignnn/checkpoint_300.pt"
        
        if not os.path.exists(model_path):
            import requests
            import zipfile
            
            zip_url = "https://ndownloader.figshare.com/files/31458811"
            zip_path = os.path.join(target_dir, "model.zip")

            response = requests.get(zip_url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)

            os.remove(zip_path)
            print("Model downloaded and unpacked successfully")

        model = ALIGNN()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

    print("Calculating descriptor for POSCAR_init...")
    if descriptor_algorithm == "RDF":
        poscar_init_descriptor = calculate_rdf(poscar_init, rdf_cutoff,rdf_bin_size)
    elif descriptor_algorithm == "ALIGNN":
        poscar_init_descriptor = calculate_alignn_batch([poscar_init], model, cutoff=8, device=device)[0]
    else:
        raise ValueError(f"Descriptor algorithm {descriptor_algorithm} not recognized.")
    
    if poscar_init_descriptor is not None:
        df_poscar_init_descriptor = pd.DataFrame([poscar_init_descriptor])
        df_poscar_init_descriptor.to_csv(
            os.path.join(descriptors_folder, subfolder, 'POSCAR_init.csv'), 
            index=False
        )

    def calculate_and_store_parallel(structures_list, ids_list, descriptor_algorithm, filename):
        """Optimized parallel calculation and storage."""
        print(f'Calculating descriptors for {filename}...')
        
        descriptors_list = []
        
        if descriptor_algorithm == "RDF":
            args_list = [(structure, rdf_cutoff, rdf_bin_size) for structure in structures_list]
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                descriptors = list(tqdm(
                    executor.map(calculate_single_rdf, args_list),
                    total=len(args_list),
                    desc=f"Processing {filename}"
                ))
            
            for i, descriptor_values in enumerate(descriptors):
                if descriptor_values is not None:
                    descriptors_list.append({
                        'ID': ids_list[i], 
                        'descriptor': descriptor_values 
                    })
        
        elif descriptor_algorithm == "ALIGNN":
            pbar = tqdm(range(0, len(structures_list), batch_size), desc=f"Processing {filename}")
            
            for i in pbar:
                batch_idx = i // batch_size + 1
                total_batches = (len(structures_list) + batch_size - 1) // batch_size

                batch = structures_list[i:i+batch_size]
                batch_ids = ids_list[i:i+batch_size]

                pbar.set_postfix(batch=f"{batch_idx}/{total_batches}")

                batch_descriptors = calculate_alignn_batch(batch, model, cutoff=8, device=device)

                for j, descriptor_values in enumerate(batch_descriptors):
                    if descriptor_values is not None:
                        descriptors_list.append({
                            'ID': batch_ids[j], 
                            'descriptor': descriptor_values 
                        })
        
        if descriptors_list:
            df_descriptors = pd.DataFrame(descriptors_list)
            if len(df_descriptors) > 0 and 'descriptor' in df_descriptors.columns:
                descriptor_columns = [f"feature_{j}" for j in range(len(df_descriptors['descriptor'].iloc[0]))]
                descriptor_df = pd.DataFrame(df_descriptors['descriptor'].to_list(), columns=descriptor_columns)
                df_descriptors = pd.concat([df_descriptors[['ID']], descriptor_df], axis=1)
                df_descriptors.to_csv(
                    os.path.join(descriptors_folder, subfolder, filename), 
                    index=False
                )
                print(f'{descriptor_algorithm} descriptors for {filename} saved successfully\n')

    print('Processing generated structures...')
    calculate_and_store_parallel(
        df_results['generated_structure'].tolist(),
        df_results['ID'].tolist(),
        descriptor_algorithm,
        'generated_structures.csv'
    )

    print('Processing relaxed structures...')
    calculate_and_store_parallel(
        df_results['relaxed_structure'].tolist(),
        df_results['ID'].tolist(),
        descriptor_algorithm,
        'relaxed_structures.csv'
    )

    print('Processing initial atoms in relaxed structures and calculating similarities...')
    descriptors_list = []
    similarities = []
    
    relaxed_structures = df_results['relaxed_structure'].tolist()
    ids = df_results['ID'].tolist()
    
    if descriptor_algorithm == "RDF":

        for i, relaxed_structure in tqdm(enumerate(relaxed_structures), 
                                        total=len(relaxed_structures),
                                        desc="Processing relaxed structures with init indices"):
            
            filtered_structure = relaxed_structure[init_indices]
            descriptor_values = calculate_rdf(filtered_structure, rdf_cutoff, rdf_bin_size)
            
            if descriptor_values is not None and poscar_init_descriptor is not None:
                cos_sim = cosine_similarity([poscar_init_descriptor], [descriptor_values])[0][0]
                
                descriptors_list.append({
                    'ID': ids[i], 
                    'descriptor': descriptor_values
                })
                
                similarities.append({
                    'ID': ids[i],
                    'cosine_similarity_with_init': cos_sim
                })
    
    elif descriptor_algorithm == "ALIGNN":

        for i in tqdm(range(0, len(relaxed_structures), batch_size),
                     desc="Processing relaxed structures with init indices"):
            
            batch = relaxed_structures[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            filtered_batch = [structure[init_indices] for structure in batch]
            
            batch_descriptors = calculate_alignn_batch(filtered_batch, model, cutoff=8, device=device)
            
            for j, descriptor_values in enumerate(batch_descriptors):
                if descriptor_values is not None and poscar_init_descriptor is not None:
                    cos_sim = cosine_similarity([poscar_init_descriptor], [descriptor_values])[0][0]
                    
                    descriptors_list.append({
                        'ID': batch_ids[j], 
                        'descriptor': descriptor_values
                    })
                    
                    similarities.append({
                        'ID': batch_ids[j],
                        'cosine_similarity_with_init': cos_sim
                    })

    if similarities:
        df_similarities = pd.DataFrame(similarities)
        df_similarities.to_csv(
            os.path.join(descriptors_folder, subfolder, 'relaxed_structures_similarities.csv'), 
            index=False
        )

    if descriptors_list:
        df_descriptors = pd.DataFrame(descriptors_list)
        if len(df_descriptors) > 0 and 'descriptor' in df_descriptors.columns:
            descriptor_columns = [f"feature_{j}" for j in range(len(df_descriptors['descriptor'].iloc[0]))]
            descriptor_df = pd.DataFrame(df_descriptors['descriptor'].to_list(), columns=descriptor_columns)
            df_descriptors = pd.concat([df_descriptors[['ID']], descriptor_df], axis=1)
            df_descriptors.to_csv(
                os.path.join(descriptors_folder, subfolder, 'relaxed_structures_init_atoms.csv'), 
                index=False
            )
            print(f'{descriptor_algorithm} descriptors for initial atoms in relaxed structures saved successfully')

    print("Descriptor calculation completed!")


def reducing_descriptors(descriptor_algorithm, 
                         reducer_algorithm,
                         poscar_init_path='../POSCAR_init', 
                         #database_filename='relaxation_results_summarized.pkl'
                        ):
    """
    Computes descriptors for structures using the specified algorithm and saves the results.

    :param descriptor_algorithm: Name of the descriptor algorithm (e.g., 'RDF' or 'ALIGNN')
    :param poscar_init_path: Path to the POSCAR_init file
    :param database_filename: Path to the file containing saved structure calculation results
    """

    #def prepare_scaled_features(csv_path, scaler):
    #    df = pd.read_csv(csv_path)
    #    ids = df["ID"]
    #    features_scaled = scaler.fit_transform(df.drop(columns=["ID"]))
    #    df_scaled = pd.DataFrame(features_scaled, columns=df.columns[1:])
    #    df_scaled.insert(0, "ID", ids)
    #    return df_scaled

    #descriptor_csv_paths = {
    #    "relaxed": f"descriptors/{descriptor_algorithm}/relaxed_structures.csv",
    #    "generated": f"descriptors/{descriptor_algorithm}/generated_structures.csv",
        #"relaxed_init": f"descriptors/{descriptor_algorithm}/relaxed_structures_init_atoms.csv"
    #}

    #scaler = StandardScaler()

    #scaled_data = {
    #    key: prepare_scaled_features(path, scaler)
    #    for key, path in descriptor_csv_paths.items()
    #}

    ### postprocessing of relaxed structures descriptors
    #### Scaling
    scaler = StandardScaler()

    file_path = f"descriptors/{descriptor_algorithm}/relaxed_structures.csv"
    df_relaxed = pd.read_csv(file_path)
    relaxed_ids = df_relaxed["ID"]
    relaxed_str_desc_scaled = scaler.fit_transform(df_relaxed.drop(columns=["ID"]))
    df_relaxed_scaled = pd.DataFrame(relaxed_str_desc_scaled, columns=df_relaxed.columns[1:])
    df_relaxed_scaled.insert(0, "ID", relaxed_ids)

    ### postprocessing of generated structures descriptors
    #### Scaling

    file_path = f"descriptors/{descriptor_algorithm}/generated_structures.csv"
    df_generated = pd.read_csv(file_path)
    generated_ids = df_generated["ID"]
    generated_str_desc_scaled = scaler.fit_transform(df_generated.drop(columns=["ID"]))
    df_generated_scaled = pd.DataFrame(generated_str_desc_scaled, columns=df_generated.columns[1:])
    df_generated_scaled.insert(0, "ID", generated_ids)

    ### postprocessing of relaxed structures with init atoms descriptors
    #### Scaling

    #file_path = f"descriptors/{descriptor_algorithm}/relaxed_structures_init_atoms.csv"
    #df_relaxed_init = pd.read_csv(file_path)
    #relaxed_init_ids = df_relaxed_init["ID"]
    #relaxed_str_init_desc_scaled = scaler.fit_transform(df_relaxed_init.drop(columns=["ID"]))
    #df_relaxed_init_scaled = pd.DataFrame(relaxed_str_init_desc_scaled, columns=df_relaxed_init.columns[1:])
    #df_relaxed_init_scaled.insert(0, "ID", relaxed_init_ids)

    if reducer_algorithm == "PCA":
        print('Reducing data')
        df_generated_reduced = reduce_dimensionality_pca(df_generated_scaled)
        df_relaxed_reduced = reduce_dimensionality_pca(df_relaxed_scaled)
        #df_relaxed_init_reduced = reduce_dimensionality_pca(df_relaxed_init_scaled)
    elif reducer_algorithm == "UMAP":
        print('Reducing data')
        df_generated_reduced = reduce_dimensionality_umap(df_generated_scaled, 5)
        df_relaxed_reduced = reduce_dimensionality_umap(df_relaxed_scaled, 5)
        #df_relaxed_init_reduced = reduce_dimensionality_umap(df_relaxed_init_scaled)
    else:
        raise ValueError(f"Reducer algorithm {reducer_algorithm} not recognized.")
    
    df_generated_reduced.to_csv(f"descriptors/{descriptor_algorithm}/generated_structures_{reducer_algorithm}.csv", index=False)
    df_relaxed_reduced.to_csv(f"descriptors/{descriptor_algorithm}/relaxed_structures_{reducer_algorithm}.csv", index=False)

    #print(df_generated_reduced)
    #print(df_relaxed_reduced)


"""Shared pydantic settings configuration."""
import json
from pathlib import Path
from typing import Union
#import matplotlib.pyplot as plt

from pydantic_settings import BaseSettings as PydanticBaseSettings


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"


def plot_learning_curve(
    results_dir: Union[str, Path], key: str = "mae", plot_train: bool = False
):
    """Plot learning curves based on json history files."""
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    with open(results_dir / "history_val.json", "r") as f:
        val = json.load(f)

    p = plt.plot(val[key], label=results_dir.name)

    if plot_train:
        # plot the training trace in the same color, lower opacity
        with open(results_dir / "history_train.json", "r") as f:
            train = json.load(f)

        c = p[0].get_color()
        plt.plot(train[key], alpha=0.5, c=c)

    plt.xlabel("epochs")
    plt.ylabel(key)

    return train, val

"""Shared model-building components."""
from typing import Optional

import torch
from torch import nn


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
from dgl.nn import AvgPooling

# from dgl.nn.functional import edge_softmax
#from pydantic.typing import Literal
from typing import Literal
from torch import nn
from torch.nn import functional as F

#from alignn.models.utils import RBFExpansion
#from alignn.utils import BaseSettings

from alignn.pretrained import get_all_models, get_figshare_model

#import os
import requests
import zipfile
#import time

from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.core.graphs import Graph


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    num_classes: int = 2

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()


        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self, in_features: int, out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        x, m = self.node_update(g, x, y)
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features,),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(config.hidden_features, config.hidden_features,)
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return h


#def get_all_models():
#    """Return the figshare links for models."""
#    return all_models


def get_figshare_model(model_name="jv_formation_energy_peratom_alignn"):
    """Get ALIGNN torch models from figshare."""

    model = ALIGNN(
        ALIGNNConfig(name="alignn", output_features=1)
    )

    device='cpu'

    model.load_state_dict(torch.load(model_name, map_location=device)["model"])
    model.to(device)
    model.eval()

    return model


def get_prediction(
    model_name="jv_formation_energy_peratom_alignn",
    atoms=None,
    cutoff=8,
):
    """Get model prediction on a single structure."""
    model = get_figshare_model(model_name)
    # print("Loading completed.")
    g, lg = Graph.atom_dgl_multigraph(atoms, cutoff=float(cutoff))
    device='cpu'
    out_data = (
        model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )
    return out_data


def find_different_optimal_structures(database_filename, descriptor_algorithm, k=5):
    """
    Determines the optimal number of clusters using silhouette_score,
    groups structures by clusters, and returns the top-k most energy-efficient
    structures from each cluster, sorted by increasing energy.

    :param database_filename: name of the .pkl file containing calculation results
    :param descriptor_algorithm: descriptor algorithm used ('RDF', 'ALIGNN', etc.)
    :param k: number of best (lowest-energy) structures to select from each cluster
    :return: DataFrame with the top-k structures from each cluster, sorted by energy
    """

    file_path = f"descriptors/{descriptor_algorithm}/relaxed_structures.csv"
    df_embeddings = pd.read_csv(file_path)
    X = df_embeddings.drop(columns=["ID"]).values
    X_scaled = StandardScaler().fit_transform(X)

    min_clusters = 2
    max_clusters = min(20, len(X_scaled) - 1)

    silhouette_scores = {}

    for n_clusters in range(min_clusters, max_clusters + 1):
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        cluster_labels = clusterer.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores[n_clusters] = score

    optimal_n_clusters = max(silhouette_scores, key=silhouette_scores.get)

    clusterer = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='complete')
    cluster_labels = clusterer.fit_predict(X_scaled)

    with open(database_filename, 'rb') as f:
        data = pkl.load(f)

    df_results = pd.DataFrame(data)
    df_results = df_results.copy()
    df_results.insert(1, 'cluster', cluster_labels)

    top_structures = []
    for cluster_id, group in df_results.groupby('cluster'):
        sorted_group = group.sort_values(by='relaxed_structure_energy')
        top_k = sorted_group.head(k)
        top_structures.append(top_k)

    df_top_all = pd.concat(top_structures)
    df_top_all['cluster_min_energy'] = df_top_all.groupby('cluster')['relaxed_structure_energy'].transform('min')
    df_top_all = df_top_all.sort_values(by=['cluster_min_energy', 'cluster', 'relaxed_structure_energy']).drop(columns='cluster_min_energy')

    df_top_all = df_top_all.reset_index(drop=True)
    df_top_all.to_pickle("different_optimal_structures.pkl")

    return df_top_all


def format_filename(descriptor_algorithm, reducer_algorithm, reduced_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation):
    parts = [
        descriptor_algorithm,
        reducer_algorithm,
        f"{str(reduced_dimensionality)}D",
    ]

    if include_enthalpy and reduced_dimensionality == 2:
        parts.append("enth")

    if structures_before_relaxation and not structures_after_relaxation:
        parts.append("before_relaxation")
    elif structures_after_relaxation and not structures_before_relaxation:
        parts.append("after_relaxation")

    return "_".join(parts) + ".png"


def plot_results(database_filename,
                 descriptor_algorithm, 
                 reducer_algorithm,
                 descriptors_dimensionality=2,
                 include_enthalpy=True,
                 structures_before_relaxation=True,
                 structures_after_relaxation=True,
                 poscar_init_path='../POSCAR_init'
                 ):
    
    os.makedirs("pictures", exist_ok=True)

    # Colormap settings
    new_colormap = LinearSegmentedColormap.from_list("", ['#c521ff', '#ff0098', '#ff7342', '#FFC521'])
    new_colormap_blue = LinearSegmentedColormap.from_list("", ['#21ffc5', '#00dcfc', '#00aeff', '#3b6eff'])


    path = f"descriptors/{descriptor_algorithm}"

    with open(database_filename, 'rb') as database:
        data = pkl.load(database)
    df_results = pd.DataFrame(data)

    if structures_before_relaxation == True:
        df_generated_reduced = pd.read_csv(f"{path}/generated_structures_{reducer_algorithm}.csv")
        
        x_col = f'{reducer_algorithm}_{descriptors_dimensionality}D_x'
        y_col = f'{reducer_algorithm}_{descriptors_dimensionality}D_y'
        x = df_generated_reduced[x_col]
        y = df_generated_reduced[y_col]

        if descriptors_dimensionality == 3:
            z_col = f'{reducer_algorithm}_{descriptors_dimensionality}D_z'
            z = df_generated_reduced[z_col]
        
        selected_columns = {
            'ID': 'ID',
            'generated_structure_energy': 'Energy',
            'generated_structure_volume': 'Volume',
            'generated_structure_SG': 'SG',
            'generated_structure_symbol': 'SG_symbol'
        }

        df_generated = df_results[list(selected_columns.keys())].rename(columns=selected_columns)

        df_generated.insert(1, 'x', x)
        df_generated.insert(2, 'y', y)

        if descriptors_dimensionality == 3:
            df_generated.insert(3, 'z', z)

    if structures_after_relaxation == True:

        df_relaxed_reduced = pd.read_csv(f"{path}/relaxed_structures_{reducer_algorithm}.csv")

        x_col = f'{reducer_algorithm}_{descriptors_dimensionality}D_x'
        y_col = f'{reducer_algorithm}_{descriptors_dimensionality}D_y'
        x = df_relaxed_reduced[x_col]
        y = df_relaxed_reduced[y_col]

        if descriptors_dimensionality == 3:
            z_col = f'{reducer_algorithm}_{descriptors_dimensionality}D_z'
            z = df_relaxed_reduced[z_col]

        df_similarity = pd.read_csv(f"{path}/relaxed_structures_similarities.csv")
        similarity = df_similarity.cosine_similarity_with_init
        
        selected_columns = {
            'ID': 'ID',
            'relaxed_structure_energy': 'Energy',
            'relaxed_structure_volume': 'Volume',
            'relaxed_structure_SG': 'SG',
            'relaxed_structure_symbol': 'SG_symbol'
        }

        df_relaxed = df_results[list(selected_columns.keys())].rename(columns=selected_columns)

        df_relaxed.insert(1, 'x', x)
        df_relaxed.insert(2, 'y', y)

        if descriptors_dimensionality == 3:
            df_relaxed.insert(3, 'z', z)

        df_relaxed['Cosine_similarity'] = similarity

    if include_enthalpy == True:

        if descriptors_dimensionality == 2:

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            ax.set_xlabel(f'{reducer_algorithm} 1', fontdict={'fontsize': 10}, labelpad=0.05)
            ax.set_ylabel(f'{reducer_algorithm} 2', fontdict={'fontsize': 10}, labelpad=0.05)
            ax.set_zlabel('Enthalpy, eV/cell', fontdict={'fontsize': 10}, labelpad=3)
            ax.tick_params(axis='x', which='major', pad=0.05, labelsize=10)
            ax.tick_params(axis='y', which='major', pad=0.05, labelbottom=True)
            ax.tick_params(axis='z', which='major', pad=3)
            ax.xaxis._axinfo["grid"].update({"linewidth":0.25})
            ax.yaxis._axinfo["grid"].update({"linewidth":0.25})
            ax.zaxis._axinfo["grid"].update({"linewidth":0.25})

            if structures_after_relaxation == True:

                new_colormap_blue = LinearSegmentedColormap.from_list("", ['#F016B1', '#1644F0'])
                similarity_colormap = new_colormap_blue
                
                scatter_relaxed = ax.scatter(
                    xs = df_relaxed.x,
                    ys = df_relaxed.y, 
                    zs = df_relaxed.Energy,
                    marker='o',   
                    s = 20,
                    c = df_relaxed.Cosine_similarity,
                    cmap = similarity_colormap,
                    label = f'Structures \nafter relaxation'
                    )
                
                cbaxes = fig.add_axes([0.9, 0.22, 0.03, 0.60]) 
                fig.colorbar(scatter_relaxed, label = 'Cosine similarity \n between generated and initial structures', pad=0.5, fraction=1, shrink=0.6, cax = cbaxes, cmap = similarity_colormap)
                ax.legend(loc=(-0.42, 0.58), frameon=False)
        
            if structures_before_relaxation == True:

                scatter_generated = ax.scatter(
                    xs = df_generated.x,
                    ys = df_generated.y, 
                    zs = df_generated.Energy,
                    marker='^',   
                    s = 20,
                    c = ['#7b70a1']*len(df_generated),
                    label = f'Structures \nbefore relaxation'
                    )
                ax.legend(loc=(-0.42, 0.58), frameon=False)
            
            if ((structures_after_relaxation == True) & (structures_before_relaxation == True)):
                plt.title(f'{reducer_algorithm} projection before and after relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)

                plt.show()

            elif ((structures_after_relaxation == True) & (structures_before_relaxation == False)):
                plt.title(f'{reducer_algorithm} projection after relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)

                plt.show()

            elif ((structures_after_relaxation == False) & (structures_before_relaxation == True)):
                plt.title(f'{reducer_algorithm} projection before relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)
                
                plt.show()
            else:
                print('Error!')
        else:
            print('Choose descriptors_dimensionality=2')

    else:

        if descriptors_dimensionality == 2:

            fig = plt.figure()
            ax = fig.add_subplot(111)

            if structures_after_relaxation == True:

                new_colormap_blue = LinearSegmentedColormap.from_list("", ['#F016B1', '#1644F0'])
                similarity_colormap = new_colormap_blue

                scatter_relaxed = ax.scatter(
                    x = df_relaxed.x,
                    y = df_relaxed.y,
                    marker='o',   
                    s = 20,
                    c = df_relaxed.Cosine_similarity,
                    cmap = similarity_colormap,
                    label = f'Structures \nafter relaxation'
                )
                ax.legend(loc=(-0.6, 0.58), frameon=False)
                fig.colorbar(scatter_relaxed, label = 'Cosine similarity \n between generated and initial structures', cmap = similarity_colormap)

            if structures_before_relaxation == True:

                scatter_generated = ax.scatter(
                    x = df_generated.x,
                    y = df_generated.y, 
                    marker='^',   
                    s = 20,
                    c = ['#7b70a1']*len(df_generated),
                    label = f'Structures \nbefore relaxation'
                    )
                ax.legend(loc=(-0.6, 0.58), frameon=False)

            plt.xlabel(f'{reducer_algorithm} 1', fontdict={'fontsize': 10}, labelpad = 8)
            plt.ylabel(f'{reducer_algorithm} 2', fontdict={'fontsize': 10}, labelpad = 0)

            if ((structures_after_relaxation == True) & (structures_before_relaxation == True)):
                plt.title(f'{reducer_algorithm} projection before and after relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)

                plt.show()

            elif ((structures_after_relaxation == True) & (structures_before_relaxation == False)):
                plt.title(f'{reducer_algorithm} projection after relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)

                plt.show()

            elif ((structures_after_relaxation == False) & (structures_before_relaxation == True)):
                plt.title(f'{reducer_algorithm} projection before relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)
                plt.close()

                plt.show()


        if descriptors_dimensionality == 3:

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            ax.set_xlabel(f'{reducer_algorithm} 1', fontdict={'fontsize': 10}, labelpad=0.05)
            ax.set_ylabel(f'{reducer_algorithm} 2', fontdict={'fontsize': 10}, labelpad=0.05)
            ax.set_zlabel(f'{reducer_algorithm} 3', fontdict={'fontsize': 10}, labelpad=3)
            ax.tick_params(axis='x', which='major', pad=0.05, labelsize=10)
            ax.tick_params(axis='y', which='major', pad=0.05, labelbottom=True)
            ax.tick_params(axis='z', which='major', pad=3)
            ax.xaxis._axinfo["grid"].update({"linewidth":0.25})
            ax.yaxis._axinfo["grid"].update({"linewidth":0.25})
            ax.zaxis._axinfo["grid"].update({"linewidth":0.25})


            if structures_after_relaxation == True:

                new_colormap_blue = LinearSegmentedColormap.from_list("", ['#F016B1', '#1644F0'])
                similarity_colormap = new_colormap_blue
                
                scatter_relaxed = ax.scatter(
                    xs = df_relaxed.x,
                    ys = df_relaxed.y, 
                    zs = df_relaxed.z,
                    marker='o',   
                    s = 20,
                    c = df_relaxed.Cosine_similarity,
                    cmap = similarity_colormap,
                    label = f'Structures \nafter relaxation'
                    )
                
                cbaxes = fig.add_axes([0.9, 0.22, 0.03, 0.60]) 
                fig.colorbar(scatter_relaxed, label = 'Cosine similarity \n between generated and initial structures', pad=0.5, fraction=1, shrink=0.6, cax = cbaxes, cmap = similarity_colormap)
                ax.legend(loc=(-0.42, 0.58), frameon=False)

            
            if structures_before_relaxation == True:

                scatter_generated = ax.scatter(
                    xs = df_generated.x,
                    ys = df_generated.y, 
                    zs = df_generated.z,
                    marker='^',   
                    s = 20,
                    c = ['#7b70a1']*len(df_generated),
                    label = f'Structures \nbefore relaxation'
                    )
                ax.legend(loc=(-0.42, 0.58), frameon=False)
            
            if ((structures_after_relaxation == True) & (structures_before_relaxation == True)):
                plt.title(f'{reducer_algorithm} projection before and after relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)
                plt.close()

                plt.show()

            elif ((structures_after_relaxation == True) & (structures_before_relaxation == False)):
                plt.title(f'{reducer_algorithm} projection after relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)
                plt.close()

                plt.show()

            elif ((structures_after_relaxation == False) & (structures_before_relaxation == True)):
                plt.title(f'{reducer_algorithm} projection before relaxation', fontsize=16, pad=20, loc='right')

                filename = format_filename(descriptor_algorithm, reducer_algorithm, descriptors_dimensionality, include_enthalpy, structures_before_relaxation, structures_after_relaxation)
                plt.savefig(f"pictures/{filename}", dpi=1200)
                plt.close()

                plt.show()
    
    return df_relaxed