![EnvXGen](EnvXGen/images/EnvXGen_logo.png)

## About EnvXGen

**EnvXGen** is a method for generating and identifying stable crystal structures when partial atomic environments are known (e.g., from experiment).

It works by placing structural elements on a spatial grid within the unit cell, guided by interatomic distance constraints. After structural relaxation, clustering, and similarity analysis, EnvXGen selects the most stable and environment-consistent configuration.

EnvXGen has been successfully tested on:
- Atomic crystals: **UH₈**, **LaH₁₀**, **H₃S**, **Y₂H₉**
- Molecular cocrystals: **CL_20+CO₂**, **CL_20+N₂O**

Furthermore, EnvXGen includes tools to assess the **diversity of generated structures** using:
- Graph Neural Network (GNN) embeddings
- Dimensionality reduction (e.g., UMAP, PCA)
- Clustering algorithms

Tests are available in the [`tests`](./tests) folder, additional data available upon request.


## 🚀 Quick Start

### Prerequisites

- **Python 3.7+** with EnvXGen package
- **SLURM** job scheduler access
- **VASP** with POTCAR files (for atomic crystals)
- **LAMMPS** with force fields (for molecular crystals)

## 📋 Generation Modes

| Mode | Description | Relaxation | Example System |
|------|-------------|------------|----------------|
| `atoms` | Atomic crystal generation | VASP | LaH₁₀ |
| `molecules` | Molecular crystal generation | LAMMPS | CL-20+CO₂ |

## ⚙️ Configuration Setup

### 1. Choose Template

**For Atomic Crystals:**
```bash
cp cfg_for_atomic_crystals_generation.yml cfg.yaml
```

**For Molecular Crystals:**
```bash
cp cfg_for_molecular_crystals_generation.yml cfg.yaml
```

### 2. Basic Configuration

```yaml
Crystal system: LaH10          # System identifier
Mode: atoms                    # atoms | molecules
Relaxation mode: vasp          # vasp | lammps
Number of generated structures: 8
Number of parallel calculations: 4
Pressure (GPa): 0             # Applied pressure
```

### 3. Grid Resolution Settings

```yaml
Number of bins along each lattice vector:
    n_bins_x: 12    # X-direction resolution
    n_bins_y: 12    # Y-direction resolution  
    n_bins_z: 12    # Z-direction resolution
```

### 4. SLURM Job Configuration

```yaml
Calculation settings:
    SBATCH --partition: cpu
    SBATCH --nodes: 1
    SBATCH --ntasks: 8
    SBATCH --time: 00:20:00
```

## 🔧 Setup Requirements

### VASP Mode (Atomic Crystals)

#### Required Files:
1. **📁 INCAR Files** → Multi-stage relaxation setup
   - `EnvXGen/inputs/vasp/incars_unfixed/INCAR_1, INCAR_2, ...` - Standard relaxation stages
   - `EnvXGen/inputs/vasp/incars_fixed/INCAR_1, INCAR_2, ...` - Fixed-atom relaxation stages
   - Number of files = `VASP steps` parameter in config

2. **📁 POTCAR Files** → `EnvXGen/inputs/vasp/potcars/`
   - `POTCAR_La`, `POTCAR_H`, etc. - Individual pseudopotentials for each element
   - Download appropriate POTCAR for all atom types in your system

3. **📄 Initial Structure** → `POSCAR_init`
   - VASP format crystal structure

#### Configuration Example:
```yaml
Initial Environment:
    Filename: POSCAR_init
    Format: vasp
    Atom types from single atoms:
        La:
            chemical symbol: La
            covalent radius: default

Placed structural elements:
    Single atoms:
        H:
            chemical symbol: H
            number of atoms: 40
            covalent radius: default
```

### LAMMPS Mode (Molecular Crystals)

#### Required Files:
1. **📁 Force Field** → `EnvXGen/inputs/lammps/ffield.reax.lg`
   - ReaxFF or other compatible force field file

2. **📁 LAMMPS Input** → `EnvXGen/inputs/lammps/lammps.in`
   - Relaxation settings and simulation parameters

3. **📄 Initial Structure** → `POSCAR_init`
   - Starting crystal structure

#### Configuration Example:
```yaml
Placed structural elements:
    Molecules from databases:
        H2O:
            name: H2O
            ase_name: H2O
            # Alternative: pubchem_cid: 2244
            number of molecules: 1
            Van der Waals radii:
                H: default
                O: default
```

## Running EnvXGen

### SLURM Submission
```bash
sbatch run_envxgen.sh
```

## 📁 Project Structure

```
your_project/
├── 📄 cfg.yaml                              # Main configuration
├── 📄 run_envxgen.sh                        # SLURM submission script  
├── 📄 POSCAR_init                           # Initial structure
├── 📁 EnvXGen/                              # Main folder with all scripts
│   └── 📁 inputs/
│       ├── 📁 vasp/
│       │   ├── 📁 incars_fixed/             # Fixed-atom relaxation stages
│       │   │   ├── 📄 INCAR_1               # Stage 1 settings
│       │   │   ├── 📄 INCAR_2               # Stage 2 settings
│       │   │   └── 📄 INCAR_n               # Stage n settings
│       │   ├── 📁 incars_unfixed/           # Standard relaxation stages  
│       │   │   ├── 📄 INCAR_1               # Stage 1 settings
│       │   │   ├── 📄 INCAR_2               # Stage 2 settings
│       │   │   └── 📄 INCAR_n               # Stage n settings
│       │   └── 📁 potcars/                  # Pseudopotentials
│       │       ├── 📄 POTCAR_La             # Lanthanum POTCAR
│       │       ├── 📄 POTCAR_H              # Hydrogen POTCAR
│       │       └── 📄 POTCAR_X              # Other elements
│       └── 📁 lammps/
│           ├── 📄 lammps.in                 # LAMMPS input
│           └── 📄 ffield.reax.lg            # ReaxFF force field
└── 📄 log                                   # Execution log
```

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| ❌ **Missing POTCAR** | Download from VASP database for all atom types |
| ❌ **LAMMPS Errors** | Verify force field compatibility with your system |
| ❌ **Job Fails** | Check `log` file and validate SLURM settings |
| ❌ **Memory Issues** | Increase `--mem-per-cpu` or reduce parallel jobs |
| ❌ **Timeout** | Increase `--time` in SLURM settings |

## Advanced Features

### Two-Stage Relaxation
```yaml
Relaxation mode: vasp_fixed_and_unfixed
VASP steps: 3  # Number of relaxation stages
```

### Custom Radii
```yaml
La:
    chemical symbol: La
    covalent radius: 1.8  # Custom value in Angstroms
```

### Molecules from Files
```yaml
Molecules from file:
    Custom_Molecule:
        name: MyMol
        Filename: molecule.vasp
        Format: vasp
        number of molecules: 2
```

---

## Results Structure

After successful completion, you'll find:
```
📁 results/
├──📁 result_vasp_fixed/          # (and/or result_vasp_unfixed, result_lammps depending on relaxation type)
│   ├──🗄️ POSCARS_*crystal_system*_vasp_fixed.pkl              # Generated structures database
│   ├──🗄️ POSCARS_*crystal_system*_vasp_fixed_relaxed.pkl      # Relaxed structures database  
│   ├──🗄️ relaxation_results_all.pkl                # Complete relaxation results
│   ├──🗄️ result_batch_0.pkl                        # Individual batch results
│   ├──🗄️ result_batch_1.pkl
│   └── ...
├──🗄️ relaxation_results_summarized.pkl         # Summary with all structure info (sorted by energy)
└──💻 Postprocessing.ipynb                      # Analysis notebook
```

## Postprocessing and Analysis

After generation completes, the `Postprocessing.ipynb` notebook will be automatically copied to your results directory. This powerful notebook provides comprehensive analysis tools:

### Available Analysis Methods

**Structure Descriptors:**
- **RDF** (Radial Distribution Function) - analyzes atomic pair correlations
- **ALIGNN** - advanced descriptors using graph neural networks

**Dimensionality Reduction:**
- **PCA** (Principal Component Analysis) - linear dimensionality reduction
- **UMAP** - non-linear manifold learning for better cluster visualization

**Analysis Capabilities:**
- Structure clustering with automatic optimal cluster number detection
- Cosine similarity calculation between generated and original structures
- 2D/3D visualization of structure projections (before and after relaxation)
- Energy characteristics analysis and comparison
- Automatic search for energetically favorable structures in different clusters


### Citation and Code Attribution

This project uses and adapts code from the following open-source projects:

#### ASE – Atomic Simulation Environment
Portions of this code are adapted from ASE:
https://gitlab.com/ase/ase
Licensed under GPLv3.
Citation: A. Hjorth Larsen et al., J. Phys.: Condens. Matter 29 273002 (2017)

#### ALIGNN – Atomistic Line Graph Neural Network
Portions of this code are adapted from ALIGNN:
https://github.com/usnistgov/alignn
Licensed under Apache 2.0.
Citation: Choudhary & DeCost (2021), npj Comput Mater 7, 185
DOI: https://doi.org/10.1038/s41524-021-00650-1


## Contact Us
Email: propad@phystech.edu
Telegram: @yanapropad


**Ready to generate your crystal structures?** Start with the basic configuration and gradually customize for your specific research needs!
