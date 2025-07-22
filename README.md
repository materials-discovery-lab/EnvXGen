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
