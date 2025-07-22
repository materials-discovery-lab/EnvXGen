#!/bin/sh
#SBATCH -p mpi
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J EnvXGen_manager
#SBATCH -e log

python -m EnvXGen