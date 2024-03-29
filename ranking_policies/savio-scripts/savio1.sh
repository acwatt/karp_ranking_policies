#!/bin/bash
#SBATCH --job-name=karp_sim_methods_compare
#SBATCH --account=fc_rankpolicy
#SBATCH --partition=savio3
#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aaron.watt@berkeley.edu
#SBATCH --output=savio-scripts/karp_sim_methods_compare_%j.out
#SBATCH --error=savio-scripts/karp_sim_methods_compare_%j.err
## Command(s) to run:
## This file run from ranking_policies directory on Savio: sbatch savio-scripts\savio1.sh
julia --threads=auto --project=./julia/KarpEstimation ./julia/KarpEstimation/KarpEstimation.jl