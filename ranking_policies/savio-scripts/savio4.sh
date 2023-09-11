#!/bin/bash
#SBATCH --job-name=savio_node_test4
#SBATCH --account=fc_rankpolicy
#SBATCH --partition=savio3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aaron.watt@berkeley.edu
#SBATCH --output=savio-scripts/savio_node_test4_%j.out
#SBATCH --error=savio-scripts/savio_node_test4_%j.err
## Command(s) to run:
## This file run from ranking_policies directory on Savio: sbatch savio-scripts\savio4.sh
julia --threads=auto --project=./julia/KarpEstimation/Simulations/turingSimulations ./julia/KarpEstimation/Utilities/savio_test.jl