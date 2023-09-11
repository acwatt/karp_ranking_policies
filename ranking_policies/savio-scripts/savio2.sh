#!/bin/bash
#SBATCH --job-name=savio_node_test1
#SBATCH --account=fc_rankpolicy
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aaron.watt@berkeley.edu
#SBATCH --output=savio-scripts/savio_node_test1_%j.out
#SBATCH --error=savio-scripts/savio_node_test1_%j.err
## Command(s) to run:
## This file run from ranking_policies directory on Savio: sbatch savio-scripts\savio2.sh
julia --threads=auto --project=./julia/KarpEstimation ./julia/KarpEstimation/Utilities/savio_test.jl