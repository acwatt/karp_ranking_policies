#!/bin/bash
# Job name that will appear in the browser and terminal
#SBATCH --job-name=savio_node_test5
#
# Account name who's credits will be charged
#SBATCH --account=fc_rankpolicy
#
# Cluster partition name
#SBATCH --partition=savio3
#
# Number of nodes to reserve (physically distinct computers, only use >1 if you have set up distributed computing)
#SBATCH --nodes=2
#
# Number of workers/`processes per node (should be multiple of # of CPUs, which is either 32 or 40 for savio3)
#SBATCH --ntasks-per-node=32
#
# Wall clock time limit for the job in HH:MM:SS
#SBATCH --time=00:20:00
#
# Email type (begin, end, error, ALL)
#SBATCH --mail-type=ALL
#
# Email where notifications will be sent
#SBATCH --mail-user=aaron.watt@berkeley.edu
#
# Output and error files that will show up in your BRC home directory
#SBATCH --output=savio-scripts/savio_node_test5_%j.out
#SBATCH --error=savio-scripts/savio_node_test5_%j.err
##
## Command(s) to run:
## This file run from ranking_policies directory on Savio via SSH terminal: sbatch savio-scripts\savio4.sh
## See [BRC computing cluster # Batch job submission] obsidian notes
julia --threads=auto ./julia/KarpEstimation/Simulations/test1.jl