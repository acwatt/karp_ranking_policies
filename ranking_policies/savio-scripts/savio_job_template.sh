```bash
#!/bin/bash
#SBATCH --job-name=karp_sim_methods_compare
#SBATCH --account=fc_rankpolicy
#SBATCH --partition=savio
#SBATCH --nodes=1
#SBATCH --time=00:00:30
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aaron.watt@berkeley.edu
## Command(s) to run:
julia --threads=auto --project=. codefile.jl