#!/bin/bash -l
#SBATCH --job-name="ebc_simulation"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alice.geminiani@unipv.it
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread
#SBATCH --mem-per-cpu=1G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

source $HOME/bsb-daint/load.sh
source $HOME/bsb-daint/nest_export.sh


srun python run_simulation.py $SLURM_NTASKS_PER_NODE $SLURM_JOB_NUM_NODES
