#!/bin/bash
#SBATCH --job-name=resize_imgs        # Job name
#SBATCH --output=./slurm_out/resize_%j.out         # Standard output file (%j expands to jobID)
#SBATCH --error=./slurm_out/resize_%j.err          # Standard error file (%j expands to jobID)
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=10          # 16 CPU cores per task
#SBATCH --mem=50GB                  # 80GB memory
#SBATCH --time=40:00:00             # Time limit: 3 hours and 30 minutes


WORKDIR=/home/nbahou/myimaging/apoDet/scripts
cd $WORKDIR

mkdir -p slurm_out

# Initialize mamba/conda
source /home/nbahou/miniforge3/etc/profile.d/conda.sh
source /home/nbahou/miniforge3/etc/profile.d/mamba.sh

# Activate your specific environment
mamba activate ERK_gpu

python 05_resize_images.py