#!/bin/bash
#SBATCH --job-name=test_modular        # Job name
#SBATCH --output=./slurm_out/modular_%j.out         # Standard output file (%j expands to jobID)
#SBATCH --error=./slurm_out/modular_%j.err          # Standard error file (%j expands to jobID)
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=16          # 16 CPU cores per task
#SBATCH --mem=100GB                  # 80GB memory
#SBATCH --time=24:00:00             # Time limit: 3 hours and 30 minutes
#SBATCH --gres=gpu:rtx4090:1        # Request 1 RTX 4090 GPU


mkdir -p slurm_out

# Initialize mamba/conda
source /home/nbahou/miniforge3/etc/profile.d/conda.sh
source /home/nbahou/miniforge3/etc/profile.d/mamba.sh

# Activate your specific environment
mamba activate ERK_gpu

# Run the Python script
python ./img_pipeline_module/pipeline.py
