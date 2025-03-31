#!/bin/bash
#SBATCH --job-name=crop        # Job name
#SBATCH --output=./slurm_out/job_%j.out         # Standard output file (%j expands to jobID)
#SBATCH --error=./slurm_out/job_%j.err          # Standard error file (%j expands to jobID)
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=16          # 16 CPU cores per task
#SBATCH --mem=80GB                  # 80GB memory
#SBATCH --time=03:30:00             # Time limit: 3 hours and 30 minutes
#SBATCH --gres=gpu:rtx4090:1        # Request 1 RTX 4090 GPU


WORKDIR=/home/nbahou/myimaging/apoDet/scripts
cd $WORKDIR

mkdir -p slurm_out

# Initialize mamba/conda
source /home/nbahou/miniforge3/etc/profile.d/conda.sh
source /home/nbahou/miniforge3/etc/profile.d/mamba.sh

# Activate your specific environment
mamba activate ERK_gpu

# Run the Python script
python 01_segment_strdst.py
python 02_tracking_btrack.py
python 03_match_annotations.py
python 04_crop_windows.py
