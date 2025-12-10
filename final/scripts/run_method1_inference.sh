#!/bin/bash
#SBATCH --job-name=method1_compression
#SBATCH --partition=reserved-takabi
#SBATCH --qos=takabi
#SBATCH --account=takabi
#SBATCH --nodelist=e2-w8545-01
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output="final/results/method1_%j.out"
#SBATCH --error="final/results/method1_%j.err"
echo "=================================================="
echo "current working directory: $(pwd)"
echo "=================================================="


# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load required modules (adjust as needed for your system)
# module load python/3.10
# module load cuda/11.8

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate your_env_name

# Navigate to the evaluation directory
# cd /home/nbirh002/nahom-final

# Run the inference script
echo "Starting inference..."
echo "current working directory: $(pwd)"
echo "src directory: $(pwd)/final/src"
echo "evaluation directory: $(pwd)/final/src/evaluation"
echo "local_inference.py: $(pwd)/final/src/evaluation/local_inference.py"
python $(pwd)/final/src/evaluation/local_inference.py

echo "End Time: $(date)"
echo "Job completed!"
