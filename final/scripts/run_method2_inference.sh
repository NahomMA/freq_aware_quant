#!/bin/bash
#SBATCH --job-name=method2_token_compression
#SBATCH --partition=reserved-takabi
#SBATCH --qos=takabi
#SBATCH --account=takabi
#SBATCH --nodelist=e2-w8545-01
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output="final/results/method2_%j.out"
#SBATCH --error="final/results/method2_%j.err"
# Frequency-Aware VLM Compression - Method 2 (Token-Level)
# Applies DCT + quantization after vision encoding
# Expected: Post-training approach shows need for training phase

echo "=================================================="
echo "current working directory: $(pwd)"
echo "=================================================="


echo "Starting inference..."
echo "current working directory: $(pwd)"
echo "src directory: $(pwd)/final/src"
echo "evaluation directory: $(pwd)/final/src/evaluation"
echo "local_inference_compressed.py: $(pwd)/final/src/evaluation/local_inference_compressed.py"
python $(pwd)/final/src/evaluation/local_inference_compressed.py --needle all --compression ultra_mild

echo "End Time: $(date)"
echo "Job completed!"

