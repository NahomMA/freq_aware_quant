# Frequency-Aware Vector Quantization for VLM Compression

**Student:** Nahom M. Birhan (nbirh002)  
**Course:** EECE 783/883 - Digital Image Processing  
**Institution:** Old Dominion University  
**Date:** December 2025

## Overview
This project implements post-training frequency-aware compression for Vision-Language Models (VLMs) using DCT-based quantization. It explores two intervention points:
1.  **Method 1 (Image-Level):** Compressing input pixels via Block DCT.
2.  **Method 2 (Token-Level):** Compressing internal visual embeddings via DCT + Quantization.

## Project Structure
```
nahom-final/
├── mileston-1/          # Proof-of-concept (ConvBench, report figures)
│   ├── src/             # Early DCT implementation
│   └── artifacts/       # Visualization outputs (spectrum, residual)
│
└── final/               # Full Evaluation (MRCR, GLM-4V)
    ├── src/             # Production implementation of Methods 1 & 2
    ├── scripts/         # HPC/SLURM job scripts
    ├── data/            # MRCR dataset (Needle-In-A-Haystack)
    └── results/         # Output logs and metrics
```

## Quick Start

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate freq_aware_quant
```

### 2. Milestone 1 (Visualizations)
Generate the frequency spectrum and residual plots used in the report:
```bash
python mileston-1/milestone_1_run.py
# Check outputs in: mileston-1/artifacts/
```

### 3. Final Evaluation (Inference)
Run the full benchmark on the MRCR dataset using GLM-4V. See `final/README.md` for detailed instructions on interactive and batch (HPC) execution.

## Warning: Hardware Requirements: you need to have the following resource the model is to bing and data is also huge
- **GPUs:** 4× NVIDIA A100 (40GB) or equivalent.
- **RAM:** ~128GB System RAM.
- **Storage:** ~50GB for model weights and dataset.

### 1. Interactive Mode
Allocating resources (Example - adjust partition/account as needed):
```bash
salloc -p <partition_name> --gres=gpu:4 --cpus-per-task=16 --time=04:00:00
```

Running the scripts:
```bash
# Method 1: Image-Level Compression
python final/src/evaluation/local_inference.py

# Method 2: Token-Level Compression
python final/src/evaluation/local_inference_compressed.py --needle 2 --compression ultra_mild
```

### 2. Batch Job (SLURM)
We provide sample scripts in `final/scripts/`. **You must modify these** to match your cluster's configuration.

**How to Modify:**
Open `final/scripts/run_method1_inference.sh` and edit the header: do the same for `final/scripts/run_method2_inference.sh`
```bash
#SBATCH --partition=your_partition   <-- Change this
#SBATCH --account=your_account       <-- Change this
#SBATCH --nodelist=node_name         <-- Remove or change this
#SBATCH --gres=gpu:4                 <-- Ensure you have 4 GPUs
```

**Finally: Job Submission: the whole it will take 6-8 hours in total**
```bash
sbatch final/scripts/run_method1_inference.sh
sbatch final/scripts/run_method2_inference.sh
```

---
**Documentation:**
- **[Milestone 1 Details](mileston-1/README.md)**: Proof-of-concept and figure generation.
- **[Final Implementation](final/README.md)**: Full-scale evaluation and HPC usage guide.
