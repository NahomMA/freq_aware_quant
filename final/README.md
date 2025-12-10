# Final Implementation: GLM-4V Evaluation

Full-scale evaluation on the MRCR (Multi-Round Co-Reference) dataset using GLM-4V-9B.

## Hardware Requirements
- **GPUs:** 4× NVIDIA A100 (40GB) or equivalent.
- **RAM:** ~128GB System RAM.
- **Storage:** ~50GB for model weights and dataset.

## Running Inference

**Prerequisite:** Render images from conversation text first.

### Step 0: Render Images
```bash
python final/scripts/render_images.py --font-path Verdana.ttf
```
This generates `final/rendered_images/{unique_id}/page_*.png` and updates `.jsonl` files with absolute image paths.

### Step 1: Interactive Mode
Allocate resources (adjust partition/account as needed):
```bash
salloc -p <partition_name> --gres=gpu:4 --cpus-per-task=16 --time=04:00:00
```

Run evaluation:
```bash
# Method 1: Image-Level Compression
python final/src/evaluation/local_inference.py

# Method 2: Token-Level Compression
python final/src/evaluation/local_inference_compressed.py --needle 2 --compression ultra_mild
```

### Step 2: Batch Job (SLURM)
We provide sample scripts in `final/scripts/`. **You must modify these** to match your cluster's configuration.

**How to Modify:**
Open `final/scripts/run_method1_inference.sh` and edit the header:
```bash
#SBATCH --partition=your_partition   <-- Change this
#SBATCH --account=your_account       <-- Change this
#SBATCH --nodelist=node_name         <-- Remove or change this
#SBATCH --gres=gpu:4                 <-- Ensure you have 4 GPUs
```

**Job Submission:**
```bash
sbatch final/scripts/run_method1_inference.sh
sbatch final/scripts/run_method2_inference.sh
```

**Check Results:**
Outputs will be saved to `final/results/`.
- `method1_*.out`: Logs and accuracy scores.
- `summary_*.json`: Aggregated metrics.

