# Milestone 1: Proof-of-Concept

Initial validation of Frequency-Aware compression using ConvBench and synthetic data. This phase focused on generating visual proof that high-frequency components (edges/noise) can be discarded while preserving low-frequency semantics.

## Contents
- **`mileton-1/src/compression/block_dct_vq.py`**: Implementation of Block DCT (14×14).
- **`mileston-1/src/rendering/renderer.py`**: Synthetic text-to-image generation.
- **`mileton-1/milestone_1_run.py`**: Main script to generate report figures.

## Usage

1. **Activate Environment**

```bash
conda env create -f environment.yml
conda activate freq_aware_quant
```

2. **Run Demo**
 You need OpenAI key 
```bash
# OpenAI API Configuration
# 
# 1. Copy this file to .env:
cp env_template.txt mileston-1/.env
#
# 2. Replace 'your_key_here' with your actual OpenAI API key
#    Get it from: https://platform.openai.com/api-keys
#
# 3. open .env file  under add mileston-1
 
OPENAI_API_KEY=your_key_here

python mileston-1/milestone_1_run.py
# Check outputs in: mileston-1/artifacts/
```

3. **View Results**
   Artifacts are saved to `mileston-1/artifacts/`:
   - `*_spectrum.png`: Frequency domain visualization.
   - `*_residual.png`: Difference between original and compressed images.
   - `*_compressed.png`: The final visual output.
