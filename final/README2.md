# Frequency-Aware Vector Quantization for Vision-Language Model Compression

Post-training compression techniques for vision-language models using DCT-based frequency decomposition and adaptive quantization. This project investigates two intervention points: image-level compression (Method 1) and token-level compression (Method 2).

## 🎯 Key Results

- **Method 1** (Image-Level DCT): Achieves 2× storage compression with comparable accuracy (35.86% vs 34.98% baseline on MRCR 2-needle)
- **Method 2** (Token-Level): Post-training manipulation yields 2.79% accuracy, indicating need for training-based approach

## 🔧 Hardware Requirements

**Critical:** This project requires significant computational resources:

- **GPUs:** 4× NVIDIA A100 (40GB) or equivalent
- **Model Size:** GLM-4V-9B (9 billion parameters)
- **Memory:** ~160GB GPU memory total (distributed across 4 GPUs)
- **Parallel Inference:** Multi-GPU parallel processing for efficiency
- **CUDA:** Version 11.8 or higher

**Note:** The GLM-4V-9B model cannot run on single consumer GPUs due to memory constraints. The implementation uses model parallelism across 4 GPUs.

## 📦 Installation

### Option 1: Conda (Recommended)

```bash
# Create environment from specification
conda env create -f environment.yml

# Activate environment
conda activate freq_quant
```

### Option 2: Pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Important:** Ensure PyTorch is installed with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Project Structure

```
nahom-final/
├── src/
│   ├── compression/
│   │   ├── block_dct_vq.py           # Method 1: Image-level DCT compression
│   │   └── vision_compressor.py      # Method 2: Token-level compression
│   └── evaluation/
│       ├── local_inference.py        # Method 1 inference script
│       └── local_inference_compressed.py  # Method 2 inference script
├── scripts/
│   ├── run_method1_inference.sh      # SLURM script for Method 1
│   ├── run_method2_inference.sh      # SLURM script for Method 2
│   └── render_images.py              # Convert text to images
├── mileston-1/                       # Initial proof-of-concept (report figures)
│   ├── src/                          # Early implementation
│   └── README.md                     # Milestone 1 documentation
├── data/
│   └── README.md                     # Data acquisition instructions
├── results/
│   ├── method1_results.json          # Pre-computed Method 1 results
│   └── method2_results.json          # Pre-computed Method 2 results
└── README.md
```

## 🚀 Quick Start

### Method 1: Image-Level DCT Compression

Applies block DCT compression (14×14 blocks, 50% coefficient retention) before vision encoding.

```bash
cd /home/nbirh002/frequncy-aware_quant/nahom-final

# Single GPU inference (for testing small samples)
python src/evaluation/local_inference.py \
    --data_path data/mrcr_sample.json \
    --output_path results/method1_output.json \
    --baseline False

# Multi-GPU parallel inference (production)
sbatch scripts/run_method1_inference.sh
```

**Expected Runtime:** ~2-3 hours on 4× A100 GPUs for full MRCR benchmark

### Method 2: Token-Level Frequency-Aware Compression

Applies DCT + quantization + pruning after vision encoding (144 → 72 tokens).

```bash
# Multi-GPU parallel inference
sbatch scripts/run_method2_inference.sh
```

**Compression Profiles Available:**
- `mild`: 8/4/2-bit quantization (conservative)
- `aggressive`: 8/4/0-bit quantization (moderate)
- `extreme`: 8/3/0-bit quantization (aggressive)

## 📊 Dataset

### MRCR Benchmark

Multi-needle Retrieval in Conversational Records for evaluating long-context VLM performance.

**Download Instructions:** See `data/README.md`

**Configurations:**
- 2-needle: Short context (0-8k tokens)
- 4-needle: Medium context (8-16k tokens)
- 8-needle: Long context (16-32k tokens)

### Sample Data

A minimal sample dataset is provided in `data/mrcr_sample.json` for testing.

### Milestone 1 Code

The `mileston-1/` directory contains the initial proof-of-concept implementation used to generate the **figures shown in the final report**. This includes:
- Early DCT compression implementation
- Frequency spectrum visualizations
- Visual quality comparison examples

See `mileston-1/README.md` for details on how these figures were generated.

## 🔬 Methodology

### Method 1: Image-Level DCT Compression

**Target:** Storage memory and I/O bandwidth reduction

**Pipeline:**
1. Render conversation history as images (Glyph rendering)
2. Partition into 14×14 pixel blocks
3. Apply 2D DCT per block (Equation 6 in paper)
4. Retain top 50% coefficients by magnitude (DC always preserved)
5. Apply inverse DCT and reconstruct
6. Feed compressed image to vision encoder

**Memory Impact:** Reduces image file size by 2×, but KV-cache unchanged

### Method 2: Token-Level Compression

**Target:** KV-cache runtime memory reduction

**Pipeline:**
1. Vision encoder outputs embeddings [B, 144, 4096]
2. Reshape to spatial grid [B, 12, 12, 4096]
3. Apply 2D DCT per embedding channel
4. Frequency-aware quantization (8-bit low, 4-bit mid, 0-bit high)
5. Energy-based token pruning (144 → 72 tokens)
6. Feed compressed tokens to LLM

**Memory Impact:** Reduces token count by 50%, directly reduces KV-cache

## 📈 Results

### Method 1: Post-Training Success

| Configuration | Baseline | Method 1 | Difference |
|--------------|----------|----------|------------|
| 2-needle     | 34.98%   | 35.86%   | +0.88pp    |
| 4-needle     | 30.00%   | 29.25%   | -0.75pp    |
| 8-needle     | 18.75%   | 18.25%   | -0.50pp    |

**Compression:** 2× storage reduction  
**Conclusion:** Post-training image-level compression is feasible

### Method 2: Training Phase Needed

| Configuration | Baseline | Method 2 | Difference |
|--------------|----------|----------|------------|
| 2-needle     | 34.98%   | 2.79%    | -32.19pp   |

**Compression:** 2× token reduction (144 → 72)  
**Conclusion:** Post-training manipulation disrupts learned embeddings; requires training-based approach with learnable codebook

## 🛠️ Implementation Details

### Model Architecture

- **Vision Encoder:** EVA-02-CLIP (3.5B parameters, 24 ViT blocks)
- **LLM:** GLM-4-9B-Chat (6B parameters, 40 decoder layers)
- **Input Resolution:** 14×14 patches → 576 tokens → downsample to 144 tokens
- **Embedding Dimension:** 4096

### Compression Parameters

**Method 1 (block_dct_vq.py):**
- Block size: 14×14 pixels
- Keep ratio: 0.5 (50% coefficients)
- DC component: Always preserved

**Method 2 (vision_compressor.py):**
- Grid size: 12×12 spatial layout
- Frequency bands: Low (<4), Mid (4-8), High (>8)
- Bit allocation: 8-bit / 4-bit / 0-bit
- Pruning: Top-72 tokens by energy

### Evaluation Metric

Accuracy computed as average SequenceMatcher similarity ratio:

```python
Accuracy = (1/N) * Σ SequenceMatcher.ratio(ground_truth_i, prediction_i)
```

## 🔍 Key Files Explained

### `src/compression/block_dct_vq.py`

Implements **Algorithm 1** from paper: Block DCT with frequency thresholding.

- `BlockDCTThresholdCompressor`: Main compression class
- `compress()`: Applies DCT, thresholding, and IDCT
- `_visualize_spectrum()`: Generates frequency spectrum plots

### `src/compression/vision_compressor.py`

Implements **Algorithm 2** from paper: Token-level frequency-aware compression.

- `VisionTokenCompressor`: PyTorch module for embedding compression
- `forward()`: Applies DCT, quantization, and pruning
- Compression profiles: mild/aggressive/extreme configurations

### `src/evaluation/local_inference.py`

Method 1 evaluation script with parallel GPU inference.

- Loads GLM-4V model across multiple GPUs
- Applies `BlockDCTThresholdCompressor` to images
- Computes SequenceMatcher-based accuracy

### `src/evaluation/local_inference_compressed.py`

Method 2 evaluation script with forward hook registration.

- Registers compression hook on vision encoder output
- Applies `VisionTokenCompressor` during inference
- Supports multiple compression profiles

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{birhan2025frequency,
  title={Frequency-Aware Vector Quantization for Vision-Language Model Memory Compression},
  author={Birhan, Nahom M.},
  journal={ODU Digital Image Processing Final Project},
  year={2025}
}
```

## 🙏 Acknowledgments

- **Glyph Pipeline:** Visual-text rendering framework ([Cheng et al., 2025](https://arxiv.org/abs/2510.17800))
- **GLM-4V Model:** Pre-trained vision-language model ([GLM Team, 2024](https://arxiv.org/abs/2406.12793))
- **MRCR Benchmark:** Multi-needle retrieval evaluation dataset
- **JPEG Compression:** DCT-based frequency decomposition inspiration ([Wallace, 1992](https://ieeexplore.ieee.org/document/125072))

## 📧 Contact

Nahom M. Birhan  
ODU ID: nbirh002  
Old Dominion University

---

**Last Updated:** December 2025  
**License:** Academic Use Only

