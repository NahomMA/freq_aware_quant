# Final Implementation: Production Evaluation

Full-scale MRCR evaluation with GLM-4V-9B on 4× A100 GPUs.

## Quick Start

```bash
conda env create -f environment.yml
conda activate freq_quant

cd src
python evaluation/local_inference.py                    # Method 1
python evaluation/local_inference_compressed.py --needle 2 --compression ultra_mild  # Method 2
```

**Model Download:** GLM-4V auto-downloads from HuggingFace (`zai-org/Glyph`) - no token configuration needed.

## Hardware ⚠️

- 4× NVIDIA A100 (40GB each)
- GLM-4V-9B: 9 billion parameters
- Runtime: 2-3 hours per method

## Results

### Method 1: Image DCT (Works Post-Training ✅)

**Target:** Storage + bandwidth  
**Compression:** 2× file size reduction

| Needle | Baseline | Method 1 | Δ |
|--------|----------|----------|---|
| 2 | 34.98% | 35.86% | +0.88pp |
| 4 | 30.00% | 29.25% | -0.75pp |
| 8 | 18.75% | 18.25% | -0.50pp |

### Method 2: Token DCT (Needs Training ⚠️)

**Target:** KV-cache memory  
**Compression:** 144 → 72 tokens (50%)

| Needle | Baseline | Method 2 | Δ |
|--------|----------|----------|---|
| 2 | 34.98% | 2.79% | -32.19pp |

**Conclusion:** Post-training embedding manipulation fails. Requires learnable codebook + fine-tuning.

## Files

```
final/
├── src/
│   ├── compression/
│   │   ├── block_dct_vq.py           # Method 1
│   │   └── vision_compressor.py      # Method 2
│   └── evaluation/
│       ├── local_inference.py        # Method 1 script
│       └── local_inference_compressed.py  # Method 2 script
├── scripts/
│   ├── run_method1_inference.sh      # SLURM: Method 1
│   └── run_method2_inference.sh      # SLURM: Method 2
├── data/processed_*needle*.jsonl     # MRCR (251×3 samples)
└── environment.yml
```

## MRCR Dataset

| Config | Context | Needles | Samples |
|--------|---------|---------|---------|
| 2-needle | 0-8k | 2 | 251 |
| 4-needle | 8-16k | 4 | 251 |
| 8-needle | 16-32k | 8 | 251 |

**Total:** 753 samples

## SLURM (Production)

```bash
sbatch scripts/run_method1_inference.sh
sbatch scripts/run_method2_inference.sh
```

Both scripts:
- Activate `freq_quant` environment
- Set `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Run from `src/` directory
- No command-line arguments needed

## Installation

```bash
# Conda (recommended)
conda env create -f environment.yml

# Pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Key Parameters

**Method 1:**
- Block size: 14×14 pixels
- Keep ratio: 50%
- DC always preserved

**Method 2:**
- Grid: 12×12 tokens
- Quantization: 8/4/0-bit (low/mid/high freq)
- Pruning: 144 → 72 tokens

## Troubleshooting

- **Import errors:** Run from `src/` directory
- **CUDA OOM:** Need 4× A100 (40GB)
- **Model download:** Auto-downloads, no auth needed

---

**For proof-of-concept:** See `../mileston-1/README.md`
