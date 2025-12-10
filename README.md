# Frequency-Aware Vector Quantization for VLM Compression

**Student:** Nahom M. Birhan (nbirh002)  
**Course:** EECE 783/883 - Digital Image Processing  
**Institution:** Old Dominion University  
**Date:** December 2025

## Overview

Post-training frequency-aware compression for vision-language models (VLMs) using DCT-based quantization. Investigates two compression intervention points: image-level (Method 1) and token-level (Method 2).

## Key Results

- **Method 1** (Image DCT): 2× compression, accuracy maintained (35.86% vs 34.98%)
- **Method 2** (Token DCT): Signals need for training phase (2.79% vs 34.98%)

## Structure

```
nahom-final/
├── mileston-1/          # Proof-of-concept (ConvBench, report figures)
│   ├── src/             # Early DCT implementation
│   ├── artifacts/       # Visualization outputs (spectrum, residual)
│   └── README.md        # Milestone 1 documentation
│
└── final/               # local post training implementation (MRCR, GLM-4V)
    ├── src/             # Methods 1 & 2 implementation
    ├── scripts/         # SLURM inference scripts
    ├── data/            # MRCR dataset
    └── README.md        # Final implementation guide
```

## Quick Start

### Milestone 1 (Report Figures)
```bash
cd mileston-1
python milestone_1_run.py  # Generate frequency visualizations
```

### Final Implementation
```bash
cd final
conda env create -f environment.yml
conda activate freq_quant

# Method 1
cd src && python evaluation/local_inference.py

# Method 2
cd src && python evaluation/local_inference_compressed.py --needle 2 --compression ultra_mild
```

## Hardware Requirements

- **GPUs:** 4× NVIDIA A100 (40GB)
- **Model:** GLM-4V-9B (9B parameters)
- **Memory:** ~160GB total (distributed)

## Documentation

- **Main README** (this file): Project overview
- **`mileston-1/README.md`**: Proof-of-concept details, figure generation
- **`final/README.md`**: Production code, full-scale evaluation



---

**For detailed implementation, see:**
- Milestone 1 (early work): `mileston-1/README.md`
- Final implementation: `final/README.md`

