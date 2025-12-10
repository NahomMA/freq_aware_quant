# Milestone 1: Proof-of-Concept

Initial validation using ConvBench + ChatGPT-4o. Generated **report figures**.

## What's Here

- `src/compression/block_dct_vq.py` - Block DCT (14×14)
- `src/rendering/renderer.py` - Text-to-image
- `src/eval/openai_eval.py` - ConvBench evaluation
- `artifacts/*.png` - **Report figures** (spectrum, residual, compressed)

## Quick Start

```bash
# Setup OpenAI API key
cp env_template.txt .env
# Edit .env and add your OpenAI API key

# Run compression demo
python milestone_1_run.py

# Output: artifacts/*.png (figures used in report)
```

## Report Figures Generated

1. **Frequency Spectrum**: `artifacts/*_spectrum.png`
2. **Compressed Images**: `artifacts/*_compressed.png`
3. **Residuals**: `artifacts/vis_residual_*.png`

## Key Parameters

| Keep Ratio | Compression | Quality |
|------------|-------------|---------|
| 0.5 (50%) | 2× | Excellent |
| 0.3 (30%) | 3.3× | Good |
| 0.1 (10%) | 10× | Degraded |

## Milestone 1 → Final

- ConvBench (50 samples) → MRCR (751 samples)
- ChatGPT-4o (API) → GLM-4V-9B (local)
- Proof-of-concept → Production code

---

**For production code:** See `../final/README.md`
