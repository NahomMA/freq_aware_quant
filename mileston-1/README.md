# Milestone 1: Proof-of-Concept

Initial validation of Frequency-Aware compression using ConvBench and synthetic data. This phase focused on generating visual proof that high-frequency components (edges/noise) can be discarded while preserving low-frequency semantics.

## Contents
- **`src/compression/block_dct_vq.py`**: Implementation of Block DCT (14×14).
- **`src/rendering/renderer.py`**: Synthetic text-to-image generation.
- **`milestone_1_run.py`**: Main script to generate report figures.

## Usage

1. **Activate Environment**
   ```bash
   conda activate freq_aware_quant
   ```

2. **Run Demo**
   ```bash
   python mileston-1/milestone_1_run.py
   ```

3. **View Results**
   Artifacts are saved to `mileston-1/artifacts/`:
   - `*_spectrum.png`: Frequency domain visualization.
   - `*_residual.png`: Difference between original and compressed images.
   - `*_compressed.png`: The final visual output.
