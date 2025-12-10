from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct
import torch


@dataclass(slots=True)
class CompressionStats:
    output_path: Path
    num_blocks: int
    block_size: int
    keep_ratio: float
    bytes_on_disk: int
    token_estimate: int


class BlockDCTThresholdCompressor:
    """
    Standard Frequency-Aware Compression (Scalar Quantization / Hard Thresholding).
    Discards high-frequency coefficients but keeps original values for retained ones.
    Fast and robust.
    """
    def __init__(self, block_size: int = 8):
        self.block_size = block_size

    def _prepare_image(self, image_or_path: str | Path | Image.Image) -> np.ndarray:
        if isinstance(image_or_path, (str, Path)):
            img = Image.open(image_or_path)
        else:
            img = image_or_path

        img = img.convert("L")
        w, h = img.size
        new_w = (w // self.block_size) * self.block_size
        new_h = (h // self.block_size) * self.block_size
        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))
        return np.array(img, dtype=float)

    def _blockify(self, img_array: np.ndarray) -> np.ndarray:
        h, w = img_array.shape
        bs = self.block_size
        blocks = img_array.reshape(h // bs, bs, w // bs, bs)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, bs, bs)
        return blocks

    def _reconstruct(self, blocks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        bs = self.block_size
        n_h, n_w = shape[0] // bs, shape[1] // bs
        grid = blocks.reshape(n_h, n_w, bs, bs)
        grid = grid.transpose(0, 2, 1, 3)
        return grid.reshape(shape)

    def _dct2(self, block_batch: np.ndarray) -> np.ndarray:
        return dct(dct(block_batch, axis=1, norm="ortho"), axis=2, norm="ortho")

    def _idct2(self, block_batch: np.ndarray) -> np.ndarray:
        return idct(idct(block_batch, axis=1, norm="ortho"), axis=2, norm="ortho")

    def _visualize_spectrum(self, original_dct: np.ndarray, kept_dct: np.ndarray, output_path: Path) -> None:
        """
        Generates a side-by-side heatmap of the average frequency spectrum
        before and after compression.
        """
        # Ensure shapes are (N, block_size, block_size)
        if original_dct.ndim == 2:
            original_dct = original_dct.reshape(-1, self.block_size, self.block_size)
        if kept_dct.ndim == 2:
            kept_dct = kept_dct.reshape(-1, self.block_size, self.block_size)

        # Compute average log-magnitude spectrum
        # We use abs() because DCT coeffs can be negative
        # We use log1p() because DC component is huge compared to high freq
        avg_spec_orig = np.log1p(np.mean(np.abs(original_dct), axis=0))
        avg_spec_kept = np.log1p(np.mean(np.abs(kept_dct), axis=0))

        # Determine common scale for heatmap to make comparison valid
        vmin = min(avg_spec_orig.min(), avg_spec_kept.min())
        vmax = max(avg_spec_orig.max(), avg_spec_kept.max())

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        im1 = axs[0].imshow(avg_spec_orig, cmap='inferno', interpolation='nearest', vmin=vmin, vmax=vmax)
        axs[0].set_title("Original Frequency Spectrum\n(Avg Log Magnitude)")
        axs[0].axis('off')
        
        im2 = axs[1].imshow(avg_spec_kept, cmap='inferno', interpolation='nearest', vmin=vmin, vmax=vmax)
        axs[1].set_title("Compressed Frequency Spectrum\n(Masked / Quantized)")
        axs[1].axis('off')

        # Add colorbar
        plt.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.8, label="Log Magnitude")
        
        vis_path = output_path.parent / f"{output_path.stem}_spectrum.png"
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def compress(
        self,
        input_image: str | Path | Image.Image,
        output_path: str | Path = None,
        *,
        keep_ratio: float = 0.5,
    ) -> Image.Image:
        arr = self._prepare_image(input_image)
        original_shape = arr.shape
        blocks = self._blockify(arr)
        dct_blocks = self._dct2(blocks).reshape(blocks.shape[0], -1)

        # Frequency-Aware Thresholding (Scalar Quantization)
        threshold = np.percentile(np.abs(dct_blocks), 100 * (1 - keep_ratio))
        
        # Always keep DC component (index 0)
        dc_mask = np.zeros_like(dct_blocks, dtype=bool)
        dc_mask[:, 0] = True
        
        # Hard thresholding: Keep coeffs > threshold OR DC
        masked_flat = np.where((np.abs(dct_blocks) >= threshold) | dc_mask, dct_blocks, 0)
        masked_blocks = masked_flat.reshape(-1, self.block_size, self.block_size)

        # Visualize the spectrum if output_path is provided
        if output_path:
            self._visualize_spectrum(dct_blocks, masked_flat, Path(output_path))

        recon = self._idct2(masked_blocks)
        recon = np.clip(recon, 0, 255).astype(np.uint8)
        reconstructed = self._reconstruct(recon, original_shape)
        
        return Image.fromarray(reconstructed, mode="L").convert("RGB")

        # output_path = Path(output_path)
        # Image.fromarray(reconstructed, mode="L").save(output_path)

        # bytes_on_disk = output_path.stat().st_size
        # num_blocks = (original_shape[0] // self.block_size) * (original_shape[1] // self.block_size)
        # token_estimate = num_blocks

        # return CompressionStats(
        #     output_path=output_path,
        #     num_blocks=num_blocks,
        #     block_size=self.block_size,
        #     keep_ratio=keep_ratio,
        #     bytes_on_disk=bytes_on_disk,
        #     token_estimate=token_estimate,
        # )
