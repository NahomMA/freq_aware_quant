"""
Frequency-Aware Vision Token Compressor for GLM-4V
Applies DCT-based compression to vision token embeddings
"""

import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class CompressionStats:
    """Track compression metrics"""
    original_tokens: int
    compressed_size_mb: float
    compression_ratio: float
    profile: str
    

class VisionTokenCompressor(nn.Module):
    """
    Compress vision token embeddings using frequency-aware quantization.
    
    Key idea: Vision tokens from images form a 2D spatial grid (e.g., 24×24).
    Apply DCT to each channel, then quantize different frequency bands differently:
    - Low frequencies (DC, general structure): Keep at high precision (8-bit)
    - Mid frequencies (details): Moderate compression (4-6 bit)
    - High frequencies (noise, fine edges): Aggressive compression (2-bit or zero)
    
    This is inspired by JPEG compression and WaveletCompressedConvolution (NeurIPS 2022).
    """
    
    def __init__(
        self,
        grid_size: int = 24,  # For GLM-4V: 576 tokens = 24×24 spatial grid
        profile: Literal["mild", "aggressive", "extreme", "baseline","ultra_mild"] = "mild",
        verbose: bool = False
    ):
        super().__init__()
        self.grid_size = grid_size
        self.profile = profile
        self.verbose = verbose
        
        # Compression profiles (frequency_threshold_low, frequency_threshold_high, bits_per_band)
        self.profiles = {
            "mild": {
                "low_thresh": 12,
                "mid_thresh": 20,
                "bits": {"low": 12, "mid": 10, "high": 8},  # Much milder: preserve text details
                "description": "Conservative: 1.5-2x compression, minimal quality loss"
            },
            "ultra_mild": {
                "low_thresh": 16,
                "mid_thresh": 24,
                "bits": {"low": 16, "mid": 14, "high": 12},  # Ultra mild: preserve text details
                "description": "Ultra mild: 1.5-2x compression, minimal quality loss"
            },
            "aggressive": {
                "low_thresh": 4,
                "mid_thresh": 8,
                "bits": {"low": 8, "mid": 4, "high": 2},
                "description": "Aggressive: 4-5x compression, moderate quality loss"
            },
            "extreme": {
                "low_thresh": 3,
                "mid_thresh": 6,
                "bits": {"low": 8, "mid": 3, "high": 0},  # 0 = discard
                "description": "Extreme: 6-8x compression, significant quality loss"
            },
            "baseline": {
            "low_thresh": 22,
            "mid_thresh": 24,  # Everything is "low frequency"
            "bits": {"low": 16, "mid": 16, "high": 16},  # Full FP16 precision
            "description": "Near-passthrough: ~1.0x, virtually no compression"
            }
        }
        
        self.config = self.profiles[profile]
        
        if verbose:
            print(f"[VisionCompressor] Profile: {profile}")
            print(f"  {self.config['description']}")
            print(f"  Low-freq (<{self.config['low_thresh']}): {self.config['bits']['low']}-bit")
            print(f"  Mid-freq ({self.config['low_thresh']}-{self.config['mid_thresh']}): {self.config['bits']['mid']}-bit")
            print(f"  High-freq (>{self.config['mid_thresh']}): {self.config['bits']['high']}-bit")
    
    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress vision token embeddings.
        
        Args:
            vision_tokens: [batch, num_tokens, hidden_dim]
                          e.g., [1, 576, 1536] for GLM-4V
        
        Returns:
            compressed_tokens: Same shape but with reduced precision (fewer unique values)
        """
        return self.compress(vision_tokens)
    
    def compress(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        """Main compression function"""
        # Handle different output shapes from vision encoder
        original_ndim = vision_tokens.ndim
        if vision_tokens.ndim == 2:
            # Shape: [num_tokens, hidden_dim] - add batch dimension
            vision_tokens = vision_tokens.unsqueeze(0)
        elif vision_tokens.ndim != 3:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {vision_tokens.shape}")
        
        batch, num_tokens, hidden_dim = vision_tokens.shape
        device = vision_tokens.device
        dtype = vision_tokens.dtype
        
        # Check if token count matches the expected 2D grid
        expected_tokens = self.grid_size ** 2
        
        # If perfect match, use 2D compression
        if num_tokens == expected_tokens:
            return self._compress_2d(vision_tokens, batch, num_tokens, hidden_dim, device, dtype, original_ndim)
        
        # Otherwise, fallback to 1D block compression to avoid scrambling spatial structure
        if self.verbose:
            print(f"[Info] Token count {num_tokens} != {self.grid_size}x{self.grid_size} ({expected_tokens}). Using 1D block compression.")
            
        return self._compress_1d_blocks(vision_tokens, batch, num_tokens, hidden_dim, device, dtype, original_ndim)

    def _compress_2d(self, vision_tokens, batch, num_tokens, hidden_dim, device, dtype, original_ndim):
        # Reshape to spatial grid: [batch, 576, 1536] → [batch, 24, 24, 1536]
        grid = vision_tokens.reshape(batch, self.grid_size, self.grid_size, hidden_dim)
        
        # Process each channel independently
        compressed_channels = []
        
        for ch_idx in range(hidden_dim):
            # Extract one channel: [batch, 24, 24]
            channel = grid[:, :, :, ch_idx]
            
            # Cast to float32 for FFT (bfloat16 not supported)
            channel_f32 = channel.float()
            
            # Apply 2D DCT (converts spatial → frequency domain)
            dct_coeffs = fft.rfft2(channel_f32)  # Returns complex values
            
            # Frequency-aware quantization
            compressed_dct = self._quantize_by_frequency(dct_coeffs, device)
            
            # Inverse DCT (converts frequency → spatial domain)
            reconstructed = fft.irfft2(compressed_dct, s=(self.grid_size, self.grid_size))
            
            compressed_channels.append(reconstructed)
        
        # Stack channels back: [batch, 24, 24, 1536]
        compressed_grid = torch.stack(compressed_channels, dim=-1)
        
        # Reshape to token format: [batch, 576, 1536]
        compressed_tokens = compressed_grid.reshape(batch, num_tokens, hidden_dim)
        
        # Cast back to original dtype
        compressed_tokens = compressed_tokens.to(dtype)
        
        # Remove batch dimension if input was 2D
        if original_ndim == 2:
            compressed_tokens = compressed_tokens.squeeze(0)
        
        return compressed_tokens

    def _compress_1d_blocks(self, vision_tokens, batch, num_tokens, hidden_dim, device, dtype, original_ndim):
        """Compress using 1D DCT on blocks"""
        block_size = self.grid_size
        
        # Pad to multiple of block_size
        pad_len = (block_size - (num_tokens % block_size)) % block_size
        if pad_len > 0:
            padding = torch.zeros(batch, pad_len, hidden_dim, device=device, dtype=dtype)
            padded_input = torch.cat([vision_tokens, padding], dim=1)
        else:
            padded_input = vision_tokens
            
        padded_tokens = padded_input.shape[1]
        num_blocks = padded_tokens // block_size
        
        # Reshape to [batch * num_blocks, block_size, hidden_dim]
        blocks = padded_input.reshape(batch * num_blocks, block_size, hidden_dim)
        
        compressed_channels = []
        for ch_idx in range(hidden_dim):
            channel = blocks[:, :, ch_idx]
            channel_f32 = channel.float()
            
            # 1D DCT
            dct_coeffs = fft.rfft(channel_f32, dim=1)
            
            # Quantize 1D
            quantized_dct = self._quantize_by_frequency_1d(dct_coeffs, device)
            
            # Inverse 1D DCT
            reconstructed = fft.irfft(quantized_dct, n=block_size, dim=1)
            compressed_channels.append(reconstructed)
            
        compressed_blocks = torch.stack(compressed_channels, dim=-1)
        
        # Reshape back
        compressed_tokens = compressed_blocks.reshape(batch, padded_tokens, hidden_dim)
        
        if pad_len > 0:
            compressed_tokens = compressed_tokens[:, :num_tokens, :]
            
        compressed_tokens = compressed_tokens.to(dtype)
        
        if original_ndim == 2:
            compressed_tokens = compressed_tokens.squeeze(0)
            
        return compressed_tokens

    def _quantize_by_frequency_1d(self, dct_coeffs: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Quantize 1D DCT coefficients based on frequency"""
        freq_bins = dct_coeffs.shape[-1]
        freq_dist = torch.arange(freq_bins, device=device).float()
        
        low_thresh = self.config["low_thresh"]
        mid_thresh = self.config["mid_thresh"]
        bits = self.config["bits"]
        
        low_mask = freq_dist < low_thresh
        mid_mask = (freq_dist >= low_thresh) & (freq_dist < mid_thresh)
        high_mask = freq_dist >= mid_thresh
        
        compressed = torch.zeros_like(dct_coeffs)
        
        if bits["low"] > 0:
            compressed[..., low_mask] = self._uniform_quantize(dct_coeffs[..., low_mask], bits["low"])
        
        if bits["mid"] > 0:
            # Check if mid_mask has any True values
            if mid_mask.any():
                compressed[..., mid_mask] = self._uniform_quantize(dct_coeffs[..., mid_mask], bits["mid"])
        
        if bits["high"] > 0:
            if high_mask.any():
                compressed[..., high_mask] = self._uniform_quantize(dct_coeffs[..., high_mask], bits["high"])
                
        return compressed
    
    def _quantize_by_frequency(self, dct_coeffs: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Apply different quantization levels to different frequency bands.
        
        Args:
            dct_coeffs: [batch, H, W] complex tensor of DCT coefficients
        
        Returns:
            compressed_dct: Same shape, but quantized
        """
        # Get spatial dimensions
        H, W = dct_coeffs.shape[-2:]
        
        # Compute frequency distance from DC component (0, 0)
        # DC = average value (most important!)
        # Low freq = general shapes
        # High freq = edges, noise
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        freq_dist = torch.sqrt(yy**2 + xx**2)
        
        # Get quantization thresholds from profile
        low_thresh = self.config["low_thresh"]
        mid_thresh = self.config["mid_thresh"]
        bits = self.config["bits"]
        
        # Create frequency band masks
        low_mask = freq_dist < low_thresh
        mid_mask = (freq_dist >= low_thresh) & (freq_dist < mid_thresh)
        high_mask = freq_dist >= mid_thresh
        
        # Initialize compressed output
        compressed = torch.zeros_like(dct_coeffs)
        
        # Quantize each frequency band with appropriate bit depth
        if bits["low"] > 0:
            compressed[..., low_mask] = self._uniform_quantize(
                dct_coeffs[..., low_mask], 
                bits["low"]
            )
        
        if bits["mid"] > 0:
            compressed[..., mid_mask] = self._uniform_quantize(
                dct_coeffs[..., mid_mask], 
                bits["mid"]
            )
        
        if bits["high"] > 0:
            compressed[..., high_mask] = self._uniform_quantize(
                dct_coeffs[..., high_mask], 
                bits["high"]
            )
        # If bits["high"] == 0, those coefficients stay zero (maximum compression)
        
        return compressed
    
    def _uniform_quantize(self, values: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Uniform quantization to N bits.
        
        Maps values to 2^bits discrete levels, then reconstructs.
        This reduces the number of unique values, enabling compression.
        """
        if values.numel() == 0:
            return values
        
        # Handle complex values (DCT produces complex coefficients)
        if values.is_complex():
            real_quantized = self._uniform_quantize(values.real, bits)
            imag_quantized = self._uniform_quantize(values.imag, bits)
            return torch.complex(real_quantized, imag_quantized)
        
        # Find value range
        v_min = values.min()
        v_max = values.max()
        
        # Handle degenerate case (all values same)
        if v_max - v_min < 1e-8:
            return values
        
        # Quantize to N bits = 2^N levels
        levels = 2 ** bits
        
        # Normalize to [0, 1]
        normalized = (values - v_min) / (v_max - v_min)
        
        # Map to integer levels
        quantized_int = (normalized * (levels - 1)).round()
        
        # Dequantize back to continuous values
        dequantized = quantized_int / (levels - 1) * (v_max - v_min) + v_min
        
        return dequantized
    
    def estimate_compression_ratio(self, vision_tokens: torch.Tensor) -> CompressionStats:
        """
        Estimate compression ratio (theoretical, based on bit counts).
        
        Args:
            vision_tokens: [batch, num_tokens, hidden_dim]
        
        Returns:
            CompressionStats with estimated savings
        """
        batch, num_tokens, hidden_dim = vision_tokens.shape
        
        # Original size (assuming FP16 = 2 bytes per value)
        original_bytes = num_tokens * hidden_dim * 2
        original_mb = original_bytes / (1024 ** 2)
        
        # Compute how many coefficients in each frequency band
        H = W = self.grid_size
        y = torch.arange(H).float()
        x = torch.arange(W).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        freq_dist = torch.sqrt(yy**2 + xx**2)
        
        low_thresh = self.config["low_thresh"]
        mid_thresh = self.config["mid_thresh"]
        bits = self.config["bits"]
        
        low_count = (freq_dist < low_thresh).sum().item()
        mid_count = ((freq_dist >= low_thresh) & (freq_dist < mid_thresh)).sum().item()
        high_count = (freq_dist >= mid_thresh).sum().item()
        
        # Compressed size (bits per coefficient / 8 = bytes)
        # Note: For complex DCT, we have ~H×W/2 coefficients per channel
        rfft_size = H * (W // 2 + 1)  # rfft2 output size
        
        # Rough estimate (real + imaginary parts)
        compressed_bytes_per_channel = (
            low_count * bits["low"] / 8 * 2 +  # ×2 for complex
            mid_count * bits["mid"] / 8 * 2 +
            high_count * bits["high"] / 8 * 2
        )
        
        compressed_bytes = compressed_bytes_per_channel * hidden_dim * batch
        compressed_mb = compressed_bytes / (1024 ** 2)
        
        ratio = original_mb / compressed_mb if compressed_mb > 0 else float('inf')
        
        return CompressionStats(
            original_tokens=num_tokens,
            compressed_size_mb=compressed_mb,
            compression_ratio=ratio,
            profile=self.profile
        )



    
 

        




