from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config import load_config
from src.data.conversations import ConversationSample, get_samples
from src.rendering.renderer import ConversationRenderer
from src.compression.block_dct_vq import BlockDCTThresholdCompressor
# from src.compression.kmeans_vq import BlockDCTVQCompressor
from src.eval.metrics import token_compression_ratio
from src.eval.openai_eval import OpenAIImageEvaluator


def _artifact_path(cfg: dict, name: str) -> Path:
    base = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts"))
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def run_demo(config_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    samples = get_samples(cfg, limit=1)
    if not samples:
        raise RuntimeError("No conversation samples available.")
    sample = samples[0]

    renderer = ConversationRenderer(cfg)
    render_path = _artifact_path(cfg, "conversation.png")
    render_result = renderer.render(sample, render_path)

    compression_cfg = cfg.get("compression", {})
    block_size = compression_cfg.get("block_size", 8)
    
    # Use Threshold (Scalar) compressor for basic profiles, or switch based on label
    compressor_threshold = BlockDCTThresholdCompressor(block_size=block_size)
    # compressor_kmeans = BlockDCTVQCompressor(block_size=block_size)

    evaluator = OpenAIImageEvaluator(cfg)
    
    # Define profiles: (Label, UseKMeans, Clusters, KeepRatio)
    profiles = [
        ("mild", False, 0, compression_cfg["keep_ratio"]["mild"]),
        ("aggressive", True, compression_cfg["clusters"]["aggressive"], compression_cfg["keep_ratio"]["aggressive"]),
    ]

    reference_text = sample.to_text()
    # Approx token count (1.3 words per token usually, or just use words for now)
    # Using a simple heuristic: len(words) * 1.3
    text_tokens = int(len(reference_text.split()) * 1.3)

    # Calculate standard ViT tokens for the rendered image (e.g. patch size 14)
    # Formula: (H/14) * (W/14)
    patch_size = 14
    orig_visual_tokens = (render_result.height // patch_size) * (render_result.width // patch_size)

    print(f"\n[Input Data]")
    print(f"Source: {sample.source}")
    print(f"Text Length: {len(reference_text)} chars")
    print(f"Estimated Text Tokens: {text_tokens}")
    print(f"Original Rendered Image: {render_result.width}x{render_result.height}")
    print(f"Standard Visual Tokens (ViT-14): {orig_visual_tokens}")
    print(f"Visual Bloat Factor: {orig_visual_tokens/text_tokens:.2f}x (Image is more expensive than text!)")

    for label, use_kmeans, clusters, keep_ratio in profiles:
        output_path = _artifact_path(cfg, f"{label}_compressed.png")        
    
        stats = compressor_threshold.compress(
            render_result.path,
            output_path,
            keep_ratio=keep_ratio,
        )
        method_name = "Scalar Thresholding"

        compression_ratio = token_compression_ratio(text_tokens, stats.token_estimate)
        vlm_result = evaluator.evaluate(stats.output_path, reference_text)

        print(f"\n=== {label.upper()} PROFILE ({method_name}) ===")
        print(f"Blocks (visual tokens): {stats.token_estimate}")
        print(f"Token Compression Ratio: {compression_ratio:.2f}x")
        print(f"File Size: {stats.bytes_on_disk / 1024:.1f} KB")
        if vlm_result.success:
            print(f"Word Recall: {vlm_result.recall:.2f}")
            print(f"VLM Output:\n{vlm_result.text}")
        else:
            print(f"VLM evaluation skipped: {vlm_result.error}")


if __name__ == "__main__":
    run_demo()
