import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

def visualize_residual(original_path: str, compressed_path: str, output_path: str):
    """
    Creates a 3-panel visualization: Original, Compressed, and Residual Error.
    """
    # Load images
    img_orig = Image.open(original_path).convert("L")
    img_comp = Image.open(compressed_path).convert("L")
    
    # Ensure sizes match (compressed might be slightly padded due to block alignment)
    w, h = img_comp.size
    img_orig = img_orig.resize((w, h))
    
    arr_orig = np.array(img_orig, dtype=float)
    arr_comp = np.array(img_comp, dtype=float)
    
    # Compute absolute residual error
    # Boost intensity (x5) to make small errors visible
    residual = np.abs(arr_orig - arr_comp) * 5
    residual = np.clip(residual, 0, 255)
    
    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    
    axs[0].imshow(arr_orig, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Original Text")
    axs[0].axis('off')
    
    axs[1].imshow(arr_comp, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title("Compressed (VQ)")
    axs[1].axis('off')
    
    # Use 'hot' colormap for error to make it stand out against black background
    im = axs[2].imshow(residual, cmap='magma', vmin=0, vmax=255)
    axs[2].set_title("Residual Error (Amplified 5x)")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved residual visualization to {output_path}")

if __name__ == "__main__":
    # Paths from your artifacts folder
    base_dir = Path("artifacts")
    orig_path = base_dir / "conversation.png"
    
    if not orig_path.exists():
        print("Original image not found. Run demo_14h.py first.")
        exit(1)

    # Visualize MILD
    mild_path = base_dir / "mild_compressed.png"
    if mild_path.exists():
        visualize_residual(str(orig_path), str(mild_path), str(base_dir / "vis_residual_mild.png"))
        print("Saved MILD residual visualization.")

    # Visualize AGGRESSIVE
    agg_path = base_dir / "aggressive_compressed.png"
    if agg_path.exists():
        visualize_residual(str(orig_path), str(agg_path), str(base_dir / "vis_residual_aggressive.png"))
        print("Saved AGGRESSIVE residual visualization.")

