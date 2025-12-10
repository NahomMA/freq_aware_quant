# Dataset Information

## MRCR Benchmark

**Multi-needle Retrieval in Conversational Records** - A benchmark for evaluating long-context vision-language models.

### Dataset Acquisition

The MRCR dataset is not included in this repository due to size constraints. Please obtain it through one of the following methods:

#### Option 1: Official Source
Contact the MRCR benchmark authors or check their official repository for dataset access.

#### Option 2: Generate from Text
Use the rendering script to generate images from conversation text:

```bash
python scripts/render_images.py \
    --input_json path/to/conversations.json \
    --output_dir data/rendered_images \
    --rendering_config glyph
```

### Dataset Structure

Expected JSON format:

```json
{
  "unique_id": "mrcr_2needle_0k_8k_sample_001",
  "question": "<|user|>What was mentioned in the first conversation?",
  "answer": "RANDOM_PREFIX The answer is ...",
  "random_string_to_prepend": "RANDOM_PREFIX",
  "image_paths": [
    "data/rendered_images/conv_001_turn_001.png",
    "data/rendered_images/conv_001_turn_002.png"
  ]
}
```

### Configurations

The MRCR benchmark has three difficulty levels:

| Configuration | Description | Token Range | Needles |
|--------------|-------------|-------------|---------|
| 2-needle | Short context | 0-8k tokens | 2 |
| 4-needle | Medium context | 8-16k tokens | 4 |
| 8-needle | Long context | 16-32k tokens | 8 |

### Sample Data

A minimal sample dataset with 2-3 entries is provided in `mrcr_sample.json` for testing purposes.

To test the pipeline without full dataset:

```bash
# Method 1 on sample data
python src/evaluation/local_inference.py \
    --data_path data/mrcr_sample.json \
    --baseline False

# Method 2 on sample data  
python src/evaluation/local_inference_compressed.py \
    --data_path data/mrcr_sample.json \
    --compression_profile mild
```

### Rendering Configuration

Images are rendered using the Glyph rendering pipeline with the following parameters:

- **DPI:** 150
- **Font:** Liberation Sans
- **Font Size:** 12pt
- **Layout:** Single column
- **Image Size:** Variable (based on content)
- **Block Size (Method 1):** 14×14 pixels for DCT

### Storage Requirements

**Full MRCR Dataset:**
- Raw images: ~5-10 GB
- Compressed (Method 1): ~2.5-5 GB (2× reduction)
- Rendered from text: ~500 MB (JSON + metadata)

**Sample Dataset:**
- Included: ~5-10 MB (2-3 samples)

### Additional Notes

1. **Image Format:** PNG (lossless) for Method 1 compression comparison
2. **Preprocessing:** Images are converted to grayscale before DCT compression
3. **Caching:** Rendered images can be cached to avoid re-rendering
4. **Parallel Rendering:** Use multiple CPUs for faster rendering of large datasets

### ConvBench (Milestone 1)

Initial validation used ConvBench with ChatGPT-4o black-box API. Results demonstrated feasibility before transitioning to local MRCR evaluation with GLM-4V.

ConvBench is not included in this submission but can be obtained from:
- Paper: "ConvBench: A Multi-Turn Conversation Evaluation Benchmark" (Wang et al., 2024)
- Dataset: Available through official channels

---

For questions about dataset access, please refer to the main README or contact the author.

