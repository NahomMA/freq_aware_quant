from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import json
import os
from PIL import Image
from tqdm import tqdm
from difflib import SequenceMatcher
from collections import defaultdict
import re
import multiprocessing as mp
import math
import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.block_dct_vq import BlockDCTThresholdCompressor

compressor = BlockDCTThresholdCompressor(block_size=14)

def grade_response(response, answer, random_string_to_prepend):
    """Grading function using SequenceMatcher"""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response_clean = response[len(random_string_to_prepend):].replace('<|user|>', '')
    answer_clean = answer[len(random_string_to_prepend):] if answer.startswith(random_string_to_prepend) else answer
    return float(SequenceMatcher(None, response_clean, answer_clean).ratio())

def extract_category(unique_id):
    """Extract category from unique_id"""
    for needle in ['2needle', '4needle', '8needle']:
        for range_str in ['0k_8k', '8k_16k', '16k_32k', '32k_64k', '64k_128k']:
            category = f"{needle}_{range_str}"
            if category in unique_id:
                return category
    return "unknown"

def get_gpu_count():
    """Detect the number of available CUDA devices."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 1

def infer_samples_on_gpu(samples, gpu_idx=None,baseline=False):
    """
    Inference for a batch of samples on the specified GPU (by gpu_idx).
    Returns a list of result dicts.
    """
    # Pin model to this GPU only (even if device_map='auto' for consistency)
    
    device = torch.device(f"cuda:{gpu_idx}" if (gpu_idx is not None and torch.cuda.is_available()) else "cpu")
    processor = AutoProcessor.from_pretrained("zai-org/Glyph")
    model = AutoModelForImageTextToText.from_pretrained(
        "zai-org/Glyph",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device.index} if device.type == "cuda" else "cpu",
    )
    model = model.eval()
    results = []
    for item in samples:
        unique_id = item['unique_id']
        question = item['question']
        answer = item['answer']
        random_prefix = item['random_string_to_prepend']
        image_paths = item['image_paths']
      

        # Load images
        # use treshold compression to compress images
        if not baseline:
            # Aggressive compression: keep only 50% of coefficients
            # images = [compressor.compress(Image.open(img_path).convert("RGB"), keep_ratio=0.5) for img_path in image_paths]
            images = []
            for img_path in image_paths:
                original_img = Image.open(img_path).convert("RGB")
                compressed_img = compressor.compress(original_img, keep_ratio=0.5)
                
                # Determine save path
                if 'rendered_images' in img_path:
                    save_path = img_path.replace('rendered_images', 'compressed_images')
                else:
                    filename = os.path.basename(img_path)
                    parent_dir = os.path.basename(os.path.dirname(img_path))
                    # Ensure we point to the correct root if structure is unexpected
                    save_path = os.path.join(project_root, 'evaluation', 'mrcr', 'compressed_images', parent_dir, filename)
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                compressed_img.save(save_path)
                images.append(compressed_img)
        else:
            images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        # images = [Image.open(img_path).convert("RGB") for img_path in image_paths]

        # Create message with multiple images
        # breakpoint()
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]
        # breakpoint()

        # Run inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        response = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Grade response
        score = grade_response(response, answer, random_prefix)
        category = extract_category(unique_id)

        # Save result
        result = {
            "unique_id": unique_id,
            "question": question,
            "answer": answer,
            "response": response,
            "score": score,
            "category": category
        }
        results.append(result)
        
        # Clean up
        del inputs, generated_ids
        torch.cuda.empty_cache()
    # Clean up model before leaving process
    del model
    return results

if __name__ == '__main__':
    # Get project root (go up from src/evaluation/ to nahom-final/)
    script_dir = os.path.dirname(__file__)  # src/evaluation/
    project_root = os.path.dirname(os.path.dirname(script_dir))  # go up two levels
    
    print("Detecting available GPUs...")
    num_gpus = get_gpu_count()
    print(f"GPUs detected: {num_gpus}")

    print("Loading processor for tokenization...")
    processor = AutoProcessor.from_pretrained("zai-org/Glyph")
    print("Processor loaded.")

    # Create results directory
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    for needle in [2, 4, 8]:
        jsonl_file = os.path.join(project_root, 'data', f'processed_{needle}needle_0-128k.jsonl')
        output_file = os.path.join(results_dir, f'local_{needle}needle_results.jsonl')

        print(f"\nProcessing {needle}-needle examples from {jsonl_file}")

        with open(jsonl_file, 'r') as f:
            data = [json.loads(line) for line in f]

        # Only use the first 10 samples for demo/test (feel free to increase for real runs!)
        # data = data[:5]
        # print(f"Loaded {len(data)} examples (limited to 10 samples)")
        

        # Parallel batching per GPU (round robin)
        chunk_size = math.ceil(len(data) / num_gpus)
        data_chunks = [
            data[i*chunk_size : (i+1)*chunk_size]
            for i in range(num_gpus)
        ]
        # Remove empty chunks (in case len(data) < num_gpus)
        data_chunks = [c for c in data_chunks if c]
        print(f"Splitting {len(data)} examples into {len(data_chunks)} chunks ({chunk_size} per GPU)")

        
        # usingg baseline and compressed inference
        # Use multiprocessing (each process gets a chunk and a GPU index)
        for baseline in [True, False]:
            mode_name = "Baseline (No Compression)" if baseline else "Compressed (keep_ratio=0.5)"
            print(f"\n=== Running {mode_name} ===")
            
            results = []
            pool_args = [(chunk, idx, baseline) for idx, chunk in enumerate(data_chunks)]
            if len(pool_args) > 1:
                with mp.get_context("spawn").Pool(len(pool_args)) as pool:
                    multi_gpu_results = pool.starmap(infer_samples_on_gpu, pool_args)
                for rchunk in multi_gpu_results:
                    results.extend(rchunk)
            else:
                # Just use one process for CPU/single GPU
                results = infer_samples_on_gpu(data_chunks[0], 0 if num_gpus > 0 else None, baseline=baseline)

            # Score aggregation by category 
            scores_by_category = defaultdict(list)
            for r in results:
                scores_by_category[r["category"]].append(r["score"])

            # Save results
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')

            # Print statistics
            print(f"\n{needle}-needle Results:")
            print(f"Overall Accuracy: {sum(r['score'] for r in results) / len(results):.4f}")
            print("\nBy Category:")
            for category, scores in sorted(scores_by_category.items()):
                avg_score = sum(scores) / len(scores)