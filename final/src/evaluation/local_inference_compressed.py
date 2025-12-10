"""
MRCR Evaluation with Frequency-Aware Vision Compression
"""

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import json
import os
import sys
import re
from PIL import Image
from tqdm import tqdm
from difflib import SequenceMatcher
from collections import defaultdict
import multiprocessing as mp
import argparse


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compression.vision_compressor import VisionTokenCompressor


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


def infer_samples_on_gpu(samples, gpu_idx=None, compression_profile=None):
    """
    Inference for a batch of samples on the specified GPU with optional compression.
    
    Args:
        samples: List of sample dicts
        gpu_idx: GPU index (None for CPU)
        compression_profile: None (no compression), "mild", "aggressive", or "extreme"
    
    Returns:
        List of result dicts
    """
    device = torch.device(f"cuda:{gpu_idx}" if (gpu_idx is not None and torch.cuda.is_available()) else "cpu")
    
    # Load model
    processor = AutoProcessor.from_pretrained("zai-org/Glyph")
    model = AutoModelForImageTextToText.from_pretrained(
        "zai-org/Glyph",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device.index} if device.type == "cuda" else "cpu",
    )
    model = model.eval()
    
    # Setup compression if requested
    compressor = None
    hook_handle = None
    
    if compression_profile:
        print(f"[GPU {gpu_idx}] Using {compression_profile} compression")
        compressor = VisionTokenCompressor(
            grid_size=24,
            profile=compression_profile,
            verbose=False  # Disable verbose for batch processing
        )
        
        # Register compression hook
        def compress_hook(module, input, output):
            try:
                # Handle tuple output (some vision encoders return (embeddings, aux_info))
                if isinstance(output, tuple):
                    compressed = compressor.compress(output[0])
                    # Return tuple with compressed embeddings
                    return (compressed,) + output[1:]
                else:
                    return compressor.compress(output)
            except Exception as e:
                print(f"[GPU {gpu_idx}] Compression error: {e}")
                print(f"[GPU {gpu_idx}] Output type: {type(output)}")
                if hasattr(output, 'shape'):
                    print(f"[GPU {gpu_idx}] Output shape: {output.shape}")
                elif isinstance(output, tuple):
                    print(f"[GPU {gpu_idx}] Tuple length: {len(output)}")
                    for i, item in enumerate(output):
                        if hasattr(item, 'shape'):
                            print(f"[GPU {gpu_idx}]   Item {i} shape: {item.shape}")
                raise
        
        hook_handle = model.model.visual.register_forward_hook(compress_hook)
    else:
        print(f"[GPU {gpu_idx}] Running BASELINE (no compression)")
    
    results = []
    
    for item in tqdm(samples, desc=f"GPU {gpu_idx}"):
        unique_id = item['unique_id']
        question = item['question']
        answer = item['answer']
        random_prefix = item['random_string_to_prepend']
        image_paths = item['image_paths']
        
        # Load images
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        
        # Create message with multiple images
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        
        messages = [{"role": "user", "content": content}]
        
        # Track memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        
        # Run inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)
        
        # Track vision tokens (total prompt tokens - text tokens)
        total_prompt_tokens = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
        
        response = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # Measure memory
        peak_memory_mb = 0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        # Grade response
        score = grade_response(response, answer, random_prefix)
        category = extract_category(unique_id)
        
        # Calculate question tokens to get vision tokens
        question_tokens = len(processor.tokenizer.encode(question))
        vision_tokens = total_prompt_tokens - question_tokens
        
        results.append({
            'unique_id': unique_id,
            'question': question,
            'answer': answer,
            'response': response,
            'score': score,
            'category': category,
            'peak_memory_mb': peak_memory_mb,
            'compression_profile': compression_profile or "baseline",
            'vision_tokens': vision_tokens,
            'total_prompt_tokens': total_prompt_tokens
        })
    
    # Cleanup
    if hook_handle:
        hook_handle.remove()
    
    return results


def run_parallel_inference(data_path, output_path, compression_profile=None):
    """Run inference across multiple GPUs"""
    # Load dataset
    with open(data_path, 'r') as f:
        if data_path.endswith('.jsonl'):
            samples = [json.loads(line) for line in f]
        else:
            samples = json.load(f)
    
    # samples = samples[:5]
    gpu_count = get_gpu_count()
    print(f"Detecting available GPUs...")
    print(f"GPUs detected: {gpu_count}")
    
    # Split samples across GPUs
    chunk_size = len(samples) // gpu_count
    chunks = [samples[i*chunk_size:(i+1)*chunk_size] for i in range(gpu_count)]
    # Add remainder to last chunk
    if len(samples) % gpu_count != 0:
        chunks[-1].extend(samples[gpu_count*chunk_size:])
    
    print(f"Splitting {len(samples)} examples into {gpu_count} chunks ({chunk_size} per GPU)")
    print()
    
    # Run inference on each GPU
    with mp.Pool(processes=gpu_count) as pool:
        results_per_gpu = pool.starmap(
            infer_samples_on_gpu,
            [(chunk, i, compression_profile) for i, chunk in enumerate(chunks)]
        )
    
    # Flatten results
    all_results = [r for gpu_results in results_per_gpu for r in gpu_results]
    
    # Save results
    with open(output_path, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')
    
    return all_results


def compute_metrics(results):
    """Compute accuracy metrics from results"""
    # Overall accuracy
    overall_acc = sum(r['score'] for r in results) / len(results) 
    
    # By category
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        cat = r['category']
        category_stats[cat]['total'] += 1
       
        category_stats[cat]['correct'] += r['score']
    
    category_acc = {
        cat: stats['correct'] / stats['total'] 
        for cat, stats in category_stats.items()
    }
    
    # Average memory
    avg_memory = sum(r.get('peak_memory_mb', 0) for r in results) / len(results) if results else 0
    
    # Average vision tokens
    avg_vision_tokens = sum(r.get('vision_tokens', 0) for r in results) / len(results) if results else 0
    
    return {
        'overall_accuracy': overall_acc,
        'category_accuracy': category_acc,
        'avg_memory_mb': avg_memory,
        'avg_vision_tokens': avg_vision_tokens,
        'num_samples': len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="MRCR Evaluation with Compression")
    parser.add_argument("--compression", type=str, default=None,
                       choices=["mild", "aggressive", "extreme", "ultra_mild", "baseline"],
                       help="Compression profile (None for baseline)")
    parser.add_argument("--needle", type=str, default="all",
                       choices=["2", "4", "8", "all"],
                       help="Which needle test to run")
    args = parser.parse_args()
    
    # Determine which datasets to process
    if args.needle == "all":
        needle_types = ["2needle", "4needle", "8needle"]
    else:
        needle_types = [f"{args.needle}needle"]
    
    # Get project root (go up from src/evaluation/ to nahom-final/)
    script_dir = os.path.dirname(__file__)  # src/evaluation/
    project_root = os.path.dirname(os.path.dirname(script_dir))  # go up two levels
    all_metrics = {}
    
    for needle in needle_types:
        data_path = os.path.join(project_root, "data", f"processed_{needle}_0-128k.jsonl")
        
        # Output path with compression profile
        profile_str = args.compression if args.compression else "baseline"
        output_path = os.path.join(project_root, "results", f"local_{needle}_results_{profile_str}.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Processing {needle} examples from {data_path}")
        print(f"Compression: {profile_str}")
        print('='*80)
        
        # Run inference
        results = run_parallel_inference(data_path, output_path, args.compression)
        
        # Compute metrics
        metrics = compute_metrics(results)
        all_metrics[needle] = metrics
        
        # Print results
        print(f"\n{needle} Results:")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Average Memory: {metrics['avg_memory_mb']:.2f} MB")
        print(f"Average Vision Tokens: {metrics['avg_vision_tokens']:.0f}")
        
        # Calculate and print compression ratio if baseline exists
        baseline_path = os.path.join(project_root, "results", "summary_baseline.json")
        if args.compression and os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                baseline_metrics = json.load(f)
            if needle in baseline_metrics:
                baseline_tokens = baseline_metrics[needle].get('avg_vision_tokens', 0)
                compressed_tokens = metrics['avg_vision_tokens']
                if compressed_tokens > 0 and baseline_tokens > 0:
                    ratio = baseline_tokens / compressed_tokens
                    print(f"Token Compression Ratio: {ratio:.2f}x ({baseline_tokens:.0f} → {compressed_tokens:.0f})")
        
        print(f"\nBy Category:")
        for cat, acc in sorted(metrics['category_accuracy'].items()):
            print(f"  {cat}: {acc:.4f}")
    
    # Save summary
    summary_path = os.path.join(project_root, "results", f"summary_{profile_str}.json")
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✅ Results saved to {summary_path}")


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()

