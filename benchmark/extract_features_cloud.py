import os
import argparse
import subprocess
import timm
import json
import shutil
import time
import logging
import requests
import csv
from io import StringIO
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('top100_feature_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_model_cache(cache_dir='./models'):
    """Set up a temporary model cache directory"""
    # Create the models directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set the HF_HOME environment variable to use this directory
    os.environ['HF_HOME'] = os.path.abspath(cache_dir)
    logger.info(f"Using model cache directory: {os.path.abspath(cache_dir)}")
    
    return cache_dir

def clean_model_cache(cache_dir='./models'):
    """Clean up the model cache to free space"""
    # Calculate initial space used
    if os.path.exists(cache_dir):
        initial_size = get_dir_size(cache_dir)
        logger.info(f"Cleaning model cache. Current size: {initial_size / (1024**2):.2f} MB")
        
        # Delete everything in the cache directory
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                logger.error(f"Error deleting {item_path}: {e}")
        
        # Calculate freed space
        if os.path.exists(cache_dir):
            final_size = get_dir_size(cache_dir)
            logger.info(f"Cache cleaned. Freed {(initial_size - final_size) / (1024**2):.2f} MB")
    else:
        logger.warning(f"Cache directory {cache_dir} does not exist")

def get_dir_size(path):
    """Calculate the total size of a directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

def get_timm_models(filter_pattern=None, pretrained_only=True):
    """Get list of available timm models"""
    models = timm.list_models(filter_pattern, pretrained=pretrained_only)
    logger.info(f"Found {len(models)} models matching the criteria")
    return models

def get_top_models(limit=100):
    """
    Get a list of top timm models based on ImageNet performance.
    Uses the results CSV from pytorch-image-models repository.
    If the data cannot be fetched, fall back to a predefined list of popular models.
    """
    # First try to get models from timm's available pretrained models
    all_pretrained = timm.list_models(pretrained=True)
    logger.info(f"Found {len(all_pretrained)} pretrained models in timm")
    
    # If we have fewer than the limit, return all available
    if len(all_pretrained) <= limit:
        logger.info(f"Returning all {len(all_pretrained)} available pretrained models")
        return all_pretrained
    
    try:
        # Try to fetch the ImageNet results CSV from GitHub
        results_url = "https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/results-imagenet.csv"
        logger.info(f"Fetching model performance data from: {results_url}")
        
        response = requests.get(results_url)
        response.raise_for_status()  # Ensure we got a valid response
        
        # Parse the CSV data
        csv_data = response.text.splitlines()
        
        # Parse header and find column indices
        header = csv_data[0].split(',')
        model_col = header.index('model')
        top1_col = header.index('top1')
        
        # Extract model performance data
        model_performance = {}
        for line in csv_data[1:]:  # Skip header
            values = line.split(',')
            model_name = values[model_col]
            
            # Clean up model name if needed (remove quotes, etc.)
            model_name = model_name.strip('"\'')
            
            # Try to get top1 accuracy
            try:
                top1_acc = float(values[top1_col])
                model_performance[model_name] = top1_acc
            except (ValueError, IndexError):
                # Skip models with invalid data
                continue
        
        logger.info(f"Parsed performance data for {len(model_performance)} models")
        
        # Filter to only include models that are available in timm with pretrained weights
        available_performance = {
            model: acc for model, acc in model_performance.items() 
            if model in all_pretrained
        }
        
        logger.info(f"Found performance data for {len(available_performance)} available pretrained models")
        
        # Sort models by accuracy (descending)
        sorted_models = sorted(available_performance.items(), key=lambda x: x[1], reverse=True)
        
        # Get top models
        top_models = [model[0] for model in sorted_models[:limit]]
        
        # If we couldn't get enough models with performance data, add other available models
        if len(top_models) < limit:
            remaining_models = [m for m in all_pretrained if m not in top_models]
            remaining_needed = limit - len(top_models)
            
            if remaining_needed > 0 and remaining_models:
                logger.info(f"Adding {remaining_needed} additional models to reach requested limit")
                top_models.extend(remaining_models[:remaining_needed])
        
        logger.info(f"Selected top {len(top_models)} models by ImageNet accuracy")
        return top_models
        
    except Exception as e:
        logger.error(f"Error getting model performance data: {e}")
        
        # Fallback to a predefined list of popular model architectures
        # with variety in architecture types and sizes
        fallback_patterns = [
            "resnet*",
            "efficientnet*",
            "vit*",
            "swin*",
            "convnext*",
            "regnety*",
            "deit*",
            "eva*",
            "beit*",
            "maxvit*",
            "dinov2*"
        ]
        
        logger.info(f"Using fallback model patterns: {fallback_patterns}")
        
        # Collect models matching these patterns
        fallback_models = []
        for pattern in fallback_patterns:
            models = timm.list_models(pattern, pretrained=True)
            fallback_models.extend(models)
        
        # Remove duplicates
        fallback_models = list(set(fallback_models))
        
        # Sort alphabetically to get consistent results
        fallback_models.sort()
        
        # Limit to requested number
        if len(fallback_models) > limit:
            fallback_models = fallback_models[:limit]
            
        logger.info(f"Selected {len(fallback_models)} fallback models")
        return fallback_models

def extract_model_features(model_name, args):
    """Extract features for a single model and clean up afterward"""
    feature_file = os.path.join(args.output_dir, f"{model_name.replace('/', '_')}_{args.feature_mode}_features.h5")
    
    if os.path.exists(feature_file) and not args.force_recompute:
        logger.info(f"Using existing feature file: {feature_file}")
        return feature_file
    
    # Record start time for performance tracking
    start_time = time.time()
    logger.info(f"Extracting features for {model_name}...")
    
    # Check available disk space before starting
    if hasattr(shutil, 'disk_usage'):  # Python 3.3+
        total, used, free = shutil.disk_usage(args.output_dir)
        logger.info(f"Disk space before extraction: {free / (1024**3):.2f} GB free")
    
    benchmark_cmd = [
        "python", "benchmark.py",
        "--model_name", model_name,
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--num_scenes", str(args.num_scenes),
        "--max_scene_search", str(args.max_scene_search),
        "--feature_mode", args.feature_mode,
        "--seed", str(args.seed),
        "--dataset_name", args.dataset_name
    ]
    
    if args.verbose:
        logger.info(f"Running command: {' '.join(benchmark_cmd)}")
    
    try:
        # Run the benchmark command
        process = subprocess.run(benchmark_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Log the output
        stdout = process.stdout.decode('utf-8')
        stderr = process.stderr.decode('utf-8')
        
        if stdout:
            logger.info(f"Benchmark output for {model_name}:\n{stdout}")
        if stderr:
            logger.warning(f"Benchmark error output for {model_name}:\n{stderr}")
        
        # Clean model cache to free space
        if not args.keep_models:
            logger.info(f"Cleaning up model cache for {model_name}")
            clean_model_cache(args.model_cache_dir)
        
        # Log completion and timing
        elapsed_time = time.time() - start_time
        logger.info(f"Completed feature extraction for {model_name} in {elapsed_time:.2f} seconds")
        
        # Check file size
        if os.path.exists(feature_file):
            file_size = os.path.getsize(feature_file) / (1024**2)
            logger.info(f"Feature file size: {file_size:.2f} MB")
            
            # Check available disk space after extraction
            if hasattr(shutil, 'disk_usage'):
                total, used, free = shutil.disk_usage(args.output_dir)
                logger.info(f"Disk space after extraction: {free / (1024**3):.2f} GB free")
        
        return feature_file
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting features for {model_name}: {e}")
        
        # Log error output
        if e.stdout:
            logger.error(f"Error stdout: {e.stdout.decode('utf-8')}")
        if e.stderr:
            logger.error(f"Error stderr: {e.stderr.decode('utf-8')}")
            
        # Clean up even if there was an error
        if not args.keep_models:
            clean_model_cache(args.model_cache_dir)
        return None
    except Exception as e:
        logger.error(f"Unexpected error for {model_name}: {e}")
        if not args.keep_models:
            clean_model_cache(args.model_cache_dir)
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract features from top 100 timm models (Cloud Version)')
    parser.add_argument('--data_path', type=str, default='../data/ASP_FixedSun/', 
                       help='Path to the ASP dataset')
    parser.add_argument('--output_dir', type=str, default='../features', 
                       help='Directory to store features')
    parser.add_argument('--model_cache_dir', type=str, default='./models', 
                       help='Directory to temporarily store model files')
    parser.add_argument('--num_scenes', type=int, default=100, 
                       help='Number of scenes to process')
    parser.add_argument('--max_scene_search', type=int, default=5000, 
                       help='Maximum number of scenes to search')
    parser.add_argument('--dataset_name', type=str, default='asp_surround_mix_noref', 
                       help='Dataset name')
    parser.add_argument('--feature_mode', type=str, default='pyramid', 
                      choices=['pyramid', 'penultimate_unpooled', 'penultimate_pooled'],
                      help='Feature extraction mode')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--force_recompute', action='store_true', 
                       help='Force recomputation')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print verbose output')
    parser.add_argument('--keep_models', action='store_true', 
                       help='Keep downloaded models (not recommended for cloud environments)')
    parser.add_argument('--cloud_status_file', type=str, default='top100_extraction_status.json', 
                      help='JSON file to track extraction status')
    parser.add_argument('--custom_models', nargs='*', 
                       help='Custom list of specific models to process instead of top 100')
    parser.add_argument('--num_models', type=int, default=100, 
                       help='Number of top models to process')
    
    args = parser.parse_args()
    
    # Set up the model cache directory
    args.model_cache_dir = setup_model_cache(args.model_cache_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load status if it exists
    status_file = os.path.join(args.output_dir, args.cloud_status_file)
    completed_models = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
                completed_models = status_data.get('completed_models', {})
                logger.info(f"Loaded status file with {len(completed_models)} completed models")
        except Exception as e:
            logger.error(f"Error loading status file: {e}")
    
    # Determine which models to process
    if args.custom_models:
        models = args.custom_models
        logger.info(f"Using {len(models)} custom-specified models")
    else:
        # Get top models
        models = get_top_models(limit=args.num_models)
        logger.info(f"Selected {len(models)} top models")
    
    # Filter out already completed models
    if completed_models and not args.force_recompute:
        remaining_models = [m for m in models if m not in completed_models]
        skipped_models = [m for m in models if m in completed_models]
        logger.info(f"Skipping {len(skipped_models)} already completed models")
        logger.info(f"Remaining models to process: {len(remaining_models)}")
        models = remaining_models
    
    logger.info(f"Processing {len(models)} models with seed {args.seed}, {args.num_scenes} scenes")
    
    # Save metadata
    metadata = {
        'seed': args.seed,
        'num_scenes': args.num_scenes,
        'dataset_name': args.dataset_name,
        'feature_mode': args.feature_mode,
        'models': models,
        'cloud_execution': True,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(args.output_dir, 'top100_extraction_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Extract features for each model
    feature_files = {}
    for i, model_name in enumerate(models):
        logger.info(f"\nProcessing model {i+1}/{len(models)}: {model_name}")
        
        try:
            # Monitor system resources
            if hasattr(shutil, 'disk_usage'):
                total, used, free = shutil.disk_usage(args.output_dir)
                if free < 5 * (1024**3):  # Less than 5 GB free
                    logger.warning(f"Low disk space warning: {free / (1024**3):.2f} GB free")
                    
                    if free < 1 * (1024**3):  # Critical - less than 1 GB free
                        logger.critical("Critical disk space shortage - pausing extraction")
                        # Wait for potential cleanup or monitoring interventation
                        time.sleep(60)  # Wait a minute
                        
                        # Check again
                        total, used, free = shutil.disk_usage(args.output_dir)
                        if free < 1 * (1024**3):
                            logger.critical("Disk space still critical - skipping this model")
                            continue
            
            # Extract features
            feature_file = extract_model_features(model_name, args)
            
            if feature_file:
                feature_files[model_name] = feature_file
                completed_models[model_name] = feature_file
                
                # Update status file after each model
                with open(status_file, 'w') as f:
                    json.dump({
                        'completed_models': completed_models,
                        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'progress': f"{i+1}/{len(models)}",
                        'remaining_models': len(models) - (i+1)
                    }, f, indent=2)
        
        except Exception as e:
            logger.error(f"Unexpected error processing {model_name}: {e}", exc_info=True)
            # Continue with the next model
            continue
    
    # Save list of extracted feature files
    with open(os.path.join(args.output_dir, 'top100_feature_files.json'), 'w') as f:
        json.dump(feature_files, f, indent=2)
    
    logger.info(f"\nExtracted features for {len(feature_files)} models")
    logger.info(f"Feature files saved in: {args.output_dir}")
    
    # Final cleanup
    clean_model_cache(args.model_cache_dir)

if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {e}", exc_info=True)