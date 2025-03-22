import os
import argparse
import subprocess
import json
import shutil
import time
import logging
import open_clip
import numpy as np
import random
import torch
from tqdm import tqdm
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('openclip_feature_extraction.log')
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

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

def get_all_openclip_models():
    """Get list of all available OpenCLIP models"""
    try:
        models = open_clip.list_pretrained()
        logger.info(f"Found {len(models)} pretrained OpenCLIP models")
        return models
    except Exception as e:
        logger.error(f"Error getting OpenCLIP models: {e}")
        return []

def extract_model_features(model_info, args):
    """Extract features for a single OpenCLIP model and clean up afterward"""
    # Parse model name and pretrained tag
    model_name, pretrained_tag = model_info
    
    # Create a model identifier for file naming
    model_identifier = f"{model_name}_{pretrained_tag}".replace('/', '_')
    feature_file = os.path.join(args.output_dir, f"{model_identifier}_{args.feature_mode}_features.h5")
    
    if os.path.exists(feature_file) and not args.force_recompute:
        logger.info(f"Using existing feature file: {feature_file}")
        return feature_file
    
    # Record start time for performance tracking
    start_time = time.time()
    logger.info(f"Extracting features for {model_name}:{pretrained_tag}...")
    
    # Check available disk space before starting
    if hasattr(shutil, 'disk_usage'):  # Python 3.3+
        total, used, free = shutil.disk_usage(args.output_dir)
        logger.info(f"Disk space before extraction: {free / (1024**3):.2f} GB free")
    
    # Build the command for openclip_extract_features.py
    benchmark_cmd = [
        "python", "openclip_extract_features.py",
        "--model_name", f"{model_name}:{pretrained_tag}",
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--num_scenes", str(args.num_scenes),
        "--max_scene_search", str(args.max_scene_search),
        "--feature_mode", args.feature_mode,
        "--seed", str(args.seed),
        "--dataset_name", args.dataset_name
    ]
    
    # Add pyramid-specific args if using pyramid mode
    if args.feature_mode == 'pyramid':
        if args.pyramid_token_mode:
            benchmark_cmd.extend(["--pyramid_token_mode", args.pyramid_token_mode])
        if args.normalize_intermediates:
            benchmark_cmd.append("--normalize_intermediates")
    
    if args.verbose:
        benchmark_cmd.append("--verbose")
        logger.info(f"Running command: {' '.join(benchmark_cmd)}")
    
    try:
        # Run the benchmark command
        process = subprocess.run(benchmark_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Log the output
        stdout = process.stdout.decode('utf-8')
        stderr = process.stderr.decode('utf-8')
        
        if stdout:
            logger.info(f"Benchmark output for {model_name}:{pretrained_tag}:\n{stdout}")
        if stderr:
            logger.warning(f"Benchmark error output for {model_name}:{pretrained_tag}:\n{stderr}")
        
        # Clean model cache to free space
        if not args.keep_models:
            logger.info(f"Cleaning up model cache for {model_name}:{pretrained_tag}")
            clean_model_cache(args.model_cache_dir)
        
        # Log completion and timing
        elapsed_time = time.time() - start_time
        logger.info(f"Completed feature extraction for {model_name}:{pretrained_tag} in {elapsed_time:.2f} seconds")
        
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
        logger.error(f"Error extracting features for {model_name}:{pretrained_tag}: {e}")
        
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
        logger.error(f"Unexpected error for {model_name}:{pretrained_tag}: {e}")
        if not args.keep_models:
            clean_model_cache(args.model_cache_dir)
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract features from OpenCLIP models (Cloud Version)')
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
    parser.add_argument('--pyramid_token_mode', type=str, default='all', choices=['cls', 'all'],
                        help='For pyramid mode: extract only cls token or all tokens')
    parser.add_argument('--normalize_intermediates', action='store_true', default=False,
                        help='Apply normalization to intermediate features')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--force_recompute', action='store_true', 
                       help='Force recomputation')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print verbose output')
    parser.add_argument('--keep_models', action='store_true', 
                       help='Keep downloaded models (not recommended for cloud environments)')
    parser.add_argument('--cloud_status_file', type=str, default='openclip_extraction_status.json', 
                      help='JSON file to track extraction status')
    parser.add_argument('--custom_models', nargs='*', 
                       help='Custom list of specific OpenCLIP models to process (format: "model_name:pretrained_tag")')
    parser.add_argument('--num_models', type=int, default=100, 
                       help='Number of top models to process')
    parser.add_argument('--all_models', action='store_true',
                       help='Process all available OpenCLIP models')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
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
        # Parse custom models in format "model_name:pretrained_tag"
        models = []
        for model_str in args.custom_models:
            if ':' in model_str:
                model_name, pretrained_tag = model_str.split(':', 1)
                models.append((model_name, pretrained_tag))
            else:
                logger.warning(f"Invalid model format: {model_str}. Expected format: 'model_name:pretrained_tag'")
        logger.info(f"Using {len(models)} custom-specified models")
    else:
        # Get all available OpenCLIP models
        all_models = get_all_openclip_models()
        
        # Process all models if requested, otherwise limit to requested number
        if args.all_models:
            models = all_models
            logger.info(f"Processing all {len(models)} OpenCLIP models")
        else:
            # Limit to requested number
            if len(all_models) > args.num_models:
                models = all_models[:args.num_models]
            else:
                models = all_models
            
            logger.info(f"Selected {len(models)} OpenCLIP models (from top {args.num_models})")
    
    # Filter out already completed models
    if completed_models and not args.force_recompute:
        model_keys = [f"{m[0]}:{m[1]}" for m in models]
        remaining_model_keys = [m for m in model_keys if m not in completed_models]
        skipped_model_keys = [m for m in model_keys if m in completed_models]
        
        # Convert back to (model_name, pretrained_tag) format
        remaining_models = []
        for model_key in remaining_model_keys:
            model_name, pretrained_tag = model_key.split(':', 1)
            remaining_models.append((model_name, pretrained_tag))
        
        logger.info(f"Skipping {len(skipped_model_keys)} already completed models")
        logger.info(f"Remaining models to process: {len(remaining_models)}")
        models = remaining_models
    
    logger.info(f"Processing {len(models)} models with seed {args.seed}, {args.num_scenes} scenes")
    
    # Save metadata
    metadata = {
        'seed': args.seed,
        'num_scenes': args.num_scenes,
        'dataset_name': args.dataset_name,
        'feature_mode': args.feature_mode,
        'models': [f"{m[0]}:{m[1]}" for m in models],
        'cloud_execution': True,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(args.output_dir, 'openclip_extraction_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Extract features for each model
    feature_files = {}
    for i, model_info in enumerate(models):
        model_name, pretrained_tag = model_info
        model_key = f"{model_name}:{pretrained_tag}"
        
        logger.info(f"\nProcessing model {i+1}/{len(models)}: {model_key}")
        
        try:
            # Monitor system resources
            if hasattr(shutil, 'disk_usage'):
                total, used, free = shutil.disk_usage(args.output_dir)
                if free < 5 * (1024**3):  # Less than 5 GB free
                    logger.warning(f"Low disk space warning: {free / (1024**3):.2f} GB free")
                    
                    if free < 1 * (1024**3):  # Critical - less than 1 GB free
                        logger.critical("Critical disk space shortage - pausing extraction")
                        # Wait for potential cleanup or monitoring intervention
                        time.sleep(60)  # Wait a minute
                        
                        # Check again
                        total, used, free = shutil.disk_usage(args.output_dir)
                        if free < 1 * (1024**3):
                            logger.critical("Disk space still critical - skipping this model")
                            continue
            
            # Extract features
            feature_file = extract_model_features(model_info, args)
            
            if feature_file:
                feature_files[model_key] = feature_file
                completed_models[model_key] = feature_file
                
                # Update status file after each model
                with open(status_file, 'w') as f:
                    json.dump({
                        'completed_models': completed_models,
                        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'progress': f"{i+1}/{len(models)}",
                        'remaining_models': len(models) - (i+1)
                    }, f, indent=2)
        
        except Exception as e:
            logger.error(f"Unexpected error processing {model_key}: {e}", exc_info=True)
            # Continue with the next model
            continue
    
    # Save list of extracted feature files
    with open(os.path.join(args.output_dir, 'openclip_feature_files.json'), 'w') as f:
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
