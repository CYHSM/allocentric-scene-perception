import os
os.environ['HF_HOME'] = '/media/markus/Elements/Github/allocentric-scene-perception/models/'
import argparse
import subprocess
import timm
import json

def get_timm_models(filter_pattern=None, pretrained_only=False):
    """Get list of available timm models"""
    return timm.list_models(filter_pattern, pretrained=pretrained_only)

def extract_model_features(model_name, args):
    """Extract features for a single model"""
    feature_file = os.path.join(args.output_dir, f"{model_name.replace('/', '_')}_{args.feature_mode}_features.h5")
    
    if os.path.exists(feature_file) and not args.force_recompute:
        print(f"Using existing feature file: {feature_file}")
        return feature_file
    
    print(f"Extracting features for {model_name}...")
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
        print(f"Running command: {' '.join(benchmark_cmd)}")
    
    try:
        subprocess.run(benchmark_cmd, check=True)
        return feature_file
    except subprocess.CalledProcessError as e:
        print(f"Error extracting features for {model_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract features from multiple models')
    parser.add_argument('--models', nargs='*', help='List of timm models to process')
    parser.add_argument('--model_pattern', type=str, help='Pattern to filter model names (e.g., "resnet*")')
    parser.add_argument('--pretrained_only', action='store_true', help='Only include models with pretrained weights')
    parser.add_argument('--data_path', type=str, default='../data/ASP_FixedSun/', help='Path to the ASP dataset')
    parser.add_argument('--output_dir', type=str, default='../features', help='Directory to store features')
    parser.add_argument('--num_scenes', type=int, default=100, help='Number of scenes to process')
    parser.add_argument('--max_scene_search', type=int, default=5000, help='Maximum number of scenes to search')
    parser.add_argument('--dataset_name', type=str, default='asp_surround_mix_noref', help='Dataset name')
    parser.add_argument('--feature_mode', type=str, default='pyramid', 
                      choices=['pyramid', 'penultimate_unpooled', 'penultimate_pooled'],
                      help='Feature extraction mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--list_all_models', action='store_true', help='List all available timm models and exit')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle listing all models
    if args.list_all_models:
        all_models = get_timm_models()
        print(f"Total available models: {len(all_models)}")
        for model in all_models:
            print(model)
        return
    
    # Determine which models to process
    if args.models:
        models = args.models
    elif args.model_pattern:
        models = get_timm_models(args.model_pattern, pretrained_only=args.pretrained_only)
        print(f"Found {len(models)} models matching pattern '{args.model_pattern}'")
    else:
        models = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'efficientnet_b0', 'efficientnet_b1', 
            'vit_small_patch16_224', 'vit_base_patch16_224',
            'convnext_tiny', 'convnext_small'
        ]
    
    print(f"Processing {len(models)} models with seed {args.seed}, {args.num_scenes} scenes")
    
    # Save metadata
    metadata = {
        'seed': args.seed,
        'num_scenes': args.num_scenes,
        'dataset_name': args.dataset_name,
        'feature_mode': args.feature_mode,
        'models': models
    }
    
    with open(os.path.join(args.output_dir, 'feature_extraction_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Extract features for each model
    feature_files = {}
    for model_name in models:
        print(f"\nProcessing model: {model_name}")
        feature_file = extract_model_features(model_name, args)
        if feature_file:
            feature_files[model_name] = feature_file
    
    # Save list of extracted feature files
    with open(os.path.join(args.output_dir, 'feature_files.json'), 'w') as f:
        json.dump(feature_files, f, indent=2)
    
    print(f"\nExtracted features for {len(feature_files)} models")
    print(f"Feature files saved in: {args.output_dir}")

if __name__ == "__main__":
    main()