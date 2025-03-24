import os
import argparse
import subprocess
import json
from tqdm import tqdm

def run_token_triplet_analysis(feature_file, output_dir, seed, token_mode='each_patch', patch_indices=None, 
                             force_recompute=False, verbose=False, visualize_all_patches=False):
    """Run token-specific triplet analysis on a feature file"""
    # Extract model name from feature file
    basename = os.path.basename(feature_file)
    if '_pyramid_' in basename:
        model_name = basename.split('_pyramid_')[0]
    elif '_penultimate_' in basename:
        model_name = basename.split('_penultimate_')[0]
    else:
        model_name = basename.split('_')[0]
    
    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name.replace('/', '_'))
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Set path for results file
    # For each_patch mode, the result will be in a subdirectory
    if token_mode == 'each_patch':
        token_results_dir = os.path.join(model_output_dir, f"{model_name.replace('/', '_')}_token_analysis")
        results_file = os.path.join(token_results_dir, f"{model_name.replace('/', '_')}_all_token_results.json")
    else:
        # For specific token modes
        token_suffix = ""
        if token_mode == 'cls':
            token_suffix = "_cls"
        elif token_mode == 'patch':
            token_suffix = f"_patch{patch_indices}"
        elif token_mode == 'all':
            token_suffix = "_all"
        
        results_file = os.path.join(model_output_dir, f"{model_name.replace('/', '_')}_all_results{token_suffix}.json")
    
    # Run triplet analysis if needed
    if not os.path.exists(results_file) or force_recompute:
        print(f"Running optimized token-specific triplet analysis for {model_name}...")
        
        # Use the optimized version of the script
        triplet_cmd = [
            "python", "triplet.py",  # Point to the optimized script
            "--h5_file", feature_file,
            "--output_dir", model_output_dir,
            "--token_mode", token_mode
        ]
        
        if seed is not None:
            triplet_cmd.extend(["--seed", str(seed)])
        
        if token_mode == 'patch' and patch_indices is not None:
            triplet_cmd.extend(["--token_index", str(patch_indices)])
        
        if token_mode == 'each_patch' and patch_indices is not None:
            triplet_cmd.extend(["--patch_indices", patch_indices])
        
        # Add option to visualize all patches 
        if visualize_all_patches:
            triplet_cmd.append("--visualize_all_patches")
        
        if verbose:
            triplet_cmd.append("--verbose")
            print(f"Running command: {' '.join(triplet_cmd)}")
        
        try:
            subprocess.run(triplet_cmd, check=True)
            return results_file
        except subprocess.CalledProcessError as e:
            print(f"Error running token-specific triplet analysis for {model_name}: {e}")
            return None
    else:
        print(f"Using existing results file: {results_file}")
        return results_file

def main():
    parser = argparse.ArgumentParser(description='Run optimized token-specific triplet analysis on feature files')
    parser.add_argument('--feature_files', nargs='*', help='List of feature files to analyze')
    parser.add_argument('--features_dir', type=str, default='../features', help='Directory containing feature files')
    parser.add_argument('--output_dir', type=str, default='../results', help='Directory to store analysis results')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--feature_files_json', type=str, help='JSON file with list of feature files')
    parser.add_argument('--token_mode', type=str, default='each_patch', 
                       choices=['all', 'cls', 'patch', 'each_patch'],
                       help='Token selection mode: all, cls, patch, or each_patch')
    parser.add_argument('--patch_indices', type=str, default=None,
                       help='Comma-separated list of patch indices to analyze (used with each_patch mode or specific index with patch mode)')
    parser.add_argument('--visualize_all_patches', action='store_true', default=False,
                      help='Create visualizations for all patch tokens (can be time-consuming)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which feature files to process
    feature_files = []
    
    if args.feature_files_json and os.path.exists(args.feature_files_json):
        with open(args.feature_files_json, 'r') as f:
            feature_files_dict = json.load(f)
            feature_files = list(feature_files_dict.values())
    elif args.feature_files:
        feature_files = args.feature_files
    else:
        # Find all feature files in the features directory
        for filename in os.listdir(args.features_dir):
            if filename.endswith('_features.h5'):
                feature_files.append(os.path.join(args.features_dir, filename))
    
    print(f"Processing {len(feature_files)} feature files with token mode: {args.token_mode}")
    
    # Extract patch index for 'patch' mode
    patch_index = None
    if args.token_mode == 'patch' and args.patch_indices:
        # Just use the first index in the list for 'patch' mode
        try:
            patch_index = int(args.patch_indices.split(',')[0])
            print(f"Using patch index: {patch_index}")
        except (ValueError, IndexError):
            print("Invalid patch index, using default (0)")
            patch_index = 0
    
    # Run triplet analysis for each feature file
    results_files = {}
    for feature_file in tqdm(feature_files, desc="Running token analysis"):
        if not os.path.exists(feature_file):
            print(f"Skipping {feature_file}: file not found")
            continue
            
        results_file = run_token_triplet_analysis(
            feature_file, 
            args.output_dir, 
            args.seed,
            token_mode=args.token_mode,
            patch_indices=args.patch_indices if args.token_mode == 'each_patch' else patch_index,
            force_recompute=args.force_recompute, 
            verbose=args.verbose,
            visualize_all_patches=args.visualize_all_patches
        )
        
        if results_file:
            # Extract model name from feature file
            basename = os.path.basename(feature_file)
            if '_pyramid_' in basename:
                model_name = basename.split('_pyramid_')[0]
            elif '_penultimate_' in basename:
                model_name = basename.split('_penultimate_')[0]
            else:
                model_name = basename.split('_')[0]
                
            results_files[model_name] = results_file
    
    # Save list of result files
    output_prefix = f"token_{args.token_mode}_" if args.token_mode != 'all' else ""
    with open(os.path.join(args.output_dir, f'{output_prefix}analysis_results.json'), 'w') as f:
        json.dump(results_files, f, indent=2)
    
    print(f"\nCompleted token-specific triplet analysis for {len(results_files)} models")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()