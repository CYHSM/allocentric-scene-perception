import os
import argparse
import subprocess
import json
from tqdm import tqdm

def run_triplet_analysis(feature_file, output_dir, seed, force_recompute=False, verbose=False):
    """Run triplet analysis on a feature file"""
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
    results_file = os.path.join(model_output_dir, f"{model_name.replace('/', '_')}_all_results.json")
    
    # Run triplet analysis if needed
    if not os.path.exists(results_file) or force_recompute:
        print(f"Running triplet analysis for {model_name}...")
        triplet_cmd = [
            "python", "triplet.py",
            "--h5_file", feature_file,
            "--output_dir", model_output_dir
        ]
        
        if seed is not None:
            triplet_cmd.extend(["--seed", str(seed)])
        
        if verbose:
            print(f"Running command: {' '.join(triplet_cmd)}")
        
        try:
            subprocess.run(triplet_cmd, check=True)
            return results_file
        except subprocess.CalledProcessError as e:
            print(f"Error running triplet analysis for {model_name}: {e}")
            return None
    else:
        print(f"Using existing results file: {results_file}")
        return results_file

def main():
    parser = argparse.ArgumentParser(description='Run triplet analysis on feature files')
    parser.add_argument('--feature_files', nargs='*', help='List of feature files to analyze')
    parser.add_argument('--features_dir', type=str, default='../features', help='Directory containing feature files')
    parser.add_argument('--output_dir', type=str, default='../results', help='Directory to store analysis results')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--feature_files_json', type=str, help='JSON file with list of feature files')
    
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
    
    print(f"Processing {len(feature_files)} feature files")
    
    # Run triplet analysis for each feature file
    results_files = {}
    for feature_file in tqdm(feature_files, desc="Running triplet analysis"):
        if not os.path.exists(feature_file):
            print(f"Skipping {feature_file}: file not found")
            continue
            
        results_file = run_triplet_analysis(feature_file, args.output_dir, args.seed, 
                                         args.force_recompute, args.verbose)
        
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
    with open(os.path.join(args.output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(results_files, f, indent=2)
    
    print(f"\nCompleted triplet analysis for {len(results_files)} models")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()