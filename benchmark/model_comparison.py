import os
import subprocess
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

def run_benchmark_and_triplet(model_name, args):
    """Run benchmark.py and triplet.py for a single model"""
    # Set paths
    feature_file = os.path.join(args.features_dir, f"{model_name.replace('/', '_')}_pyramid_features.h5")
    results_dir = os.path.join(args.results_dir, model_name.replace('/', '_'))
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Run benchmark.py if needed
    if not os.path.exists(feature_file) or args.force_recompute:
        print(f"Extracting features for {model_name}...")
        benchmark_cmd = [
            "python", "benchmark.py",
            "--model_name", model_name,
            "--data_path", args.data_path,
            "--output_dir", args.features_dir,
            "--num_scenes", str(args.num_scenes),
            "--max_scene_search", str(args.max_scene_search),
            "--feature_mode", "pyramid",
            "--seed", str(args.seed),
            "--dataset_name", args.dataset_name
        ]
        
        if args.verbose:
            print(f"Running command: {' '.join(benchmark_cmd)}")
        
        try:
            subprocess.run(benchmark_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting features for {model_name}: {e}")
            return None
    else:
        print(f"Using existing feature file: {feature_file}")
    
    # Step 2: Run triplet.py
    results_file = os.path.join(results_dir, f"{model_name.replace('/', '_')}_all_results.json")
    
    if not os.path.exists(results_file) or args.force_recompute:
        print(f"Running triplet evaluation for {model_name}...")
        triplet_cmd = [
            "python", "triplet.py",
            "--h5_file", feature_file,
            "--output_dir", results_dir
        ]
        
        if args.verbose:
            print(f"Running command: {' '.join(triplet_cmd)}")
        
        try:
            subprocess.run(triplet_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running triplet evaluation for {model_name}: {e}")
            return None
    else:
        print(f"Using existing results file: {results_file}")
    
    # Return the path to the results file
    return results_file

def collect_results(model_results_files):
    """Collect results from all models into a single DataFrame"""
    all_results = []
    
    for model_name, results_file in model_results_files.items():
        if not results_file or not os.path.exists(results_file):
            print(f"Skipping {model_name}: results file not found")
            continue
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Process each level result
            for level_result in data['level_results']:
                # Extract layer depth from level name if possible
                if '_' in level_result['level_name'] and level_result['level_name'].split('_')[1].isdigit():
                    layer_depth = int(level_result['level_name'].split('_')[1])
                else:
                    # For non-standard level names, use index in level list
                    layer_depth = data['level_results'].index(level_result)
                
                # Add model name and layer depth
                level_result['model'] = model_name
                level_result['layer_depth'] = layer_depth
                
                all_results.append(level_result)
        except Exception as e:
            print(f"Error processing results for {model_name}: {e}")
    
    # Convert to DataFrame
    if all_results:
        return pd.DataFrame(all_results)
    else:
        return None

def create_scatter_plot(df, output_dir, metric='complete_accuracy'):
    """Create a scatter plot showing model performance across layers"""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    # Extract models and sort them
    models = sorted(df['model'].unique())
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Set up color palette
    cmap = plt.cm.get_cmap('viridis')
    
    # Get min and max layer depth for color normalization
    min_depth = df['layer_depth'].min()
    max_depth = df['layer_depth'].max()
    norm = plt.Normalize(min_depth, max_depth)
    
    # Plot each model's layers as points
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        
        # Sort by layer depth
        model_data = model_data.sort_values('layer_depth')
        
        # Create scatter plot
        scatter = plt.scatter(
            [i] * len(model_data),  # x position = model index
            model_data[metric],     # y position = metric value
            c=model_data['layer_depth'],  # color = layer depth
            s=120,                  # point size
            cmap='viridis',         # colormap
            norm=norm,              # normalize colors across all models
            alpha=0.8               # transparency
        )
        
        # Connect the dots to show progression through layers
        plt.plot(
            [i] * len(model_data),
            model_data[metric],
            'k-',
            alpha=0.3
        )
        
        # Add layer name labels
        for _, row in model_data.iterrows():
            plt.text(
                i + 0.1,           # x position (offset)
                row[metric],       # y position
                row['level_name'],  # label
                fontsize=8,
                ha='left',
                va='center'
            )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Layer Depth (lower = earlier layer)', fontsize=12)
    
    # Add title and labels
    metric_name = metric.replace('_', ' ').title()
    plt.title(f'Model Layer Performance - {metric_name}', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    
    # Set x-axis ticks to model names
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f'model_layer_scatter_{metric}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Scatter plot saved to: {output_file}")

def main(args):
    # Create output directories
    os.makedirs(args.features_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set models to process
    models = args.models if args.models else [
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'efficientnet_b0', 'efficientnet_b1', 
        'vit_small_patch16_224', 'vit_base_patch16_224',
        'convnext_tiny', 'convnext_small'
    ]
    # models = args.models if args.models else [
    #     'resnet18', 'resnet34', 'resnet50'
    # ]

    print(f"Processing {len(models)} models with seed {args.seed}, {args.num_scenes} scenes")
    
    # Run benchmark and triplet for each model
    model_results_files = {}
    for model_name in tqdm(models, desc="Processing models"):
        print(f"\nProcessing model: {model_name}")
        results_file = run_benchmark_and_triplet(model_name, args)
        model_results_files[model_name] = results_file
    
    # Collect results
    print("\nCollecting results...")
    results_df = collect_results(model_results_files)
    
    if results_df is not None:
        # Save combined results
        csv_file = os.path.join(args.results_dir, 'all_models_results.csv')
        results_df.to_csv(csv_file, index=False)
        print(f"Combined results saved to: {csv_file}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_scatter_plot(results_df, args.results_dir, 'complete_accuracy')
        create_scatter_plot(results_df, args.results_dir, 'partial_accuracy')
        create_scatter_plot(results_df, args.results_dir, 'roc_auc')
        if 'cohens_d_ap_an' in results_df.columns:
            create_scatter_plot(results_df, args.results_dir, 'cohens_d_ap_an')
    else:
        print("No results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple timm models and analyze triplet metrics')
    
    parser.add_argument('--models', nargs='*', default=None, help='List of timm models to process')
    parser.add_argument('--data_path', type=str, default='../data/ASP_FixedSun/', help='Path to the ASP dataset')
    parser.add_argument('--features_dir', type=str, default='../features', help='Directory to store features')
    parser.add_argument('--results_dir', type=str, default='../analysis', help='Directory to store results')
    parser.add_argument('--num_scenes', type=int, default=100, help='Number of scenes to process')
    parser.add_argument('--max_scene_search', type=int, default=5000, help='Maximum number of scenes to search')
    parser.add_argument('--dataset_name', type=str, default='asp_surround_mix_noref', help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--force_recompute', action='store_true', help='Force recomputation')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()
    main(args)
