import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

def collect_results(result_files):
    """Collect results from multiple model files into a single DataFrame"""
    all_results = []
    
    for model_name, result_file in result_files.items():
        if not os.path.exists(result_file):
            print(f"Skipping {model_name}: results file not found")
            continue
        
        try:
            with open(result_file, 'r') as f:
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
    """Create a scatter plot showing model performance across layers, sorted by best performance"""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    # Get the best performance for each model
    best_performances = df.groupby('model')[metric].max().reset_index()
    
    # Sort models by their best performance (descending)
    sorted_models = best_performances.sort_values(metric, ascending=False)['model'].tolist()
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Create a custom colormap with 5 beautiful, distinct colors
    # Going from early layers (blues) to late layers (reds)
    custom_colors = ['#1A5B92', '#48A9A6', '#F9DB6D', '#E07A5F', '#D62246']

    # Plot each model's layers as points
    for i, model in enumerate(sorted_models):
        model_data = df[df['model'] == model]
        
        # Sort by layer depth
        model_data = model_data.sort_values('layer_depth')
        
        # Get model-specific layer depths
        model_min_depth = model_data['layer_depth'].min()
        model_max_depth = model_data['layer_depth'].max()
        
        # Calculate number of colors to use (minimum 3, maximum 5)
        n_layers = len(model_data)
        n_colors = min(max(3, n_layers), len(custom_colors))
        
        # Create a mapping specific to this model's layer depths
        # This ensures the first layer is always the first color and the last layer is always the last color
        model_depth_bins = np.linspace(model_min_depth, model_max_depth, n_colors)
        
        # Plot each layer with its appropriate color
        for _, row in model_data.iterrows():
            # Find the closest bin for this layer depth
            depth = row['layer_depth']
            # Calculate which bin this depth falls into
            if n_colors == 1:  # Edge case: only one layer
                color_idx = 0
            else:
                # Find the bin index for this depth
                bin_idx = np.digitize(depth, model_depth_bins) - 1
                # Clip to valid range
                color_idx = min(max(0, bin_idx), n_colors-1)
            
            color = custom_colors[color_idx]
            
            plt.scatter(
                i,                      # x position = model index
                row[metric],            # y position = metric value
                c=color,                # color based on layer depth
                s=120,                  # point size
                alpha=0.8,              # transparency
                edgecolor='white',      # white border
                linewidth=0.5           # thin border
            )
        
        # Connect the dots to show progression through layers
        plt.plot(
            [i] * len(model_data),
            model_data[metric],
            'k-',
            alpha=0.3
        )

    # Create a custom legend
    legend_elements = []
    for i in range(len(custom_colors)):
        if i >= len(custom_colors):
            break
            
        if i == 0:
            label = "Early layer"
        elif i == len(custom_colors) - 1:
            label = "Late layer"
        elif i == 1:
            label = "Early-middle layer"
        elif i == len(custom_colors) - 2:
            label = "Middle-late layer"
        else:
            label = "Middle layer"
        
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=custom_colors[i], 
                                         markersize=10, label=label))

    # Add the legend
    plt.legend(handles=legend_elements, loc='upper right', title='Layer Depth')
    
    # Add title and labels
    metric_name = metric.replace('_', ' ').title()
    plt.title(f'Model Layer Performance - {metric_name} (Sorted by Best Performance)', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    
    # Set x-axis ticks to model names
    plt.xticks(range(len(sorted_models)), sorted_models, rotation=45, ha='right', fontsize=5)
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f'model_layer_scatter_{metric}_sorted.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Sorted scatter plot saved to: {output_file}")

def create_performance_heatmap(df, output_dir):
    """Create a heatmap comparing model performance across metrics"""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    # Get unique models
    models = sorted(df['model'].unique())
    
    # For each model, get the best layer's performance
    model_metrics = []
    metrics = ['complete_accuracy', 'partial_accuracy', 'roc_auc']
    
    for model in models:
        model_data = df[df['model'] == model]
        best_layer = model_data.loc[model_data['complete_accuracy'].idxmax()]
        
        metrics_dict = {
            'model': model,
            'best_layer': best_layer['level_name']
        }
        
        for metric in metrics:
            metrics_dict[metric] = best_layer[metric]
        
        model_metrics.append(metrics_dict)
    
    # Create DataFrame
    model_metrics_df = pd.DataFrame(model_metrics)
    
    # Sort models by complete_accuracy (descending)
    sorted_models = model_metrics_df.sort_values('complete_accuracy', ascending=False)['model'].tolist()
    
    # Create heatmap data
    heatmap_data = np.zeros((len(sorted_models), len(metrics)))
    for i, model in enumerate(sorted_models):
        model_row = model_metrics_df[model_metrics_df['model'] == model]
        for j, metric in enumerate(metrics):
            heatmap_data[i, j] = model_row[metric].values[0]
    
    # Create heatmap
    plt.figure(figsize=(12, len(sorted_models) * 0.8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=[m.replace('_', ' ').title() for m in metrics],
                yticklabels=sorted_models)
    plt.title('Best Layer Performance by Model and Metric (Sorted by Complete Accuracy)')
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'model_performance_heatmap_sorted.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Sorted performance heatmap saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize triplet analysis results')
    parser.add_argument('--results_dir', type=str, default='../results', help='Directory containing analysis results')
    parser.add_argument('--output_dir', type=str, default='../visualizations', help='Directory to store visualizations')
    parser.add_argument('--results_files_json', type=str, help='JSON file with list of result files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get results files
    results_files = {}
    
    if args.results_files_json and os.path.exists(args.results_files_json):
        with open(args.results_files_json, 'r') as f:
            results_files = json.load(f)
    else:
        # Find all results files
        for root, dirs, files in os.walk(args.results_dir):
            for file in files:
                if file.endswith('_all_results.json'):
                    model_name = file.replace('_all_results.json', '')
                    results_files[model_name] = os.path.join(root, file)
    
    print(f"Processing results for {len(results_files)} models")
    
    # Collect results into DataFrame
    results_df = collect_results(results_files)
    
    if results_df is not None:
        # Save combined results
        csv_file = os.path.join(args.output_dir, 'all_models_results.csv')
        results_df.to_csv(csv_file, index=False)
        print(f"Combined results saved to: {csv_file}")
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Create scatter plots for different metrics
        for metric in ['complete_accuracy', 'partial_accuracy', 'roc_auc', 'cohens_d_ap_an']:
            if metric in results_df.columns:
                create_scatter_plot(results_df, args.output_dir, metric)
        
        # Create performance heatmap
        create_performance_heatmap(results_df, args.output_dir)
        
        # Create bar chart of best performing layers, sorted by performance
        best_layers = results_df.loc[results_df.groupby('model')['complete_accuracy'].idxmax()]
        best_layers_sorted = best_layers.sort_values('complete_accuracy', ascending=False)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='model', y='complete_accuracy', data=best_layers_sorted, palette='viridis')
        plt.title('Best Layer Performance by Model (Sorted)')
        plt.xlabel('Model')
        plt.ylabel('Complete Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'best_layer_performance_sorted.png'), dpi=300)
        plt.close()
        
        print("Visualizations complete!")
    else:
        print("No results collected. Check input files.")

if __name__ == "__main__":
    main()