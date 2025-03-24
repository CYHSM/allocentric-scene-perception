import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

def collect_token_results(token_dir):
    """Collect token-specific results from a model's token analysis directory"""
    if not os.path.exists(token_dir):
        print(f"Token directory {token_dir} not found")
        return None
    
    # Check for token results file
    token_results_file = os.path.join(token_dir, os.path.basename(token_dir).split('_token_analysis')[0] + '_all_token_results.json')
    
    if not os.path.exists(token_results_file):
        print(f"Token results file {token_results_file} not found")
        return None
    
    # Load token results
    try:
        with open(token_results_file, 'r') as f:
            token_data = json.load(f)
        
        # Create DataFrame from token results
        token_results = token_data.get('token_results', [])
        
        if not token_results:
            print("No token results found in the file")
            return None
            
        token_df = pd.DataFrame(token_results)
        return token_df
    
    except Exception as e:
        print(f"Error loading token results: {e}")
        return None

def plot_token_performance(token_df, model_name, level_name, output_dir):
    """Plot performance metrics for different tokens at a specific level"""
    # Filter to the specific level
    level_df = token_df[token_df['original_level_name'] == level_name]
    
    if level_df.empty:
        print(f"No data for level {level_name}")
        return
    
    # Extract token information
    level_df['token_type'] = level_df.apply(
        lambda row: 'CLS' if row['token_mode'] == 'cls' else 
                   f"Patch {row['token_index']}" if row['token_mode'] == 'patch' else 
                   'All', axis=1
    )
    
    # Sort by token type (CLS first, then patches by index)
    level_df['sort_key'] = level_df.apply(
        lambda row: -1 if row['token_mode'] == 'cls' else 
                  row['token_index'] if row['token_mode'] == 'patch' else 
                  999, axis=1
    )
    
    level_df = level_df.sort_values('sort_key')
    
    # Plot metrics across tokens
    metrics = ['complete_accuracy', 'partial_accuracy', 'roc_auc', 'cohens_d_ap_an']
    metric_names = ['Complete Accuracy', 'Partial Accuracy', 'ROC AUC', "Cohen's d"]
    
    # Create a grid of subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get token labels for x-axis
    token_labels = level_df['token_type'].tolist()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Create bar chart
        sns.barplot(x='token_type', y=metric, data=level_df, ax=ax, palette='viridis')
        
        # Set titles and labels
        ax.set_title(f'{name} by Token - {model_name} - {level_name}', fontsize=14)
        ax.set_xlabel('Token', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{level_name}_token_comparison.png'), dpi=300)
    plt.close()
    
    # Now create a heatmap of all tokens vs metrics
    # Convert to wide format for heatmap
    plt.figure(figsize=(10, 8))
    heatmap_data = []
    
    for metric, name in zip(metrics, metric_names):
        heatmap_data.append(level_df[metric].values)
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=token_labels, yticklabels=metric_names)
    plt.title(f'Token Performance Metrics - {model_name} - {level_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{level_name}_token_heatmap.png'), dpi=300)
    plt.close()

def plot_level_token_matrix(token_df, model_name, output_dir, metric='complete_accuracy'):
    """Plot a 2D matrix of level vs token performance for a specific metric"""
    # Transform data for the heatmap
    token_types = {}
    
    # Identify unique levels and tokens
    for _, row in token_df.iterrows():
        level = row['original_level_name']
        
        if row['token_mode'] == 'cls':
            token = 'CLS'
        elif row['token_mode'] == 'patch':
            token = f"P{row['token_index']}"
        else:
            token = 'All'
            
        token_types[token] = row['token_mode']
    
    # Get unique levels and tokens, sorted appropriately
    levels = sorted(token_df['original_level_name'].unique(), 
                  key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 999)
    
    # Sort tokens: CLS first, then patches by index
    tokens = []
    if 'CLS' in token_types:
        tokens.append('CLS')
    
    patch_tokens = sorted([t for t in token_types if t.startswith('P')], 
                        key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    tokens.extend(patch_tokens)
    
    if 'All' in token_types:
        tokens.append('All')
    
    # Create the matrix
    heatmap_data = np.zeros((len(levels), len(tokens)))
    mask = np.ones_like(heatmap_data, dtype=bool)
    
    # Fill in the data
    for i, level in enumerate(levels):
        level_data = token_df[token_df['original_level_name'] == level]
        
        for j, token in enumerate(tokens):
            if token == 'CLS':
                token_data = level_data[level_data['token_mode'] == 'cls']
            elif token.startswith('P'):
                token_idx = int(token[1:])
                token_data = level_data[(level_data['token_mode'] == 'patch') & 
                                      (level_data['token_index'] == token_idx)]
            else:  # 'All'
                token_data = level_data[level_data['token_mode'] == 'all']
            
            if not token_data.empty:
                heatmap_data[i, j] = token_data[metric].values[0]
                mask[i, j] = False
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=tokens, yticklabels=levels, mask=mask)
    
    metric_name = metric.replace('_', ' ').title()
    plt.title(f'{metric_name} by Level and Token - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_level_token_matrix_{metric}.png'), dpi=300)
    plt.close()

def visualize_token_performance(results_dir, output_dir):
    """Find and visualize token-specific results"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find token analysis directories
    token_dirs = []
    for root, dirs, files in os.walk(results_dir):
        for d in dirs:
            if d.endswith('_token_analysis'):
                token_dirs.append(os.path.join(root, d))
    
    if not token_dirs:
        print("No token analysis directories found")
        return
    
    print(f"Found {len(token_dirs)} token analysis directories")
    
    # Process each token directory
    for token_dir in token_dirs:
        model_name = os.path.basename(token_dir).split('_token_analysis')[0]
        print(f"Processing token results for {model_name}")
        
        # Collect token results
        token_df = collect_token_results(token_dir)
        
        if token_df is None:
            continue
        
        # Create model-specific output directory
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Get unique levels
        levels = token_df['original_level_name'].unique()
        
        # For each level, create token comparison plots
        for level in levels:
            plot_token_performance(token_df, model_name, level, model_output_dir)
        
        # Create level vs token matrix plots
        for metric in ['complete_accuracy', 'partial_accuracy', 'roc_auc', 'cohens_d_ap_an']:
            plot_level_token_matrix(token_df, model_name, model_output_dir, metric)
        
        # Create token distribution visualization
        token_types = token_df['token_mode'].value_counts()
        print(f"Token distribution for {model_name}: {token_types.to_dict()}")
        
        # Save token data as CSV
        token_df.to_csv(os.path.join(model_output_dir, f'{model_name}_token_analysis.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description='Visualize token-specific triplet analysis results')
    parser.add_argument('--results_dir', type=str, default='../results', help='Directory containing analysis results')
    parser.add_argument('--output_dir', type=str, default='../visualizations/token_analysis', help='Directory to store visualizations')
    
    args = parser.parse_args()
    
    # Visualize token-specific performance
    visualize_token_performance(args.results_dir, args.output_dir)
    
    print(f"Token visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()