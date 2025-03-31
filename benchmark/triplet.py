import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json
from tqdm import tqdm
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc
import random
import torch
import gc  # Garbage collector

def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_metadata_from_h5(h5_file_path):
    """Extract metadata from H5 file without loading features"""
    with h5py.File(h5_file_path, 'r') as f:
        # Get metadata
        feature_mode = f.attrs.get('feature_mode', 'unknown')
        
        # Try to get the seed from metadata
        seed = None
        if 'seed' in f.attrs:
            try:
                seed = int(f.attrs['seed'])
                print(f"Found seed in h5 file: {seed}")
            except:
                print("Could not parse seed from h5 file")
        
        # Get all feature levels
        if feature_mode == 'pyramid' or any(key.startswith('level_') for key in f.keys()):
            feature_mode = 'pyramid'
            feature_levels = [key for key in f.keys() if key.startswith('level_')]
            
            # Sort levels by their numeric index
            try:
                feature_levels = sorted(feature_levels, 
                                      key=lambda x: int(x.split('_')[1]))
            except:
                # If sorting fails, keep original order
                pass
        else:  # Not pyramid mode - add single feature set
            if feature_mode in ['penultimate_pooled', 'penultimate_unpooled'] and 'features' in f:
                feature_levels = ['features']
            elif len(f.keys()) > 0:
                # Just use the first group that's not scene_mapping
                feature_levels = [key for key in f.keys() if key != 'scene_mapping']
    
    return feature_mode, seed, feature_levels

def load_full_layer_features(h5_file_path, level_name):
    """
    Load all features for a single layer at once.
    Returns a dictionary mapping scene indices to feature arrays with all tokens.
    """
    full_features_by_scene = {}
    scene_mapping = {}
    
    with h5py.File(h5_file_path, 'r') as f:
        # Get scene mapping if available
        if 'scene_mapping' in f:
            for key in f['scene_mapping'].attrs:
                scene_mapping[key] = f['scene_mapping'].attrs[key]
        
        # Get the feature group for this level
        if level_name in f:
            feature_group = f[level_name]
            
            # Process features for this level
            for key in feature_group.keys():
                # Extract scene ID from key
                if '_t' in key:
                    scene_id, t_id = key.split('_t')
                    
                    # Get original scene index if available
                    if scene_id in scene_mapping:
                        scene_idx = int(scene_mapping[scene_id])
                    else:
                        # Extract numeric part from scene_id
                        scene_idx = int(scene_id.split('_')[1].lstrip('0'))
                    
                    # Load feature with all tokens
                    feature = np.array(feature_group[key])
                    
                    # Initialize list for this scene if needed
                    if scene_idx not in full_features_by_scene:
                        full_features_by_scene[scene_idx] = []
                    
                    # Add to scene's feature list
                    full_features_by_scene[scene_idx].append(feature)
            
            # Convert lists of features to numpy arrays
            for scene_idx in full_features_by_scene:
                full_features_by_scene[scene_idx] = np.array(full_features_by_scene[scene_idx])
    
    return full_features_by_scene

def extract_token_features(full_features_by_scene, token_mode, token_index):
    """
    Extract token-specific features from full features
    
    Parameters:
        full_features_by_scene: Dictionary mapping scene indices to feature arrays with all tokens
        token_mode: How to select tokens ('all', 'cls', 'patch')
        token_index: Which specific token index to use if mode is 'patch'
    
    Returns:
        token_features_by_scene: Dictionary mapping scene indices to token-specific feature arrays
    """
    token_features_by_scene = {}
    
    for scene_idx, features_array in full_features_by_scene.items():
        token_features = []
        
        for feature in features_array:
            # Apply token selection based on mode
            if len(feature.shape) > 2:  # For features with shape [B, S, D] or [B, C, H, W]
                if feature.shape[0] == 1:  # If batch dimension is 1
                    # Check if it looks like transformer features [B, S, D]
                    if len(feature.shape) == 3:
                        # Extract tokens based on mode
                        if token_mode == 'cls':
                            # Extract CLS token (first token, index 0)
                            selected_feature = feature[0, 0, :]
                        elif token_mode == 'patch':
                            # Extract specific patch token
                            if token_index < feature.shape[1]:
                                selected_feature = feature[0, token_index, :]
                            else:
                                # Using last token if index is out of bounds
                                selected_feature = feature[0, -1, :]
                        else:  # 'all' mode or fallback
                            # Use all tokens, flatten them
                            selected_feature = feature.reshape(feature.shape[0], -1).squeeze()
                    else:
                        # CNN-style features [B, C, H, W]
                        if token_mode in ['cls', 'patch']:
                            # CNN features don't have tokens, using all features
                            selected_feature = feature.reshape(feature.shape[0], -1).squeeze()
                        else:
                            selected_feature = feature.reshape(feature.shape[0], -1).squeeze()
                else:
                    # If batch dimension is not 1, just flatten
                    selected_feature = feature.reshape(feature.shape[0], -1)
            else:
                # For already flattened features
                selected_feature = feature.squeeze()
            
            token_features.append(selected_feature)
        
        token_features_by_scene[scene_idx] = np.array(token_features)
    
    return token_features_by_scene

def estimate_num_tokens(h5_file_path, level_name):
    """
    Estimate the number of tokens in the feature map at a given level.
    Returns the maximum possible token index and shape.
    """
    with h5py.File(h5_file_path, 'r') as f:
        if level_name in f:
            # Try to find a dataset
            for key in f[level_name].keys():
                if '_t' in key:
                    # Found a dataset, check shape
                    feature = np.array(f[level_name][key])
                    shape = feature.shape
                    
                    # For transformer features [B, S, D]
                    if len(shape) == 3:
                        # Return number of tokens (sequence length)
                        return shape[1], shape
                    else:
                        # Return 0 for non-transformer models
                        return 0, shape
    
    # Fallback if no features found
    return 0, None

def sample_triplets_per_image(features_by_scene, rng=None, sample_size=None):
    """
    Sample triplets for evaluation with each image being used once as an anchor.
    This creates more triplets than the original approach which used each scene once.
    
    Parameters:
    features_by_scene: Dictionary mapping scene indices to feature arrays
    rng: Random number generator (optional)
    sample_size: Maximum number of triplets to sample (optional)
    
    Returns:
    triplets: List of (anchor, positive, negative) feature tuples
    metadata: List of (anchor_scene, anchor_idx, pos_idx, neg_scene, neg_idx) metadata tuples
    """
    triplets = []
    metadata = []
    
    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random
    
    # Filter scenes that have at least 2 images
    valid_scenes = [scene_idx for scene_idx, features in features_by_scene.items() 
                   if len(features) >= 2]
    
    if len(valid_scenes) < 2:
        raise ValueError("Need at least 2 scenes with 2+ images each to create triplets")
    
    # Calculate total number of possible triplets
    total_images = sum(len(features_by_scene[scene]) for scene in valid_scenes)
    
    # Create a list of all (scene_idx, image_idx) pairs
    all_image_pairs = []
    for scene_idx in valid_scenes:
        num_images = len(features_by_scene[scene_idx])
        all_image_pairs.extend([(scene_idx, img_idx) for img_idx in range(num_images)])
    
    # Limit sample size if requested
    if sample_size is not None and sample_size < len(all_image_pairs):
        sampled_pairs = rng.choice(all_image_pairs, size=sample_size, replace=False)
    else:
        sampled_pairs = all_image_pairs
    
    # Use each selected image once as an anchor
    for anchor_scene, anchor_idx in sampled_pairs:
        # Choose a different image from the same scene as positive
        scene_images = len(features_by_scene[anchor_scene])
        other_indices = [i for i in range(scene_images) if i != anchor_idx]
        pos_idx = rng.choice(other_indices)
        
        # Sample negative scene
        neg_scene = rng.choice([s for s in valid_scenes if s != anchor_scene])
        
        # Sample negative from negative scene
        neg_idx = rng.choice(len(features_by_scene[neg_scene]))
        
        # Get feature vectors
        anchor = features_by_scene[anchor_scene][anchor_idx]
        positive = features_by_scene[anchor_scene][pos_idx]
        negative = features_by_scene[neg_scene][neg_idx]
        
        # Add to lists
        triplets.append((anchor, positive, negative))
        metadata.append((anchor_scene, anchor_idx, pos_idx, neg_scene, neg_idx))
    
    return triplets, metadata

def evaluate_triplets_complete(triplets, similarity_metric='cosine'):
    """
    Evaluate triplets completely, checking all pairwise relationships
    """
    complete_correct = 0
    partial_correct = 0
    similarity_matrices = []
    
    for anchor, positive, negative in triplets:
        # Normalize vectors for cosine similarity
        anchor_norm = anchor / (np.linalg.norm(anchor) + 1e-10)
        positive_norm = positive / (np.linalg.norm(positive) + 1e-10)
        negative_norm = negative / (np.linalg.norm(negative) + 1e-10)
        
        # Compute all pairwise similarities
        sim_a_p = float(np.dot(anchor_norm, positive_norm))
        sim_a_n = float(np.dot(anchor_norm, negative_norm))
        sim_p_n = float(np.dot(positive_norm, negative_norm))
        
        # Full correctness: All within-scene similarities > all across-scene similarities
        if sim_a_p > sim_a_n and sim_a_p > sim_p_n:
            complete_correct += 1
        
        # Partial correctness: Anchor-positive > anchor-negative (original criterion)
        if sim_a_p > sim_a_n:
            partial_correct += 1
        
        # Record similarities
        similarity_matrices.append((sim_a_p, sim_a_n, sim_p_n))
    
    complete_accuracy = complete_correct / len(triplets) if triplets else 0
    partial_accuracy = partial_correct / len(triplets) if triplets else 0
    
    return complete_accuracy, partial_accuracy, similarity_matrices

def plot_similarity_distributions(similarity_matrices, output_dir, model_name, level_name, token_suffix="", skip_visualization=False):
    """Create plot of similarity distributions"""
    # Unpack similarities
    same_scene_sims = [s[0] for s in similarity_matrices]  # anchor-positive
    diff_scenes_sims = [s[1] for s in similarity_matrices] + [s[2] for s in similarity_matrices]  # anchor-negative + positive-negative
    
    # Calculate ROC curve for same scene vs different scene similarities
    y_true = [1] * len(same_scene_sims) + [0] * len(diff_scenes_sims)
    y_score = same_scene_sims + diff_scenes_sims
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Skip visualization if requested (except for CLS token or all tokens mode)
    if skip_visualization and token_suffix not in ["_cls", "_all", ""]:
        return float(roc_auc)
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    sns.kdeplot(same_scene_sims, label='Within Scene', fill=True)
    sns.kdeplot(diff_scenes_sims, label='Across Scenes', fill=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title(f'Distribution of Feature Similarities - {model_name} - {level_name}{token_suffix}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{level_name}{token_suffix}_triplet_similarity_dist.png'), dpi=300)
    plt.close()
    
    # Create violin plot of all three similarity types
    plt.figure(figsize=(10, 6))
    sim_df = pd.DataFrame({
        'Same Scene (Anchor-Positive)': [s[0] for s in similarity_matrices],
        'Diff Scenes (Anchor-Negative)': [s[1] for s in similarity_matrices],
        'Diff Scenes (Positive-Negative)': [s[2] for s in similarity_matrices]
    })
    
    # Melt for seaborn
    sim_df_melted = pd.melt(sim_df, var_name='Comparison', value_name='Similarity')
    
    # Create violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Comparison', y='Similarity', data=sim_df_melted)
    plt.xticks(rotation=15)
    plt.title(f'Pairwise Similarity Distributions - {model_name} - {level_name}{token_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{level_name}{token_suffix}_triplet_violins.png'), dpi=300)
    plt.close()
    
    # ROC curve plot
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} - {level_name}{token_suffix}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'{model_name}_{level_name}{token_suffix}_triplet_roc.png'), dpi=300)
    plt.close()
    
    return float(roc_auc)


def compute_similarity_deltas(similarity_matrices):
    """
    Compute similarity deltas for analysis with corrected Cohen's d calculations
    
    Args:
        similarity_matrices: List of tuples containing (sim_a_p, sim_a_n, sim_p_n)
            sim_a_p: anchor-positive similarity
            sim_a_n: anchor-negative similarity
            sim_p_n: positive-negative similarity
    
    Returns:
        Tuple containing:
        - ap_an_deltas: List of differences between anchor-positive and anchor-negative similarities
        - ap_pn_deltas: List of differences between anchor-positive and positive-negative similarities
        - same_scene_sims: List of anchor-positive similarities (same scene)
        - diff_scene_sims: List of anchor-negative and positive-negative similarities (different scenes)
    """
    # Delta between anchor-positive and anchor-negative
    ap_an_deltas = [s[0] - s[1] for s in similarity_matrices]
    
    # Delta between anchor-positive and positive-negative
    ap_pn_deltas = [s[0] - s[2] for s in similarity_matrices]
    
    # Extract similarity groups for Cohen's d calculation
    same_scene_sims = [s[0] for s in similarity_matrices]  # anchor-positive similarities
    diff_scene_sims = [s[1] for s in similarity_matrices] + [s[2] for s in similarity_matrices]  # anchor-negative + positive-negative
    
    # Return deltas and raw similarity lists
    return ap_an_deltas, ap_pn_deltas, same_scene_sims, diff_scene_sims


def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size with pooled standard deviation.
    
    Args:
        group1: First group of values (e.g., same-scene similarities)
        group2: Second group of values (e.g., different-scene similarities)
    
    Returns:
        Cohen's d effect size
    """
    # Calculate means
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    # Calculate sample sizes
    n1 = len(group1)
    n2 = len(group2)
    
    # Calculate variances
    var1 = np.var(group1, ddof=1)  # Using sample variance (ddof=1)
    var2 = np.var(group2, ddof=1)
    
    # Calculate pooled standard deviation
    # Formula: sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    # Add small epsilon to avoid division by zero
    d = (mean1 - mean2) / (pooled_std + 1e-10)
    
    return d, pooled_std

def process_token_features(token_features, args, output_dir, model_name, level_name, 
                          token_mode, token_index, rng=None, max_triplets=None, skip_visualization=False):
    """
    Process a single token's features and return results
    
    Parameters:
        token_features: Dictionary mapping scene indices to token-specific feature arrays
        args: Command line arguments
        output_dir: Directory to save output files
        model_name: Name of the model
        level_name: Name of the layer level
        token_mode: How tokens are selected ('all', 'cls', 'patch')
        token_index: Which patch token index to use if token_mode is 'patch'
        rng: Random number generator
        max_triplets: Maximum number of triplets to sample
        skip_visualization: Whether to skip creating visualization plots
    
    Returns:
        dict: Results dictionary with all metrics
    """
    # Create token suffix for file naming
    if token_mode == 'cls':
        token_suffix = "_cls"
    elif token_mode == 'patch':
        token_suffix = f"_patch{token_index}"
    else:
        token_suffix = "_all"
    
    # Sample triplets (one triplet per image, or up to max_triplets)
    triplets, metadata = sample_triplets_per_image(token_features, rng=rng, sample_size=max_triplets)
    
    # Evaluate triplets
    complete_acc, partial_acc, sim_matrices = evaluate_triplets_complete(triplets)
    
    # Free memory
    triplets = None
    
    # Plot similarity distributions - skip visualization for patch tokens if requested
    roc_auc = plot_similarity_distributions(sim_matrices, output_dir, model_name, level_name, token_suffix, skip_visualization)
    
    # Compute similarity deltas for analysis - UPDATED: also returns raw similarities
    ap_an_deltas, ap_pn_deltas, same_scene_sims, diff_scene_sims = compute_similarity_deltas(sim_matrices)
    
    # Compute statistical significance (one-sample t-test against 0)
    t_ap_an, p_ap_an = stats.ttest_1samp(ap_an_deltas, 0)
    t_ap_pn, p_ap_pn = stats.ttest_1samp(ap_pn_deltas, 0)
    
    # Calculate effect sizes (Cohen's d) - CORRECTED version using pooled standard deviation
    cohens_d_ap_an, pooled_std_ap_an = calculate_cohens_d(same_scene_sims, [s[1] for s in sim_matrices])
    cohens_d_ap_pn, pooled_std_ap_pn = calculate_cohens_d(same_scene_sims, [s[2] for s in sim_matrices])
    
    # Calculate overall Cohen's d (same scene vs. all different scenes)
    cohens_d_overall, pooled_std_overall = calculate_cohens_d(same_scene_sims, diff_scene_sims)
    
    # Average of both same-scene vs different-scene comparisons
    avg_delta = np.mean(ap_an_deltas + ap_pn_deltas)
    
    # Return results
    return {
        'level_name': f"{level_name}{token_suffix}",
        'original_level_name': level_name,
        'token_mode': token_mode,
        'token_index': token_index,
        'num_triplets': len(sim_matrices),
        'complete_accuracy': float(complete_acc),
        'partial_accuracy': float(partial_acc),
        'roc_auc': float(roc_auc),
        'mean_same_scene_similarity': float(np.mean(same_scene_sims)),
        'mean_diff_scene_similarity_a_n': float(np.mean([s[1] for s in sim_matrices])),
        'mean_diff_scene_similarity_p_n': float(np.mean([s[2] for s in sim_matrices])),
        'mean_delta_ap_an': float(np.mean(ap_an_deltas)),
        'mean_delta_ap_pn': float(np.mean(ap_pn_deltas)),
        'avg_same_diff_delta': float(avg_delta),
        't_statistic_ap_an': float(t_ap_an),
        'p_value_ap_an': float(p_ap_an),
        'cohens_d_ap_an': float(cohens_d_ap_an),
        'pooled_std_ap_an': float(pooled_std_ap_an),
        't_statistic_ap_pn': float(t_ap_pn),
        'p_value_ap_pn': float(p_ap_pn),
        'cohens_d_ap_pn': float(cohens_d_ap_pn),
        'pooled_std_ap_pn': float(pooled_std_ap_pn),
        'cohens_d_overall': float(cohens_d_overall),
        'pooled_std_overall': float(pooled_std_overall)
    }

def plot_patch_grid(num_tokens, token_results, output_dir, model_name, level_name):
    """
    Create a 2x2 plot showing performance metrics for patch positions.
    
    Parameters:
        num_tokens: Number of tokens in the model (including CLS token)
        token_results: List of result dictionaries for all tokens
        output_dir: Directory to save output files
        model_name: Name of the model
        level_name: Name of the layer level
    """
    # Extract patch tokens and their performance
    patch_performances = {}
    
    # Get the metric values for patch tokens
    for result in token_results:
        if result['token_mode'] == 'patch':
            patch_idx = result['token_index']
            patch_performances[patch_idx] = {
                'complete_accuracy': result['complete_accuracy'],
                'partial_accuracy': result['partial_accuracy'],
                'roc_auc': result['roc_auc'],
                'cohens_d_ap_an': result['cohens_d_ap_an']
            }
    
    if not patch_performances:
        print("No patch tokens found in results")
        return
    
    # Determine the grid dimensions
    # For transformer models, we need to figure out the grid shape
    # First, subtract 1 for the CLS token to get number of patch tokens
    num_patches = num_tokens - 1
    
    # Try to estimate a square-ish grid
    grid_size = int(np.sqrt(num_patches))
    
    # Handle non-perfect squares by adjusting
    if grid_size * grid_size == num_patches:
        # Perfect square
        grid_width = grid_height = grid_size
    else:
        # Try to find the best rectangular grid
        grid_width = grid_size
        grid_height = (num_patches + grid_width - 1) // grid_width  # Ceiling division
    
    # Create a grid of patch indices
    grid = np.zeros((grid_height, grid_width))
    # Fill with -1 as default (for empty spots)
    grid.fill(-1)
    
    # Fill in the grid with patch indices
    for i in range(min(num_patches, grid_width * grid_height)):
        # Calculate row and column (assuming row-major order)
        row = i // grid_width
        col = i % grid_width
        # Patches start at index 1 (0 is CLS token)
        patch_idx = i + 1
        # Add to grid if in bounds
        if row < grid_height and col < grid_width:
            grid[row, col] = patch_idx
    
    # Create performance grids for each metric
    metrics = ['complete_accuracy', 'partial_accuracy', 'roc_auc', 'cohens_d_ap_an']
    metric_names = ['Complete Accuracy', 'Partial Accuracy', 'ROC AUC', "Cohen's d"]
    performance_grids = {}
    
    for metric in metrics:
        performance_grid = np.zeros_like(grid, dtype=float)
        performance_grid.fill(np.nan)  # Fill with NaN for missing values
        
        # Fill in the performance values
        for i in range(min(num_patches, grid_width * grid_height)):
            row = i // grid_width
            col = i % grid_width
            patch_idx = i + 1
            
            if patch_idx in patch_performances:
                performance_grid[row, col] = patch_performances[patch_idx][metric]
        
        performance_grids[metric] = performance_grid
    
    # Create figure with 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    # Plot heatmaps for each metric
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Create a masked array to handle missing values
        masked_performance = np.ma.masked_invalid(performance_grids[metric])
        
        # Set proper vmin/vmax based on the metric
        if metric == 'cohens_d_ap_an':
            # Cohen's d can have a wider range
            vmin, vmax = -1, 2
        else:
            # Accuracy metrics range from 0 to 1
            vmin, vmax = 0, 1
        
        # Create a heatmap of performance
        im = ax.imshow(masked_performance, cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # Add color bar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(name)
        
        # Add performance values to cells (only if grid is not too dense)
        # if grid_width <= 8 and grid_height <= 8:
        for i in range(grid_height):
            for j in range(grid_width):
                val = performance_grids[metric][i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color='white' if val > 0.5 else 'black',
                            fontsize=6)
        
        ax.set_title(f'{name} by Patch Position')
        ax.set_xticks(np.arange(grid_width))
        ax.set_yticks(np.arange(grid_height))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    
    # Add a note about CLS token
    if any(r['token_mode'] == 'cls' for r in token_results):
        fig.text(0.5, 0.01, "Note: CLS token (index 0) is not shown in the grid", 
                ha='center', fontsize=12, fontstyle='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # Save to the parent directory 
    parent_dir = os.path.dirname(output_dir)
    plt.savefig(os.path.join(parent_dir, f'{model_name}_{level_name}_patch_metrics.png'), dpi=300)
    plt.close()

def plot_layer_token_summary(layer_results, output_dir, model_name, level_name):
    """
    Create a summary plot for a single layer showing performance across all tokens.
    This is called after all tokens for a layer have been processed.
    """
    if not layer_results:
        print(f"No results to plot for layer {level_name}")
        return
    
    # Extract token information
    tokens = []
    token_indices = []
    complete_accs = []
    partial_accs = []
    roc_aucs = []
    cohens_ds = []
    token_modes = []
    
    # Process and sort results
    for result in layer_results:
        token_type = "CLS" if result['token_mode'] == 'cls' else \
                    f"P{result['token_index']}" if result['token_mode'] == 'patch' else \
                    "All"
        
        tokens.append(token_type)
        token_indices.append(result['token_index'] if result['token_mode'] == 'patch' else 
                           -1 if result['token_mode'] == 'cls' else -2)
        token_modes.append(result['token_mode'])
        complete_accs.append(result['complete_accuracy'])
        partial_accs.append(result['partial_accuracy'])
        roc_aucs.append(result['roc_auc'])
        cohens_ds.append(result['cohens_d_ap_an'])
    
    # Count how many patch tokens we have
    num_patches = sum(1 for mode in token_modes if mode == 'patch')
    
    # Create the clean line chart for metrics (for both key tokens and patch overview)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # ====== First plot: Focus on CLS and All tokens + top patches ======
    # Prepare data for special tokens (CLS, ALL, and a few top patches)
    special_tokens = []
    special_indices = []
    special_complete = []
    special_partial = []
    special_roc = []
    special_cohens = []
    
    # First add CLS and All tokens
    for i, mode in enumerate(token_modes):
        if mode in ['cls', 'all']:
            special_tokens.append(tokens[i])
            special_indices.append(i)
            special_complete.append(complete_accs[i])
            special_partial.append(partial_accs[i])
            special_roc.append(roc_aucs[i])
            special_cohens.append(cohens_ds[i])
    
    # Then find the top 5 performing patch tokens by complete accuracy
    patch_indices = [i for i, mode in enumerate(token_modes) if mode == 'patch']
    top_patches = sorted(patch_indices, key=lambda i: complete_accs[i], reverse=True)[:5]
    
    for i in top_patches:
        special_tokens.append(tokens[i])
        special_indices.append(i)
        special_complete.append(complete_accs[i])
        special_partial.append(partial_accs[i])
        special_roc.append(roc_aucs[i])
        special_cohens.append(cohens_ds[i])
    
    # Sort by token type so CLS comes first, then All, then patches
    sorted_together = sorted(zip(special_tokens, special_indices, special_complete, 
                               special_partial, special_roc, special_cohens),
                           key=lambda x: (0 if x[0] == 'CLS' else 
                                        1 if x[0] == 'All' else 2, x[0]))
    
    special_tokens = [x[0] for x in sorted_together]
    special_indices = [x[1] for x in sorted_together]
    special_complete = [x[2] for x in sorted_together]
    special_partial = [x[3] for x in sorted_together]
    special_roc = [x[4] for x in sorted_together]
    special_cohens = [x[5] for x in sorted_together]
    
    # Plot metrics for special tokens
    x = np.arange(len(special_tokens))
    
    # Create line plot with markers
    ax1.plot(x, special_complete, 'o-', linewidth=2, label='Complete Accuracy', color='#1f77b4')
    ax1.plot(x, special_partial, 's-', linewidth=2, label='Partial Accuracy', color='#ff7f0e')
    ax1.plot(x, special_roc, '^-', linewidth=2, label='ROC AUC', color='#2ca02c')
    ax1.plot(x, special_cohens, 'D-', linewidth=2, label="Cohen's d", color='#d62728')
    
    # Add grid and labels
    ax1.grid(alpha=0.3)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title(f'Key Token Performance - {model_name} - {level_name}', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(special_tokens)
    ax1.legend(fontsize=12)
    
    # Add value labels above points
    for i, (c, p, r) in enumerate(zip(special_complete, special_partial, special_roc)):
        ax1.annotate(f'{c:.3f}', (i, c), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Despine - remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ====== Second plot: Focus on patch tokens only ======
    # Filter to only patch tokens
    patch_indices = [i for i, mode in enumerate(token_modes) if mode == 'patch']
    
    if patch_indices:
        # Sort patches by index
        patch_indices.sort(key=lambda i: token_indices[i])
        
        patch_tokens = [token_indices[i] for i in patch_indices]
        patch_complete = [complete_accs[i] for i in patch_indices]
        patch_partial = [partial_accs[i] for i in patch_indices]
        patch_roc = [roc_aucs[i] for i in patch_indices]
        patch_cohens = [cohens_ds[i] for i in patch_indices]
        
        # Create line plot for patches
        ax2.plot(patch_tokens, patch_complete, '-', linewidth=1.5, label='Complete Accuracy', color='#1f77b4', alpha=0.9)
        ax2.plot(patch_tokens, patch_partial, '-', linewidth=1.5, label='Partial Accuracy', color='#ff7f0e', alpha=0.9)
        ax2.plot(patch_tokens, patch_roc, '-', linewidth=1.5, label='ROC AUC', color='#2ca02c', alpha=0.9)
        ax2.plot(patch_tokens, patch_cohens, '-', linewidth=1.5, label="Cohen's d", color='#d62728', alpha=0.9)
        
        # Add mean lines
        mean_complete = np.mean(patch_complete)
        mean_partial = np.mean(patch_partial)
        mean_roc = np.mean(patch_roc)
        mean_cohens = np.mean(patch_cohens)
        
        ax2.axhline(y=mean_complete, color='#1f77b4', linestyle='--', alpha=0.5)
        ax2.axhline(y=mean_partial, color='#ff7f0e', linestyle='--', alpha=0.5)
        ax2.axhline(y=mean_roc, color='#2ca02c', linestyle='--', alpha=0.5)
        ax2.axhline(y=mean_cohens, color='#d62728', linestyle='--', alpha=0.5)
        
        # Add annotations for means
        ax2.text(patch_tokens[-1] + 1, mean_complete, f'Mean: {mean_complete:.3f}', 
                va='center', ha='left', color='#1f77b4', fontsize=9)
        ax2.text(patch_tokens[-1] + 1, mean_partial, f'Mean: {mean_partial:.3f}', 
                va='center', ha='left', color='#ff7f0e', fontsize=9)
        ax2.text(patch_tokens[-1] + 1, mean_roc, f'Mean: {mean_roc:.3f}', 
                va='center', ha='left', color='#2ca02c', fontsize=9)
        ax2.text(patch_tokens[-1] + 1, mean_cohens, f'Mean: {mean_cohens:.3f}', 
                va='center', ha='left', color='#d62728', fontsize=9)
        
        # Add grid and labels
        ax2.grid(alpha=0.3)
        ax2.set_xlabel('Patch Index', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title(f'All Patch Tokens ({num_patches}) - {model_name} - {level_name}', fontsize=14)
        ax2.legend(fontsize=12)
        
        # Despine - remove top and right spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Add a note about number of patches
        fig.text(0.5, 0.01, f"Note: Showing performance for {num_patches} patch tokens (indices 1-{max(patch_tokens)})", 
                ha='center', fontsize=10, fontstyle='italic')
    else:
        ax2.text(0.5, 0.5, "No patch tokens found in the results", 
                ha='center', va='center', fontsize=14, fontstyle='italic')
        
        # Despine for this plot too
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save to the parent directory
    parent_dir = os.path.dirname(output_dir)
    plt.savefig(os.path.join(parent_dir, f'{model_name}_{level_name}_token_summary.png'), dpi=300)
    plt.close()
    
    # Create additional visualizations for patch grid mapping
    if num_patches > 0:
        # Estimate total tokens including CLS
        plot_patch_grid(num_patches + 1, layer_results, output_dir, model_name, level_name)
    
    # Print summary of token performance for this layer
    print(f"\nSummary for {model_name} - {level_name}:")
    
    # First show CLS and All tokens
    for mode in ['cls', 'all']:
        for i, m in enumerate(token_modes):
            if m == mode:
                print(f"  {tokens[i]}: Complete Acc={complete_accs[i]:.3f}, Partial Acc={partial_accs[i]:.3f}, ROC AUC={roc_aucs[i]:.3f}")
    
    # Then show average for patch tokens
    patch_complete = [complete_accs[i] for i in range(len(token_modes)) if token_modes[i] == 'patch']
    patch_partial = [partial_accs[i] for i in range(len(token_modes)) if token_modes[i] == 'patch']
    patch_roc = [roc_aucs[i] for i in range(len(token_modes)) if token_modes[i] == 'patch']
    
    if patch_complete:
        print(f"  Patches (avg of {len(patch_complete)}): Complete Acc={np.mean(patch_complete):.3f}, " + 
             f"Partial Acc={np.mean(patch_partial):.3f}, ROC AUC={np.mean(patch_roc):.3f}")

def plot_level_comparison(all_results, output_dir, model_name, token_suffix=""):
    """Create comparison plots across all levels"""
    # Extract data for plotting, filter by token suffix
    filtered_results = [r for r in all_results if token_suffix in r['level_name']]
    
    if not filtered_results:
        print(f"No results found with token suffix {token_suffix}")
        return None
    
    # Get levels without token suffix for display
    levels = [r['original_level_name'] for r in filtered_results]
    
    # Try to order levels by layer depth
    ordered_levels = []
    ordered_indices = []
    
    # Extract stride information if available (for level_X_strideY format)
    for i, level in enumerate(levels):
        if '_stride_' in level:
            try:
                stride = int(level.split('_stride_')[1])
                ordered_levels.append((level, stride, i))
            except:
                ordered_levels.append((level, 999, i))  # Default high stride for unknown
        else:
            try:
                level_num = int(level.split('_')[1])
                ordered_levels.append((level, level_num, i))
            except:
                ordered_levels.append((level, 999, i))  # Default high stride for unknown
    
    # Sort by stride
    ordered_levels.sort(key=lambda x: x[1])
    
    # Get sorted levels and indices
    levels = [x[0] for x in ordered_levels]
    ordered_indices = [x[2] for x in ordered_levels]
    
    # Extract metrics
    complete_accs = [filtered_results[i]['complete_accuracy'] for i in ordered_indices]
    partial_accs = [filtered_results[i]['partial_accuracy'] for i in ordered_indices]
    roc_aucs = [filtered_results[i]['roc_auc'] for i in ordered_indices]
    cohens_d_ap_an = [filtered_results[i]['cohens_d_ap_an'] for i in ordered_indices]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Level': levels,
        'Complete Accuracy': complete_accs,
        'Partial Accuracy': partial_accs,
        'ROC AUC': roc_aucs,
        "Cohen's d": cohens_d_ap_an
    })
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, f'{model_name}_level_comparison{token_suffix}.csv'), index=False)
    
    # Create bar chart of accuracy metrics
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(levels))
    width = 0.25
    
    plt.bar(x - width, complete_accs, width, label='Complete Accuracy')
    plt.bar(x, partial_accs, width, label='Partial Accuracy')
    plt.bar(x + width, roc_aucs, width, label='ROC AUC')
    
    plt.ylabel('Score')
    plt.xlabel('Feature Level')
    plt.title(f'Performance Metrics by Feature Level - {model_name}{token_suffix}')
    plt.xticks(x, levels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_level_comparison{token_suffix}.png'), dpi=300)
    plt.close()
    
    # Create line plot for metrics
    plt.figure(figsize=(12, 6))
    
    plt.plot(levels, complete_accs, 'o-', label='Complete Accuracy')
    plt.plot(levels, partial_accs, 's-', label='Partial Accuracy')
    plt.plot(levels, roc_aucs, '^-', label='ROC AUC')
    
    plt.ylabel('Score')
    plt.xlabel('Feature Level')
    plt.title(f'Performance Metrics by Feature Level - {model_name}{token_suffix}')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_level_comparison{token_suffix}_line.png'), dpi=300)
    plt.close()
    
    # Create heatmap
    data_matrix = np.array([complete_accs, partial_accs, roc_aucs, cohens_d_ap_an])
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=levels, yticklabels=['Complete Acc', 'Partial Acc', 'ROC AUC', "Cohen's d"])
    plt.title(f'Performance Metrics by Feature Level - {model_name}{token_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_level_heatmap{token_suffix}.png'), dpi=300)
    plt.close()
    
    return df

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed for reproducibility if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        set_seed(args.seed)
        # Create a seeded random number generator
        rng = np.random.RandomState(args.seed)
    else:
        rng = None
    
    # Extract model name from h5 file
    h5_basename = os.path.basename(args.h5_file)
    if '_pyramid_' in h5_basename:
        model_name = h5_basename.split('_pyramid_')[0]
    elif '_penultimate_' in h5_basename:
        model_name = h5_basename.split('_penultimate_')[0]
    else:
        # Fallback to original method if pattern not found
        model_name = h5_basename.split('_')[0]
    print(f"Using model name: {model_name}")
    
    # Get metadata and feature levels without loading features
    print(f"Extracting metadata from {args.h5_file}")
    feature_mode, h5_seed, feature_levels = get_metadata_from_h5(args.h5_file)
    print(f"Found {len(feature_levels)} feature levels, feature mode: {feature_mode}")
    
    # If we weren't given a seed but there's one in the h5 file, use that
    if args.seed is None and h5_seed is not None:
        print(f"Using seed from h5 file: {h5_seed}")
        set_seed(h5_seed)
        rng = np.random.RandomState(h5_seed)
    
    if not feature_levels:
        print("No features found!")
        return
    
    if args.token_mode == 'each_patch':
        # Run analysis for all token positions
        all_token_results = []
        
        # Estimate number of tokens for the first level
        level_to_check = feature_levels[0] if len(feature_levels) > 0 else feature_levels[0]
        num_tokens, feature_shape = estimate_num_tokens(args.h5_file, level_to_check)
        
        if num_tokens == 0:
            print(f"This model doesn't appear to have separate tokens. Feature shape: {feature_shape}")
            print("Defaulting to 'all' token mode")
            token_modes = [('all', 0)]
        else:
            print(f"Detected {num_tokens} tokens in feature map")
            
            # Create token modes for CLS and each patch
            token_modes = [('cls', 0)]  # CLS token
            token_modes.append(('all', 0))  # All tokens
            
            # Patch tokens - users can specify which indices to analyze
            if args.patch_indices:
                indices = [int(idx) for idx in args.patch_indices.split(',')]
                token_modes.extend([('patch', idx) for idx in indices if idx < num_tokens])
            else:
                # Run for all patch tokens
                token_modes.extend([('patch', idx) for idx in range(1, num_tokens)])
        
        # Set up output subdirectory for token-specific results
        token_output_dir = os.path.join(args.output_dir, f"{model_name}_token_analysis")
        os.makedirs(token_output_dir, exist_ok=True)
        
        # Process each level
        for level_idx, level_name in enumerate(feature_levels):
            print(f"\n======== Processing level {level_idx+1}/{len(feature_levels)}: {level_name} ========")
            
            # Load all features for this layer ONCE
            print(f"Loading full features for level {level_name}...")
            full_features = load_full_layer_features(args.h5_file, level_name)
            
            # Dictionary to store results for this level
            level_results = []
            
            # Process each token mode for this level
            total_tokens = len(token_modes)
            for token_idx, (token_mode, token_index) in enumerate(token_modes):
                # Show progress for this level
                print(f"\nToken {token_idx+1}/{total_tokens}: {token_mode} {token_index if token_mode=='patch' else ''}")
                print(f"Progress: {token_idx}/{total_tokens} tokens processed, {total_tokens-token_idx} remaining")
                
                # Skip processing if this level doesn't support the token mode
                if token_mode in ['cls', 'patch']:
                    num_tokens, _ = estimate_num_tokens(args.h5_file, level_name)
                    if num_tokens == 0:
                        print(f"Skipping {token_mode} analysis for level {level_name} - no token structure detected")
                        continue
                    if token_mode == 'patch' and token_index >= num_tokens:
                        print(f"Skipping patch index {token_index} for level {level_name} - out of bounds")
                        continue
                
                # Extract token-specific features for this token
                print(f"Extracting features for token: {token_mode} {token_index if token_mode=='patch' else ''}...")
                token_features = extract_token_features(full_features, token_mode, token_index)
                
                # Determine whether to skip visualization
                # Only create visualizations for CLS token, 'all' token mode, and up to 3 patch tokens for demonstration
                skip_visualization = (token_mode == 'patch' and token_index > 0 and not args.visualize_all_patches)
                
                # Process these token features
                print(f"Analyzing token features...")
                level_result = process_token_features(
                    token_features,
                    args,
                    token_output_dir,
                    model_name,
                    level_name,
                    token_mode,
                    token_index,
                    rng=rng,
                    max_triplets=args.max_triplets,
                    skip_visualization=skip_visualization
                )
                
                # Add to results lists
                level_results.append(level_result)
                all_token_results.append(level_result)
                
                # Free memory for token features
                token_features = None
                gc.collect()
            
            # Free memory for full features
            full_features = None
            gc.collect()
            
            # Create and display a summary plot for this level after all tokens are processed
            if level_results:
                print(f"\nCreating summary plot for level {level_name} with {len(level_results)} tokens...")
                plot_layer_token_summary(level_results, token_output_dir, model_name, level_name)
        
        # Create comparison plots for each token mode across all levels
        if len(all_token_results) > 1:
            print("\nCreating cross-level comparisons...")
            # Create comparison for CLS tokens
            cls_df = plot_level_comparison(all_token_results, token_output_dir, model_name, token_suffix="_cls")
            
            # Create comparison for "all" tokens mode
            all_df = plot_level_comparison(all_token_results, token_output_dir, model_name, token_suffix="_all")
            
            # Create comparison for a subset of patch indices (just the first few)
            if not args.visualize_all_patches:
                # Only visualize the first few patch tokens (0, 1, 2, 3)
                for token_idx in range(1, min(4, num_tokens)):
                    if any(r['token_mode'] == 'patch' and r['token_index'] == token_idx for r in all_token_results):
                        patch_df = plot_level_comparison(all_token_results, token_output_dir, model_name, token_suffix=f"_patch{token_idx}")
            else:
                # Visualize all patch tokens if requested
                for token_idx in range(1, num_tokens):
                    if any(r['token_mode'] == 'patch' and r['token_index'] == token_idx for r in all_token_results):
                        patch_df = plot_level_comparison(all_token_results, token_output_dir, model_name, token_suffix=f"_patch{token_idx}")
        
        # Save combined token results
        combined_token_results = {
            'model_name': model_name,
            'feature_mode': feature_mode,
            'num_levels': len(feature_levels),
            'seed': h5_seed if args.seed is None else args.seed,
            'token_analysis': True,
            'num_tokens': num_tokens,
            'token_results': all_token_results
        }
        
        with open(os.path.join(token_output_dir, f'{model_name}_all_token_results.json'), 'w') as f:
            json.dump(combined_token_results, f, indent=2)
        
        print(f"\nToken-specific results saved to {token_output_dir}")
        
    else:
        # Process each level individually with the specified token mode
        all_results = []
        
        for level_idx, level_name in enumerate(feature_levels):
            print(f"\nProcessing level {level_idx+1}/{len(feature_levels)}: {level_name}")
            
            # Load all features for this layer
            full_features = load_full_layer_features(args.h5_file, level_name)
            
            # Extract token-specific features
            token_features = extract_token_features(full_features, args.token_mode, args.token_index)
            
            # Free memory for full features
            full_features = None
            gc.collect()
            
            # Process token features
            level_result = process_token_features(
                token_features,
                args,
                args.output_dir,
                model_name,
                level_name,
                args.token_mode,
                args.token_index,
                rng=rng,
                max_triplets=args.max_triplets
            )
            
            all_results.append(level_result)
            
            # Force garbage collection
            token_features = None
            gc.collect()
        
        # Create comparison plots across levels
        if len(all_results) > 1:
            print("\nCreating level comparison visualizations...")
            token_suffix = ""
            if args.token_mode == 'cls':
                token_suffix = "_cls"
            elif args.token_mode == 'patch':
                token_suffix = f"_patch{args.token_index}"
            elif args.token_mode == 'all':
                token_suffix = "_all"
            
            level_df = plot_level_comparison(all_results, args.output_dir, model_name, token_suffix=token_suffix)
            
            # Print summary of level comparison
            print("\nLevel Comparison Summary:")
            if level_df is not None:
                print(level_df.to_string(index=False))
        
        # Save combined results
        token_suffix = ""
        if args.token_mode == 'cls':
            token_suffix = "_cls"
        elif args.token_mode == 'patch':
            token_suffix = f"_patch{args.token_index}"
        elif args.token_mode == 'all':
            token_suffix = "_all"
            
        combined_results = {
            'model_name': model_name,
            'feature_mode': feature_mode,
            'num_levels': len(feature_levels),
            'seed': h5_seed if args.seed is None else args.seed,
            'token_mode': args.token_mode,
            'token_index': args.token_index,
            'level_results': all_results
        }
        
        with open(os.path.join(args.output_dir, f'{model_name}_all_results{token_suffix}.json'), 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Memory-efficient triplet analysis with token-specific evaluation')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to H5 file with features')
    parser.add_argument('--output_dir', type=str, default='./triplet_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--max_triplets', type=int, default=None, 
                       help='Maximum number of triplets to sample (default: use all valid scenes)')
    parser.add_argument('--token_mode', type=str, default='all', 
                      choices=['all', 'cls', 'patch', 'each_patch'],
                      help='How to select tokens for analysis')
    parser.add_argument('--token_index', type=int, default=0,
                      help='Which patch token to use if token_mode is patch (default: 0)')
    parser.add_argument('--patch_indices', type=str, default=None,
                      help='Comma-separated list of patch indices to analyze (used with each_patch mode)')
    parser.add_argument('--visualize_all_patches', action='store_true', default=False,
                      help='Create visualizations for all patch tokens (can be time-consuming)')
    
    args = parser.parse_args()
    main(args)