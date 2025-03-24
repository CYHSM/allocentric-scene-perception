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

def load_single_level_features(h5_file_path, level_name):
    """
    Load features from a single level, memory efficient
    
    Returns:
        - features_by_scene: Dictionary mapping scene indices to feature arrays
        - scene_mapping: Dictionary mapping scene IDs to scene indices
    """
    features_by_scene = {}
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
                    
                    # Load feature
                    feature = np.array(feature_group[key])
                    
                    # Initialize list for this scene if needed
                    if scene_idx not in features_by_scene:
                        features_by_scene[scene_idx] = []
                    
                    # Flatten if needed
                    if len(feature.shape) > 2:
                        # print(feature.shape)
                        feature = feature.reshape(feature.shape[0], -1)
                        #feature = feature[0,1,:]

                    # Add to scene's feature list
                    features_by_scene[scene_idx].append(feature.squeeze())
            
            # Convert lists of features to numpy arrays
            for scene_idx in features_by_scene:
                features_by_scene[scene_idx] = np.array(features_by_scene[scene_idx])
    
    return features_by_scene

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
    print(f"Total valid images across {len(valid_scenes)} scenes: {total_images}")
    
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
    
    print(f"Sampling {len(sampled_pairs)} image pairs as anchors")
    
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

def sample_triplets(features_by_scene, rng=None, sample_size=None):
    """
    Sample triplets for evaluation with each scene being used exactly once as an anchor.
    Optionally limit to a sample_size number of triplets.
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
    
    # Limit sample size if requested
    if sample_size is not None and sample_size < len(valid_scenes):
        valid_scenes = rng.choice(valid_scenes, size=sample_size, replace=False)
    
    # Use each selected scene once as an anchor
    for anchor_scene in valid_scenes:
        # Sample anchor and positive from same scene
        anchor_idx, pos_idx = rng.choice(len(features_by_scene[anchor_scene]), size=2, replace=False)
        
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

def plot_similarity_distributions(similarity_matrices, output_dir, model_name, level_name):
    """Create plot of similarity distributions"""
    # Unpack similarities
    same_scene_sims = [s[0] for s in similarity_matrices]  # anchor-positive
    diff_scenes_sims = [s[1] for s in similarity_matrices] + [s[2] for s in similarity_matrices]  # anchor-negative + positive-negative
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(same_scene_sims, label='Within Scene', fill=True)
    sns.kdeplot(diff_scenes_sims, label='Across Scenes', fill=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title(f'Distribution of Feature Similarities - {model_name} - {level_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{level_name}_triplet_similarity_dist.png'), dpi=300)
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
    plt.title(f'Pairwise Similarity Distributions - {model_name} - {level_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_{level_name}_triplet_violins.png'), dpi=300)
    plt.close()
    
    # Calculate ROC curve for same scene vs different scene similarities
    y_true = [1] * len(same_scene_sims) + [0] * len(diff_scenes_sims)
    y_score = same_scene_sims + diff_scenes_sims
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} - {level_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'{model_name}_{level_name}_triplet_roc.png'), dpi=300)
    plt.close()
    
    return float(roc_auc)

def compute_similarity_deltas(similarity_matrices):
    """Compute similarity deltas for analysis"""
    # Delta between anchor-positive and anchor-negative
    ap_an_deltas = [s[0] - s[1] for s in similarity_matrices]
    
    # Delta between anchor-positive and positive-negative
    ap_pn_deltas = [s[0] - s[2] for s in similarity_matrices]
    
    # Return both sets of deltas
    return ap_an_deltas, ap_pn_deltas

def process_level(h5_file_path, level_name, args, output_dir, model_name, rng=None, max_triplets=None):
    """Process a single feature level and return results"""
    print(f"\nProcessing level: {level_name}")
    
    # Load features for this level only
    print(f"Loading features for level {level_name}...")
    features_by_scene = load_single_level_features(h5_file_path, level_name)
    
    # Count valid scenes (with at least 2 images)
    valid_scenes = [scene_idx for scene_idx, features in features_by_scene.items() if len(features) >= 2]
    print(f"Found {len(valid_scenes)} valid scenes with 2+ images")
    
    # Sample triplets (one triplet per image, or up to max_triplets)
    print(f"Sampling triplets with each image in {len(valid_scenes)} scenes as anchors...")
    triplets, metadata = sample_triplets_per_image(features_by_scene, rng=rng, sample_size=max_triplets)
    print(f"Generated {len(triplets)} triplets")
    
    # Free up memory from features we no longer need
    features_by_scene = None
    gc.collect()
    
    # Evaluate triplets
    print(f"Evaluating {len(triplets)} triplets...")
    complete_acc, partial_acc, sim_matrices = evaluate_triplets_complete(triplets)
    
    # Free up memory from triplets we no longer need
    triplets = None
    gc.collect()
    
    # Plot similarity distributions
    print("Creating visualizations...")
    roc_auc = plot_similarity_distributions(sim_matrices, output_dir, model_name, level_name)
    
    # Compute similarity deltas for analysis
    ap_an_deltas, ap_pn_deltas = compute_similarity_deltas(sim_matrices)
    
    # Compute statistical significance (one-sample t-test against 0)
    t_ap_an, p_ap_an = stats.ttest_1samp(ap_an_deltas, 0)
    t_ap_pn, p_ap_pn = stats.ttest_1samp(ap_pn_deltas, 0)
    
    # Calculate effect sizes (Cohen's d)
    cohens_d_ap_an = np.mean(ap_an_deltas) / (np.std(ap_an_deltas) + 1e-10)
    cohens_d_ap_pn = np.mean(ap_pn_deltas) / (np.std(ap_pn_deltas) + 1e-10)
    
    # Average of both same-scene vs different-scene comparisons
    avg_delta = np.mean(ap_an_deltas + ap_pn_deltas)
    
    # Print level results
    print(f"  Complete Accuracy: {complete_acc:.4f}")
    print(f"  Partial Accuracy: {partial_acc:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Mean AP-AN delta: {np.mean(ap_an_deltas):.4f}")
    print(f"  Mean AP-PN delta: {np.mean(ap_pn_deltas):.4f}")
    print(f"  Effect size (AP vs AN): Cohen's d = {cohens_d_ap_an:.4f}")
    
    # Return results
    return {
        'level_name': level_name,
        'num_triplets': len(sim_matrices),
        'complete_accuracy': float(complete_acc),
        'partial_accuracy': float(partial_acc),
        'roc_auc': float(roc_auc),
        'mean_same_scene_similarity': float(np.mean([s[0] for s in sim_matrices])),
        'mean_diff_scene_similarity_a_n': float(np.mean([s[1] for s in sim_matrices])),
        'mean_diff_scene_similarity_p_n': float(np.mean([s[2] for s in sim_matrices])),
        'mean_delta_ap_an': float(np.mean(ap_an_deltas)),
        'mean_delta_ap_pn': float(np.mean(ap_pn_deltas)),
        'avg_same_diff_delta': float(avg_delta),
        't_statistic_ap_an': float(t_ap_an),
        'p_value_ap_an': float(p_ap_an),
        'cohens_d_ap_an': float(cohens_d_ap_an),
        't_statistic_ap_pn': float(t_ap_pn),
        'p_value_ap_pn': float(p_ap_pn),
        'cohens_d_ap_pn': float(cohens_d_ap_pn)
    }

def plot_level_comparison(all_results, output_dir, model_name):
    """Create comparison plots across all levels"""
    # Implementation is the same as original
    # Extract data for plotting
    levels = [r['level_name'] for r in all_results]
    
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
            ordered_levels.append((level, 999, i))  # Default high stride for unknown
    
    # Sort by stride
    ordered_levels.sort(key=lambda x: x[1])
    
    # Get sorted levels and indices
    levels = [x[0] for x in ordered_levels]
    ordered_indices = [x[2] for x in ordered_levels]
    
    # Extract metrics
    complete_accs = [all_results[i]['complete_accuracy'] for i in ordered_indices]
    partial_accs = [all_results[i]['partial_accuracy'] for i in ordered_indices]
    roc_aucs = [all_results[i]['roc_auc'] for i in ordered_indices]
    cohens_d_ap_an = [all_results[i]['cohens_d_ap_an'] for i in ordered_indices]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Level': levels,
        'Complete Accuracy': complete_accs,
        'Partial Accuracy': partial_accs,
        'ROC AUC': roc_aucs,
        "Cohen's d": cohens_d_ap_an
    })
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, f'{model_name}_level_comparison.csv'), index=False)
    
    # Create bar chart of accuracy metrics
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(levels))
    width = 0.25
    
    plt.bar(x - width, complete_accs, width, label='Complete Accuracy')
    plt.bar(x, partial_accs, width, label='Partial Accuracy')
    plt.bar(x + width, roc_aucs, width, label='ROC AUC')
    
    plt.ylabel('Score')
    plt.xlabel('Feature Level')
    plt.title(f'Performance Metrics by Feature Level - {model_name}')
    plt.xticks(x, levels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_level_comparison.png'), dpi=300)
    plt.close()
    
    # Create line plot for metrics
    plt.figure(figsize=(12, 6))
    
    plt.plot(levels, complete_accs, 'o-', label='Complete Accuracy')
    plt.plot(levels, partial_accs, 's-', label='Partial Accuracy')
    plt.plot(levels, roc_aucs, '^-', label='ROC AUC')
    
    plt.ylabel('Score')
    plt.xlabel('Feature Level')
    plt.title(f'Performance Metrics by Feature Level - {model_name}')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_level_comparison_line.png'), dpi=300)
    plt.close()
    
    # Create heatmap
    data_matrix = np.array([complete_accs, partial_accs, roc_aucs, cohens_d_ap_an])
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=levels, yticklabels=['Complete Acc', 'Partial Acc', 'ROC AUC', "Cohen's d"])
    plt.title(f'Performance Metrics by Feature Level - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_level_heatmap.png'), dpi=300)
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
    
    # Process each level individually to reduce memory usage
    all_results = []
    
    for level_name in feature_levels:
        # Process one level at a time
        level_result = process_level(
            args.h5_file, 
            level_name, 
            args, 
            args.output_dir, 
            model_name, 
            rng=rng,
            max_triplets=args.max_triplets
        )
        all_results.append(level_result)
        
        # Force garbage collection
        gc.collect()
    
    # Create comparison plots across levels
    if len(all_results) > 1:
        print("\nCreating level comparison visualizations...")
        level_df = plot_level_comparison(all_results, args.output_dir, model_name)
        
        # Print summary of level comparison
        print("\nLevel Comparison Summary:")
        print(level_df.to_string(index=False))
    
    # Save combined results
    combined_results = {
        'model_name': model_name,
        'feature_mode': feature_mode,
        'num_levels': len(feature_levels),
        'seed': h5_seed if args.seed is None else args.seed,
        'level_results': all_results
    }
    
    with open(os.path.join(args.output_dir, f'{model_name}_all_results.json'), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Memory-efficient triplet analysis')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to H5 file with features')
    parser.add_argument('--output_dir', type=str, default='./triplet_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--max_triplets', type=int, default=None, 
                       help='Maximum number of triplets to sample (default: use all valid scenes)')
    
    args = parser.parse_args()
    main(args)
