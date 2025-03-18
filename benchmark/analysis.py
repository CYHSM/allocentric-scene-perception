import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score
from scipy import stats
import argparse
import json
from tqdm import tqdm
import pandas as pd

def load_features(h5_file_path):
    """
    Load features from H5 file and organize by scene
    
    Returns:
        - features_by_scene: Dictionary mapping scene indices to arrays of features
        - scene_ids: List of scene IDs
        - feature_mode: Type of features (penultimate_pooled, etc.)
        - actual_scenes: List of actual scene indices (if available)
    """
    features_by_scene = {}
    actual_scenes = None
    
    with h5py.File(h5_file_path, 'r') as f:
        # Get metadata
        feature_mode = f.attrs.get('feature_mode', 'unknown')
        
        # Try to get actual scene indices
        if 'actual_scenes' in f.attrs:
            try:
                actual_scenes = json.loads(f.attrs['actual_scenes'])
            except:
                pass
        
        # Check if we have a scene mapping
        scene_mapping = {}
        if 'scene_mapping' in f:
            for key in f['scene_mapping'].attrs:
                scene_mapping[key] = f['scene_mapping'].attrs[key]
        
        # Determine where features are stored based on feature mode
        if feature_mode in ['penultimate_pooled', 'penultimate_unpooled']:
            feature_group = f['features']
            
            # Process each feature dataset
            for key in feature_group.keys():
                # Extract scene ID from key (e.g., "scene_001_t0")
                if '_t' in key:
                    scene_id, t_id = key.split('_t')
                    
                    # Get original scene index if available
                    if scene_id in scene_mapping:
                        scene_idx = scene_mapping[scene_id]
                    else:
                        # Extract numeric part from scene_id (e.g., "scene_001" -> 1)
                        scene_idx = int(scene_id.split('_')[1].lstrip('0'))
                    
                    # Load feature
                    feature = np.array(feature_group[key])
                    
                    # Initialize list for this scene if needed
                    if scene_idx not in features_by_scene:
                        features_by_scene[scene_idx] = []
                    
                    # For unpooled features (spatial feature maps), global average pool
                    # if len(feature.shape) > 2:
                    #     # Average over spatial dimensions for each channel
                    #     pooled_dims = tuple(range(2, len(feature.shape)))
                    #     feature = np.mean(feature, axis=pooled_dims)
                    
                    # Flatten if needed
                    if len(feature.shape) > 2:
                        feature = feature.reshape(feature.shape[0], -1)
                    
                    # Add to scene's feature list
                    features_by_scene[scene_idx].append(feature.squeeze())
        
        else:  # 'pyramid' or other
            # Find feature levels
            feature_levels = []
            for key in f.keys():
                if key.startswith('level_'):
                    feature_levels.append(key)
            
            # Use last level by default
            if feature_levels:
                feature_group = f[feature_levels[-1]]
                
                # Process each feature dataset
                for key in feature_group.keys():
                    # Extract scene ID from key
                    if '_t' in key:
                        scene_id, t_id = key.split('_t')
                        
                        # Get original scene index if available
                        if scene_id in scene_mapping:
                            scene_idx = scene_mapping[scene_id]
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
                            feature = feature.reshape(feature.shape[0], -1)
                        
                        # Add to scene's feature list
                        features_by_scene[scene_idx].append(feature.squeeze())
    
    # Convert lists of features to numpy arrays
    for scene_idx in features_by_scene:
        features_by_scene[scene_idx] = np.array(features_by_scene[scene_idx])
    
    return features_by_scene, list(features_by_scene.keys()), feature_mode, actual_scenes

def compute_within_scene_similarity(features_by_scene):
    """
    Compute similarity between all pairs of features within each scene
    
    Returns:
        - within_similarities: Array of similarity scores
        - within_sim_by_scene: Dictionary mapping scene indices to similarity scores
    """
    within_similarities = []
    within_sim_by_scene = {}
    
    for scene_idx, features in features_by_scene.items():
        # Normalize features
        norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Compute similarity matrix
        sim_matrix = np.dot(norm_features, norm_features.T)
        
        # Extract unique pairs (excluding self-similarity)
        mask = np.triu(np.ones_like(sim_matrix), k=1).astype(bool)
        scene_similarities = sim_matrix[mask]
        
        # Store similarities
        within_similarities.extend(scene_similarities)
        within_sim_by_scene[scene_idx] = scene_similarities
    
    return np.array(within_similarities), within_sim_by_scene

def compute_across_scene_similarity(features_by_scene, max_pairs=10000000):
    """
    Compute similarity between features from different scenes
    
    Arguments:
        - features_by_scene: Dictionary mapping scene indices to arrays of features
        - max_pairs: Maximum number of pairs to sample (for memory efficiency)
    
    Returns:
        - across_similarities: Array of similarity scores
    """
    across_similarities = []
    scene_ids = list(features_by_scene.keys())
    
    # Count total possible pairs
    total_pairs = 0
    for i in range(len(scene_ids)):
        for j in range(i+1, len(scene_ids)):
            total_pairs += features_by_scene[scene_ids[i]].shape[0] * features_by_scene[scene_ids[j]].shape[0]
    
    # Determine sampling ratio if needed
    sampling_ratio = 1.0
    if total_pairs > max_pairs:
        sampling_ratio = max_pairs / total_pairs
        print(f"Sampling {sampling_ratio:.2%} of across-scene pairs ({max_pairs}/{total_pairs})")
    
    # Compute similarities between scenes
    for i in range(len(scene_ids)):
        scene1 = scene_ids[i]
        features1 = features_by_scene[scene1]
        norm_features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
        
        for j in range(i+1, len(scene_ids)):
            scene2 = scene_ids[j]
            features2 = features_by_scene[scene2]
            norm_features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
            
            # Compute similarity matrix between scenes
            sim_matrix = np.dot(norm_features1, norm_features2.T)
            
            # Sample pairs if needed
            if sampling_ratio < 1.0:
                flat_sim = sim_matrix.flatten()
                indices = np.random.choice(
                    len(flat_sim), 
                    size=int(len(flat_sim) * sampling_ratio), 
                    replace=False
                )
                sampled_sim = flat_sim[indices]
                across_similarities.extend(sampled_sim)
            else:
                across_similarities.extend(sim_matrix.flatten())
    
    return np.array(across_similarities)

def create_similarity_plots(within_similarities, across_similarities, output_dir, model_name):
    """Create plots comparing within-scene and across-scene similarities"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Histogram
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    plt.hist(within_similarities, bins=bins, alpha=0.5, label='Within Scene', density=True)
    plt.hist(across_similarities, bins=bins, alpha=0.5, label='Across Scenes', density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title(f'Distribution of Feature Similarities - {model_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_similarity_hist.png'), dpi=300)
    plt.close()
    
    # Violin plot
    plt.figure(figsize=(8, 6))
    data = {
        'Within Scene': within_similarities,
        'Across Scenes': across_similarities
    }
    df = pd.DataFrame({
        'Similarity': np.concatenate([within_similarities, across_similarities]),
        'Type': ['Within Scene'] * len(within_similarities) + ['Across Scenes'] * len(across_similarities)
    })
    sns.violinplot(x='Type', y='Similarity', data=df)
    plt.title(f'Distribution of Feature Similarities - {model_name}')
    plt.savefig(os.path.join(output_dir, f'{model_name}_similarity_violin.png'), dpi=300)
    plt.close()
    
    # CDF plot
    plt.figure(figsize=(10, 6))
    for name, data in [('Within Scene', within_similarities), ('Across Scenes', across_similarities)]:
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cumulative, label=name)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Cumulative Probability')
    plt.title(f'Cumulative Distribution of Similarities - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{model_name}_similarity_cdf.png'), dpi=300)
    plt.close()
    
    # Calculate statistics
    mean_within = np.mean(within_similarities)
    mean_across = np.mean(across_similarities)
    median_within = np.median(within_similarities)
    median_across = np.median(across_similarities)
    
    # Run t-test
    t_stat, p_value = stats.ttest_ind(within_similarities, across_similarities, equal_var=False)
    
    # Calculate effect size (Cohen's d)
    mean_diff = mean_within - mean_across
    pooled_std = np.sqrt((np.std(within_similarities) ** 2 + np.std(across_similarities) ** 2) / 2)
    cohens_d = mean_diff / pooled_std
    
    # Save statistics
    stats_df = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'T-statistic', 'P-value', 'Effect size (Cohen\'s d)'],
        'Within Scene': [mean_within, median_within, t_stat, p_value, cohens_d],
        'Across Scenes': [mean_across, median_across, np.nan, np.nan, np.nan],
        'Difference': [mean_within - mean_across, median_within - median_across, np.nan, np.nan, np.nan]
    })
    stats_df.to_csv(os.path.join(output_dir, f'{model_name}_similarity_stats.csv'), index=False)
    
    print(f"--- Statistics for {model_name} ---")
    print(f"Mean within-scene similarity: {mean_within:.4f}")
    print(f"Mean across-scene similarity: {mean_across:.4f}")
    print(f"Difference: {mean_within - mean_across:.4f}")
    print(f"T-test: t={t_stat:.4f}, p={p_value:.8f}")
    print(f"Effect size (Cohen's d): {cohens_d:.4f}")
    
    return mean_within, mean_across, cohens_d

def visualize_feature_space(features_by_scene, output_dir, model_name, method='tsne', n_samples=5000):
    """Create dimensionality reduction visualization of feature space"""
    all_features = []
    scene_labels = []
    
    # Collect features and labels
    for scene_idx, features in features_by_scene.items():
        all_features.append(features)
        scene_labels.extend([scene_idx] * len(features))
    
    all_features = np.vstack(all_features)
    scene_labels = np.array(scene_labels)
    
    # Sample if there are too many points
    if len(all_features) > n_samples:
        indices = np.random.choice(len(all_features), size=n_samples, replace=False)
        all_features = all_features[indices]
        scene_labels = scene_labels[indices]
    
    # Normalize features
    normalized_features = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        title_prefix = 't-SNE'
        reducer = TSNE(n_components=2, random_state=42)
    else:  # UMAP
        title_prefix = 'UMAP'
        reducer = umap.UMAP(random_state=42)
    
    embeddings = reducer.fit_transform(normalized_features)
    
    # Calculate silhouette score
    silhouette = silhouette_score(embeddings, scene_labels)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=scene_labels, cmap='tab20', alpha=0.7, s=10)
    plt.title(f'{title_prefix} Visualization of Features - {model_name}\nSilhouette Score: {silhouette:.3f}')
    plt.colorbar(scatter, label='Scene ID')
    plt.savefig(os.path.join(output_dir, f'{model_name}_{method}_visualization.png'), dpi=300)
    plt.close()
    
    return silhouette

def analyze_temporal_structure(features_by_scene, output_dir, model_name):
    """Analyze how similarity changes with temporal distance within scenes"""
    all_temporal_diffs = []
    all_similarities = []
    
    for scene_idx, features in features_by_scene.items():
        if len(features) < 2:
            continue
            
        # Normalize features
        norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        # Compute similarity matrix
        sim_matrix = np.dot(norm_features, norm_features.T)
        
        # For each pair of timesteps, record temporal distance and similarity
        n_timesteps = len(features)
        for i in range(n_timesteps):
            for j in range(i+1, n_timesteps):
                temporal_diff = j - i
                similarity = sim_matrix[i, j]
                
                all_temporal_diffs.append(temporal_diff)
                all_similarities.append(similarity)
    
    # Convert to arrays
    all_temporal_diffs = np.array(all_temporal_diffs)
    all_similarities = np.array(all_similarities)
    
    # Group by temporal difference
    unique_diffs = np.unique(all_temporal_diffs)
    mean_similarities = []
    std_similarities = []
    
    for diff in unique_diffs:
        mask = all_temporal_diffs == diff
        mean_sim = np.mean(all_similarities[mask])
        std_sim = np.std(all_similarities[mask])
        
        mean_similarities.append(mean_sim)
        std_similarities.append(std_sim)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(unique_diffs, mean_similarities, yerr=std_similarities, fmt='o-')
    plt.xlabel('Temporal Distance')
    plt.ylabel('Mean Cosine Similarity')
    plt.title(f'Feature Similarity vs. Temporal Distance - {model_name}')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{model_name}_temporal_similarity.png'), dpi=300)
    plt.close()
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(all_temporal_diffs, all_similarities)
    
    print(f"Correlation between temporal distance and similarity: r={corr:.4f}, p={p_val:.8f}")
    
    return corr

def analyze_within_across_scene_consistency(within_sim_by_scene, across_similarities, output_dir, model_name):
    """Analyze which scenes have highest/lowest within-scene similarity"""
    # Calculate mean within-scene similarity for each scene
    scene_mean_sim = {}
    for scene_idx, similarities in within_sim_by_scene.items():
        scene_mean_sim[scene_idx] = np.mean(similarities)
    
    # Calculate mean across-scene similarity
    mean_across = np.mean(across_similarities)
    
    # Create sorted DataFrame
    df = pd.DataFrame({
        'Scene': list(scene_mean_sim.keys()),
        'Mean Within-Scene Similarity': list(scene_mean_sim.values()),
        'Difference from Across-Scene Mean': [s - mean_across for s in scene_mean_sim.values()]
    })
    df = df.sort_values('Mean Within-Scene Similarity', ascending=False)
    
    # Save results
    df.to_csv(os.path.join(output_dir, f'{model_name}_scene_consistency.csv'), index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    plt.bar(df['Scene'], df['Mean Within-Scene Similarity'])
    plt.axhline(mean_across, color='r', linestyle='--', label='Mean Across-Scene Similarity')
    plt.xlabel('Scene')
    plt.ylabel('Mean Within-Scene Similarity')
    plt.title(f'Within-Scene Similarity by Scene - {model_name}')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_scene_similarity_ranking.png'), dpi=300)
    plt.close()
    
    return df

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.h5_file}")
    features_by_scene, scene_ids, feature_mode, actual_scenes = load_features(args.h5_file)
    print(f"Loaded features for {len(features_by_scene)} scenes with feature mode: {feature_mode}")
    
    # Get model name from file
    model_name = os.path.basename(args.h5_file).split('_')[0]
    
    # Compute similarities
    print("Computing within-scene similarities...")
    within_similarities, within_sim_by_scene = compute_within_scene_similarity(features_by_scene)
    
    print("Computing across-scene similarities...")
    across_similarities = compute_across_scene_similarity(features_by_scene, max_pairs=args.max_pairs)
    
    # Create similarity plots
    print("Creating similarity plots...")
    mean_within, mean_across, cohens_d = create_similarity_plots(
        within_similarities, across_similarities, args.output_dir, model_name
    )
    
    # Visualize feature space
    print("Visualizing feature space...")
    silhouette = visualize_feature_space(
        features_by_scene, args.output_dir, model_name, method=args.vis_method, n_samples=args.n_samples
    )
    
    # Analyze temporal structure
    print("Analyzing temporal structure...")
    temporal_corr = analyze_temporal_structure(features_by_scene, args.output_dir, model_name)
    
    # Analyze scene consistency
    print("Analyzing scene consistency...")
    scene_df = analyze_within_across_scene_consistency(
        within_sim_by_scene, across_similarities, args.output_dir, model_name
    )
    
    # Save summary results
    summary = {
        'model_name': model_name,
        'feature_mode': feature_mode,
        'num_scenes': len(features_by_scene),
        'mean_within_scene_similarity': float(mean_within),
        'mean_across_scene_similarity': float(mean_across),
        'similarity_difference': float(mean_within - mean_across),
        'cohens_d': float(cohens_d),
        'silhouette_score': float(silhouette),
        'temporal_correlation': float(temporal_corr),
        'top_consistent_scene': int(scene_df.iloc[0]['Scene']),
        'lowest_consistent_scene': int(scene_df.iloc[-1]['Scene'])
    }
    
    with open(os.path.join(args.output_dir, f'{model_name}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    
    # Print key findings
    print("\nKey Findings:")
    print(f"• Within-scene similarity: {mean_within:.4f}, Across-scene similarity: {mean_across:.4f}")
    print(f"• Difference: {mean_within - mean_across:.4f} (Cohen's d = {cohens_d:.4f})")
    print(f"• Feature space silhouette score: {silhouette:.4f}")
    print(f"• Temporal distance correlation: {temporal_corr:.4f}")
    print(f"• Most consistent scene: {scene_df.iloc[0]['Scene']} (similarity: {scene_df.iloc[0]['Mean Within-Scene Similarity']:.4f})")
    print(f"• Least consistent scene: {scene_df.iloc[-1]['Scene']} (similarity: {scene_df.iloc[-1]['Mean Within-Scene Similarity']:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze model features from ASP scenes')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to H5 file with features')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='Output directory')
    parser.add_argument('--vis_method', type=str, default='tsne', choices=['tsne', 'umap'], help='Visualization method')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples for visualization')
    parser.add_argument('--max_pairs', type=int, default=10000000, help='Maximum number of pairs for across-scene similarity')
    
    args = parser.parse_args()
    main(args)
