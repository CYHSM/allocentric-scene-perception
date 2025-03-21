import os
import torch
import numpy as np
import open_clip
from tqdm import tqdm
import argparse
import h5py
import json
from torchvision import transforms
import random
from typing import Dict, List, Optional, Tuple, Union

import sys
sys.path.append('../')
from data import ASP

def set_seed(seed):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(args):
    # Set seed for reproducibility
    print(f"Setting random seed to {args.seed}")
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use CPU instead of GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up dataset first to get the actual scene indices
    print(f"Setting up dataset: {args.dataset_name}")
    dataset = ASP(
        scene_base=args.data_path,
        scenes=range(args.max_scene_search),  # Search through this many scenes
        num_timesteps=args.num_timesteps,
        dataset_name=args.dataset_name,
        seed=args.seed  # Pass the seed to ASP dataset
    )
    
    # Get the filtered scene list
    actual_scenes = dataset.scenes
    if args.num_scenes < len(actual_scenes):
        # Limit to the requested number of scenes
        actual_scenes = actual_scenes[:args.num_scenes]
    
    print(f"Found {len(actual_scenes)} scenes matching criteria: {args.dataset_name}")
    print(f"First few scene indices: {actual_scenes[:5]}...")
    
    # Calculate the maximum number of digits needed for zero-padding
    max_digits = len(str(max(actual_scenes)))
    print(f"Using {max_digits} digits for scene numbering")
    
    # Set up model based on feature extraction mode
    print(f"Loading OpenCLIP model: {args.model_name}")
    
    # Parse model name and pretrained tag
    if ':' in args.model_name:
        model_name, pretrained_tag = args.model_name.split(':', 1)
    else:
        model_name, pretrained_tag = args.model_name, 'openai'
    
    # Get available models for reference
    if args.verbose:
        print("Available OpenCLIP models:")
        pretrained_models = open_clip.list_pretrained()
        for model_info in pretrained_models:
            print(f"- {model_info}")
    
    # Create model and transforms
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained_tag,
        precision='fp32',
        device=device,
        jit=False,
        force_quick_gelu=False
    )
    
    # Extract the vision model from CLIP
    if hasattr(model, 'visual'):
        vision_model = model.visual
    else:
        vision_model = model
    
    vision_model = vision_model.to(device).eval()
    
    # Determine feature dimensions
    feature_dim = None
    if hasattr(vision_model, 'output_dim'):
        feature_dim = vision_model.output_dim
    elif hasattr(vision_model, 'embed_dim'):
        feature_dim = vision_model.embed_dim
    elif hasattr(vision_model, 'width'):
        feature_dim = vision_model.width
    else:
        # Fallback: run a forward pass to determine output dimension
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            try:
                output = vision_model(dummy_input)
                if isinstance(output, tuple):
                    output = output[0]  # Some models return multiple outputs
                feature_dim = output.shape[1]
            except Exception as e:
                print(f"Could not determine feature dimension automatically: {e}")
                feature_dim = "unknown"
    
    print(f"Feature dimension: {feature_dim}")
    
    # Check if model supports forward_intermediates
    has_intermediates_api = hasattr(model, 'forward_intermediates')
    
    if args.feature_mode == 'pyramid' and not has_intermediates_api:
        print("Warning: This OpenCLIP version might not support the intermediate features API.")
        print("Proceeding with pyramid mode extraction, but results may vary.")
    
    # File for saving features
    model_name_safe = args.model_name.replace('/', '_').replace(':', '_')
    output_file = os.path.join(args.output_dir, f"openclip_{model_name_safe}_{args.feature_mode}_features.h5")
    h5_file = h5py.File(output_file, 'w')
    
    # Create a mapping from scene index to padded string representation
    scene_id_map = {idx: f"scene_{idx:0{max_digits}d}" for idx in actual_scenes}
    
    # Save the scene id mapping in the HDF5 file
    scene_map_group = h5_file.create_group("scene_mapping")
    for idx, padded_id in scene_id_map.items():
        scene_map_group.attrs[padded_id] = idx
    
    # Save metadata
    input_size = (3, 224, 224)  # Default for most OpenCLIP models
    
    # Try to get actual input size from the model
    if hasattr(vision_model, 'image_size'):
        if isinstance(vision_model.image_size, (tuple, list)):
            input_size = (3, vision_model.image_size[0], vision_model.image_size[1])
        else:
            input_size = (3, vision_model.image_size, vision_model.image_size)
    elif hasattr(vision_model, 'input_resolution'):
        input_size = (3, vision_model.input_resolution, vision_model.input_resolution)
    
    # Create metadata
    metadata = {
        'model_name': args.model_name,
        'model_architecture': type(vision_model).__name__,
        'input_size': input_size,
        'feature_mode': args.feature_mode,
        'dataset_name': args.dataset_name,
        'actual_scenes': actual_scenes,
        'num_scenes': len(actual_scenes),
        'num_timesteps': args.num_timesteps,
        'device': str(device),
        'max_digits': max_digits,
        'seed': args.seed,
        'feature_dim': str(feature_dim),
        'pyramid_token_mode': args.pyramid_token_mode if args.feature_mode == 'pyramid' else 'n/a',
        'normalize_intermediates': str(args.normalize_intermediates),
    }
    
    # Store metadata in the HDF5 file as attributes
    for key, value in metadata.items():
        if isinstance(value, list):
            h5_file.attrs[key] = json.dumps(value)
        else:
            h5_file.attrs[key] = str(value)
    
    # For pyramid mode, create a group for each layer of interest
    # For non-pyramid modes, create a single features group
    if args.feature_mode == 'pyramid':
        features_group = h5_file.create_group("image_intermediates")
    else:
        features_group = h5_file.create_group("features")
    
    # Create custom dataset for specific scenes
    class CustomSceneASP(ASP):
        def __init__(self, scene_base, scene_idx, num_timesteps, dataset_name, seed=None):
            super().__init__(scene_base=scene_base, scenes=[scene_idx], 
                         num_timesteps=num_timesteps, dataset_name=dataset_name, seed=seed)
            self.specific_scene_idx = scene_idx
        
        def __getitem__(self, idx):
            # Use the specific scene directly
            return self.get_random_samples_within_scene(specific_scene=self.specific_scene_idx)
    
    # Process all filtered scenes
    print(f"Processing {len(actual_scenes)} scenes")
    
    for i, scene_idx in enumerate(tqdm(actual_scenes)):
        # Create dataset for this specific scene
        scene_dataset = CustomSceneASP(
            scene_base=args.data_path,
            scene_idx=scene_idx,
            num_timesteps=args.num_timesteps,
            dataset_name=args.dataset_name,
            seed=args.seed
        )
        
        # Get scene data
        img_stack, mask_stack, info = scene_dataset[0]
        
        # Get the zero-padded scene ID
        padded_scene_id = scene_id_map[scene_idx]
        
        # Create a group for this scene's metadata (only if saving scene info)
        if args.save_scene_info:
            scene_info_group = h5_file.create_group(f"{padded_scene_id}_info")
            
            # Store scene metadata - handle different data types properly
            for key, value in info.items():
                try:
                    # For numeric data
                    if isinstance(value, (int, float, bool, np.number)) or (
                        isinstance(value, (list, np.ndarray)) and 
                        all(isinstance(x, (int, float, bool, np.number)) for x in value)
                    ):
                        scene_info_group.create_dataset(key, data=np.array(value))
                    
                    # For string data or mixed lists
                    else:
                        # Convert to list of bytes for storage in HDF5
                        if isinstance(value, list):
                            # Convert each item to string and then to bytes
                            bytes_data = [str(item).encode('utf-8') for item in value]
                            scene_info_group.create_dataset(
                                key, 
                                data=np.array(bytes_data, dtype=h5py.string_dtype())
                            )
                        else:
                            # Single string
                            scene_info_group.create_dataset(
                                key, 
                                data=np.array(str(value).encode('utf-8'), dtype=h5py.string_dtype())
                            )
                
                except Exception as e:
                    if args.verbose:
                        print(f"Error saving metadata for key {key}: {e}")
                    # Save as JSON string as a fallback
                    try:
                        json_data = json.dumps(value)
                        scene_info_group.create_dataset(
                            key, 
                            data=np.array(json_data.encode('utf-8'), dtype=h5py.string_dtype())
                        )
                    except:
                        if args.verbose:
                            print(f"Could not save {key} - skipping")
        
        # Process each image in the scene
        for t in range(img_stack.shape[0]):
            # Get a single image
            img = img_stack[t]
            
            # Apply OpenCLIP preprocessing
            # Convert from [C, H, W] to PIL for OpenCLIP preprocessing
            img_pil = transforms.ToPILImage()(img)
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
            
            # Extract features based on mode
            with torch.no_grad():
                if args.feature_mode == 'pyramid':
                    # Use the forward_intermediates method to get features from different layers
                    # This is the new API added in the OpenCLIP update
                    
                    # Determine which indices to extract
                    if args.out_indices[0] == -1:
                        indices = None
                    elif hasattr(vision_model, 'transformer') and hasattr(vision_model.transformer, 'resblocks'):
                        num_layers = len(vision_model.transformer.resblocks)
                        indices = args.out_indices or [num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
                    else:
                        indices = args.out_indices or 4  # Default to last 4 layers
                    
                    # Set output format based on whether we want CLS token or full feature map
                    output_fmt = 'NLC'  # Default to NLC format for transformer models
                    if isinstance(vision_model, open_clip.model.ModifiedResNet):
                        output_fmt = 'NCHW'  # Use NCHW for ResNet models
                    
                    # Call forward_intermediates on the model
                    if hasattr(model, 'forward_intermediates'):
                        outputs = model.forward_intermediates(
                            image=img_tensor,
                            image_indices=indices,
                            normalize_intermediates=args.normalize_intermediates,
                            intermediates_only=True,
                            image_output_fmt=output_fmt,
                            image_output_extra_tokens=args.pyramid_token_mode == 'all'
                        )
                        
                        # Get the intermediate features
                        intermediates = outputs.get('image_intermediates', [])
                        prefix_features = outputs.get('image_intermediates_prefix', [])
                        
                        # Store each intermediate feature
                        for i, feature in enumerate(intermediates):
                            # Convert to numpy
                            feature_data = feature.cpu().numpy()
                            
                            # If we want all tokens and there are prefix tokens available, concatenate them
                            if args.pyramid_token_mode == 'all' and i < len(prefix_features):
                                # Prefix tokens have shape [batch, num_prefix, dim]
                                # Feature tokens have shape [batch, num_patches, dim] 
                                # or [batch, channels, height, width] for CNNs
                                prefix_data = prefix_features[i].cpu()
                                
                                # For transformer models with NLC format, we can concatenate along sequence dimension
                                if output_fmt == 'NLC':
                                    # Concatenate along the sequence/patch dimension
                                    combined = torch.cat([prefix_data, feature.cpu()], dim=1)
                                    feature_data = combined.numpy()
                            
                            # Create dataset name based on layer index
                            layer_name = f"layer_{i}"
                            
                            # Create the layer group if it doesn't exist
                            if layer_name not in features_group:
                                features_group.create_group(layer_name)
                            
                            # Save the feature
                            features_group[layer_name].create_dataset(
                                f"{padded_scene_id}_t{t}", 
                                data=feature_data,
                                compression="gzip"
                            )
                            
                        # Store extra metadata about whether we combined tokens
                        if args.pyramid_token_mode == 'all' and len(prefix_features) > 0:
                            h5_file.attrs["combined_prefix_tokens"] = "True"
                            h5_file.attrs["prefix_token_position"] = "prefix"  # Indicates prefix tokens are at beginning
                                
                    else:
                        # Fallback for older OpenCLIP versions
                        # Just extract the image features
                        features = vision_model(img_tensor)
                        
                        # Handle different return types
                        if isinstance(features, tuple):
                            features = features[0]  # Some models return multiple outputs
                        
                        # Save features
                        feature_data = features.cpu().numpy()
                        features_group.create_dataset(
                            f"{padded_scene_id}_t{t}", 
                            data=feature_data,
                            compression="gzip"
                        )
                        
                else:  # penultimate_pooled or penultimate_unpooled
                    # For non-pyramid modes, just extract the final features
                    features = vision_model(img_tensor)
                    
                    # Handle different return types
                    if isinstance(features, tuple):
                        features = features[0]  # Some models return multiple outputs
                    
                    # Save pooled features
                    feature_data = features.cpu().numpy()
                    features_group.create_dataset(
                        f"{padded_scene_id}_t{t}", 
                        data=feature_data,
                        compression="gzip"
                    )
    
    h5_file.close()
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from OpenCLIP models for ASP scenes')
    parser.add_argument('--model_name', type=str, default='ViT-B-32:openai', 
                        help='Name of the OpenCLIP model (format: model_name:pretrained_tag)')
    parser.add_argument('--data_path', type=str, default='../data/ASP_FixedSun/', help='Path to the ASP dataset')
    parser.add_argument('--output_dir', type=str, default='./features', help='Output directory')
    parser.add_argument('--num_scenes', type=int, default=1000, 
                        help='Number of scenes to process after filtering')
    parser.add_argument('--max_scene_search', type=int, default=5000,
                        help='Maximum number of scenes to search for matching criteria')
    parser.add_argument('--num_timesteps', type=int, default=3, help='Number of timesteps per scene')
    parser.add_argument('--feature_mode', type=str, default='pyramid', 
                      choices=['pyramid', 'penultimate_unpooled', 'penultimate_pooled'],
                      help='Feature extraction mode: pyramid (multiple layers), penultimate_unpooled, or penultimate_pooled')
    parser.add_argument('--out_indices', nargs='*', type=int, default=None, 
                        help='Indices of transformer layers to extract (only for pyramid mode)')
    parser.add_argument('--pyramid_token_mode', type=str, default='cls', choices=['cls', 'all'],
                        help='For pyramid mode: extract only cls token or all tokens')
    parser.add_argument('--normalize_intermediates', action='store_true', default=False,
                        help='Apply normalization to intermediate features')
    parser.add_argument('--dataset_name', type=str, default='asp_surround_mix_noref', 
                        help='Dataset name')
    parser.add_argument('--save_scene_info', action='store_true', default=False,
                        help='Save scene metadata (can make files larger)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)