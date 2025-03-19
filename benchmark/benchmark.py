import os
import torch
import numpy as np
import timm
from tqdm import tqdm
import argparse
import h5py
import json
from torchvision import transforms
import random

import sys
sys.path.append('/home/markus/Documents/Github/allocentric-scene-perception')
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
    device = torch.device('cpu')
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
    # This ensures scenes are sorted correctly (scene_001 comes before scene_002, etc.)
    max_digits = len(str(max(actual_scenes)))
    print(f"Using {max_digits} digits for scene numbering")
    
    # Set up model based on feature extraction mode
    print(f"Loading model: {args.model_name}")
    
    if args.feature_mode == 'pyramid':
        # Use features_only for feature pyramid (multiple layers)
        model = timm.create_model(
            args.model_name, 
            pretrained=True, 
            features_only=True,
            out_indices=args.out_indices
        )
        is_feature_pyramid = True
    elif args.feature_mode == 'penultimate_unpooled':
        # Regular model for forward_features (unpooled penultimate features)
        model = timm.create_model(args.model_name, pretrained=True)
        is_feature_pyramid = False
    elif args.feature_mode == 'penultimate_pooled':
        # Model with removed classifier but kept pooling
        model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
        is_feature_pyramid = False
    else:
        raise ValueError(f"Unknown feature mode: {args.feature_mode}")
    
    model = model.to(device).eval()
    
    # Resolve data configuration for the specific model
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    
    # Create transform using the resolved data config
    print("Using model-specific data transform")
    transform = timm.data.create_transform(**data_config)
    
    print(f"Transform details: {transform}")
    
    # Get feature info for pyramid mode
    if is_feature_pyramid:
        feature_info = model.feature_info
        print(f"Feature channels: {feature_info.channels()}")
        print(f"Feature reduction: {feature_info.reduction()}")
    else:
        # Try to determine feature dimension for penultimate features
        try:
            if args.feature_mode == 'penultimate_pooled':
                # Get shape from forward pass
                with torch.no_grad():
                    # Use input size from data config
                    input_size = data_config.get('input_size', (3, 224, 224))
                    dummy_input = torch.randn(1, *input_size).to(device)
                    dummy_output = model(dummy_input)
                    feature_dim = dummy_output.shape[1]
            else:  # penultimate_unpooled
                # Try to get feature dimension from model's feature_info if available
                if hasattr(model, 'feature_info'):
                    feature_dim = model.feature_info.channels()[-1]
                else:
                    # This is a fallback and might not be accurate for all models
                    feature_dim = model.num_features if hasattr(model, 'num_features') else "unknown"
        except Exception as e:
            feature_dim = "unknown"
            if args.verbose:
                print(f"Could not determine feature dimension: {e}")
        
        print(f"Penultimate feature dimension: {feature_dim}")
    
    # File for saving features
    output_file = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_{args.feature_mode}_features.h5")
    h5_file = h5py.File(output_file, 'w')
    
    # Create a mapping from scene index to padded string representation
    scene_id_map = {idx: f"scene_{idx:0{max_digits}d}" for idx in actual_scenes}
    
    # Save the scene id mapping in the HDF5 file
    scene_map_group = h5_file.create_group("scene_mapping")
    for idx, padded_id in scene_id_map.items():
        scene_map_group.attrs[padded_id] = idx
    
    # Save metadata
    metadata = {
        'model_name': args.model_name,
        'input_size': data_config.get('input_size', (3, 224, 224)),
        'feature_mode': args.feature_mode,
        'dataset_name': args.dataset_name,
        'actual_scenes': actual_scenes,
        'num_scenes': len(actual_scenes),
        'num_timesteps': args.num_timesteps,
        'device': 'cpu',
        'max_digits': max_digits,
        'seed': args.seed,  # Save the seed in metadata
        'data_transform_config': {
            'mean': data_config.get('mean', None),
            'std': data_config.get('std', None),
            'interpolation': data_config.get('interpolation', None),
            'crop_pct': data_config.get('crop_pct', None)
        }
    }
    
    # Add feature pyramid specific metadata
    if is_feature_pyramid:
        metadata['feature_channels'] = feature_info.channels()
        metadata['feature_reduction'] = feature_info.reduction()
    else:
        metadata['feature_dim'] = str(feature_dim)
    
    # Store metadata in the HDF5 file as attributes
    for key, value in metadata.items():
        if isinstance(value, list):
            h5_file.attrs[key] = json.dumps(value)
        else:
            h5_file.attrs[key] = str(value)
    
    # Create a group for features
    if is_feature_pyramid:
        # Create a group for each feature level
        feature_groups = {}
        for i, (channels, reduction) in enumerate(zip(feature_info.channels(), feature_info.reduction())):
            feature_name = f"level_{i}_stride_{reduction}"
            feature_groups[feature_name] = h5_file.create_group(feature_name)
    else:
        # Single group for penultimate features
        features_group = h5_file.create_group("features")
    
    # Process all filtered scenes
    print(f"Processing {len(actual_scenes)} scenes")
    
    # Create custom dataset for specific scenes
    class CustomSceneASP(ASP):
        def __init__(self, scene_base, scene_idx, num_timesteps, dataset_name, seed=None):
            super().__init__(scene_base=scene_base, scenes=[scene_idx], 
                         num_timesteps=num_timesteps, dataset_name=dataset_name, seed=seed)
            self.specific_scene_idx = scene_idx
        
        def __getitem__(self, idx):
            # Use the specific scene directly
            return self.get_samples_within_scene(specific_scene=self.specific_scene_idx)
    
    for i, scene_idx in enumerate(tqdm(actual_scenes)):
        # Create dataset for this specific scene
        scene_dataset = CustomSceneASP(
            scene_base=args.data_path,
            scene_idx=scene_idx,
            num_timesteps=args.num_timesteps,
            dataset_name=args.dataset_name,
            seed=args.seed  # Pass the seed to CustomSceneASP
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
            # Get a single image and add batch dimension
            img = img_stack[t:t+1]
            
            # Apply transform
            img = transform(img).to(device)
            
            # Extract features based on mode
            with torch.no_grad():
                if is_feature_pyramid:
                    # Get feature pyramid
                    features_list = model(img)
                    
                    # Store features for each level
                    for i, feature in enumerate(features_list):
                        feature_name = f"level_{i}_stride_{feature_info.reduction()[i]}"
                        
                        # Save raw feature maps with zero-padded scene ID
                        feature_data = feature.cpu().numpy()
                        feature_groups[feature_name].create_dataset(
                            f"{padded_scene_id}_t{t}", 
                            data=feature_data,
                            compression="gzip"
                        )
                elif args.feature_mode == 'penultimate_unpooled':
                    # Get unpooled penultimate features using forward_features
                    features = model.forward_features(img)
                    
                    # Save raw feature maps with zero-padded scene ID
                    feature_data = features.cpu().numpy()
                    features_group.create_dataset(
                        f"{padded_scene_id}_t{t}", 
                        data=feature_data,
                        compression="gzip"
                    )
                else:  # penultimate_pooled
                    # Get pooled penultimate features
                    features = model(img)
                    
                    # Save pooled features with zero-padded scene ID
                    feature_data = features.cpu().numpy()
                    features_group.create_dataset(
                        f"{padded_scene_id}_t{t}", 
                        data=feature_data,
                        compression="gzip"
                    )
    
    h5_file.close()
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features from timm models for ASP scenes')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Name of the timm model')
    parser.add_argument('--data_path', type=str, default='/data/asp/', help='Path to the ASP dataset')
    parser.add_argument('--output_dir', type=str, default='./features', help='Output directory')
    parser.add_argument('--num_scenes', type=int, default=1000, 
                        help='Number of scenes to process after filtering')
    parser.add_argument('--max_scene_search', type=int, default=5000,
                        help='Maximum number of scenes to search for matching criteria')
    parser.add_argument('--num_timesteps', type=int, default=3, help='Number of timesteps per scene')
    parser.add_argument('--feature_mode', type=str, default='penultimate_pooled', 
                      choices=['pyramid', 'penultimate_unpooled', 'penultimate_pooled'],
                      help='Feature extraction mode: pyramid (multiple layers), penultimate_unpooled, or penultimate_pooled')
    parser.add_argument('--out_indices', nargs='*', type=int, default=None, 
                        help='Indices of feature levels to extract (only for pyramid mode)')
    parser.add_argument('--input_size', type=int, default=None,
                        help='Input image size (default: 224 for most models)')
    parser.add_argument('--dataset_name', type=str, default='asp_surround_green_ref', 
                        help='Dataset name')
    parser.add_argument('--save_scene_info', action='store_true', default=False,
                        help='Save scene metadata (can make files larger)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)