import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import pytorch_lightning as pl
import torch
import torch.distributed
from torchvision import transforms as T
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ASP(Dataset):
    def __init__(self, scene_base = '/data/asp/', scenes=range(0, 800), samples_per_scene=range(0, 16), num_timesteps=8, dataset_name='asp_surround', seed=None): 
        self.scene_base = scene_base
        self.scenes = scenes
        self.samples_per_scene = samples_per_scene
        self.num_timesteps = num_timesteps
        
        # Set the base random seed
        self.base_seed = seed if seed is not None else 42
        
        # Create a global RNG for scene selection
        self.global_rng = np.random.RandomState(self.base_seed)
            
        self.setup(dataset_name)
        self.image_transform = T.Resize((64, 64))

    def setup(self, dataset_name):
        # Choose samples: (0,16) are snapshots from the surround condition, (16,32) are snapshots from the within condition
        if 'surround' in dataset_name:
            self.samples_per_scene = range(0, 16)
        if 'within' in dataset_name:
            self.samples_per_scene = range(16, 32)
        if 'both' in dataset_name:
            self.samples_per_scene = range(0, 32)
        chosen_scenes = []

        # Choose color of objects
        color = -1
        if 'mix' in dataset_name:
            color = 0
        elif 'green' in dataset_name:
            color = 1
        elif 'white' in dataset_name:
            color = 2

        # Helper function
        def chose_color(chosen_scenes, color, scene, df_scene):
            if color > -1:
                if df_scene['# Color'][0] == color:
                    chosen_scenes.append(scene)
            else:
                chosen_scenes.append(scene)
            return chosen_scenes

        # Choose reference frame 
        for scene in self.scenes:
            df_scene = pd.read_csv(self.scene_base + 'props_scene{}.csv'.format(scene))
            # If no reference frame is specified, use both
            if ('ref' not in dataset_name) and ('noref' not in dataset_name):
                chosen_scenes = chose_color(chosen_scenes, color, scene, df_scene)

            # If reference frame is specified, use only the specified one
            if not df_scene['GlobalRef'][0] and ('noref' in dataset_name):
                chosen_scenes = chose_color(chosen_scenes, color, scene, df_scene)
            if df_scene['GlobalRef'][0] and ('ref' in dataset_name) and ('no' not in dataset_name):
                chosen_scenes = chose_color(chosen_scenes, color, scene, df_scene)

        self.scenes = chosen_scenes
        # print('Number of chosen scenes for {}: {}'.format(dataset_name, len(self.scenes)))

    def read_img(self, path):
        image = Image.open(path + '_rgb.jpg').convert('RGB')
        # image = image.resize((64, 64))
        image = torch.tensor(np.array(image)) / 255
        image = image.unsqueeze(0)
        return image

    def read_mask(self, path):
        mask = Image.open(path + '_inst.png').convert('L')
        mask_array = np.array(mask)
        mask_tensor = torch.tensor(mask_array, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(mask_tensor, num_classes=10)
        one_hot = one_hot.unsqueeze(0)
        
        return one_hot

    def __getitem__(self, idx):
        return self.get_samples_within_scene() 

    def get_scene_specific_rng(self, scene_idx):
        """Create a deterministic RNG for a specific scene"""
        # Combine the base seed with the scene index to get a scene-specific seed
        scene_seed = self.base_seed * 10000 + scene_idx
        return np.random.RandomState(scene_seed)
        
    def get_random_samples_within_scene(self, specific_scene=None):
        """
        Get random samples from within a scene.
        
        Unlike get_samples_within_scene which takes consecutive samples,
        this function takes random samples from the available timesteps.
        
        Args:
            specific_scene: If provided, use this specific scene. 
                        If None, randomly select a scene.
        """
        if specific_scene is None:
            # Use the global RNG to select a random scene
            rand_scene = self.global_rng.choice(self.scenes)
        else:
            # Use the specified scene
            rand_scene = specific_scene
        
        # Create a scene-specific RNG for consistent sampling within this scene
        scene_rng = self.get_scene_specific_rng(rand_scene)
        
        # Get all samples for this scene
        df_scene = pd.read_csv(self.scene_base + 'props_scene{}.csv'.format(rand_scene))
        df_scene = df_scene.iloc[self.samples_per_scene]
        
        # Select random indices without replacement
        indices = scene_rng.choice(len(self.samples_per_scene), size=self.num_timesteps, replace=False)
        selected_rows = df_scene.iloc[indices]
        
        # print(f"rand Scene: {rand_scene}, df: {selected_rows['img_path']}")

        # Loop through selected rows and stack images
        for i, row in enumerate(selected_rows.iterrows()):
            row = row[1]  # Get the Series from the tuple
            row_split = row['img_path'].split('/')
            row_correct = self.scene_base + row_split[-2] + os.path.sep + row_split[-1]
            image = self.read_img(row_correct)
            mask = self.read_mask(row_correct)
            if i == 0:
                img_stack = image
                mask_stack = mask
            else:
                img_stack = torch.cat((img_stack, image), dim=0)
                mask_stack = torch.cat((mask_stack, mask), dim=0)
        
        mask_stack = rearrange(mask_stack, "t h w k -> t k h w")
        mask_stack = self.image_transform(mask_stack)
        mask_stack = mask_stack.unsqueeze(-1)

        img_stack = rearrange(img_stack, "t h w c -> t c h w")

        return img_stack, mask_stack, selected_rows.to_dict(orient='list')


    def get_samples_within_scene(self, specific_scene=None):
        """
        Get samples from within a scene.
        
        Args:
            specific_scene: If provided, use this specific scene. 
                           If None, randomly select a scene.
        """
        if specific_scene is None:
            # Use the global RNG to select a random scene
            rand_scene = self.global_rng.choice(self.scenes)
        else:
            # Use the specified scene
            rand_scene = specific_scene
        
        # Create a scene-specific RNG for consistent sampling within this scene
        scene_rng = self.get_scene_specific_rng(rand_scene)
        
        df_scene = pd.read_csv(self.scene_base + 'props_scene{}.csv'.format(rand_scene))
        df_scene = df_scene.iloc[self.samples_per_scene]

        # Use the scene-specific RNG to determine the starting index
        # This ensures that the same scene always uses the same images
        rand_indices_start = scene_rng.randint(0, len(self.samples_per_scene) - self.num_timesteps)
        df_scene = df_scene.iloc[rand_indices_start : rand_indices_start + self.num_timesteps]

        # print(f"rand Scene: {rand_scene}, df: {df_scene['img_path']}")

        # Loop through rand_rows and stack images
        for i in range(0, df_scene.shape[0]):
            row = df_scene.iloc[i, :]
            row_split = row['img_path'].split('/')
            row_correct = self.scene_base + row_split[-2] + os.path.sep + row_split[-1]
            image = self.read_img(row_correct)
            mask = self.read_mask(row_correct)            
            if i == 0:
                img_stack = image
                mask_stack = mask
            else:
                img_stack = torch.cat((img_stack, image), dim=0)
                mask_stack = torch.cat((mask_stack, mask), dim=0)
        mask_stack = rearrange(mask_stack, "t h w k -> t k h w")
        mask_stack = self.image_transform(mask_stack)
        mask_stack = mask_stack.unsqueeze(-1)

        img_stack = rearrange(img_stack, "t h w c -> t c h w")

        return img_stack, mask_stack, df_scene.to_dict(orient='list')

    def get_samples_across_scenes(self):
        # Select scenes using the global RNG
        rand_scenes = self.global_rng.choice(self.scenes, self.num_timesteps)
        
        for i, rand_scene in enumerate(rand_scenes):
            # Get scene-specific RNG for each scene
            scene_rng = self.get_scene_specific_rng(rand_scene)
            
            df_scene = pd.read_csv(self.scene_base + 'props_scene{}.csv'.format(rand_scene))
            df_scene = df_scene.iloc[self.samples_per_scene]
            
            # Use the consistent scene_rng-determined index
            scene_rand_index = scene_rng.randint(0, len(self.samples_per_scene))
            df_scene = df_scene.iloc[scene_rand_index]
            
            row = df_scene['img_path']
            row_split = row.split('/')
            row_correct = self.scene_base + row_split[-2] + os.path.sep + row_split[-1]
            image = self.read_img(row_correct)
            mask = self.read_mask(row_correct)
            if i == 0:
                img_stack = image
                mask_stack = mask
            else:
                img_stack = torch.cat((img_stack, image), dim=0)
                mask_stack = torch.cat((mask_stack, mask), dim=0)
        mask_stack = rearrange(mask_stack, "t h w k -> t k h w")
        mask_stack = self.image_transform(mask_stack)
        mask_stack = mask_stack.unsqueeze(-1)

        img_stack = rearrange(img_stack, "t h w c -> t c h w")

        return img_stack, mask_stack

    def __len__(self):
        return (len(self.scenes) * len(self.samples_per_scene))


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_gpus: int,
        train_batch_size: int,
        train_dataset_size: int,
        val_batch_size: int,
        val_dataset_size: int,
        num_train_workers: int = 0,
        num_timesteps: int = 6,
        dataset: str = 'asp_surround',
        dataset_path: str = '/data/',
        seed: int = None,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_train_workers = num_train_workers
        self.train_dataset_size = train_dataset_size
        self.val_dataset_size = val_dataset_size
        self.n_gpus = n_gpus
        self.num_timesteps = num_timesteps
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        # Fix train and test set. For reproducibility, this should not be changed!
        train_scenes = 5000
        test_scenes = 1000
        # -----------------------

        self.train_dataset = ASP(scenes=range(0, train_scenes), scene_base=self.dataset_path, num_timesteps=self.num_timesteps, dataset_name=self.dataset, seed=self.seed)
        self.val_dataset = ASP(scenes=range(train_scenes, train_scenes + test_scenes), scene_base=self.dataset_path, num_timesteps=self.num_timesteps, dataset_name=self.dataset, seed=self.seed)   
        self.max_entities = 6 # Used for calculating ARI

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False, # Dataloader already shuffles samples
            drop_last=False,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )