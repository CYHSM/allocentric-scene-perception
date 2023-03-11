import random

import pytorch_lightning as pl
import torch
import torch.distributed
from einops import rearrange, repeat
from torch import Tensor
import numpy as np
import seaborn as sns
import warnings
from scipy.ndimage.filters import gaussian_filter
from PIL import Image as PILImage
import matplotlib.pyplot as plt

from helper import compute_ari
import math
import secrets
import time
import wandb


LOG_FREQ = 500
def _generate_color_palette(num_masks: int, bg_color=(0.5, 0.5, 0.5)):
    palette = [bg_color] + sns.color_palette('hls', num_masks-1)
    return torch.tensor(palette)

class WandbCallback(pl.Callback):
    """Here we just log a ton of stuff to make for easier debugging."""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.state = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        batch, mask, info = batch
        weighted_pixels = outputs["weighted_pixels"]
        weights_softmax = outputs["weights_softmax"]
        pixels = outputs["pixels"]

        log_iteration_time(trainer, self.batch_size)
        log_scalar(trainer, "train/loss", outputs["loss"], freq=1)
        log_scalar(trainer, "train/p_loss", outputs["p_loss"], freq=1)
        log_scalar(trainer, "train/l2o_loss", outputs["l2o_loss"], freq=1)
        log_scalar(trainer, "train/l2f_loss", outputs["l2f_loss"], freq=1)

        log_video(trainer, "videos/train_dataset", batch, freq=LOG_FREQ)

        # shape is (b, t, w, h, c)
        weighted_pixels_video = weighted_pixels.permute((0, 1, 4, 2, 3))
        log_video(trainer, "videos/weighted_pixels", weighted_pixels_video, freq=LOG_FREQ)

        # show videos of the masks for a single batch element (so a grid of the 16 object masks)
        weights_softmax_video = weights_softmax[0].unsqueeze(-1).permute((1, 0, 4, 2, 3))
        log_video(trainer, "videos/weights_softmax", weights_softmax_video, freq=LOG_FREQ)

        # show videos of the pixels for each object for a single batch element (so a grid of the 16 object masks)
        pixels_video = pixels[0].permute((1, 0, 4, 2, 3))
        log_video(trainer, "videos/pixels", pixels_video, freq=LOG_FREQ)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        batch, mask, info = batch

        latents = outputs["latents"] # (mec, lec, hc, hc_transformed, ph, pr)
        weighted_pixels = outputs["weighted_pixels"]
        weights_softmax = outputs["weights_softmax"]
        pixels = outputs["pixels"]

        # Save state to calculate stuff across batches, ratemaps etc...
        self.state.append((latents, info))
        
        # Compute ARI
        ari_fg = compute_ari(mask.cpu(), weights_softmax.cpu(), fg_only=True, max_num_entities=trainer.datamodule.max_entities)
        ari_full = compute_ari(mask.cpu(), weights_softmax.cpu(), fg_only=False, max_num_entities=trainer.datamodule.max_entities)
        log_scalar(trainer, "val/ARI_Foreground", ari_fg.mean(), freq=1, offset=batch_idx)
        log_scalar(trainer, "val/ARI_Full", ari_full.mean(), freq=1, offset=batch_idx)

        log_scalar(trainer, "val/loss", outputs["loss"], freq=1, offset=batch_idx)

        if batch_idx > 2:
            return
        
        log_video(trainer, "val/dataset", batch, freq=1, offset=batch_idx)

        # shape is (b, t, w, h, c)
        weighted_pixels = weighted_pixels.permute((0, 1, 4, 2, 3))
        log_video(trainer, "val/weighted_pixels", weighted_pixels[:16], freq=1, offset=batch_idx)
        weights_softmax_video = weights_softmax[0].unsqueeze(-1).permute((1, 0, 4, 2, 3))
        log_video(trainer, "val/weights_softmax", weights_softmax_video, freq=1, offset=batch_idx)

        # show videos of the pixels for each object for a single batch element (so a grid of the 16 object masks)
        pixels = pixels[0].permute((1, 0, 4, 2, 3))
        log_video(trainer, "val/pixels", pixels, freq=1, offset=batch_idx)

        # Log segmentation mask
        colors = _generate_color_palette(weights_softmax.shape[2])
        segmentation = _generate_segmentation(weights_softmax, colors, bg_idx=weights_softmax.mean(dim=(0,1,3,4)).argmax()).permute(0, 1, 4, 2, 3)
        log_video(trainer, "val/segmentation", segmentation, freq=1, offset=batch_idx)

        # Log true mask
        colors_t = _generate_color_palette(trainer.datamodule.max_entities)
        segmentation = _generate_segmentation(mask[..., 0], colors_t).permute(0, 1, 4, 2, 3)
        log_video(trainer, "val/segmentation_true", segmentation, freq=1, offset=batch_idx)

        # Log latents
        # Shape is (b, K or T, 32, 2)
        log_image(trainer, "latents/mec", latents[0][0, ...].to(torch.float32), freq=1, offset=batch_idx)
        log_image(trainer, "latents/lec", latents[1][0, ...].to(torch.float32), freq=1, offset=batch_idx)
        log_image(trainer, "latents/hc_t0", latents[2][0, 0, ...].to(torch.float32), freq=1, offset=batch_idx)
        log_image(trainer, "latents/hc_t-1", latents[2][0, -1, ...].to(torch.float32), freq=1, offset=batch_idx)
        log_image(trainer, "latents/hc_transformed_t0", latents[3][0, 0, ...].to(torch.float32), freq=1, offset=batch_idx)
        log_image(trainer, "latents/hc_transformed_t-1", latents[3][0, -1, ...].to(torch.float32), freq=1, offset=batch_idx)
        log_image(trainer, "latents/ph", latents[4][0, ...].to(torch.float32), freq=1, offset=batch_idx)
        log_image(trainer, "latents/pr", latents[5][0, ...].to(torch.float32), freq=1, offset=batch_idx)


def _generate_segmentation(weights: Tensor, colors: Tensor, bg_idx=0):
    """Generate a segmentation mask visualization."""
    if bg_idx != 0:
        # swap idx 0 and bg_idx
        colors[[0, bg_idx], :] = colors[[bg_idx, 0], :]

    colors = colors.to(weights.device)
    # weights should have shape b, t, k, h, w
    b, t, k, h, w = weights.shape
    assert len(colors) == k
    # colors should have shape k, 3
    ce = colors.view(1, 1, k, 1, 1, 3).expand(b, t, k, h, w, 3)
    wa = weights.argmax(dim=2)
    we = wa.view(b, t, 1, h, w, 1).expand(b, t, 1, h, w, 3)
    return torch.gather(ce, 2, we).view(b, t, h, w, 3)



"""A little utility library to make for easy logging to wandb."""
"""Taken from https://gitlab.com/generally-intelligent/simone"""
last_time = None
last_step = None
LOG_BACKOFF_POINT = 5000
LOG_BACKOFF_FACTOR = 20


def make_video_grid(
    tensor,
    num_images_per_row: int = 10,
    padding: int = 2,
    pad_value: int = 0,
):
    """
    This is a repurposed implementation of `make_grid` from torchvision to work with videos.
    """
    n_maps, sequence_length, num_channels, height, width = tensor.size()
    x_maps = min(num_images_per_row, n_maps)
    y_maps = int(math.ceil(float(n_maps) / x_maps))
    height, width = int(height + padding), int(width + padding)
    grid = tensor.new_full(
        (sequence_length, num_channels, height * y_maps + padding, width * x_maps + padding), pad_value
    )
    k = 0
    for y in range(y_maps):
        for x in range(x_maps):
            if k >= n_maps:
                break
            grid.narrow(2, y * height + padding, height - padding).narrow(
                3, x * width + padding, width - padding
            ).copy_(tensor[k])
            k += 1
    return grid


def effective_freq(step, freq):
    # Simple logging backoff logic.
    if step > LOG_BACKOFF_POINT and freq != 1:
        freq *= LOG_BACKOFF_FACTOR
    return freq


def check_log_interval(step, freq):
    freq = effective_freq(step, freq)
    return step % freq == 0


def download_file(run_id, project_name, filename=None):
    api = wandb.Api()
    run = api.run(f"sourceress/{project_name}/{run_id}")
    # We save to a random directory to avoid contention issues if there's
    # potentially multiple processes downloading the same file at the same time.
    # TODO: clean these files up
    path = run.file(filename).download(replace=True, root=f"./data/{secrets.token_hex(10)}").name
    return path


def log_histogram(trainer, tag, value, freq=20):
    if not trainer.is_global_zero:
        return
    if not check_log_interval(trainer.global_step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    trainer.logger.experiment.log({tag: value}, step=trainer.global_step)


def log_video(trainer, tag, batch, freq=10, normalize=False, offset=0):
    # Expects b, t, c, h, w
    if not trainer.is_global_zero:
        return
    if not check_log_interval(trainer.global_step, freq):
        return

    if normalize:
        min_v = torch.min(batch)
        range_v = torch.max(batch) - min_v
        if range_v > 0:
            batch = (batch - min_v) / range_v
        else:
            batch = torch.zeros(batch.size())

    # batch = preprocess_batch(batch).permute(0, 2, 1, 3, 4) + .5
    frames = make_video_grid(batch, num_images_per_row=4, pad_value=1)

    # This should be in range 0-1
    if type(frames) == torch.Tensor:
        frames = frames.detach()
    frames = (frames * 255).clamp(0, 255).to(torch.uint8)
    frames = frames.cpu()
    trainer.logger.experiment.log({tag: wandb.Video(frames, fps=1, format="gif")}, step=trainer.global_step + offset)

def logo_video(trainer, tag, batch, freq=10, normalize=False, offset=0):
    if normalize:
        min_v = torch.min(batch)
        range_v = torch.max(batch) - min_v
        if range_v > 0:
            batch = (batch - min_v) / range_v
        else:
            batch = torch.zeros(batch.size())

    # batch = preprocess_batch(batch).permute(0, 2, 1, 3, 4) + .5
    frames = make_video_grid(batch, num_images_per_row=16, pad_value=1)

    # This should be in range 0-1
    if type(frames) == torch.Tensor:
        frames = frames.detach()
    frames = (frames * 255).clamp(0, 255).to(torch.uint8)
    frames = frames.cpu()
    trainer.logger.experiment.log({tag: wandb.Video(frames, fps=0.3, format="gif")}, step=offset)


def log_image(trainer, tag, value, freq=20, offset=0):
    if not trainer.is_global_zero:
        return
    if not check_log_interval(trainer.global_step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    trainer.logger.experiment.log({tag: wandb.Image(value)}, step=trainer.global_step + offset)


def log_table(trainer, tag, value, freq=20):
    # value should be a 1d tensor, in this current implementation. can add more columns in the future.
    if not trainer.is_global_zero:
        return
    if not check_log_interval(trainer.global_step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    columns = ["test"]
    rows = [[x] for x in value]
    table = wandb.Table(data=rows, columns=columns)
    trainer.logger.experiment.log({tag: table}, step=trainer.global_step)


def log_scalar(trainer, tag, value, freq=20, offset=0):
    if not trainer.is_global_zero:
        return
    if not check_log_interval(trainer.global_step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    trainer.logger.experiment.log({tag: value}, step=trainer.global_step + offset)


def log_iteration_time(trainer, batch_size, freq=10):
    """Call this once per training iteration."""
    if not trainer.is_global_zero:
        return
    global last_time
    global last_step
    if not check_log_interval(trainer.global_step, freq):
        return

    if last_time is None:
        last_time = time.time()
        last_step = trainer.global_step
    else:
        if trainer.global_step == last_step:
            return
        dt = (time.time() - last_time) / (trainer.global_step - last_step)
        last_time = time.time()
        last_step = trainer.global_step
        log_scalar(trainer, "timings/iterations-per-sec", 1 / dt, freq=1)
        log_scalar(trainer, "timings/samples-per-sec", batch_size / dt, freq=1)


def watch(model, freq=50):
    # wandb.watch(model, log="all")
    wandb.watch(model, "all", log_freq=freq)
