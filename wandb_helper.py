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

import math
import secrets
import time
import wandb


LOG_FREQ = 500
def _generate_color_palette(num_masks: int, bg_color=(0.5, 0.5, 0.5)):
    palette = [bg_color] + sns.color_palette('hls', num_masks-1)
    return torch.tensor(palette)

class WandbCallback(pl.Callback):
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

        weighted_pixels_video = weighted_pixels.permute((0, 1, 4, 2, 3))
        log_video(trainer, "videos/weighted_pixels", weighted_pixels_video, freq=LOG_FREQ)

        weights_softmax_video = weights_softmax[0].unsqueeze(-1).permute((1, 0, 4, 2, 3))
        weights_softmax_video = weights_softmax_video.repeat(1, 1, 3, 1, 1)
        log_video(trainer, "videos/weights_softmax", weights_softmax_video, freq=LOG_FREQ)

        pixels_video = pixels[0].permute((1, 0, 4, 2, 3))
        log_video(trainer, "videos/pixels", pixels_video, freq=LOG_FREQ)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        batch, mask, info = batch

        latents = outputs["latents"]
        weighted_pixels = outputs["weighted_pixels"]
        weights_softmax = outputs["weights_softmax"]
        pixels = outputs["pixels"]

        self.state.append((latents, info))
        
        # ari_fg = compute_ari(mask.cpu(), weights_softmax.cpu(), fg_only=True, max_num_entities=trainer.datamodule.max_entities)
        # ari_full = compute_ari(mask.cpu(), weights_softmax.cpu(), fg_only=False, max_num_entities=trainer.datamodule.max_entities)
        # log_scalar(trainer, "val/ARI_Foreground", ari_fg.mean(), freq=1, offset=batch_idx)
        # log_scalar(trainer, "val/ARI_Full", ari_full.mean(), freq=1, offset=batch_idx)

        log_scalar(trainer, "val/loss", outputs["loss"], freq=1, offset=batch_idx)

        if batch_idx > 2:
            return
        
        log_video(trainer, "val/dataset", batch, freq=1, offset=batch_idx)

        weighted_pixels = weighted_pixels.permute((0, 1, 4, 2, 3))
        log_video(trainer, "val/weighted_pixels", weighted_pixels[:16], freq=1, offset=batch_idx)
        
        weights_softmax_video = weights_softmax[0].unsqueeze(-1).permute((1, 0, 4, 2, 3))
        weights_softmax_video = weights_softmax_video.repeat(1, 1, 3, 1, 1)
        log_video(trainer, "val/weights_softmax", weights_softmax_video, freq=1, offset=batch_idx)

        pixels = pixels[0].permute((1, 0, 4, 2, 3))
        log_video(trainer, "val/pixels", pixels, freq=1, offset=batch_idx)

        colors = _generate_color_palette(weights_softmax.shape[2])
        segmentation = _generate_segmentation(weights_softmax, colors, bg_idx=weights_softmax.mean(dim=(0,1,3,4)).argmax()).permute(0, 1, 4, 2, 3)
        log_video(trainer, "val/segmentation", segmentation, freq=1, offset=batch_idx)

        # colors_t = _generate_color_palette(trainer.datamodule.max_entities)
        # segmentation = _generate_segmentation(mask[..., 0], colors_t).permute(0, 1, 4, 2, 3)
        # log_video(trainer, "val/segmentation_true", segmentation, freq=1, offset=batch_idx)

        # log_image(trainer, "latents/mec", latents[0][0, ...].to(torch.float32), freq=1, offset=batch_idx)
        # log_image(trainer, "latents/lec", latents[1][0, ...].to(torch.float32), freq=1, offset=batch_idx)
        # log_image(trainer, "latents/hc_t0", latents[2][0, 0, ...].to(torch.float32), freq=1, offset=batch_idx)
        # log_image(trainer, "latents/hc_t-1", latents[2][0, -1, ...].to(torch.float32), freq=1, offset=batch_idx)
        # log_image(trainer, "latents/hc_transformed_t0", latents[3][0, 0, ...].to(torch.float32), freq=1, offset=batch_idx)
        # log_image(trainer, "latents/hc_transformed_t-1", latents[3][0, -1, ...].to(torch.float32), freq=1, offset=batch_idx)
        # log_image(trainer, "latents/ph", latents[4][0, ...].to(torch.float32), freq=1, offset=batch_idx)
        # log_image(trainer, "latents/pr", latents[5][0, ...].to(torch.float32), freq=1, offset=batch_idx)


def _generate_segmentation(weights: Tensor, colors: Tensor, bg_idx=0):
    if bg_idx != 0:
        colors[[0, bg_idx], :] = colors[[bg_idx, 0], :]

    colors = colors.to(weights.device)
    b, t, k, h, w = weights.shape
    assert len(colors) == k
    ce = colors.view(1, 1, k, 1, 1, 3).expand(b, t, k, h, w, 3)
    wa = weights.argmax(dim=2)
    we = wa.view(b, t, 1, h, w, 1).expand(b, t, 1, h, w, 3)
    return torch.gather(ce, 2, we).view(b, t, h, w, 3)


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
    if step > LOG_BACKOFF_POINT and freq != 1:
        freq *= LOG_BACKOFF_FACTOR
    return freq


def check_log_interval(step, freq):
    freq = effective_freq(step, freq)
    return step % freq == 0


def download_file(run_id, project_name, filename=None):
    api = wandb.Api()
    run = api.run(f"sourceress/{project_name}/{run_id}")
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

    frames = make_video_grid(batch, num_images_per_row=4, pad_value=1)

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

    frames = make_video_grid(batch, num_images_per_row=16, pad_value=1)

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
    wandb.watch(model, "all", log_freq=freq)