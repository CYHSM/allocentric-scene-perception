import random
from functools import partial

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from torch import Tensor
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import Conv2d
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer

from loss import p_loss
from helper import PositionalEncoding3D

class Model(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        transformer_layers,
        use_transformer,
        num_timesteps,
        p_loss,
        p_weight,
        p_reduction,
        p_anneal,
        activation,
        l1f_weight,
        l1o_weight,
        map_size,
        anneal_lr,
        weight_decay,
        full_decode,
        K_down,
        xy_resolution,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        # This saves the input args into self.hparams
        self.save_hyperparameters()

        self.visualcortex = VisualCortex(128)
        self.split_forward = Split(128, 320)
        self.hc = Hippocampus(320)
        self.split_backward = Split(320, self.hparams.map_size)
        self.decoder = Decoder()

    def forward(self, input_videos: Tensor, full_res_decode: bool = False):
        """The model has 5 stages:
        --- Encoding: ---
        - a convolutional layers that extracts features from the input video
        - a perirhinal / parahippocampal split into space / time features + a MEC / LEC processing stage
        - a HC layer which integrates the MEC and LEC features
        --- Decoding: ---
        - a MEC / LEC split into space / time features + a perirhinal / parahippocampal processing stage
        - a reconstruction layer that reconstructs the input video
        """
        b, t, c, h, w = input_videos.shape
        # Encode input using visual cortex areas V1, V2, V4 and IT
        v1, v2, v4, it = self.visualcortex(input_videos)
        # Split V4 & IT into MEC and LEC
        (_, _), (mec, lec) = self.split_forward(v4, it)  # (ph, pr), (mec, lec)
        # Combine MEC and LEC into HC
        hc_transformed, hc = self.hc(mec, lec)
        # For feedback, split HC into MEC and LEC
        (_, _), (ph, pr) = self.split_backward(hc_transformed)  # (mec, lec), (ph, pr)
        # Decode pixelwise - most biologically unrealistic part of the model. Could be more realistic if it decodes patch/saccade wise
        pixels, weights, weights_softmax, weighted_pixels, time_indexes = self.decoder(
            ph, pr, full_res_decode
        )

        return {
            "pixels": pixels,
            "weights_softmax": weights_softmax,
            "weighted_pixels": weighted_pixels,
            "latents": (mec, lec, hc, hc_transformed, ph, pr),
            "time_indexes": time_indexes,
        }

    def loss(
        self,
        target: Tensor,
        pixels: Tensor,
        weights_softmax: Tensor,
        weighted_pixels: Tensor,
        time_indexes: Tensor,
        full_res_decode: bool,
        latents,
        **_,
    ):
        target = rearrange(target, "b t c h w -> b t h w c")
        if not full_res_decode:
            # downsample targets to match model output
            target = target[:, :, ::2, ::2, :]
            target = target.index_select(dim=1, index=time_indexes)

        # Compute the individual loss terms
        if self.hparams.p_anneal > 0:
            p = min(
                self.hparams.p_loss,
                (float(self.trainer.global_step + 1) / self.hparams.p_anneal) + 0.5,
            )
        elif self.hparams.p_anneal < 0:
            p = max(
                0.5,
                self.hparams.p_loss
                - (float(self.trainer.global_step + 1) / abs(self.hparams.p_anneal)),
            )
        else:
            p = self.hparams.p_loss

        # Combine p_loss and p_loss_prospective and p_loss_retrospective
        p_term = p_loss(
            target, weighted_pixels, p=p, reduction=self.hparams.p_reduction
        )

        # L2 loss on last two latents, PH & PR
        l1o_term = latents[-1].abs().mean()
        l1f_term = latents[-2].abs().mean()

        # Apply loss weights to get weighted loss terms
        losses = {
            "l2o_loss": l1o_term * self.hparams.l1o_weight,
            "l2f_loss": l1f_term * self.hparams.l1f_weight,
            "p_loss": self.hparams.p_weight * p_term.mean(),
        }
        # Sum the weighted loss terms to get the total loss
        losses["loss"] = sum(losses.values())
        return losses

    def step(self, batch, full_res_decode: bool = False):
        videos, mask, info = batch
        outputs = self(videos, full_res_decode)
        losses = self.loss(videos, full_res_decode=full_res_decode, **outputs)
        return {**losses, **outputs}

    def training_step(self, batch: Tensor, batch_idx: int):
        outputs = self.step(batch, full_res_decode=self.hparams.full_decode)
        return outputs

    def validation_step(self, batch: Tensor, batch_idx: int):
        outputs = self.step(batch, full_res_decode=True)
        self.log("val/loss", outputs["loss"], prog_bar=True, sync_dist=True)
        return outputs

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_foo(epoch):
            if self.hparams.anneal_lr:
                lr_scale = 0.95**epoch
            else:
                lr_scale = 1.0

            return lr_scale

        scheduler = LambdaLR(optimizer, lr_lambda=lr_foo)

        return [optimizer], [scheduler]


class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.mlp = MLP(
            in_features=self.hparams.map_size * 2 + 3,
            out_features=4,
            hidden_features=[512 for _ in range(5)],
            activation=self.hparams.activation,
        )

    def forward(
        self,
        object_latents: Tensor,
        temporal_latents: Tensor,
        full_size_decode: bool = False,
    ):
        T = temporal_latents.shape[1]  # self.hparams.num_timesteps
        Kd = self.hparams.K_down
        batch_size = object_latents.shape[0]
        if full_size_decode:
            OUTPUT_RES = self.hparams.xy_resolution
            Td = T
        else:
            OUTPUT_RES = self.hparams.xy_resolution // 2
            Td = T // 2
        assert object_latents.shape == (batch_size, 16, self.hparams.map_size)
        assert temporal_latents.shape == (batch_size, T, self.hparams.map_size)
        time_indexes = torch.tensor(
            sorted(random.sample(range(T), Td)), device=self.device
        )

        temporal_latents = temporal_latents.index_select(dim=1, index=time_indexes)
        object_indexes = torch.tensor(
            range(Kd), device=self.device
        ) 
        object_latents = object_latents.index_select(dim=1, index=object_indexes)

        object_latents = repeat(
            object_latents, "b k c -> b t k h w c", t=Td, h=OUTPUT_RES, w=OUTPUT_RES
        )
        temporal_latents = repeat(
            temporal_latents, "b t c -> b t k h w c", k=Kd, h=OUTPUT_RES, w=OUTPUT_RES
        )

        desired_shape = [batch_size, Td, Kd, OUTPUT_RES, OUTPUT_RES, 1]
        x_encoding, y_encoding, t_encoding = _build_xyt_indicators(
            desired_shape,
            time_indexes,
            self.device,
            object_latents.dtype,
            full_size_decode,
            T, self.hparams.xy_resolution,
        )

        x = torch.cat(
            [object_latents, temporal_latents, t_encoding, x_encoding, y_encoding],
            dim=5,
        )  # What, When, When, Where, Where

        x = rearrange(
            x, "b td k h w c -> (b td k h w) c", c=2 * self.hparams.map_size + 3
        )
        x = self.mlp(x)
        x = rearrange(
            x,
            "(b td k h w) c -> b td k h w c",
            td=Td,
            k=Kd,
            h=OUTPUT_RES,
            w=OUTPUT_RES,
            c=4,
        )
        pixels = x[..., 0:3]
        weights = x[..., 3]

        weights_sum = (-1 * weights[:, :, 1:, ...]).sum(dim=2)
        weights = torch.cat((weights_sum.unsqueeze(2), weights[:, :, 1:, ...]), dim=2)

        weights = torch.nn.functional.layer_norm(
            weights, [Td, Kd, OUTPUT_RES, OUTPUT_RES]
        )
        weights_softmax = torch.nn.functional.softmax(weights, dim=2)
        weighted_pixels = (pixels * weights_softmax.unsqueeze(-1)).sum(dim=2)

        assert weighted_pixels.shape == (batch_size, Td, OUTPUT_RES, OUTPUT_RES, 3)

        return pixels, weights, weights_softmax, weighted_pixels, time_indexes


def _create_position_encoding(range: Tensor, target_shape, dim):
    """Create a tensor of shape `target_shape` that is filled with values from `range` along `dim`."""
    assert len(range.shape) == 1
    assert len(range) == target_shape[dim]

    view_shape = [1 for _ in target_shape]
    view_shape[dim] = target_shape[dim]
    range = range.view(view_shape)
    encoding = range.expand(target_shape)
    assert encoding.shape == tuple(target_shape)
    return encoding


def _build_xyt_indicators(
    desired_shape,
    time_indexes,
    device,
    dtype,
    full_size_decode: bool,
    T, xy_resolution,
):
    # Form the T, X, Y indicators
    t_linspace = torch.linspace(0, 1, T, device=device, dtype=dtype)
    t_linspace = t_linspace.index_select(dim=0, index=time_indexes)
    t_encoding = _create_position_encoding(t_linspace, desired_shape, dim=1)

    xy_linspace = torch.linspace(-1, 1, xy_resolution, device=device, dtype=dtype)
    if not full_size_decode:
        # we decode every other pixel
        xy_linspace = xy_linspace[::2]
    x_encoding = _create_position_encoding(xy_linspace, desired_shape, dim=3)
    y_encoding = _create_position_encoding(xy_linspace, desired_shape, dim=4)
    return x_encoding, y_encoding, t_encoding


class VisualCortex(pl.LightningModule):
    """The encoder is a simple stack of convolutional layers."""

    def __init__(self, out_channels):
        super().__init__()
        self.save_hyperparameters()
        # Activation function
        if self.hparams.activation == "relu":
            self.activation = nn.ReLU()
        elif self.hparams.activation == "linear":
            self.activation = nn.Identity()
        elif self.hparams.activation == "leaky":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Activation function not supported")

        conv_layer = partial(Conv2d, kernel_size=4, stride=2, padding=(1, 1))
        self.v1 = conv_layer(in_channels=3, out_channels=out_channels)
        self.v2 = conv_layer(in_channels=out_channels, out_channels=out_channels)
        self.v4 = conv_layer(in_channels=out_channels, out_channels=out_channels)
        self.it = conv_layer(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x: Tensor):
        b, t, c, h, w = x.shape
        x = rearrange(
            x, "b t c h w -> (b t) c h w", t=t, c=3, h=self.hparams.xy_resolution, w=self.hparams.xy_resolution
        )

        v1 = self.activation(self.v1(x))
        v2 = self.activation(self.v2(v1))
        v4 = self.activation(self.v4(v2))
        it = self.activation(self.it(v2))

        v4 = rearrange(v4, "(b t) c h w -> b t (h w) c", b=b, t=t)
        it = rearrange(it, "(b t) c h w -> b t (h w) c", b=b, t=t)

        return v1, v2, v4, it


class Split(pl.LightningModule):
    """Splits into ventral and dorsal streams"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.save_hyperparameters()
        num_rsc = in_channels // 4
        self.rsc = nn.Parameter(torch.randn(num_rsc))

        self.linear_space = MLP(
            in_features=in_channels + num_rsc,
            out_features=out_channels,
            hidden_features=[1024],
            activation=self.hparams.activation,
        )
        self.linear_time = MLP(
            in_features=in_channels,
            out_features=out_channels,
            hidden_features=[1024],
            activation=self.hparams.activation,
        )

    def forward(self, x: Tensor, x_two=None):
        b, t, k, c = x.shape

        # Aggregate the temporal info to get the spatial features
        time_avg = torch.mean(x, dim=1)
        time_avg = rearrange(time_avg, "b k c -> (b k) c", b=b, k=k)
        # Concat with rsc parameters
        time_avg = torch.cat([time_avg, self.rsc.repeat(b * k, 1)], dim=1)

        # Aggregate the spatial info to get the temporal features
        if x_two is None:
            space_avg = torch.mean(x, dim=2)
        else:
            space_avg = torch.mean(x_two, dim=2)
        space_avg = rearrange(space_avg, "b t c -> (b t) c", b=b, t=t)

        # Apply the MLPs
        across_space = self.linear_space(time_avg)
        across_space = rearrange(across_space, "(b k) c -> b k c", b=b, k=k)
        across_time = self.linear_time(space_avg)
        across_time = rearrange(across_time, "(b t) c -> b t c", b=b, t=t)

        return (time_avg, space_avg), (across_space, across_time)


class Hippocampus(pl.LightningModule):
    """Integrates MEC and LEC information"""

    def __init__(self, in_channels):
        super().__init__()
        self.save_hyperparameters()
        # Activation function
        if self.hparams.activation == "relu":
            self.activation = nn.ReLU()
        elif self.hparams.activation == "linear":
            self.activation = nn.Identity()
        elif self.hparams.activation == "leaky":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Activation function not supported")

        self.in_channels = in_channels
        self.out_channels = 320
        self.xy_after_conv = 8
        self.xy_after_transformer = self.xy_after_conv // 2 

        if self.hparams.use_transformer:
            # this template layer will get cloned inside the TransformerEncoder modules below.
            encoder_layer_template = TransformerEncoderLayer(
                d_model=self.out_channels,
                nhead=5,
                dim_feedforward=1024,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation=self.activation,
            )
            self.linear_layer = torch.nn.Linear(
                in_features=self.in_channels, out_features=self.out_channels, bias=False
            )

            self.transformer_1 = TransformerEncoder(
                encoder_layer=encoder_layer_template,
                num_layers=self.hparams.transformer_layers,
            )
            self.transformer_2 = TransformerEncoder(
                encoder_layer=encoder_layer_template,
                num_layers=self.hparams.transformer_layers,
            )

            self.position_encoding_1 = PositionalEncoding3D(self.out_channels)
            self.position_encoding_2 = PositionalEncoding3D(self.out_channels)
        else:
            self.linear_layer = torch.nn.Linear(
                in_features=self.in_channels, out_features=self.out_channels, bias=False
            )

            self.recurrent_1 = torch.nn.GRU(
                input_size=self.in_channels,
                hidden_size=self.out_channels,
                bias=True,
                batch_first=True,
                num_layers=self.hparams.transformer_layers,
                bidirectional=False,
            )
            self.recurrent_2 = torch.nn.GRU(
                input_size=self.in_channels,
                hidden_size=self.out_channels,
                bias=True,
                batch_first=True,
                num_layers=self.hparams.transformer_layers,
                bidirectional=False,
            )

            self.position_encoding_1 = PositionalEncoding3D(self.out_channels)
            self.position_encoding_2 = PositionalEncoding3D(self.out_channels)

    def forward(self, x1: Tensor, x2: Tensor):
        b, k, c2 = x1.shape
        b, t, c1 = x2.shape

        hc = torch.einsum("bkc,btc->btkc", x1, x2)

        # Hippocampus integrates mec and lec information
        if self.hparams.use_transformer:
            # x = repeat(x, "b c -> b t h w c", b=b, t=t, h=xy_after_conv, w=xy_after_conv)
            x = rearrange(
                hc,
                "b t (h w) c -> b t h w c",
                b=b,
                t=t,
                h=self.xy_after_conv,
                w=self.xy_after_conv,
            )

            # apply linear transformation to project ENCODER_CONV_CHANNELS to TRANSFORMER_CHANNELS
            x = self.linear_layer(x)

            # apply 3d position encoding before going through the first transformer
            x = x + self.position_encoding_1(x)
            x = rearrange(x, "b t h w c -> b (t h w) c", b=b, t=t, h=self.xy_after_conv, w=self.xy_after_conv, c=self.out_channels)  # fmt: skip
            x = self.transformer_1(x)

            # Original repo scaling
            x = rearrange(x, "b (t h w) c -> (b t) c h w", b=b, t=t, h=self.xy_after_conv, w=self.xy_after_conv, c=self.out_channels)  # fmt: skip
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2) * 2
            x = rearrange(x, "(b t) c h w -> b t h w c", b=b, t=t, h=self.xy_after_transformer, w=self.xy_after_transformer, c=self.out_channels)  # fmt: skip

            x = x + self.position_encoding_2(x)
            x = rearrange(x, "b t h w c -> b (t h w) c", b=b, t=t, h=self.xy_after_transformer, w=self.xy_after_transformer, c=self.out_channels)  # fmt: skip
            x = self.transformer_2(x)
            x = rearrange(x, "b (t h w) c -> b t (h w) c", b=b, t=t, h=self.xy_after_transformer, w=self.xy_after_transformer, c=self.out_channels)  # fmt: skip
        else:
            x = rearrange(
                hc,
                "b t (h w) c -> b t h w c",
                b=b,
                t=t,
                h=self.xy_after_conv,
                w=self.xy_after_conv,
            )

            # apply linear transformation to project ENCODER_CONV_CHANNELS to TRANSFORMER_CHANNELS
            x = self.linear_layer(x)

            # apply 3d position encoding before going through the first transformer
            x = x + self.position_encoding_1(x)
            x = rearrange(x, "b t h w c -> b (t h w) c", b=b, t=t, h=self.xy_after_conv, w=self.xy_after_conv, c=self.out_channels)  # fmt: skip
            x, _ = self.recurrent_1(x)

            # Original repo scaling
            x = rearrange(x, "b (t h w) c -> (b t) c h w", b=b, t=t, h=self.xy_after_conv, w=xy_after_conv, c=self.out_channels)  # fmt: skip
            x = F.avg_pool2d(x, kernel_size=2) * 2
            x = rearrange(x, "(b t) c h w -> b t h w c", b=b, t=t, h=self.xy_after_transformer, w=self.xy_after_transformer, c=self.out_channels)  # fmt: skip

            x = x + self.position_encoding_2(x)
            x = rearrange(x, "b t h w c -> b (t h w) c", b=b, t=t, h=xy_after_transformer, w=xy_after_transformer, c=self.out_channels)  # fmt: skip
            x, _ = self.recurrent_2(x)
            x = rearrange(x, "b (t h w) c -> b t (h w) c", b=b, t=t, h=xy_after_transformer, w=xy_after_transformer, c=self.out_channels)  # fmt: skip

        return x, hc


class MLP(nn.Module):
    """Create a MLP with `len(hidden_features)` hidden layers, each with `hidden_features[i]` features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features,
        activation: nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        if not isinstance(activation, torch.nn.Module):
            if activation == "relu":
                activation = nn.ReLU()
            elif activation == "linear":
                activation = nn.Identity()
            elif activation == "leaky":
                activation = nn.LeakyReLU()
            else:
                raise ValueError("Activation function not supported")

        layers: List[nn.Module] = []
        last_size = in_features
        for size in hidden_features:
            layers.append(nn.Linear(last_size, size))
            last_size = size
            layers.append(activation)
        # Don't put an activation after the last layer
        layers.append(nn.Linear(last_size, out_features))

        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.sequential(x)
