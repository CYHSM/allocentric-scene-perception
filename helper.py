import argparse
import numpy as np
import torch
import torch.nn as nn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
def all_valid_asp_datasets():
    arena_views = ['surround', 'within', 'both']
    colors = ['mix', 'green', 'white']
    arena_landmark = ['ref', 'noref']
    
    all_valid = []
    for a in arena_views:
        ds_name = f'asp_{a}'
        all_valid.append(ds_name)
        for c in colors:
            ds_name = f'asp_{a}_{c}'
            all_valid.append(ds_name)
            for l in arena_landmark:
                ds_name = f'asp_{a}_{c}_{l}'
                all_valid.append(ds_name)
    return all_valid




# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Implementation of the adjusted Rand index."""

import tensorflow.compat.v1 as tf

MAX_NUM_ENTITIES = 11


def compute_ari(mask, weights_softmax, fg_only=True, max_num_entities = MAX_NUM_ENTITIES):
    # (not original Deepmind code)
    # This is based on the example in: https://github.com/deepmind/multi_object_datasets/blob/master/README.md
    # weights_softmax has shape b, t, num_objects, w, h

    # Ground-truth segmentation masks are always returned in the canonical
    # [batch_size, T, max_num_entities, height, width, channels] format. To use these
    # as an input for `segmentation_metrics.adjusted_rand_index`, we need them in
    # the [batch_size, n_points, n_true_groups] format,
    # where n_true_groups == max_num_entities. We implement this reshape below.
    # Note that 'oh' denotes 'one-hot'.
    batch_size, T, K, h, w = weights_softmax.shape
    desired_shape = [batch_size, T * h * w, max_num_entities]
    true_groups_oh = tf.transpose(mask, [0, 1, 3, 4, 5, 2])
    true_groups_oh = tf.reshape(true_groups_oh, desired_shape)

    desired_shape = [batch_size, T * h * w, K]
    prediction = weights_softmax.permute(0, 1, 3, 4, 2).reshape(desired_shape)
    prediction = tf.convert_to_tensor(prediction.detach().numpy())

    # This function takes a one-hot true label, and a softmax prediction.
    # If the true label for a point is a zero vector, that point is not included in the score.
    # Thus we mask out the first object, assumed to the background, to get just the foreground object score.
    if fg_only:
      ari = adjusted_rand_index(true_groups_oh[..., 1:], prediction)
    else:
      ari = adjusted_rand_index(true_groups_oh, prediction)
    return ari.numpy()


def adjusted_rand_index(true_mask, pred_mask, name="ari_score"):
    r"""Computes the adjusted Rand index (ARI), a clustering similarity score.
    This implementation ignores points with no cluster label in `true_mask` (i.e.
    those points for which `true_mask` is a zero vector). In the context of
    segmentation, that means this function can ignore points in an image
    corresponding to the background (i.e. not to an object).
    Args:
      true_mask: `Tensor` of shape [batch_size, n_points, n_true_groups].
        The true cluster assignment encoded as one-hot.
      pred_mask: `Tensor` of shape [batch_size, n_points, n_pred_groups].
        The predicted cluster assignment encoded as categorical probabilities.
        This function works on the argmax over axis 2.
      name: str. Name of this operation (defaults to "ari_score").
    Returns:
      ARI scores as a tf.float32 `Tensor` of shape [batch_size].
    Raises:
      ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
        We've chosen not to handle the special cases that can occur when you have
        one cluster per datapoint (which would be unusual).
    References:
      Lawrence Hubert, Phipps Arabie. 1985. "Comparing partitions"
        https://link.springer.com/article/10.1007/BF01908075
      Wikipedia
        https://en.wikipedia.org/wiki/Rand_index
      Scikit Learn
        http://scikit-learn.org/stable/modules/generated/\
        sklearn.metrics.adjusted_rand_score.html
    """
    with tf.name_scope(name):
        _, n_points, n_true_groups = true_mask.shape.as_list()
        n_pred_groups = pred_mask.shape.as_list()[-1]
        if n_points <= n_true_groups and n_points <= n_pred_groups:
            # This rules out the n_true_groups == n_pred_groups == n_points
            # corner case, and also n_true_groups == n_pred_groups == 0, since
            # that would imply n_points == 0 too.
            # The sklearn implementation has a corner-case branch which does
            # handle this. We chose not to support these cases to avoid counting
            # distinct clusters just to check if we have one cluster per datapoint.
            raise ValueError(
                "adjusted_rand_index requires n_groups < n_points. We don't handle "
                "the special cases that can occur when you have one cluster "
                "per datapoint."
            )

        true_group_ids = tf.argmax(true_mask, -1)
        pred_group_ids = tf.argmax(pred_mask, -1)
        # We convert true and predicted clusters to one-hot ('oh') representations.
        true_mask_oh = tf.cast(true_mask, tf.float32)  # already one-hot
        pred_mask_oh = tf.one_hot(pred_group_ids, n_pred_groups)  # returns float32

        n_points = tf.cast(tf.reduce_sum(true_mask_oh, axis=[1, 2]), tf.float32)

        nij = tf.einsum("bji,bjk->bki", pred_mask_oh, true_mask_oh)
        a = tf.reduce_sum(nij, axis=1)
        b = tf.reduce_sum(nij, axis=2)

        rindex = tf.reduce_sum(nij * (nij - 1), axis=[1, 2])
        aindex = tf.reduce_sum(a * (a - 1), axis=1)
        bindex = tf.reduce_sum(b * (b - 1), axis=1)
        expected_rindex = aindex * bindex / (n_points * (n_points - 1))
        max_rindex = (aindex + bindex) / 2
        ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

        # The case where n_true_groups == n_pred_groups == 1 needs to be
        # special-cased (to return 1) as the above formula gives a divide-by-zero.
        # This might not work when true_mask has values that do not sum to one:
        both_single_cluster = tf.logical_and(_all_equal(true_group_ids), _all_equal(pred_group_ids))
        return tf.where(both_single_cluster, tf.ones_like(ari), ari)


def _all_equal(values):
    """Whether values are all equal along the final axis."""
    return tf.reduce_all(tf.equal(values, values[..., :1]), axis=-1)


"""from https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py"""


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels, freq_base=1.0, freq_scale=10000.0):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = freq_base / (freq_scale ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)

        The encodings for each of (x, y, z) are stacked in the channel dim.
        So the first ch/3 channels are the x encodings.
        For each of x, y, z, the sin, cos embeddings are also stacked in the channel dim.
        So the first ch/6 are x_sin, and the second ch/6 are x_cos.
        They start at high freq and go to low freq.
        So ch0 is the highest freq x sin encoding.
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)
