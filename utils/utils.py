# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# use the same colormap everywhere
COLORMAP = ['#76B900', '#F3BD00','#814B9D','#5C5C5C','#214B9D', '#e48aa7', '#7c4848', '#78A2EB', '#ff1397', '#bee7dd']
# ["#ea5545", "#f46a9b", "#ef9b20", "#edbf33", "#ede15b", "#bdcf32", "#87bc45", "#27aeef", "#b33dc6"]
# ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
# ["#b30000", "#7c1158", "#4421af", "#1a53ff", "#0d88e6", "#00b7c7", "#5ad45a", "#8be04e", "#ebdc78"]
# ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]
# ["#115f9a", "#1984c5", "#22a7f0", "#48b5c4", "#76c68f", "#a6d75b", "#c9e52f", "#d0ee11", "#d0f400"]
# ["#d7e1ee", "#cbd6e4", "#bfcbdb", "#b3bfd1", "#a4a2a8", "#df8879", "#c86558", "#b04238", "#991f17"]
# ["#2e2b28", "#3b3734", "#474440", "#54504c", "#6b506b", "#ab3da9", "#de25da", "#eb44e8", "#ff80ff"]

##### Utility functions #####

import tensorflow as tf
import pickle
import os
import datetime
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import pandas as pd
import io

import sys
sys.path.append('../')


import core.runtime as _runtime

import sionna as sn
from .e2e_model import E2E_Model
from .parameters import Parameters
from utils.model_weights import compute_lr_multipliers
from sionna.utils import ebnodb2no, expand_to_rank

import h5py

from utils import dbg

# tf.debugging.set_log_device_placement(True)
# tf.debugging.enable_check_numerics()


def init_loss_dict(block_config, loss_fields=None, total_key: str = "Total"):
    """
    Build a fresh `loss_dict` whose top-level keys are the *block names* (not types)
    for every block whose type ends with 'Loss'.

    Behavior:
      - If `loss_fields` provides a schema for a given loss type, initialize that block
        as a dict with those fields (all 0.0).
      - Otherwise, initialize that block as a scalar 0.0.
      - Add a global '<total_key>' with 0.0 (defaults to 'total').

    Parameters
    ----------
    block_config : list[dict]
        Network graph blocks (each with at least 'type' and 'name').
    loss_fields : list[dict] | None
        Optional schema describing per-type fields, e.g.:
        [
          {"type":"EchoLoss", "fields":["Echoness","Alpha","AoA","aZoA","Tau","BGP","L1","Total"]},
          {"type":"ChLoss",   "fields":["MSE","Total"]},
          {"type":"LLRLoss",  "fields":["BCE","Total"]},
        ]
        If omitted, EchoLoss defaults to a detailed dict and others default to scalars.
    total_key : str
        Name of the global total key to add to the dict (default: 'total').

    Returns
    -------
    dict
    """
    # Build a mapping {loss_type: [field1, field2, ...]}
    if block_config is None and loss_fields is None:
        return {
            "llr_loss": 0.0,
            "ch_loss": 0.0,
            total_key: 0.0,
        }

    # Build a mapping {loss_type: [field1, field2, ...]}
    schema = {}
    if loss_fields:
        for item in loss_fields:
            t = item.get("type")
            fields = item.get("fields")
            if t and isinstance(fields, (list, tuple)) and len(fields) > 0:
                schema[t] = list(fields)

    # Reasonable default if no schema provided for EchoLoss
    if "EchoLoss" not in schema:
        schema["EchoLoss"] = [
            "Echoness", "Alpha", "AoA", "aZoA", "Tau", "BGP", "L1", "Total"
        ]

    loss_dict = {}

    for b in block_config:
        t = b.get("type", "")
        name = b.get("name", "")
        if not name or not isinstance(t, str):
            continue

        if t.endswith("Loss"):
            # If we have a field schema for this loss type, make a dict of zeros.
            # Otherwise, a scalar 0.0.
            if t in schema:
                loss_dict[name] = {field: 0.0 for field in schema[t]}
            else:
                loss_dict[name] = 0.0

    # global total (overall objective used for optimization/backprop)
    loss_dict[total_key] = 0.0
    return loss_dict

from collections import Counter

def show_duplicate_weight_names(model):
    names = [w.name for w in model.weights]
    counts = Counter(names)
    dupes = {name: count for name, count in counts.items() if count > 1}

    print(f"total weights: {len(names)}")
    print(f"unique names : {len(counts)}")

    if not dupes:
        print("No duplicate weight names found in model.weights")
        return

    print("\nDuplicate weight names:")
    for name, count in sorted(dupes.items()):
        print(f"{count}x  {name}")

    print("\nAll repeated entries:")
    for name in sorted(dupes):
        print(f"\n{name}")
        for i, w in enumerate(model.weights):
            if w.name == name:
                print(f"  [{i}] shape={tuple(w.shape)} dtype={w.dtype}")

def save_weights(system, model_path, save_format='h5', v=0):
    """Save model weights.

    This function saves the weights of a Keras model ``system`` to the
    path as provided by ``model_path``.

    Parameters
    ----------
        system: Keras model
            A model containing the weights to be stored.

        model_path: str
            Defining the path where the weights are stored.

        save_format: str
            defining the save file format: 'pkl' (neural_rx default without extention), 'tf', 'h5'

    """
    base, ext = os.path.splitext(model_path)


    # --- check all weights for NaNs or Infs
    has_issue = False
    for w in system.weights:
        if not tf.executing_eagerly():
            nan_mask = tf.math.reduce_any(tf.math.is_nan(w))
            inf_mask = tf.math.reduce_any(tf.math.is_inf(w))
            if nan_mask or inf_mask:
                has_issue = True
                print(f"[save_weights] Warning: Detected NaN/Inf in weight '{w.name}'")
        else:
            # eager-safe numeric check
            arr = w.numpy()
            if not tf.math.reduce_all(tf.math.is_finite(arr)):
                has_issue = True
                print(f"[save_weights] Warning: Detected NaN/Inf in weight '{w.name}'")

    if has_issue:
        print("[save_weights] Save aborted: at least one weight contains NaN or Inf.")
        return

    if v>1:
        show_duplicate_weight_names(system)

    try:
        if save_format=='pkl':
            weights = system.get_weights()
            with open(model_path, 'wb') as f:
                pickle.dump(weights, f)
        elif save_format=='tf' or save_format=='h5':
            if not ext:
                model_path += '.h5'
            
            system.save_weights(model_path, save_format=save_format)
        else:
            raise ValueError(f"[save_weights] Error: Format '{save_format}' not supported. Supported formats are 'pkl', 'tf', and 'h5'.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def load_weights(system, model_path, skip_mismatch=True, by_name=True):
    """Load model weights.

    This function loads the weights of a Keras model ``system`` from a file
    provided by ``model_path``.

    Parameters
    ----------
        system: Keras model
            The target model into which the weights are loaded.

        model_path: str
            Defining the path where the weights are stored.

    """
    # print(f"loading {model_path}")
    extension = os.path.splitext(model_path)[1]
    # print(f'[load_weights] extension:{extension}')
    if not extension:
        with open(model_path, 'rb') as f:
            weights = pickle.load(f)
        system.set_weights(weights)
    elif extension=='.keras':
        system.load_weights(model_path)
    elif extension=='.h5':
        system.load_weights(model_path, skip_mismatch=skip_mismatch, by_name=by_name)
    else:
        raise ValueError(f"[load_weights] Error: extension '{extension}' not supported. Supported formats are 'pkl'(without extention), 'keras', and 'h5'.")


class TriangularDistributionSampler:
    # pylint: disable=line-too-long
    r"""
    Class for sampling from a triangular distribution.

    Used to train different number of users and putting the
    focus on more complex cases (=more users).

    Parameters
    -----------
    minimum : float
        Lower limit.

    maximum : float
        Upper limit.

    dtype : tf.DType
        Dtype for the output.
        Default to `tf.float32`.

    Input
    ------
    shape : tf.TensorShape
        Shape for the output.

    Output
    -------
    : shape, dtype
        Tensor of random samples with shape ``shape`` and following a
        triangular distribution with lower bound ``minimum`` and upper bound
        ``maximum``.
    """

    def __init__(self, minimum, maximum, dtype=tf.float32):
        self._dtype = dtype
        if dtype.is_integer:
            self._dtype_f = tf.float32
            self._a = tf.cast(minimum, tf.float32)
            self._b = tf.cast(maximum, tf.float32)
        else:
            self._dtype_f = dtype
            self._a = tf.cast(minimum, dtype)
            self._b = tf.cast(maximum, dtype)

    def __call__(self, shape):
        u = tf.random.uniform(  shape=shape,
                                minval=0.0,
                                maxval=1.0,
                                dtype=self._dtype_f)

        x = self._a + tf.sqrt(u)*(self._b - self._a)

        if self._dtype.is_integer:
            x = tf.cast(tf.floor(x), self._dtype)

        return x


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    From : https://www.tensorflow.org/tensorboard/image_summaries
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

import zlib

def _short_tag(full_name: str | None, idx: int, *, maxlen: int = 32, last_parts: int = 4) -> str:
    base = (full_name or f"var_{idx}").split(":")[0]
    segs = base.split("/")
    tail = "/".join(segs[-last_parts:]) if last_parts > 0 else segs[-1]
    if len(tail) <= maxlen:
        return tail.replace(" ", "_")
    h = format(zlib.crc32(base.encode()), "08x")[:4]
    return (tail.replace(" ", "_"))[-(maxlen - 5):] + h

def grads_to_dict(
    grads,
    variables,
    *,
    maxlen: int = 32,
    last_parts: int = 4,
    include_name: str | None = "_detect",   #  only include vars whose name contains this
    ignore_case: bool = True,
):
    """
    Return (grad_dict: {short_tag -> grad32}, global_norm: scalar).
    If include_name is provided, only variables with names containing that substring are kept.
    """
    # Normalize filter text once (works fine inside or outside @tf.function)
    filt = include_name.lower() if (include_name and ignore_case) else include_name

    out = {}
    for i, (g, v) in enumerate(zip(grads, variables)):
        if g is None:
            continue

        vname = getattr(v, "name", None) or f"var_{i}"
        cmp_name = vname.lower() if (ignore_case and vname is not None) else vname

        # Apply substring filter
        if filt is not None and (cmp_name is None or filt not in cmp_name):
            continue

        # Make grads dense for logging
        if isinstance(g, tf.IndexedSlices):
            g = tf.convert_to_tensor(g)
        g = tf.cast(g, tf.float32)

        tag = _short_tag(vname, i, maxlen=maxlen, last_parts=last_parts)
        out[tag] = g

    vals = list(out.values())
    gnorm = tf.linalg.global_norm(vals) if vals else tf.constant(0.0, tf.float32)
    return out, tf.cast(gnorm, tf.float32)




def training_loop(model, label, filename, training_logdir, training_seed,
                  training_schedule, eval_ebno_db_arr, min_num_tx, max_num_tx,
                  sys_parameters, mcs_arr_training_idx,
                  mcs_training_snr_db_offset=None, mcs_training_probs=None,
                  weight_saving_schedule=None, transfer_loaded=False, xla=False, save_format='pkl',
                  grad_log_include_name='detect',
                  log_grads=False):
    # pylint: disable=line-too-long
    r"""
    Training loop used to train a system ``model``.

    Input
    ------
    model : Keras Model
        The model to train

    label : str
        Label to id the training in Tensorboard plots

    filename : str
        Location to store the weights of the system

    training_logdir : str
        Directory to log training data for Tensorboard.

    training_seed : int
        Seed used initializing trainings.

    training_schedule : dictionary
        Training schedule.
        Dictionary with the training parameters. Each entry is a list.
        The training loops over these parameters, i.e., performs num_iter[i]
        SGD iterations for the ith set of parameters

    min_training_snr_db : [num_tx_values], float
        Lower bound of the SNR range for training in dB.

    max_training_snr_db : [num_tx_values], float
        Upper bound of the SNR range for training in dB.

    eval_ebno_db_arr : list
        EbNo points in dB the model loss is evaluated during training every 1k
        iterations.

    min_num_tx : int
        Minimum number of transmitters.

    max_num_tx : int
        Maximum number of transmitters.

    sys_parameters: dict
        System parameters for training and evaluation.

    mcs_arr_training_idx : list
        Specifies the indices of the mcs_index list the model is trained for.

    mcs_training_snr_db_offset : list
        Specifies the MCS-specific SNR offsets; outer list for different number
        of UEs, inner list for all MCSs (must be of same length as mcs_index
        list)

    mcs_training_probs : list
        Probabilities for sampling MCS indices; outer list for different number
        of UEs, inner list for all MCSs (must sum up to one).
        Defaults to None, which will result in uniform probabilities.

    xla: bool
        If True, training runs in XLA mode.
    """

    print(f"Training with mixed MCS from arr. idx {mcs_arr_training_idx}. Eval EbNo at {eval_ebno_db_arr} dB.")

    # Set the seed for reproducible trainings
    tf.random.set_seed(training_seed)

    # Sampler for the number of transmitters
    num_tx_sampler = TriangularDistributionSampler(min_num_tx,
                                                   max_num_tx+1,
                                                   dtype=tf.int64)


    # Adam optimizer is used
    optimizer = tf.keras.optimizers.Adam()

    # Enable XLA compatibility when xla==True
    sn.config.xla_compat = xla

    if mcs_training_snr_db_offset is not None:
        mcs_training_snr_db_offset = tf.constant(mcs_training_snr_db_offset,
                                                 dtype=tf.float32)

    # Compiled training over multiple iterations
    @tf.function(jit_compile=xla)
    def _compile_step(batch_size, min_snr_db, max_snr_db, double_readout,
                      weighting_double_readout, apply_multiloss, train_tx,
                      lr_muls, step):

        # Remark: applying multiloss changes computation graph fundamentally
        # thus we need to re-trace the graph and set the value inside of the
        # compiled function
        # print("Applying multiloss: ", apply_multiloss)
        if sys_parameters.system=='nrx' or sys_parameters.system=='mdx':
            model._receiver._neural_rx._cgnn.apply_multiloss = apply_multiloss

        # set constellation to trainable
        # print("Constellation is trainable: ", train_tx)
        for tx_ in model._transmitters:
            tx_._mapper.constellation.trainable = train_tx

        # if sys_parameters.system=='mdx':
        # bce_rel = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+2)]
        # loss_chest_all = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
        # loss_tilde_all = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
        loss_data = 0.
        loss_chest = 0.

        if hasattr(sys_parameters, "block_config") and hasattr(sys_parameters, "loss_fields"):
            loss_dict = init_loss_dict(sys_parameters.block_config, sys_parameters.loss_fields)
        else:
            loss_dict = init_loss_dict(None, None)
            
        for _ in tf.range(step, dtype=tf.int64):
            num_tx = num_tx_sampler(())

            # randomly sample elements from mcs_arr_training_idx
            mcs_arr_training_idx_ = tf.constant(mcs_arr_training_idx,
                                                dtype=tf.int32)
            if mcs_training_probs is None:
                # uniform distribution
                mcs_arr_idx = tf.random.uniform(
                                            (batch_size, max_num_tx),
                                            maxval=len(mcs_arr_training_idx),
                                            dtype=tf.int32)
                mcs_arr_idx = tf.gather(mcs_arr_training_idx_,
                                        indices=mcs_arr_idx)
            else:
                # generate non-uniform distribution
                mcs_probs = tf.constant(mcs_training_probs, dtype=tf.float32)
                mcs_probs = tf.gather(mcs_probs,
                                      indices=[num_tx - min_num_tx], axis=0)
                mcs_probs_ = tf.concat([[0.0], tf.squeeze(mcs_probs)], axis=0)
                mcs_cdf = tf.math.cumsum(mcs_probs_ / tf.reduce_sum(mcs_probs_))
                rand_samples = tf.random.uniform((batch_size, max_num_tx),
                                                 maxval=1.0, dtype=tf.float32)
                condition = tf.logical_and(
                    tf.greater_equal(expand_to_rank(rand_samples, 3, axis=-1),
                                     expand_to_rank(mcs_cdf[:-1], 3, axis=0)),
                    tf.less(expand_to_rank(rand_samples, 3, axis=-1),
                            expand_to_rank(mcs_cdf[1:], 3, axis=0)))
                mcs_arr_idx = expand_to_rank(mcs_arr_training_idx, 3, axis=0) * tf.cast(condition, dtype=tf.int32)
                mcs_arr_idx = tf.reduce_sum(mcs_arr_idx, axis=-1)

            # one-hot mask of depth corresponding to number of supported MCSs
            # mcs_arr_idx:[batch_size, max_num_tx] random idx for each element
            # mcs_ue_mask: [batch_size, max_num_tx, depth], depth being num supported MCSs
            mcs_ue_mask = tf.one_hot(mcs_arr_idx,
                                     depth=len(sys_parameters.mcs_index))

            snr_db = tf.random.uniform( shape=[batch_size],
                                        minval=min_snr_db[num_tx - min_num_tx],
                                        maxval=max_snr_db[num_tx - min_num_tx])

            # apply MCS-specific SNR [dB] offsets
            if mcs_training_snr_db_offset is not None:
                # select snr offset for number of Tx
                _mcs_training_snr_db_offsets = tf.gather(
                                                    mcs_training_snr_db_offset,
                                                    indices=[num_tx-1], axis=0)
                # select snr offset for MCS
                _mcs_training_snr_db_offsets = tf.squeeze(tf.gather(
                                                _mcs_training_snr_db_offsets,
                                                indices=mcs_arr_idx, axis=1))
                # select offsets for active DMRS
                active_dmrs = model._active_dmrs_mask(batch_size, num_tx,
                                                      sys_parameters.max_num_tx)
                _mcs_training_snr_db_offsets *= active_dmrs
                # compute sum of all active DMRS SNR offsets
                _mcs_training_snr_db_offsets = tf.reduce_sum(
                                                _mcs_training_snr_db_offsets,
                                                axis=1)
                # Add offset to snr_db
                snr_db += _mcs_training_snr_db_offsets
            else:
                active_dmrs = None

            # if batch_size>1:
            #     print(f"[utils] snr_db[0]:{snr_db[0]}")

            g1 = weighting_double_readout[0]
            g2 = weighting_double_readout[1]
            with tf.GradientTape() as tape:
                if sys_parameters.system=='mdx':
                    loss_dict = model(batch_size, snr_db, num_tx,
                                                mcs_ue_mask=mcs_ue_mask,
                                                active_dmrs=active_dmrs)

                    if double_readout:
                        loss = loss_dict["llr_loss"] + g1*loss_dict["ch_loss"]
                    else:
                        loss = loss_data

                if sys_parameters.system=='nrx':
                    loss_data, loss_chest = model(batch_size, snr_db, num_tx,
                            mcs_ue_mask=mcs_ue_mask,
                            active_dmrs=active_dmrs)
                    
                    if double_readout:
                        loss = loss_data + g1*loss_chest
                    else:
                        loss = loss_data
                if sys_parameters.system=='deep_echo':
                    loss_dict = model(batch_size, snr_db, num_tx,
                            mcs_ue_mask=mcs_ue_mask,
                            active_dmrs=active_dmrs)
                    
                    loss = loss_dict["Total"]

            grads = tape.gradient(loss, model.trainable_weights)



            if lr_muls is None:
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            else:
                # Scale the gradients according to the multipliers
                scaled_grads = [g * m if g is not None else None for g, m in zip(grads, lr_muls)]
                # Apply the scaled gradients
                optimizer.apply_gradients(zip(scaled_grads, model.trainable_weights))

        # # ---- debug: check for NaN/Inf ---------
        # for v, g in zip(model.trainable_weights, grads):
        #     if g is None:
        #         continue
        #     g_dense = tf.convert_to_tensor(g)
        #     if not tf.reduce_all(tf.math.is_finite(g_dense)):
        #         print(f"[grad_check] Non-finite grad for {v.name}", flush=True)

        # if not tf.reduce_all(tf.math.is_finite(loss)):
        #     print(f"[grad_check] Non-finite loss before apply_gradients: {loss}", flush=True)
        # for v, g in zip(model.trainable_weights, grads):
        #     if g is None:
        #         continue
        #     tf.debugging.assert_all_finite(g, f"NaN/Inf grad for {v.name}")
        # tf.debugging.assert_all_finite(loss, "NaN/Inf loss before apply_gradients")
        # # --------- end debug

        grad_dict = grads_to_dict(grads, model.trainable_variables, include_name=grad_log_include_name)
        if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
            return loss_dict, grad_dict
        # if sys_parameters.system=='mdx':
        #     return loss_data, loss_chest, loss, bce_rel, loss_chest_all, grad_dict
        if sys_parameters.system=='nrx':
            return loss_data, loss_chest, loss, grad_dict


    ## Implements an outer/large training step
    def _training_step(global_iter, num_iterations, batch_size,
                       min_snr_db, max_snr_db, double_readout,
                       weighting_double_readout, apply_multiloss,
                       train_tx,lr_muls,cgnn_it_sch):

        # Check if it's a scalar
        is_scalar = tf.equal(tf.size(weighting_double_readout), 1)
        # Assign values based on the condition
        g1, g2 = tf.cond(
            is_scalar,
            lambda: (weighting_double_readout, weighting_double_readout),  # If scalar, copy to g1 and g2
            lambda: (weighting_double_readout[0], weighting_double_readout[1])  # If list, assign directly
        )

        weighting_double_readout = [g1, g2]

        cgnn_step, num_it, git = cgnn_it_sch
        step = 100
        if hasattr(sys_parameters, 'num_iter_step'):
            step = sys_parameters.num_iter_step
        for _ in tf.range(int(num_iterations/step), dtype=tf.int64):

            if global_iter<=1: # write the graph trace once
                logdir = os.path.join(training_logdir, f"_graph")
                dumpdir = os.path.join(training_logdir, f"_dump")

                writer = tf.summary.create_file_writer(logdir)
                tf.summary.trace_on(graph=True, profiler=True)
            try:
                if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
                    loss_dict, grads = _compile_step(batch_size,
                                        min_snr_db, max_snr_db,
                                        double_readout,
                                        weighting_double_readout,
                                        apply_multiloss,
                                        train_tx,
                                        lr_muls,
                                        step)
                # if sys_parameters.system=='mdx':
                #     loss_data, loss_chest, loss, bce_rel, loss_chest_all, grads = \
                #                         _compile_step(batch_size,
                #                             min_snr_db, max_snr_db,
                #                             double_readout,
                #                             weighting_double_readout,
                #                             apply_multiloss,
                #                             train_tx,
                #                             lr_muls,
                #                             step)
                if sys_parameters.system=='nrx':
                    loss_data, loss_chest, loss, grads = _compile_step(batch_size,
                                        min_snr_db, max_snr_db,
                                        double_readout,
                                        weighting_double_readout,
                                        apply_multiloss,
                                        train_tx,
                                        lr_muls,
                                        step)
            except Exception as e:
                print(f"\n\nError occurred:{e}")
                if global_iter<=1: # write the graph trace once
                    with writer.as_default():
                        tf.summary.trace_export(name="faulty_graph", step=0, profiler_outdir=logdir)
                        print(f"Partially compiled graph has been logged at {logdir}")

            if global_iter<=1: # write the graph trace once
                writer.close()


            global_iter += step


            if sys_parameters.system=='nrx':
                tf.summary.scalar(f"Loss", loss_data, step=global_iter)
                tf.summary.scalar(f"Loss Ch. Est.", loss_chest, step=global_iter)
                tf.summary.scalar(f"Total Loss", loss, step=global_iter)
            # elif sys_parameters.system=='mdx':
            #     tf.summary.scalar(f"Loss", loss_data, step=global_iter)
            #     tf.summary.scalar(f"Loss Ch. Est.", loss_chest, step=global_iter)
            #     tf.summary.scalar(f"Total Loss", loss, step=global_iter)                
            #     if isinstance(bce_rel, list):
            #         for i, value in enumerate(bce_rel):
            #             tf.summary.scalar(f"BCE rel. PerfCh {i}", value, step=global_iter)
            #         for i, value in enumerate(loss_chest_all):
            #             tf.summary.scalar(f"CHEst {i}", value, step=global_iter)

            elif sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
                # for name, value in loss_dict.items():
                #     if isinstance(value, dict):
                #         # handle nested dicts (e.g. EchoLoss)
                #         for sub_name, sub_value in value.items():
                #             sub_value = tf.convert_to_tensor(sub_value)
                #             sub_value = tf.reshape(sub_value, [])
                #             tf.summary.scalar(f"{name}/{sub_name}", sub_value, step=global_iter)
                #     else:
                #         # handle scalar values
                #         value = tf.convert_to_tensor(value)
                #         value = tf.reshape(value, [])
                #         tf.summary.scalar(f"{name}", value, step=global_iter)
                loss = loss_dict
                loss_data = 0.
                loss_chest = 0.

            else:
                msg = tf.constant(f"Unimplemented system type: {sys_parameters.system}", dtype=tf.string)
                tf.print(msg)
                raise NotImplementedError(f"System type '{sys_parameters.system}' is not implemented.")

            if weight_saving_schedule is not None and global_iter in weight_saving_schedule:          
                print(f"Saving weights after {global_iter} iterations")                         
                save_weights(model, filename + f"_{global_iter}_iter",save_format=save_format)                      

        return global_iter, loss, loss_data, loss_chest, num_it, git, grads

    # XLA compilation function for evaluation of model performance
    # Set different mcs_arr_idx as integer to trigger XLA re-tracing.
    @tf.function(jit_compile=xla)
    def eval_model_xla(batch_size, eval_snr_db, max_num_tx, mcs_arr_idx):

        # if sys_parameters.system=='mdx':
        #     bce_rel = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
        #     loss_chest_all = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
        #     loss_tilde_all = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
        #     loss_data_mcs, loss_chest, bce_rel, loss_chest_all, loss_tilde, loss_tilde_all = model(batch_size, _snr_db, num_tx=max_num_tx,
        #                             mcs_arr_eval_idx=mcs_arr_idx)
        #     return loss_data_mcs, loss_chest, bce_rel, loss_chest_all
        
        if sys_parameters.system=='nrx':

            loss_data_mcs, loss_chest = model(batch_size, eval_snr_db, num_tx=max_num_tx,
                        mcs_arr_eval_idx=mcs_arr_idx)
            return loss_data_mcs, loss_chest

        if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
            loss_dict = model(batch_size, eval_snr_db, num_tx=max_num_tx,
                        mcs_arr_eval_idx=mcs_arr_idx)
            return loss_dict

    ## Logs loss and learning rate
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(training_logdir, f"{label}-{current_time}")
    summary_writer = tf.summary.create_file_writer(logdir)

    with summary_writer.as_default():
        # save config
        tf.summary.text("config",
                        sys_parameters.config_str,
                        step=0)
        ## Training loop
        global_iter = tf.zeros((), tf.int64)
        for i, num_iterations in enumerate(training_schedule["num_iter"]):

            # read-in training schedule parameters
            num_iterations = int(num_iterations)
            lr = training_schedule["learning_rate"][i]
            batch_size = training_schedule["batch_size"][i]
            train_tx = training_schedule["train_tx"][i]
            double_readout = training_schedule["double_readout"][i]
            apply_multiloss = training_schedule["apply_multiloss"][i]
            weighting_double_readout = tf.constant(
                    training_schedule["weighting_double_readout"][i],
                    tf.float32)

            # Range for the SNR [dB]
            min_snr_db = tf.constant(
                    training_schedule["min_training_snr_db"][i], tf.float32)
            max_snr_db = tf.constant(
                    training_schedule["max_training_snr_db"][i], tf.float32)

            # cgnn steps
            if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
                cgnn_step=-1
            else:
                if "cgnn_steps" in training_schedule:
                    cgnn_step = int(training_schedule["cgnn_steps"][i])
                    # if cgnn_step > 0:
                    #     model._receiver._neural_rx.num_it = 1
                    # if cgnn_step==0:
                    #     model._receiver._neural_rx.num_it = np.random.randint(1, sys_parameters.num_nrx_iter + 1)
                    # if cgnn_step==-1:
                    #     model._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter
                else:
                    # model._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter
                    cgnn_step=-1

            num_it=1
            git=0
            # lr multiplier for transfer learning
            lr_muls=None
            if transfer_loaded and hasattr(sys_parameters, 'lr_mul'):
                lr_mul = sys_parameters.lr_mul[i]
                lr_muls = compute_lr_multipliers(model, sys_parameters.transfer_weights_path, start_token="neural_pusch_receiver/cgnnofdm", lr_m=lr_mul)

            # Set the learning rate
            optimizer.learning_rate.assign(lr)
            # Train
            num_iter_global = num_iterations \
                              // sys_parameters.num_iter_train_save
            for _ in range(num_iter_global):
                
                global_iter, loss, loss_data, loss_chest, num_it, git, grads_weights = \
                            _training_step(global_iter,
                                           sys_parameters.num_iter_train_save,
                                           batch_size,
                                           min_snr_db, max_snr_db,
                                           double_readout,
                                           weighting_double_readout,
                                           apply_multiloss,
                                           train_tx,
                                           lr_muls,
                                           [cgnn_step,num_it,git])
                
                if all(tf.reduce_all(tf.math.is_finite(w)).numpy() for w in model.weights):
                    # Save the trained model
                    save_weights(model, filename,save_format=save_format)
                else:
                    print("Warning: Model contains NaN or Inf weights, not saving.",flush=True)


                # Log progress and model performance
                for mcs_i, mcs_arr_idx in enumerate(mcs_arr_training_idx):
                    eval_ebno_mcsi = eval_ebno_db_arr[mcs_arr_idx]
                    multi_snr = isinstance(eval_ebno_mcsi, (list, tuple))
                    eval_ebno_list = eval_ebno_mcsi if multi_snr else [eval_ebno_mcsi]

                    for eval_ebno_val in eval_ebno_list:
                        snr_suffix = f" snr={float(eval_ebno_val):.1f}dB" if multi_snr else ""
                        # compute ebno_db for current MCS
                        if not sys_parameters.ebno:
                            # convert EbNo to SNR
                            eval_no = ebnodb2no(
                                eval_ebno_val,
                                model._transmitters[mcs_arr_idx]._num_bits_per_symbol,
                                model._transmitters[mcs_arr_idx]._target_coderate,
                                model._transmitters[mcs_arr_idx]._resource_grid)
                            eval_snr_db = - 10.0 * tf.math.log(eval_no) / tf.math.log(10.0)
                        else:
                            # model takes in EbNo (not SNR)
                            eval_snr_db = eval_ebno_val
                        # if sys_parameters.system=='mdx':
                        #     loss_data_mcs, loss_chestv, bce_rel, loss_chest_allv = eval_model_xla(batch_size, _snr_db,
                        #                                 max_num_tx, mcs_arr_idx)
                        #     tf.summary.scalar(
                        #                 f"Eval CE loss / mcs_arr_idx=" + str(mcs_arr_idx),
                        #                 loss_data_mcs, step=global_iter)
                        #     tf.summary.scalar(
                        #                 f"Eval CHEst loss/ mcs_arr_idx=" + str(mcs_arr_idx),
                        #                 loss_chestv, step=global_iter)
                        #     if isinstance(bce_rel, list):
                        #         for i, value in enumerate(bce_rel):
                        #             tf.summary.scalar(f"Eval BCE rel. PerfCh {i} mcs:" + str(mcs_arr_idx), value, step=global_iter)
                        #         for i, value in enumerate(loss_chest_allv):
                        #             tf.summary.scalar(f"Eval CHEst {i} mcs:" + str(mcs_arr_idx), value, step=global_iter)

                        if sys_parameters.system=='nrx':
                            loss_data_mcs, loss_chestv = eval_model_xla(batch_size, eval_snr_db,
                                                        max_num_tx, mcs_arr_idx)
                            tf.summary.scalar(
                                        f"Eval CE loss / mcs_arr_idx={mcs_arr_idx}{snr_suffix}",
                                        loss_data_mcs, step=global_iter)
                            tf.summary.scalar(
                                        f"Eval CHEst loss/ mcs_arr_idx={mcs_arr_idx}{snr_suffix}",
                                        loss_chestv, step=global_iter)



                        if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
                            loss_dict_mcs = eval_model_xla(batch_size, eval_snr_db,
                                                        max_num_tx, mcs_arr_idx)

                            for name, value in loss_dict_mcs.items():
                                if isinstance(value, dict):
                                    # handle nested dicts (e.g. EchoLoss)
                                    # log only first mcs for echoloss eval
                                    # if mcs_i==0:
                                    for sub_name, sub_value in value.items():
                                        sub_value = tf.convert_to_tensor(sub_value)
                                        sub_value = tf.reshape(sub_value, [])
                                        tf.summary.scalar(f"{name}/{sub_name}/Eval:{mcs_arr_idx}{snr_suffix}", sub_value, step=global_iter)
                                else:
                                    # handle scalar values
                                    value = tf.convert_to_tensor(value)
                                    value = tf.reshape(value, [])
                                    tf.summary.scalar(f"{name}/ Eval mcs:{mcs_arr_idx}{snr_suffix}", value, step=global_iter)

                if sys_parameters.system=='nrx':
                    tf.summary.scalar(f"Loss", loss_data, step=global_iter)
                    tf.summary.scalar(f"Loss Ch. Est.", loss_chest,
                                    step=global_iter)
                    tf.summary.scalar(f"Total Loss", loss, step=global_iter)

                if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx': # loss is dict of losses

                    for name, value in loss.items():
                        if isinstance(value, dict):
                            # handle nested dicts (e.g. EchoLoss)
                            for sub_name, sub_value in value.items():
                                sub_value = tf.convert_to_tensor(sub_value)
                                sub_value = tf.reshape(sub_value, [])
                                tf.summary.scalar(f"{name}/{sub_name}", sub_value, step=global_iter)
                        else:
                            # handle scalar values
                            value = tf.convert_to_tensor(value)
                            value = tf.reshape(value, [])
                            tf.summary.scalar(f"{name}", value, step=global_iter)

                    # log grads
                    if log_grads:

                        grad_dict, global_norm = grads_weights

                        tf.summary.scalar("grads/global_norm", global_norm, step=global_iter)

                        for name in sorted(grad_dict.keys()):
                            g = grad_dict[name]
                            tf.summary.scalar(f"grads/norm/{name}", tf.norm(g), step=global_iter)

                        # if int(global_iter) % 100 == 0:
                        #     for name in sorted(grad_dict.keys()):
                        #         tf.summary.histogram(f"grads/hist/{name}", grad_dict[name], step=global_iter)

                # --------- change num_it
                # if not sys_parameters.system=='deep_echo':
                #     if cgnn_step>0:
                #         if model._receiver._neural_rx.num_it+1<=sys_parameters.num_nrx_iter:
                #             model._receiver._neural_rx.num_it = model._receiver._neural_rx.num_it + 1
                #         else:
                #             model._receiver._neural_rx.num_it = 1
                #     if cgnn_step==0:
                #         model._receiver._neural_rx.num_it = np.random.randint(1, sys_parameters.num_nrx_iter + 1)



# ________________________________________________________________________
# ---------------------------Training Loop--------------------------------
# ________________________________________________________________________
# import os, datetime, numpy as np, tensorflow as tf






# # --- helpers: safe scalar logging (skip NaN/Inf), and profiling blocks --------
# def _is_finite_scalar(x):
#     x = tf.convert_to_tensor(x)
#     x = tf.reshape(x, [])  # scalar
#     return tf.reduce_all(tf.math.is_finite(x))

# def _safe_scalar(tag, value, step, writer):
#     """Log scalar only if finite."""
#     if _is_finite_scalar(value):
#         with writer.as_default():
#             tf.summary.scalar(tag, tf.reshape(value, []), step=step)

# class _MaybeProfiler:
#     """Start/stop TF profiler for a short window without touching scalar writer."""
#     def __init__(self, profile_dir, start=True):
#         self.profile_dir = profile_dir
#         self._started = False
#         self._enabled = start

#     def start(self):
#         if self._enabled and not self._started:
#             try:
#                 tf.profiler.experimental.start(self.profile_dir)
#                 self._started = True
#             except Exception as e:
#                 print(f"[profiler] start failed: {e}")

#     def stop(self):
#         if self._enabled and self._started:
#             try:
#                 tf.profiler.experimental.stop()
#             except Exception as e:
#                 print(f"[profiler] stop failed: {e}")
#             self._started = False

# ------------------------------------------------------------------------------

# def training_loop_(model, label, filename, training_logdir, training_seed,
#                   training_schedule, eval_ebno_db_arr, min_num_tx, max_num_tx,
#                   sys_parameters, mcs_arr_training_idx,
#                   mcs_training_snr_db_offset=None, mcs_training_probs=None,
#                   weight_saving_schedule=None, transfer_loaded=False, xla=False, save_format='pkl'):
#     # pylint: disable=line-too-long
#     r"""
#     Training loop used to train a system ``model``.

#     All train/* and eval/* scalars are written into ONE run directory:
#         training_logdir
#     Graph and profiler artifacts are written into:
#         training_logdir + "_graph"
#         training_logdir + "_profile"
#     """

#     print(f"Training with mixed MCS from arr. idx {mcs_arr_training_idx}. Eval EbNo at {eval_ebno_db_arr} dB.")
#     tf.random.set_seed(training_seed)

#     # Sampler for the number of transmitters (assume provided utility)
#     num_tx_sampler = TriangularDistributionSampler(min_num_tx, max_num_tx+1, dtype=tf.int64)

#     optimizer = tf.keras.optimizers.Adam()
#     sn.config.xla_compat = xla  # keep your flag

#     if mcs_training_snr_db_offset is not None:
#         mcs_training_snr_db_offset = tf.constant(mcs_training_snr_db_offset, dtype=tf.float32)

#     # -------------------------- compiled inner step ---------------------------
#     @tf.function(jit_compile=xla)
#     def _compile_step(batch_size, min_snr_db, max_snr_db, double_readout,
#                       weighting_double_readout, apply_multiloss, train_tx,
#                       lr_muls, step):
#         # Configure multi-loss inside graph (kept as your logic)
#         if sys_parameters.system=='nrx':
#             model._receiver._neural_rx._cgnn.apply_multiloss = apply_multiloss

#         # Set constellation trainable flag
#         for tx_ in model._transmitters:
#             tx_._mapper.constellation.trainable = train_tx

#         bce_rel = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+2)]
#         loss_chest_all = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
#         loss_tilde_all = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
#         loss_data = 0.
#         loss_chest = 0.
#         loss_dict = {
#             "EchoLoss": 0.,
#             "ChLoss": 0.,
#             "LLRLoss": 0.,
#             "Total": 0.,
#         }
#         for _ in tf.range(step, dtype=tf.int64):
#             num_tx = num_tx_sampler(())

#             # sample MCS per-UE
#             mcs_arr_training_idx_ = tf.constant(mcs_arr_training_idx, dtype=tf.int32)
#             if mcs_training_probs is None:
#                 mcs_arr_idx = tf.random.uniform((batch_size, max_num_tx),
#                                                 maxval=len(mcs_arr_training_idx),
#                                                 dtype=tf.int32)
#                 mcs_arr_idx = tf.gather(mcs_arr_training_idx_, indices=mcs_arr_idx)
#             else:
#                 mcs_probs = tf.constant(mcs_training_probs, dtype=tf.float32)
#                 mcs_probs = tf.gather(mcs_probs, indices=[num_tx - min_num_tx], axis=0)
#                 mcs_probs_ = tf.concat([[0.0], tf.squeeze(mcs_probs)], axis=0)
#                 mcs_cdf = tf.math.cumsum(mcs_probs_ / tf.reduce_sum(mcs_probs_))
#                 rand_samples = tf.random.uniform((batch_size, max_num_tx), maxval=1.0, dtype=tf.float32)
#                 condition = tf.logical_and(
#                     tf.greater_equal(expand_to_rank(rand_samples, 3, axis=-1),
#                                      expand_to_rank(mcs_cdf[:-1], 3, axis=0)),
#                     tf.less(expand_to_rank(rand_samples, 3, axis=-1),
#                             expand_to_rank(mcs_cdf[1:], 3, axis=0)))
#                 mcs_arr_idx = expand_to_rank(mcs_arr_training_idx, 3, axis=0) * tf.cast(condition, dtype=tf.int32)
#                 mcs_arr_idx = tf.reduce_sum(mcs_arr_idx, axis=-1)

#             # one-hot MCS
#             mcs_ue_mask = tf.one_hot(mcs_arr_idx, depth=len(sys_parameters.mcs_index))

#             # SNR sampling
#             snr_db = tf.random.uniform(shape=[batch_size],
#                                        minval=min_snr_db[num_tx - min_num_tx],
#                                        maxval=max_snr_db[num_tx - min_num_tx])

#             # MCS-specific SNR offsets
#             if mcs_training_snr_db_offset is not None:
#                 _mcs_training_snr_db_offsets = tf.gather(mcs_training_snr_db_offset, indices=[num_tx-1], axis=0)
#                 _mcs_training_snr_db_offsets = tf.squeeze(tf.gather(_mcs_training_snr_db_offsets, indices=mcs_arr_idx, axis=1))
#                 active_dmrs = model._active_dmrs_mask(batch_size, num_tx, sys_parameters.max_num_tx)
#                 _mcs_training_snr_db_offsets *= active_dmrs
#                 _mcs_training_snr_db_offsets = tf.reduce_sum(_mcs_training_snr_db_offsets, axis=1)
#                 snr_db += _mcs_training_snr_db_offsets
#             else:
#                 active_dmrs = None

#             g1, g2 = weighting_double_readout[0], weighting_double_readout[1]
#             with tf.GradientTape() as tape:
#                 # if sys_parameters.system=='mdx':
#                 #     loss_data, loss_chest, bce_rel, loss_chest_all, loss_tilde, loss_tilde_all = model(
#                 #         batch_size, snr_db, num_tx, mcs_ue_mask=mcs_ue_mask, active_dmrs=active_dmrs)
#                 #     loss = loss_data + g1*loss_chest if double_readout else loss_data

#                 if sys_parameters.system=='nrx':
#                     loss_data, loss_chest = model(batch_size, snr_db, num_tx,
#                                                   mcs_ue_mask=mcs_ue_mask, active_dmrs=active_dmrs)
#                     loss = loss_data + g1*loss_chest if double_readout else loss_data

#                 elif sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
#                     loss_dict = model(batch_size, snr_db, num_tx,
#                                       mcs_ue_mask=mcs_ue_mask, active_dmrs=active_dmrs)
#                     loss = loss_dict["Total"]

#                 else:
#                     raise NotImplementedError(f"System type '{sys_parameters.system}' is not implemented.")

#             grads = tape.gradient(loss, model.trainable_weights)
#             if lr_muls is None:
#                 optimizer.apply_gradients(zip(grads, model.trainable_weights))
#             else:
#                 scaled_grads = [g * m if g is not None else None for g, m in zip(grads, lr_muls)]
#                 optimizer.apply_gradients(zip(scaled_grads, model.trainable_weights))

#         if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
#             return loss_dict
#         # if sys_parameters.system=='mdx':
#         #     return loss_data, loss_chest, loss, bce_rel, loss_chest_all
#         if sys_parameters.system=='nrx':
#             return loss_data, loss_chest, loss

#     # ----------------------- compiled evaluation step -------------------------
#     @tf.function(jit_compile=xla)
#     def eval_model_xla(batch_size, _snr_db, max_num_tx, mcs_arr_idx):
#         # if sys_parameters.system=='mdx':
#         #     bce_rel = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
#         #     loss_chest_all = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
#         #     loss_tilde_all = [tf.constant(0.0, dtype=tf.float32) for _ in range(sys_parameters.num_nrx_iter+1)]
#         #     loss_data_mcs, loss_chest, bce_rel, loss_chest_all, loss_tilde, loss_tilde_all = model(
#         #         batch_size, _snr_db, num_tx=max_num_tx, mcs_arr_eval_idx=mcs_arr_idx)
#         #     return loss_data_mcs, loss_chest, bce_rel, loss_chest_all

#         if sys_parameters.system=='nrx':
#             loss_data_mcs, loss_chest = model(batch_size, _snr_db, num_tx=max_num_tx, mcs_arr_eval_idx=mcs_arr_idx)
#             return loss_data_mcs, loss_chest

#         if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
#             loss_dict = model(batch_size, _snr_db, num_tx=max_num_tx, mcs_arr_eval_idx=mcs_arr_idx)
#             return loss_dict

#     # ------------------- writers: single scalar writer; separate graph/prof ----
#     # DO NOT change training_logdir: all scalar summaries (train/*, eval/*) go here
#     scalar_writer = tf.summary.create_file_writer(training_logdir)

#     # Graph/trace and profiler go to separate dirs (different files)
#     graph_dir   = training_logdir + "_graph"
#     profile_dir = training_logdir + "_profile"
#     os.makedirs(graph_dir, exist_ok=True)
#     os.makedirs(profile_dir, exist_ok=True)

#     # Optional: export graph once (separate dir)
#     try:
#         with tf.summary.create_file_writer(graph_dir).as_default():
#             tf.summary.text("config", sys_parameters.config_str, step=0)
#             tf.summary.trace_on(graph=True, profiler=False)
#             # Make a tiny dry-run trace for the captured graph (no heavy work)
#             tf.summary.trace_export(name="keras_graph", step=0)  # graph files live in graph_dir
#     except Exception as e:
#         print(f"[graph-trace] export failed: {e}")

#     # Profiler controller (short window at start if requested)
#     profile_k_steps = int(getattr(sys_parameters, "profile_k_steps", 0))  # e.g., set to 200 to profile first 200 steps
#     profiler = _MaybeProfiler(profile_dir, start=(profile_k_steps > 0))

#     # ------------------------------- outer loop -------------------------------
#     global_iter = 0  # python int for clean summary steps

#     # Write config as text into the scalar run once (same single event file/dir)
#     with scalar_writer.as_default():
#         tf.summary.text("config", sys_parameters.config_str, step=0)

#     for i, num_iterations in enumerate(training_schedule["num_iter"]):
#         num_iterations = int(num_iterations)
#         lr = training_schedule["learning_rate"][i]
#         batch_size = training_schedule["batch_size"][i]
#         train_tx = training_schedule["train_tx"][i]
#         double_readout = training_schedule["double_readout"][i]
#         apply_multiloss = training_schedule["apply_multiloss"][i]
#         weighting_double_readout = tf.constant(training_schedule["weighting_double_readout"][i], tf.float32)

#         min_snr_db = tf.constant(training_schedule["min_training_snr_db"][i], tf.float32)
#         max_snr_db = tf.constant(training_schedule["max_training_snr_db"][i], tf.float32)

#         # CGNN steps (kept same behavior)
#         if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
#             cgnn_step = -1
#         else:
#             if "cgnn_steps" in training_schedule:
#                 cgnn_step = int(training_schedule["cgnn_steps"][i])
#                 # if cgnn_step > 0:
#                 #     model._receiver._neural_rx.num_it = 1
#                 # if cgnn_step == 0:
#                 #     model._receiver._neural_rx.num_it = np.random.randint(1, sys_parameters.num_nrx_iter + 1)
#                 # if cgnn_step == -1:
#                 #     model._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter
#             else:
#                 # model._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter
#                 cgnn_step = -1

#         num_it, git = 1, 0

#         # Optional transfer learning lr multipliers
#         lr_muls = None
#         if transfer_loaded and hasattr(sys_parameters, 'lr_mul'):
#             lr_mul = sys_parameters.lr_mul[i]
#             lr_muls = compute_lr_multipliers(
#                 model, sys_parameters.transfer_weights_path,
#                 start_token="neural_pusch_receiver/cgnnofdm",
#                 lr_m=lr_mul
#             )

#         optimizer.learning_rate.assign(lr)

#         # chunked updates
#         step_chunk = getattr(sys_parameters, 'num_iter_step', 100)
#         num_iter_global = num_iterations // getattr(sys_parameters, 'num_iter_train_save', 1000)
#         for _ in range(num_iter_global):
#             # Optional: start profiler at the very beginning window
#             if profiler._enabled and not profiler._started and global_iter == 0:
#                 profiler.start()

#             # --- perform one "save interval" block, inside which we call compiled chunks
#             loops_this_block = getattr(sys_parameters, 'num_iter_train_save', 1000) // step_chunk
#             for _inner in range(loops_this_block):
#                 # mark this chunk for profiler timeline
#                 try:
#                     tf.profiler.experimental.Trace('train_chunk',
#                                                    step_num=int(global_iter),
#                                                    _r=1)
#                 except Exception:
#                     pass  # Trace is best-effort

#                 # Run compiled step
#                 if sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
#                     loss_out = _compile_step(batch_size, min_snr_db, max_snr_db,
#                                              double_readout, [weighting_double_readout if tf.size(weighting_double_readout)==1 else weighting_double_readout[0],
#                                                               weighting_double_readout if tf.size(weighting_double_readout)==1 else weighting_double_readout[1]],
#                                              apply_multiloss, train_tx, lr_muls, step_chunk)
#                     loss = loss_out
#                     loss_data = 0.0
#                     loss_chest = 0.0

#                 # elif sys_parameters.system=='mdx':
#                 #     loss_data, loss_chest, loss, bce_rel, loss_chest_all = _compile_step(
#                 #         batch_size, min_snr_db, max_snr_db,
#                 #         double_readout,
#                 #         [weighting_double_readout if tf.size(weighting_double_readout)==1 else weighting_double_readout[0],
#                 #          weighting_double_readout if tf.size(weighting_double_readout)==1 else weighting_double_readout[1]],
#                 #         apply_multiloss, train_tx, lr_muls, step_chunk)

#                 elif sys_parameters.system=='nrx':
#                     loss_data, loss_chest, loss = _compile_step(
#                         batch_size, min_snr_db, max_snr_db,
#                         double_readout,
#                         [weighting_double_readout if tf.size(weighting_double_readout)==1 else weighting_double_readout[0],
#                          weighting_double_readout if tf.size(weighting_double_readout)==1 else weighting_double_readout[1]],
#                         apply_multiloss, train_tx, lr_muls, step_chunk)

#                 global_iter += step_chunk

#                 # --- SCALAR LOGS: single writer (same file/dir), prefixed tags
#                 if sys_parameters.system in ( 'nrx'):
#                     _safe_scalar("train/Loss",       loss_data,  global_iter, scalar_writer)
#                     _safe_scalar("train/Loss_ChEst", loss_chest, global_iter, scalar_writer)
#                     _safe_scalar("train/Total_Loss", loss,       global_iter, scalar_writer)
#                     # if sys_parameters.system=='mdx' and isinstance(bce_rel, list):
#                     #     for i_b, v_bce in enumerate(bce_rel):
#                     #         _safe_scalar(f"train/BCE_rel_PerfCh_{i_b}", v_bce, global_iter, scalar_writer)
#                     #     for i_c, v_ch in enumerate(loss_chest_all):
#                     #         _safe_scalar(f"train/CHEst_{i_c}", v_ch, global_iter, scalar_writer)

#                 elif sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
#                     for name, value in loss.items():
#                         _safe_scalar(f"train/{name}", value, global_iter, scalar_writer)

#                 # --- Optional save / guard on NaN weights
#                 if weight_saving_schedule is not None and global_iter in weight_saving_schedule:
#                     print(f"Saving weights after {global_iter} iterations")
#                     save_weights(model, filename + f"_{global_iter}_iter", save_format=save_format)

#                 if all(tf.reduce_all(tf.math.is_finite(w)).numpy() for w in model.weights):
#                     save_weights(model, filename, save_format=save_format)
#                 else:
#                     print("Warning: Model contains NaN or Inf weights, not saving.", flush=True)

#                 # --- Eval loop: log into same writer with 'eval/' prefix
#                 for mcs_arr_idx in mcs_arr_training_idx:
#                     if not sys_parameters.ebno:
#                         _no = ebnodb2no(
#                             eval_ebno_db_arr[mcs_arr_idx],
#                             model._transmitters[mcs_arr_idx]._num_bits_per_symbol,
#                             model._transmitters[mcs_arr_idx]._target_coderate,
#                             model._transmitters[mcs_arr_idx]._resource_grid)
#                         _snr_db = - 10.0 * tf.math.log(_no) / tf.math.log(10.0)
#                     else:
#                         _snr_db = eval_ebno_db_arr[mcs_arr_idx]

#                     # if sys_parameters.system=='mdx':
#                     #     loss_data_mcs, loss_chestv, bce_rel_v, loss_chest_all_v = eval_model_xla(
#                     #         batch_size, _snr_db, max_num_tx, mcs_arr_idx)
#                     #     _safe_scalar(f"eval/CE_loss_mcs{mcs_arr_idx}",   loss_data_mcs, global_iter, scalar_writer)
#                     #     _safe_scalar(f"eval/CHEst_loss_mcs{mcs_arr_idx}", loss_chestv,   global_iter, scalar_writer)
#                     #     if isinstance(bce_rel_v, list):
#                     #         for i_b, v_bce in enumerate(bce_rel_v):
#                     #             _safe_scalar(f"eval/BCE_rel_PerfCh_{i_b}_mcs{mcs_arr_idx}", v_bce, global_iter, scalar_writer)
#                     #         for i_c, v_ch in enumerate(loss_chest_all_v):
#                     #             _safe_scalar(f"eval/CHEst_{i_c}_mcs{mcs_arr_idx}", v_ch, global_iter, scalar_writer)

#                     if sys_parameters.system=='nrx':
#                         loss_data_mcs, loss_chestv = eval_model_xla(batch_size, _snr_db, max_num_tx, mcs_arr_idx)
#                         _safe_scalar(f"eval/CE_loss_mcs{mcs_arr_idx}",   loss_data_mcs, global_iter, scalar_writer)
#                         _safe_scalar(f"eval/CHEst_loss_mcs{mcs_arr_idx}", loss_chestv,   global_iter, scalar_writer)

#                     elif sys_parameters.system=='deep_echo' or sys_parameters.system=='mdx':
#                         loss_dict_mcs = eval_model_xla(batch_size, _snr_db, max_num_tx, mcs_arr_idx)
#                         for name, value in loss_dict_mcs.items():
#                             _safe_scalar(f"eval/{name}_mcs{mcs_arr_idx}", value, global_iter, scalar_writer)

#                 # ---- CGNN iteration schedule (unchanged logic)
#                 # if not sys_parameters.system=='deep_echo':
#                 #     if cgnn_step > 0:
#                 #         if model._receiver._neural_rx.num_it + 1 <= sys_parameters.num_nrx_iter:
#                 #             model._receiver._neural_rx.num_it += 1
#                 #         else:
#                 #             model._receiver._neural_rx.num_it = 1
#                 #     if cgnn_step == 0:
#                 #         model._receiver._neural_rx.num_it = np.random.randint(1, sys_parameters.num_nrx_iter + 1)

#                 # Stop profiler after first window if requested
#                 if profiler._enabled and profiler._started and global_iter >= profile_k_steps:
#                     profiler.stop()

#     # Make sure profiler is stopped
#     profiler.stop()

# ________________________________________________________________________
# -----------------------End Training Loop--------------------------------
# ________________________________________________________________________




def calculate_goodput(pe, pusch_transmitter, verbose=False):
    """
    Calculates goodput in [info bits / resource element].
    See (24) in https://arxiv.org/pdf/2009.05261.pdf

    Input
    -----
    pe: float or ndarray
        error rate (can be BLER or BER)
    pusch_transmitter: PUSCHTransmitter
        PUSCHTransmitter containing the resource_grid used for encoding

    verbose: bool
        Defaults to False. If True, additional information is provided.

    Output
    ------
    gp_baseline: float or ndarray
        Goodput of baseline with pilots for each given value of ``pe``.

    gp_e2e: float or ndarray
        Goodput of E2E system without pilots for each given value of ``pe``.
    """

    # number of info bits per slot
    num_info_bits = pusch_transmitter._pusch_configs[0].tb_size

    # total number of REs in the grid
    rg_type = pusch_transmitter._resource_grid.build_type_grid()
    num_res = tf.reduce_prod(rg_type.shape[-2:]).numpy()

    # number of pilots (ignore 0 pilots from other streams)
    # focus on first user and assume that other users have same DMRS config
    eps = 1e-6
    num_pilots = tf.reduce_sum(tf.where(
                    tf.math.abs(pusch_transmitter.pilot_pattern.pilots[0])>eps,
                               1,0)).numpy()

    #remove empty pilots (from different CDM groups)
    num_empty_pilots = tf.reduce_sum(tf.where(
                    tf.math.abs(pusch_transmitter.pilot_pattern.pilots[0])<eps,
                             1,0)).numpy()

    # pilots are used, but ignore empty pilots
    gp_baseline = (1 - pe) * num_info_bits / (num_res - num_empty_pilots)
    # pilot positions are not transmitted
    gp_e2e = (1 - pe)*num_info_bits / (num_res - num_pilots - num_empty_pilots)

    if verbose:
        print(f"------------------------------")
        print(f"Total number of REs: {num_res}")
        print(f"Total number of payload bits: {num_info_bits}")
        print(f"Number of pilots: {num_pilots}")
        print(f"Number of empty pilots: {num_empty_pilots}")
        print(f"Goodput w. pilots: {gp_baseline} [info. bits / RE]")
        print(f"Goodput w.o. pilots: {gp_e2e} [info. bits / RE]")
    return gp_baseline, gp_e2e

def plot_results(config_name, show_ber=False, xlim=None, ylim=None,
                 sim_idx=None, num_tx_eval=None, fig=None, color_offset=0,
                 labels=None, mcs_arr_eval_idx=0, line_styles=None, axis=None, show=[True,True,True,True, True, True],
                 results_dir=None):
    # pylint: disable=line-too-long
    r"""
    Visualize results

    Parameters
    ----------
    config_name : str
        Name of the config file to be visualized

    show_ber : bool
        If True, the BER instead of the BLER is shown.

    xlim: [float, float]
        xlims of figure.

    ylim: [float, float]
        ylims of figure.

    sim_idx: list of ints
        Indices of results to be plotted. If set to `None`, all results will be
        shown.

    num_tx_eval : int
        Plot only results for ``num_tx_eval`` active users.

    fig: None of figure
        If None, a new figure will be created.

    color_offset: int
        Skip first colors in colormap.

    labels: list of str | None
        If not None, will be used as labels for the legend of the figure.

    mcs_arr_eval_idx: int
        Selects the MCS index (element index of the mcs_index list) of the
        results to be plotted, defaults to 0.
    """

    def remove_trailing_zeros(snrs_,ers_):

        last_non_zero = len(ers_) - 1
        while last_non_zero >= 0 and ers_[last_non_zero] == 0:
            last_non_zero -= 1


        if last_non_zero >= 0:  
            ers_trimmed = ers_[:last_non_zero + 1]
            snrs_trimmed = snrs_[:last_non_zero + 1]
        else:
            ers_trimmed, snrs_trimmed = [], [] 
        return snrs_trimmed, ers_trimmed

    show_title, show_x_label, show_y_label, show_legend, show_x_ticks, show_y_ticks = show


    sys_parameters = Parameters(config_name,
                                training=False,
                                system='dummy') # dummy system)
    
    filename = f"../results/{sys_parameters.label}_results"
    if results_dir is not None:
        filename = f"{results_dir[0]}{sys_parameters.label}{results_dir[1]}_results"

    # print(f"results file name:{filename}")

    if num_tx_eval is None:
        num_tx_eval = sys_parameters.max_num_tx

    if sim_idx is not None:
        assert isinstance(sim_idx, (int, list, tuple)),\
            "sim_idx must be list of ints."
        # wrap in to list of int is provided
        if isinstance(sim_idx, int):
            sim_idx = [sim_idx]

    if fig is None and axis is None:
        # generate new figure
        fig, ax = plt.subplots(figsize=(12,8));
    if fig is not None and axis is None:
        ax = fig.gca()
    if fig is None and axis is not None:
        ax = axis
    if fig is not None and axis is not None:
        fig = None
        ax = axis
    
    if exists(filename):
        SNRs = None
        with open(filename,'rb') as f:
            data = pickle.load(f)
            if len(data)==3:
                snrs, BERs, BLERs = data
                BIT_ERRORs = None 
                BLOCK_ERRORs = None 
                NB_BITs = None 
                NB_BLOCKs = None
            if len(data) == 7:
                snrs, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs = data
            if len(data) == 8:
                snrs, BERs, BLERs, BIT_ERRORs, BLOCK_ERRORs, NB_BITs, NB_BLOCKs, SNRs = data

        if show_ber:
            ERs = BERs
        else:
            ERs = BLERs

        if sim_idx is None:
            sim_idx=np.arange(len(ERs))

        idx = 0
        l_idx = 0 # index of label
        for e in ERs:
            # only consider num_tx_eval entries
            if num_tx_eval == e[1]:
                if idx in sim_idx:
                    # "len(e)==2" implements backwards
                    # compatibility for non-MCS-specific results
                    if len(e) == 2 or mcs_arr_eval_idx == e[2]:
                        if labels is None:
                            l = e[0] # use label from result file
                        else:
                            l_style = "-"
                            color = COLORMAP[idx+color_offset]
                            marker=''     
                            marker_size = 3                       
                            if line_styles is not None:
                                style_ = line_styles[l_idx]
                                if isinstance(style_,list):
                                    if len(style_)>0:
                                        l_style = style_[0]
                                    if len(style_)>1:     
                                        color = COLORMAP[style_[1]]
                                    if len(style_)>2: 
                                        marker=style_[2]
                                    if len(style_)>3: 
                                        marker_size = style_[3]
                                else:
                                    l_style = style_
                            # use custom label
                            l = labels[l_idx]
                            l_idx += 1 # increase label index
                        if SNRs is not None:
                            if e in SNRs:
                                snrs_ = SNRs[e]
                                # print("key{e} exists!")
                            else:
                                snrs_ = snrs
                        else:
                            snrs_ = snrs
                        ers_ = ERs[e]
                        snrs_,ers_ = remove_trailing_zeros(snrs_,ers_)

                        if not l.startswith('-'):
                            ax.semilogy(snrs_, ers_, label=l,
                                        color=color,
                                        linewidth=2.0,
                                        marker=marker,
                                        ms=marker_size,
                                        linestyle=l_style)
                        else:
                            print(f"skipped key:{e}, label:{l}")

                        idx += 1
                        # Add y-ticks at every decade
                        # Adjust the range based on your data's min and max y-values
                        y_min, y_max = ax.get_ylim()  # Get current y-axis limits
                        if y_min>0:
                            decades = np.logspace(np.floor(np.log10(y_min)), np.ceil(np.log10(y_max)), 
                                                num=int(np.ceil(np.log10(y_max) - np.floor(np.log10(y_min)) + 1)))
                            ax.set_yticks(decades)
    else:
        print("No results found")

    text_size = 10 # 15
    if show_title:
        title = f"5G NR PUSCH {num_tx_eval}x{sys_parameters.num_rx_antennas} "\
                f"MU-MIMO, {sys_parameters.channel_type}-Channel, " \
                f"MCS={sys_parameters.mcs_index[mcs_arr_eval_idx]}, "\
                f"PRBs={sys_parameters.n_size_bwp}"
        ax.set_title(title, fontsize=text_size)

    if show_x_ticks:
        ax.tick_params(axis='x', labelsize=text_size)

    if show_y_ticks:
        ax.tick_params(axis='y', labelsize=text_size)


    ax.grid(True, which="both")
    
    if show_x_label:
        # ax.set_xlabel("SNR [dB]", fontsize=text_size)
        ax.set_xlabel(r"$\mathrm{E_b/N_0}$ [dB]", fontsize=text_size)
    if show_y_label:
        if show_ber:
            ax.set_ylabel("BER", fontsize=text_size)
        else:
            ax.set_ylabel("TBLER", fontsize=text_size)

    if show_legend:
        # ax.legend(loc="lower left", fontsize=15);
        # ax.legend(loc="upper right", fontsize=15);
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncols=2) #up right
        # plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncols=2) #below
        plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', ncols=2) #above center

    if xlim is not None:
        ax.set_xlim(xlim);
    else:
        ax.set_xlim([min(snrs),max(snrs)]);

    if ylim is not None:
        ax.set_ylim(ylim);

    return fig

def export_csv(config_name, num_tx_eval):
    """Export results to csv for pgfplots etc.

    Parameters
    ----------
    config_name : str
        Name of the config file to be visualized

    num_tx_eval : int
        Plot only results for ``num_tx_eval`` active users.
    """
    sys_parameters = Parameters(config_name,
                                training=True,
                                system='dummy') # dummy system)
    filename = f"../results/{sys_parameters.label}_results"

    with open(filename,'rb') as f:
        snrs, BERs, BLERs = pickle.load(f)

    a = snrs
    for label in BLERs:
        if num_tx_eval == label[1]:
            a = np.column_stack((snrs, BLERs[label].numpy()))
            if label[0]=="Baseline - LS/lin+LMMSE":
                l = "lslin+lmmse"
            elif label[0]=="Neural Receiver":
                l = "nn"
            elif label[0]=="Baseline - Perf. CSI & K-Best":
                l = "perfcsi"
            elif label[0]=="Baseline - LMMSE+K-Best":
                l = "lmmse+kbest"
            else:
                pass
            # Save into individual file via pandas DataFrame
            df = pd.DataFrame(a, columns=['snr', "bler"])

            # Export the DataFrame to a CSV file
            df.to_csv(f"{sys_parameters.label}_{num_tx_eval}_{l}.csv",
                      index=False)

def plot_gp(config_name, num_tx_eval=None, xlim=None, ylim=None, fig=None,
            sim_idx=None, color_offset=0, labels=None, verbose=False):
    """Calculate and plot goodput from simulated error rates.
    Parameters
    ----------
    config_name : str
        Name of the config file to be visualized

    num_tx_eval : int
        Plot only results for ``num_tx_eval`` active users.

    xlim: [float, float]
        xlims of figure.

    ylim: [float, float]
        ylims of figure.

    fig: None of figure
        If None, a new figure will be created.

    sim_idx: list of ints
        Indices of results to be plotted. If set to `None`, all results will be
        shown.

    color_offset: int
        Skip first colors in colormap.

   labels: list of str | None
        If not None, will be used as labels for the legend of the figure.

    verbose: bool
        Defaults to False. If True, additional information is provided.

    """
    sys_parameters = Parameters(config_name,
                                training=False,
                                system='nn') # dummy system to init transmitter
    filename = f"../results/{sys_parameters.label}_results"

    if num_tx_eval is None:
        num_tx_eval = sys_parameters.max_num_tx

    if sim_idx is not None:
        assert isinstance(sim_idx, (int, list, tuple)),\
            "sim_idx must be list of ints."
        # wrap in to list of int is provided
        if isinstance(sim_idx, int):
            sim_idx = [sim_idx]

    if fig is None:
        # generate new figure
        fig, ax = plt.subplots(figsize=(12,8));
    else:
        ax = fig.gca()

    if exists(filename):
        with open(filename,'rb') as f:
            snrs, BERs, BLERs = pickle.load(f)

        if sim_idx is None:
            sim_idx=np.arange(len(BLERs))

        idx = 0
        l_idx = 0 # index of label
        for e in BLERs:
            # only consider num_tx_eval entries
            if num_tx_eval == e[1]:
                if idx in sim_idx:
                    if labels is None:
                        l = e[0] # use label from result file
                    else:
                        # use custom label
                        l = labels[l_idx]
                        l_idx += 1 # increase label index
                    gp_bs, gp_e2e = calculate_goodput(
                                            BLERs[e],
                                            sys_parameters.transmitters[0],
                                            verbose=verbose)
                    # ignore DMRS overhead if pilots are masked (i.e., not used)
                    if e[0]=="Neural Receiver" and sys_parameters.mask_pilots:
                        print("masked pilots detected: "\
                              "ignoring DMRS overhead for NRX results")
                        gp = gp_e2e
                    else: # baseline uses pilots
                        gp = gp_bs
                    ax.plot(snrs, gp,
                            label=l, color=COLORMAP[idx+color_offset],
                            linewidth=3.0)
                idx += 1
    else:
        print("No results found")

    title = f"Goodput: {num_tx_eval}x{sys_parameters.num_rx_antennas} " \
            f"MU-MIMO, {sys_parameters.channel_type}-Channel, "\
            f"MCS={sys_parameters.mcs_index}, PRBs={sys_parameters.n_size_bwp}"

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.grid(True, which="both")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("SNR [dB]", fontsize=15)
    ax.set_ylabel("[info. bits / RE]", fontsize=15)
    ax.legend(loc="lower left", fontsize=15);

    if xlim is not None:
        ax.set_xlim(xlim);
    else:
        ax.set_xlim([min(snrs),max(snrs)]);

    if ylim is not None:
        ax.set_ylim(ylim);

    return fig


def export_constellation(config_name, fn="custom_constellation"):
    """Export custom constellation from trained neural-rx model.

    Parameters
    ----------
    config_name: str
        Name of the config to load.

    fn: str
        Name of the exported csv file containing the data points.

    Output
    ------
    cs: ndarray of complex
        Custom constellation points

    labels: ndarray of ints
        Labels for each point in cs
    """

    # load system_parameters and model
    sys_parameters = Parameters(config_name,
                                training=True,
                                system='nn')

    model = E2E_Model(sys_parameters, training=False)

    # init weights
    model(1,1.);
    filename = f'../weights/{sys_parameters.label}_weights'
    load_weights(model, filename)

    # get constellation points
    cs = model._transmitters[0]._mapper.constellation.points.numpy()

    # symbols are labeled in ascending order
    m = int(np.log2(len(cs)))
    labels = np.zeros((len(cs), m))
    for idx in range(len(cs)):
        labels[idx,:] = sn.fec.utils.int2bin(idx, m)

    # generate dictionary for export
    r = {}
    for idx,(l,c) in enumerate(zip(labels,cs)):
        r.update({f"{idx}": {"constellation": c, "label": l}})

    # Export the DataFrame to a CSV file
    df = pd.DataFrame(r)
    df.to_csv(fn+".csv", index=False)

    return cs, labels

def sample_along_trajectory(waypoints, num_points, velocity):
    """Sample user positions on a trajectory defined by the waypoints.

    The function samples in total num_points positions.
    Further each position has an individual velocity vector.

    Parameters
    ----------
    waypoints: list of [3] float
        Waypoints defining the trajectory.

    num_points: int
        Defines how many discrete positions shall be sampled.

    velocity: float
        UE velocity.

    Outputs
    -------
    rx_positions: list of [3] floats
        Contains all `num_points` user positions.

    rx_velocity: list of [3] floats
        User velocity for each position.

    total_distance: float
        Total path length of all segments.

    """
    num_segments = len(waypoints) - 1
    waypoints = np.array(waypoints)

    # calculate length and direction of each segment
    directions = np.roll(waypoints, -1, 0) - waypoints
    distances = np.sqrt(np.sum(np.abs(directions)**2, axis=1, keepdims=True))
    directions /= distances

    # ignore last entry (returns to starting position)
    distances = distances[:-1,...]
    directions = directions[:-1,...]

    # total length of trajectory
    total_distance = np.sum(distances)
    sample_distance = total_distance / num_points

    # sample discrete positions
    rx_positions = []
    rx_velocities = []
    for i in range(num_segments):
        num_points_segm = int(np.round(distances[i]/sample_distance))
        # initial position from waypoint
        p = waypoints[i]
        for _ in range(num_points_segm):
            rx_positions.append(np.copy(p))
            p += directions[i] * sample_distance
            rx_velocities.append(directions[i] * velocity)

    # remove last points in case more than num_points are added due to rounding
    rx_positions = rx_positions[:num_points]
    rx_velocities = rx_velocities[:num_points]
    return rx_positions, rx_velocities, total_distance


######################################
### Utilities for tf_records exporting
######################################

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(a, tau):
    a_bytes = tf.io.serialize_tensor(a)
    tau_bytes = tf.io.serialize_tensor(tau)
    feature = {
        'a': _bytes_feature(a_bytes.numpy()),
        'tau': _bytes_feature(tau_bytes.numpy())
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
