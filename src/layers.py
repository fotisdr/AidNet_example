# coding: utf-8
"""
Custom layers that define the DNN architecture.

Fotios Drakopoulos, UCL, January 2026
"""

__author__ = 'fotisdr'


import tensorflow as tf
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Conv1D, Layer, Reshape, Lambda, Concatenate
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import backend as K

import numpy as np
import math

from activations import *

@register_keras_serializable(package=__name__)
class ConvEncoderBlockAntialiased(Layer):
    """
    A 1D convolutional encoder block with anti-aliasing
    """

    def __init__(self, channels, kernel_size, padding, activation, 
                 strides, use_skip_connections = True,
                 activity_regularizer=None, kernel_regularizer=None, 
                 kernel_initializer='glorot_uniform', use_blurpool=False, 
                 use_firwin=False, filter_kernel=16, **kwargs):
        super().__init__(**kwargs)

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.strides = strides
        self.use_skip_connections = use_skip_connections
        
        self.activity_regularizer = activity_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        
        self.use_blurpool = use_blurpool
        self.use_firwin = use_firwin
        self.filter_kernel = filter_kernel

        # Use only 1-strided convolutions
        self.conv = Conv1D(
            channels,
            kernel_size,
            padding=padding,
            strides=1,
            activity_regularizer = activity_regularizer,
            kernel_regularizer = kernel_regularizer,
            kernel_initializer = kernel_initializer,
            name=f"{self.name}_conv",
        )
        # Downsampling layer
        if use_blurpool: # blurpoool downsampling
            self.ds = BlurPool1D(channels=channels,
                               kernel_size=self.filter_kernel, 
                               stride=strides,
                               name=f"{self.name}_blurpool")
        elif use_firwin: # firwin downsampling
            self.ds = FirWin1D(channels=channels,
                               kernel_size=self.filter_kernel, 
                               stride=strides,
                               name=f"{self.name}_firwin")
        else: # Decimate by the stride (e.g. a factor of 2) - Adopted from Wave-U-Net 
            self.ds = Lambda(lambda x: x[:,::strides,:], name=f"{self.name}_decimate")
        # Activation function
        self.out = get_activation(activation, name=f"{self.name}_out")

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.out(outputs)
        if self.use_skip_connections:
            skip = outputs # the output after the activation function is used as the skip connection 
        outputs = self.ds(outputs)

        if self.use_skip_connections:
            return outputs, skip
        else:
            return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "activation": self.activation,
                "strides": self.strides,
                "activity_regularizer": self.activity_regularizer,
                "kernel_regularizer": self.kernel_regularizer,
                "kernel_initializer": self.kernel_initializer,
                "use_blurpool": self.use_blurpool,
                "use_firwin": self.use_firwin,
                "filter_kernel": self.filter_kernel, 
            }
        )
        return config

@register_keras_serializable(package=__name__)
class UpSample1D(Layer):
    """
    A simple wrapper of `tf.keras.layers.UpSampling2D` to expose linear and nearest
    neighbor interpolation for 1D inputs, which are not available in
    `tf.keras.layers.UpSampling1D` at the moment

    Parameters
    ----------
    See `tf.keras.layers.Upsampling2D`
    """

    def __init__(
        self, size, interpolation="nearest", data_format="channels_last", **kwargs
    ):
        super().__init__(**kwargs)

        self.size = size
        self.interpolation = interpolation
        self.data_format = data_format
        self.upsample = tf.keras.layers.UpSampling2D(
            size=(size, 1), data_format=data_format, interpolation=interpolation
        )

    def call(self, inputs):
        input_shape = int_shape(inputs)
        outputs = tf.keras.layers.Reshape((-1, 1, input_shape[2]))(inputs) # Replaced input_shape[1] with -1
        outputs = self.upsample(outputs)
        outputs = tf.keras.layers.Reshape((-1, input_shape[2]))(outputs) # Replaced self.size * input_shape[1] with -1
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "interpolation": self.interpolation,
                "data_format": self.data_format,
            }
        )
        return config

@register_keras_serializable(package=__name__)
class ConvDecoderBlock(Layer):
    """
    A 1D convolutional decoder block with optional skip connection
    """

    def __init__(
        self,
        channels,
        kernel_size,
        padding,
        activation,
        strides,
        interpolation,
        activity_regularizer=None, 
        kernel_regularizer=None, 
        kernel_initializer='glorot_uniform', 
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.strides = strides
        self.interpolation = interpolation
        self.activity_regularizer = activity_regularizer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer

        self.upsample = UpSample1D(
            size=strides, interpolation=interpolation, data_format="channels_last"
        )
        self.concat = Concatenate(axis=-1)
        self.conv = Conv1D(
            channels, kernel_size, padding=padding, strides=1, 
            activity_regularizer = activity_regularizer,
            kernel_regularizer = kernel_regularizer,
            kernel_initializer = kernel_initializer,
            name=f"{self.name}_conv"
        )
        self.out = get_activation(activation, name=f"{self.name}_out")

    def call(self, inputs, skip_connection=None):
        outputs = self.upsample(inputs)

        if skip_connection is not None:
            outputs = self.concat([outputs, skip_connection])

        outputs = self.conv(outputs)
        return self.out(outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "activation": self.activation,
                "strides": self.strides,
                "interpolation": self.interpolation,
                "activity_regularizer": self.activity_regularizer,
                "kernel_regularizer": self.kernel_regularizer,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config
