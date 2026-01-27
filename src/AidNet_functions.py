# -*- coding: utf-8 -*-
"""
This file contains all the supplementary functions needed to execute the
ICNet example script in Python.

@author: Fotios Drakopoulos, UCL, June 2024
"""

from activations import *
from layers import *

from typing import List, Optional, Tuple, Union
from glob import glob
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sp_io
import scipy.signal as sp_sig

import tensorflow as tf
from tensorflow.keras.models import load_model, Model


def rms(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute RMS energy of a matrix over the specified axis.
    If axis = None then the RMS is computed over all axes.
    """
    sq = np.mean(np.square(x), axis = axis)
    return np.sqrt(sq)

def resample_and_filter(signal: np.ndarray, fs_signal: float, fs_target: float,
                         filter_order: int = 8, axis: int = 0) -> np.ndarray:
    """
    Resample an audio signal to the fs_target sampling frequency.
    If the signal gets downsampled, a low-pass filter is first applied
    to avoid aliasing. 
    Returns the resampled (and filtered) audio signal.
    """
    if fs_target < fs_signal: # downsampling
        sos = sp_sig.butter(filter_order, 0.99*(fs_target/2), btype='low', analog=False, 
                            fs=fs_signal, output='sos') # low-pass filtering
        # Apply the low-pass digital filter
        signal = sp_sig.sosfiltfilt(sos, signal, axis=axis)
    # Resample the signal
    if fs_target != fs_signal: 
        signal = sp_sig.resample_poly(signal, int(fs_target), int(fs_signal), axis=axis)

    return signal

def wavfile_read(wavfile: str, fs: float = None) -> Tuple[np.ndarray, float]:
    """
    Read a wavfile and normalize it to +/-1.
    If fs is given, the signal is resampled to the given sampling frequency.
    Returns the sound signal and the corresponding sampling frequency.
    """
    fs_signal, signal = sp_io.wavfile.read(wavfile)
    if not fs:
        fs=fs_signal

    if signal.dtype != 'float32' and signal.dtype != 'float64':
        if signal.dtype == 'int16':
            nb_bits = 16 # -> 16-bit wav files
        elif signal.dtype == 'int32':
            nb_bits = 32 # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        signal = signal / (max_nb_bit + 1.0) # scale the signal to [-1.0,1.0]

    if fs_signal != fs :
        signalr = resample_and_filter(signal, fs_signal, fs)
    else:
        signalr = signal

    return signalr, fs_signal

def wavfile_write(signal: np.ndarray, wavfile: str, fs_signal: float, 
                  fs_wavfile: float = None, dtype: str = 'int16') -> None:
    """
    Write an audio signal to a wavfile.
    If fs_signal and wavfile are both given (and are different), 
    the signal is resampled to the fs_wavfile sampling frequency.
    """
    if fs_wavfile and fs_wavfile != fs_signal:
        signalr = resample_and_filter(signal, fs_signal, fs_wavfile)
    else:
        fs_wavfile = fs_signal
        signalr = signal

    if 'float' in str(signalr.dtype) and 'int' in dtype:
        nb_bits = int(dtype[3:]) # -> 16-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        signalr *= max_nb_bit
        signalr = signalr.astype(dtype)
        
    sp_io.wavfile.write(wavfile, fs_wavfile, signalr)
    
    return None

def pad_along_1dimension(data: np.ndarray, npad_left: int = 0, npad_right: int = 0, 
                         axis: int = 0) -> np.ndarray:
    """
    Pads data with zeros on both sides along the dimension defined.
    """
    # define the npad tuple
    npad = [(0, 0)] * data.ndim
    npad[axis] = (npad_left,npad_right)
    # pad across the defined axis
    data = np.pad(data, npad, mode='constant', constant_values=0)
    
    return data

def slice_1dsignal(signal: np.ndarray, window_size: int, winshift: int, minlength: int = 0, 
                   left_context: int = 2048, right_context: int = 0) -> np.ndarray:
    """Return windows of the given signal by sweeping in stride fractions of window. Slices
    that are less than minlength are omitted. Input signal must be a 1D-shaped array.

    Args:
      signal: A one-dimensional input waveform
      window_size: The size of each window of data from the signal. 
        If window_size = 0 then the window size is matched to the audio size.
      winshift: How much to shift the window as we progress down the signal. 
        If winshift = window_size then the overlap between windows is 0.
      minlength: Drop (final) windows that have less than this number of samples.
      left_context: How much context to add (from earlier parts of the signal) before
        the current window. (Or add zeros if not enough signal)
      right_context: Like left, but to the right of the current window.
    
    Returns:
      A 3D tensor of size [num_frames x window_size x 1]
    """
    assert len(signal.shape) == 1, "signal must be a 1D-shaped array"

    # concatenate zeros to beginning for adding context
    n_samples = signal.shape[0]
    num_slices = (n_samples)
    slices = [] # initialize empty array 
    
    if window_size == 0:
        window_size = n_samples
        winshift = window_size

    for beg_i in range(0, n_samples, winshift):
        beg_i_context = beg_i - left_context
        end_i = beg_i + window_size + right_context
        if n_samples - beg_i < minlength :
            break
        if beg_i_context < 0 and end_i <= n_samples:
            slice_ = np.concatenate((np.zeros((1, left_context - beg_i)),np.array([signal[:end_i]])), axis=1)
        elif end_i <= n_samples: # beg_i_context >= 0
            slice_ = np.array([signal[beg_i_context:end_i]])
        elif beg_i_context < 0: # end_i > n_samples
            slice_ = np.concatenate((np.zeros((1, left_context - beg_i)),np.array([signal]), np.zeros((1, end_i - n_samples))), axis=1)
        else :
            slice_ = np.concatenate((np.array([signal[beg_i_context:]]), np.zeros((1, end_i - n_samples))), axis=1)

        slices.append(slice_)
    slices = np.vstack(slices)
    slices = np.expand_dims(slices, axis=2) # the CNN will need 3D data
    
    return slices

def process_audio(model_path: str, audio: np.ndarray, print_summary: bool = False, 
                             context_size_left: int = 4096, context_size_right: int = 4096, 
                             fs_audio: float = 24414.0625, p0: float = 2e-5, 
                             audio_normalisation: float = 1/25, n_encoder_layers: int = 9,
                             ) -> Tuple[np.ndarray, np.ndarray]:

    '''
    Function to process a sound input (audio) using the provided AidNet model (model_path).

    Args:
      model_path: Where to find the model and weight files
      audio: The audio signal that will be used as input
      print_summary: Print the DNN model summary
      context_size_left, context_size_right: Samples to be cropped out from both sides 
      of the processed audio signal (if provided)
      fs_audio, p0, audio_normalisation, n_encoder_layers: 
      Model parameters fixed to the specific AidNet model version. See the corresponding 
      config.yaml file for more details.

    Returns:
      audio_processed: The AidNet-processed sound, based on the provided arguments.
    '''

    ## Define the input/output parameters
    n_batches = audio.shape[0]
    audio_processed = np.zeros(audio.shape,dtype=audio.dtype)
    ## Recalibrate the audio input for the model
    audio *= audio_normalisation
    ## Load the DNN model
    model_file = model_path + "/model_weights.hdf5"
    model = load_model(model_file, custom_objects={}, compile=False)  
    if print_summary:
        model.summary(line_length = 125)

    ## Simulate the model responses
    for i_batch in range(n_batches):
        stim_input = tf.convert_to_tensor(audio[i_batch:i_batch+1])
        audio_processed[i_batch:i_batch+1] = model.predict(stim_input, verbose=0)

    ## Crop out the context
    if context_size_left:
        audio_processed = audio_processed[:,context_size_left:]
    if context_size_right:
        audio_processed = audio_processed[:,:-context_size_right]
    ## Recalibrate the audio signals
    audio /= audio_normalisation
    audio_processed /= audio_normalisation

    return audio_processed
