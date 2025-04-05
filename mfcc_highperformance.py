import os

import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
from scipy.fftpack import dct
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

FRAME_LEN_MS = 25  # Frame length in milliseconds
FRAME_SHIFT_MS = 10  # Frame shift in milliseconds
FFT_SIZE = 512  # Number of FFT points
PRE_EMPHASIS_COEFF = 0.97  # Pre-emphasis filter coefficient
NUM_MFCC_COEFFS = 20  # Number of MFCC coefficients to retain
NUM_MEL_FILTERS = 40  # Number of Mel filters
LOW_FREQ_HZ = 20  # Minimum frequency for Mel filter
HIGH_FREQ_DIVISOR = 2  # High frequency limit = sampling rate / this value, Nyquist
MEL_SCALE_FACTOR = 2595  # Scaling factor for Mel conversion
MEL_REFERENCE_FREQ = 700  # Reference frequency in Hz
MS_TO_SEC = 1000
DB_CONVERSION_FACTOR = 10
FFT_BIN_OFFSET = 1  # To account for FFT bin calculation
EXTRA_MEL_POINTS = 2  # Extra points for filterbank calculation


def compute_spectrum(frame):
    """
    This function calculates the power spectrum of audio frame.
    The power spectrum shows how much energy (power) is present at different frequencies.
    
    Steps:
    1. Perform the Fast Fourier Transform (FFT) to convert the frame from time-domain to frequency-domain.
    2. Compute the magnitude of the FFT result.
    3. Square the magnitude to get the power spectrum.
    
    Parameters:
    - frame (numpy array): The audio frame to analyze.
    
    Returns:
    - power_spec (numpy array): The power spectrum of the frame.
    """
    freq_domain = np.fft.rfft(frame, n=FFT_SIZE)
    return np.abs(freq_domain) ** 2  # Power spectrum (magnitude squared)

def compute_mfcc(row):
    """
    This function calculates the Mel-Frequency Cepstral Coefficients (MFCCs) 
    from the log energy of the Mel-filtered spectrum.

    Steps:
    1. Perform the Discrete Cosine Transform (DCT) on the input data.
    2. Take only the first 20 coefficients since they contain the most useful information.
    
    Parameters:
    - row (numpy array): The log Mel-filterbank energies.
    
    Returns:
    - mfcc_coeffs (numpy array): The first 20 MFCC coefficients.
    """
    return dct(row, type=2, norm="ortho")[:NUM_MFCC_COEFFS]

def load_audio(filename):
    """Loads an audio file and normalizes it to floating-point format."""
    try:
        fs, wavdata = wavfile.read(filename)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    wavdata = wavdata.astype(np.float32) / np.max(np.abs(wavdata))
    if wavdata.ndim > 1:
        wavdata = wavdata[:, 0]  # Use only one channel if stereo

    return fs, wavdata

def preprocess_audio(wavdata, fs):
    """Applies pre-emphasis filtering and pads the audio."""
    frame_len_samples = int(FRAME_LEN_MS * fs / MS_TO_SEC)
    frame_shift_samples = int(FRAME_SHIFT_MS * fs / MS_TO_SEC)

    first_frame = 1
    total_frames = int(np.ceil((len(wavdata) - frame_len_samples) / frame_shift_samples)) + first_frame

    pad_length = (total_frames - first_frame) * frame_shift_samples + frame_len_samples - len(wavdata)
    pad_data = np.pad(wavdata, (0, pad_length), mode='constant')

    pad_data = np.append(pad_data[0], pad_data[1:] - PRE_EMPHASIS_COEFF * pad_data[:-1])

    return pad_data, frame_len_samples, frame_shift_samples, total_frames

def process_frame(i, pad_data, frame_len_samples, frame_shift_samples, window_func):
    """Extracts a single frame and applies a Hamming window."""
    return pad_data[i * frame_shift_samples:i * frame_shift_samples + frame_len_samples] * window_func

def compute_mel_filterbank(fs):
    """Computes the Mel filterbank matrix."""
    high_freq_hz = fs // HIGH_FREQ_DIVISOR
    low_freq_mel = MEL_SCALE_FACTOR * np.log10(1 + LOW_FREQ_HZ / MEL_REFERENCE_FREQ)
    high_freq_mel = MEL_SCALE_FACTOR * np.log10(1 + high_freq_hz / MEL_REFERENCE_FREQ)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, NUM_MEL_FILTERS + EXTRA_MEL_POINTS)
    hz_points = MEL_REFERENCE_FREQ * (DB_CONVERSION_FACTOR ** (mel_points / MEL_SCALE_FACTOR) - 1)

    bins = np.floor((FFT_SIZE + FFT_BIN_OFFSET) * hz_points / fs).astype(int)

    fbank = np.zeros((NUM_MEL_FILTERS, FFT_SIZE // 2 + 1))
    for m in range(1, NUM_MEL_FILTERS + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        fbank[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
        fbank[m - 1, f_m:f_m_plus] = (f_m_plus - np.arange(f_m, f_m_plus)) / (f_m_plus - f_m)

    return fbank

def plot_mfcc(mfcc, fs):
    """Plots the MFCC coefficients."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc.T, x_axis='time', sr=fs, cmap='viridis')
    plt.colorbar(label="MFCC Coefficients")
    plt.title("MFCCs of Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficients")
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <wav_file>")
        sys.exit(1)

    filename = sys.argv[1]
    fs, wavdata = load_audio(filename)
    
    pad_data, frame_len_samples, frame_shift_samples, total_frames = preprocess_audio(wavdata, fs)
    window_func = np.hamming(frame_len_samples)

    with ThreadPoolExecutor() as executor:
        frame_data = np.array(list(executor.map(lambda i: process_frame(i, pad_data, frame_len_samples, frame_shift_samples, window_func), range(total_frames))))

    with ProcessPoolExecutor() as executor:
        power_spec = np.array(list(executor.map(compute_spectrum, frame_data)))

    fbank = compute_mel_filterbank(fs)
    filter_bank = np.dot(power_spec, fbank.T)
    epsilon = np.finfo(float).eps
    filter_bank = np.where(filter_bank == 0, epsilon, filter_bank)

    log_fbank = DB_CONVERSION_FACTOR * np.log10(filter_bank + epsilon)
    # Convert to decibels

    with ProcessPoolExecutor() as executor:
        mfcc = np.array(list(executor.map(compute_mfcc, log_fbank)))

    # Замість збереження повної матриці MFCC, рахуємо mean + std:
    mfcc_mean = mfcc.mean(axis=0)
    mfcc_std = mfcc.std(axis=0)
    mfcc_vector = np.concatenate([mfcc_mean, mfcc_std])

    # Зберігаємо результат у файл (один рядок)
    np.savetxt("mfcc_feature_vector.txt", mfcc_vector[None], delimiter=",", fmt="%.9f")
    print("Збережено вектор MFCC (mean + std) у mfcc_feature_vector.txt")

    # Uncomment to visualize MFCCs
    # plot_mfcc(mfcc, fs)

if __name__ == "__main__":
    main()