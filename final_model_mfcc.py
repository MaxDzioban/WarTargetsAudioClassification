# в папці dataset є у підпапках є файли з аудіо кожного класу,
# потрібно побудувати по них матрицію. потім взяти з неї std dev and mean 
# потім позакидати в модельку

import os
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
    freq_domain = np.fft.rfft(frame, n=FFT_SIZE)
    return np.abs(freq_domain) ** 2

def compute_mfcc(row):
    return dct(row, type=2, norm="ortho")[:NUM_MFCC_COEFFS]

def load_audio(filename):
    fs, wavdata = wavfile.read(filename)
    wavdata = wavdata.astype(np.float32) / np.max(np.abs(wavdata))
    if wavdata.ndim > 1:
        wavdata = wavdata[:, 0]
    return fs, wavdata

def preprocess_audio(wavdata, fs):
    frame_len_samples = int(FRAME_LEN_MS * fs / MS_TO_SEC)
    frame_shift_samples = int(FRAME_SHIFT_MS * fs / MS_TO_SEC)
    total_frames = int(np.ceil((len(wavdata) - frame_len_samples) / frame_shift_samples)) + 1
    pad_length = (total_frames - 1) * frame_shift_samples + frame_len_samples - len(wavdata)
    pad_data = np.pad(wavdata, (0, pad_length), mode='constant')
    pad_data = np.append(pad_data[0], pad_data[1:] - PRE_EMPHASIS_COEFF * pad_data[:-1])
    return pad_data, frame_len_samples, frame_shift_samples, total_frames

def process_frame(i, pad_data, frame_len_samples, frame_shift_samples, window_func):
    return pad_data[i * frame_shift_samples:i * frame_shift_samples + frame_len_samples] * window_func

def compute_mel_filterbank(fs):
    high_freq_hz = fs // HIGH_FREQ_DIVISOR
    low_freq_mel = MEL_SCALE_FACTOR * np.log10(1 + LOW_FREQ_HZ / MEL_REFERENCE_FREQ)
    high_freq_mel = MEL_SCALE_FACTOR * np.log10(1 + high_freq_hz / MEL_REFERENCE_FREQ)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, NUM_MEL_FILTERS + 2)
    hz_points = MEL_REFERENCE_FREQ * (10 ** (mel_points / MEL_SCALE_FACTOR) - 1)
    bins = np.floor((FFT_SIZE + 1) * hz_points / fs).astype(int)
    fbank = np.zeros((NUM_MEL_FILTERS, FFT_SIZE // 2 + 1))
    for m in range(1, NUM_MEL_FILTERS + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        fbank[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
        fbank[m - 1, f_m:f_m_plus] = (f_m_plus - np.arange(f_m, f_m_plus)) / (f_m_plus - f_m)
    return fbank

def extract_feature_vector(filepath):
    fs, wavdata = load_audio(filepath)
    pad_data, frame_len_samples, frame_shift_samples, total_frames = preprocess_audio(wavdata, fs)
    window_func = np.hamming(frame_len_samples)

    frames = [process_frame(i, pad_data, frame_len_samples, frame_shift_samples, window_func) for i in range(total_frames)]
    power_spec = [compute_spectrum(f) for f in frames]

    fbank = compute_mel_filterbank(fs)
    filter_bank = np.dot(power_spec, fbank.T)
    filter_bank = np.where(filter_bank == 0, np.finfo(float).eps, filter_bank)
    log_fbank = DB_CONVERSION_FACTOR * np.log10(filter_bank)
    mfcc = np.array([compute_mfcc(row) for row in log_fbank])
    mfcc_mean = mfcc.mean(axis=0)
    mfcc_std = mfcc.std(axis=0)
    return np.concatenate([mfcc_mean, mfcc_std])

def process_dataset_split_by_class(root_dir):
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        output_file = None
        if "AR15" in folder.upper():
            output_file = "ar15_mfcc.csv"
        elif "SPEECH" in folder.upper():
            output_file = "speech_mfcc.csv"
        elif "TANK" in folder.upper():
            output_file = "tank_mfcc.csv"
        else:
            print(f"⚠️ Пропускаю папку: {folder} (невідомий клас)")
            continue

        samples = []

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                filepath = os.path.join(folder_path, file)
                try:
                    features = extract_feature_vector(filepath)
                    samples.append(features)
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {e}")

        samples = np.array(samples)
        np.savetxt(output_file, samples, delimiter=",", fmt="%.6f")
        print(f"✅ Збережено {len(samples)} записів у {output_file} з {folder}")

def main():
    root_dir = "/Users/max/Downloads/Breast-Cancer-Wisconsin-Diagnostic-Dataset-Analysis-main/done/DATASET_new"
    process_dataset_split_by_class(root_dir)
    print("🎯 Успішно створено x.csv (AR15), y.csv (SPEECH), z.csv (TANK)")

if __name__ == "__main__":
    main()
