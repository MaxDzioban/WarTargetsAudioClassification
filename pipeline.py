import argparse
import os
from mp3_to_wav import split_mp3_to_chunks
from final_model_mfcc import process_flat_wav_folder
from pretrained_model import predict_with_pretrained
from split_wav_to_chunks import split_wav_to_chunks

def run_pipeline(filepath):
    if filepath.endswith(".mp3"):
        output_dir = os.path.splitext(filepath)[0] + "_chunks"
        split_mp3_to_chunks(filepath, output_dir)
        mfcc_csv_path = os.path.splitext(filepath)[0] + "_mfcc.csv"
        process_flat_wav_folder(output_dir, mfcc_csv_path)
        results = predict_with_pretrained(mfcc_csv_path)
    elif filepath.endswith(".wav"):
        output_dir = os.path.splitext(filepath)[0] + "_chunks"
        split_wav_to_chunks(filepath, output_dir)
        mfcc_csv_path = os.path.splitext(filepath)[0] + "_mfcc.csv"
        process_flat_wav_folder(output_dir, mfcc_csv_path)
        results = predict_with_pretrained(mfcc_csv_path)
    else:
        raise TypeError("Unsupported file type")
    for idx, res in enumerate(results):
        print(f"Sample {idx+1}: {res}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict class from MP3 or WAV")
    parser.add_argument("file", nargs="+", help="Paths to mp3 or wav file")
    args = parser.parse_args()
    for f in args.file:
        run_pipeline(f)

