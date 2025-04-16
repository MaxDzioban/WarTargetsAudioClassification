import argparse
import os
from mp3_to_wav import split_mp3_to_chunks
from final_model_mfcc import process_flat_wav_folder
from pretrained_model import predict_with_pretrained
from split_wav_to_chunks import split_wav_to_chunks
def main():
    parser = argparse.ArgumentParser(description="Predict class from MP3 or WAV")
    parser.add_argument("file", nargs="+", help="Paths to mp3 or wav file")
    args = parser.parse_args()

    for f in args.file:
        if f.endswith(".mp3"):
            output_dir = os.path.splitext(f)[0] + "_chunks"
            split_mp3_to_chunks(f, output_dir)
            mfcc_csv_path = os.path.splitext(f)[0] + "_mfcc.csv"
            process_flat_wav_folder(output_dir, mfcc_csv_path)
            predict_with_pretrained(mfcc_csv_path)
        elif f.endswith(".wav"):
            print(f"Handling WAV file: {f}")
            output_dir = os.path.splitext(f)[0] + "_chunks"
            split_wav_to_chunks(f, output_dir)
            mfcc_csv_path = os.path.splitext(f)[0] + "_mfcc.csv"
            process_flat_wav_folder(output_dir, mfcc_csv_path)
            predict_with_pretrained(mfcc_csv_path)
        else:
            return TypeError

if __name__ == "__main__":
    main()
