import os
import subprocess
import argparse
from math import ceil

def get_duration(filename):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)

def split_wav_to_chunks(input_file, output_dir, chunk_duration=6):
    os.makedirs(output_dir, exist_ok=True)

    duration = get_duration(input_file)
    num_chunks = ceil(duration / chunk_duration)

    for i in range(num_chunks):
        start = i * chunk_duration
        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(input_file))[0]}_{i+1:03}.wav"
        )

        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", input_file,
            "-ss", str(start),
            "-t", str(chunk_duration),
            output_path
        ])

        print(f"Saved: {output_path}")

    print("All chunks saved!")

def main():
    parser = argparse.ArgumentParser(description="Split WAV file into smaller chunks.")
    parser.add_argument("--input", required=True, help="Path to input WAV file")
    parser.add_argument("--output", required=True, help="Output directory for WAV chunks")
    parser.add_argument("--duration", type=int, default=6, help="Duration of each chunk in seconds (default: 6)")
    args = parser.parse_args()

    split_wav_to_chunks(args.input, args.output, args.duration)

if __name__ == "__main__":
    main()
