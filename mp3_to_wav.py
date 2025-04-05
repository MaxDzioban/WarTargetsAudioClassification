import os
import subprocess
from math import ceil

# === НАЛАШТУВАННЯ ===
input_file = "/Users/max/Downloads/радіоперехват2.mp3"  # Замінити на шлях до твого mp3
output_dir = "radio_speech2"
chunk_duration = 6  # у секундах

# === СТВОРЕННЯ ТЕКИ ===
os.makedirs(output_dir, exist_ok=True)

# === ОТРИМАТИ ТРИВАЛІСТЬ ЗАПИСУ ===
def get_duration(filename):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

duration = get_duration(input_file)
num_chunks = ceil(duration / chunk_duration)

# === РОЗБИТТЯ І ЗБЕРЕЖЕННЯ ===
for i in range(num_chunks):
    start = i * chunk_duration
    output_path = os.path.join(output_dir, f"chunk_tank_{i+1:03}.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_file,
        "-ss", str(start),
        "-t", str(chunk_duration),
        output_path
    ])
    print(f"✅ Збережено: {output_path}")

print("🎉 Готово! Усі частини записані.")
