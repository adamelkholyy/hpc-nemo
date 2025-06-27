import os 
import subprocess
import argparse
import time

# initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--folder", help="name of the target audio folder", required=True
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="large-v3",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
)

parser.add_argument(
    "--language",
    type=str,
    default="en",
    help="Language spoken in the audio, specify None to perform language detection",
)

args = parser.parse_args()

# get audio file paths
cwd = os.getcwd()
audio_folder = os.path.join(cwd, args.folder)
audio_files = [f for f in os.listdir(audio_folder) if f.lower().endswith('.mp3') or f.lower().endswith('.wav')]
print(f"Transcribing and diarizing {len(audio_files)} audio files in {audio_folder}...")

# process each audio file
file_counter = 1
for file in audio_files:
    print(f"Processing {file}...")
    file_path = os.path.join(audio_folder, file)
    command = f"python diarize_parallel.py -a {file_path} --no-stem --suppress_numerals --whisper-model {args.model_name} --language {args.language} --batch-size {args.batch_size}"

    start = time.time()
    subprocess.run(command, shell=True)
    end = time.time()

    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"{file_counter}\t{file}{((file_counter/len(audio_files))*100):.2f}%\t{end:.2f}s" + "\n")

    file_counter += 1

print(f"Successfully transcribed and diarized {len(audio_files)} audio files in {audio_folder}.")
