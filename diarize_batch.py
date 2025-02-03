import os 
import subprocess
import argparse

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
for file in audio_files:
    print(f"Processing {file}...")
    file_path = os.path.join(audio_folder, file)
    command = f"python diarize.py -a {file_path} --no-stem --suppress_numerals --whisper-model {args.model_name} --language {args.language} --batch-size {args.batch_size}"
    subprocess.run(command, shell=True)

print(f"Successfully transcribed and diarized {len(audio_files)} audio files in {audio_folder}.")
