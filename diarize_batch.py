import os 
import subprocess
import argparse
import time

# format estimated finish time into HH:MM:SS
def format_time(seconds):
    total_hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(total_hours):02}:{int(minutes):02}:{int(secs):02}"


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
audio_folder = args.folder

# separate untranscribed files
processed_files = [f[:-4] for f in os.listdir(audio_folder) if f.lower().endswith('txt')]
unprocessed_files = [f for f in os.listdir(audio_folder) if f.lower().endswith('.mp3') or f.lower().endswith('.wav')]
total_processed_files = len(processed_files)
total_unprocessed_files = len(unprocessed_files)

print(f"Transcribing and diarizing {total_unprocessed_files} audio files in {audio_folder}...")

file_counter = 1
total_time = 100
for file in unprocessed_files:
 
    # calculate ETA using rolling average time
    average_time = total_time / file_counter
    eta_seconds = average_time * (total_unprocessed_files - file_counter)
    eta_formatted = format_time(eta_seconds)
    print(f"Transcribing and diarizing {file}...")

    # transcribe and diarize file
    file_path = os.path.join(audio_folder, file)
    command = f"python diarize_parallel.py -a {file_path} --no-stem --suppress_numerals --whisper-model {args.model_name} --language {args.language} --batch-size {args.batch_size}"
    start = time.time()
    subprocess.run(command, shell=True)
    end = time.time()

    # append timing info to logfile
    with open("diarize_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{file_counter + total_processed_files}\t{file}\t{end:.2f}s\t{(((total_processed_files + file_counter)/(total_unprocessed_files + total_processed_files))*100):.2f}%\t{eta_formatted}" + "\n")

    print(f"Transcribed and diarized {file} in {end:.2f}s")
    file_counter += 1
    total_time += end

print(f"Successfully transcribed and diarized {file_counter} audio files in {audio_folder}.")
