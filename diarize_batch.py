import os
import subprocess
import time
import logging

from helpers import initialise_parser


# Format a time in seconds into hours, mins and seconds
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    mins, secs = divmod(remainder, 60)
    return int(hours), int(mins), int(secs)


parser = initialise_parser()
parser.add_argument(
    "-f",
    "--folder",
    help="name of the folder conatining files to transcribe and diarize",
    required=True,
)

parser.add_argument(
    "--no-parallel",
    dest="no_parallel",
    action=store_true,
    help="turn off parallel transcription and diarization",
)

args = parser.parse_args()

# Generate command from args
command = [
    "python",
    "diarize.py" if args.no_parallel else "diarize_parallel.py", 
    "--whisper-model", args.model_name,
    "--language", args.language,
    "--batch-size", args.batch_size,
]

if not args.stemming:
    command.append("--no-stem")
if args.surpress_numerals:
    command.append("--surpress-numerals")

# Add NeMo parameters to command accordingly
for argname, nemo_param in args.nemo_params.items():
    command += [f"--{argname}", nemo_param]

command = list(map(str, command))


cwd = os.getcwd()
audio_folder = os.path.join(cwd, args.folder)
audio_files = [
    f
    for f in os.listdir(audio_folder)
    if f.lower().endswith(".mp3") or f.lower().endswith(".wav")
]
num_files = len(audio_files)

logging.info(f"Found {num_files} files in {audio_folder} to transcribe and diarize")


# Process each audio file
total_time = 0
for i, file in enumerate(audio_files):
    logging.info(f" ({i+1}/{num_files}) Transcribing and diarizing {file}")
    filepath = os.path.join(audio_folder, file)

    start_time = time.time()
    subprocess.run(command + ["-a", filepath])
    time_taken = time.time() - start_time

    total_time += time_taken
    hrs, mins, secs = format_time(time_taken)
    logging.info(f"Transcription and diarization complete in {mins}m {secs}s")


total_hrs, total_mins, total_secs = format_time(total_time)
logging.info(
    f"Successfully transcribed and diarized {len(audio_files)} audio files in {audio_folder} in {total_hrs}h {total_mins}m {total_secs}s"
)
