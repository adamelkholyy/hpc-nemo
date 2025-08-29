import argparse
import logging
import os
import time

from DiarizePipeline import DiarizePipeline
from helpers import format_timestamp, whisper_langs
from torch.cuda import is_available

# TODO: Add softformer once merged from whisper-diarization

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
    "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
    "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="large-v3",
    help="Select which Whisper model to use",
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
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if is_available() else "cpu",
    help="If you have a GPU use 'cuda', otherwise 'cpu'. Leave blank for automatic detection.",
)

parser.add_argument(
    "--parallel",
    action="store_true",
    dest="parallel",
    default=False,
    help="Enable parallel NeMo diarization during Whisper transcription",
)

parser.set_defaults(nemo_params={})

# Custom args action class for setting NeMO diarization params
class AddNemoParam(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.nemo_params[self.dest] = values


parser.add_argument(
    "--num-speakers",
    type=int,
    action=AddNemoParam,
    help="Specify number of speakers in audio. Default is 0 for automatic detection",
)

parser.add_argument(
    "--domain-type",
    choices=["telephonic", "meeting", "general"],
    action=AddNemoParam,
    help="Type of diarization model to use. Options are as follows (default is 'telephonic')"
    "\n- 'telephonic': Suitable for telephone recordings involving 2-8 speakers in a session and may not show the best performance on the other types of acoustic conditions or dialogues"
    "\n- 'meeting': Suitable for 3-5 speakers participating in a meeting and may not show the best performance on other types of dialogues"
    "\n- 'general': Optimized to show balanced performances on various types of domain. VAD is optimized on multilingual ASR datasets and diarizer is optimized on DIHARD3 development set",
)

args = parser.parse_args()

# Process args and NeMo params
diarize_args = {k: v for k, v in args.__dict__.items() if v is not None}
audio = diarize_args.pop("audio")
nemo_params = diarize_args.pop("nemo_params")

pipeline = DiarizePipeline(**diarize_args, **nemo_params)


# Run diarization pipeline for audio dir or file
if os.path.isdir(audio):
    audio_files = [
        f for f in os.listdir(audio) if f.endswith(".mp3") or f.endswith(".wav")
    ]
    num_files = len(audio_files)

    assert num_files > 0, f"No .mp3 or .wav files found in {audio}"
    logging.info(f"Found {num_files} audio files to diarize in {audio}")

    start_time = time.time()
    for i, file in enumerate(audio_files):
        logging.info(f"Diarizing {file} ({i+1}/{num_files})")
        pipeline.run(file)

    end_time = time.time() - start_time
    logging.info(
        f"Successfully diarized {num_files} files in {format_timestamp(end_time)}"
    )

elif audio.endswith(".wav") or audio.endswith(".mp3"):
    pipeline.run(audio)
else:
    raise ValueError(
        f"Invalid --audio argument '{audio}' : must be either a directory containing .mp3/.wav files or a .mp3/.wav file"
    )
