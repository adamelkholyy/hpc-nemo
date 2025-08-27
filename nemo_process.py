import argparse
import os

import torch
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from pydub import AudioSegment

from helpers import create_config
from json import loads as json_loads

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
parser.add_argument(
    "--nemo-params",
    dest="nemo_params",
    help="NeMo model parameters (passed from diarize_parallel.py)",
)

args = parser.parse_args()
nemo_params = json_loads(args.nemo_params.replace("'", '"'))

# convert audio to mono for NeMo combatibility
sound = AudioSegment.from_file(args.audio).set_channels(1)
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path, **nemo_params)).to(args.device)
msdd_model.diarize()
