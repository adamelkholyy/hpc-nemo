import json
import os
import tempfile
from typing import Literal

import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
from omegaconf import OmegaConf


class MSDDDiarizer:
    def __init__(
        self,
        device: str | torch.device,
        num_speakers: int | None = None,
        domain_type: Literal["telephonic", "meeting", "general"] = "telephonic",
        vad_model: str | os.PathLike = "vad_multilingual_marblenet",
        speaker_model: Literal[
            "titanet_large", "ecapa_tdnn", "speakerverification_speakernet"
        ] = "titanet_large",
    ):
        """

        Create NeMo diarization model.

        Args:
            device (str | torch.device):
                PyTorch device to run diarizer on.
            num_speakers (int | None, optional):
                Number of speakers in audio. Default is ``None`` for automatic detection.
            domain_type (Literal["telephonic", "meeting", "general"], optional):
                Type of diarization model to use. Default is ``"telephonic"``.
                - ``"telephonic"``: Suitable for telephone recordings involving 2–8 speakers in a session.
                May not show the best performance on other types of acoustic conditions or dialogues.
                - ``"meeting"``: Suitable for 3–5 speakers participating in a meeting.
                May not show the best performance on other types of dialogues.
                - ``"general"``: Optimized to show balanced performance across various domains.
                VAD is optimized on multilingual ASR datasets, and the diarizer is optimized
                on the DIHARD3 development set.
            vad_model (str | os.PathLike, optional):
                Model to use for Voice Activity Detection (VAD). Default is ``"vad_multilingual_marblenet"``.
            speaker_model (Literal["titanet_large", "ecapa_tdnn", "speakerverification_speakernet"], optional):
                Model to use for speaker embeddings. Default is ``"titanet_large"``.

        """

        self.num_speakers = num_speakers
        self.domain_type = domain_type
        self.vad_model = vad_model
        self.speaker_model = speaker_model
        self.model: NeuralDiarizer = NeuralDiarizer(cfg=self.create_config()).to(device)


    def create_config(self):
        config = OmegaConf.load(
            os.path.join(
                os.path.dirname(__file__), f"diar_infer_{self.domain_type}.yaml"
            )
        )

        self.max_speakers = config.diarizer.clustering.parameters.max_num_speakers
        self.batch_size = config.batch_size

        # Set num CPU workers used for splitting audio to 0
        config.num_workers = 0

        config.diarizer.out_dir = None
        config.diarizer.manifest_filepath = None
        config.diarizer.vad.model_path = self.vad_model
        config.diarizer.speaker_embeddings.model_path = self.speaker_model
        config.diarizer.clustering.parameters.oracle_num_speakers = (
            True if self.num_speakers else False
        )

        return config


    def diarize(self, audio: torch.Tensor):
        with tempfile.TemporaryDirectory() as temp_path:
            torchaudio.save(
                os.path.join(temp_path, "mono_file.wav"),
                audio,
                16000,
                channels_first=True,
            )

            manifest_path = os.path.join(temp_path, "manifest.json")
            meta = {
                "audio_filepath": os.path.join(temp_path, "mono_file.wav"),
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "rttm_filepath": None,
                "uem_filepath": None,
                "num_speakers": self.num_speakers,
            }

            with open(manifest_path, "w") as f:
                json.dump(meta, f)

            self.model._initialize_configs(
                manifest_path=manifest_path,
                tmpdir=temp_path,
                max_speakers=self.max_speakers,
                batch_size=self.batch_size,
                num_speakers=self.num_speakers,
                num_workers=0,
                verbose=True,
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.out_dir = (
                temp_path
            )
            self.model.clustering_embedding.clus_diar_model._diarizer_params.manifest_filepath = (
                manifest_path
            )
            self.model.msdd_model.cfg.test_ds.manifest_filepath = manifest_path
            self.model.diarize()

            pred_labels_clus = rttm_to_labels(
                os.path.join(temp_path, "pred_rttms", "mono_file.rttm")
            )

            labels = []
            for label in pred_labels_clus:
                start, end, speaker = label.split()
                start, end = float(start), float(end)
                start, end = int(start * 1000), int(end * 1000)
                labels.append((start, end, int(speaker.split("_")[1])))

            labels = sorted(labels, key=lambda x: x[0])

        return labels
