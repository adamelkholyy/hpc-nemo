import logging
import multiprocessing as mp
import os
import re
import time

import faster_whisper
import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from diarization import MSDDDiarizer
from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    format_timestamp,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    write_srt,
)


class DiarizePipeline:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        language: str | None = None,
        batch_size: int = 8,
        suppress_numerals: bool = False,
        stemming: bool = True,
        parallel: bool = False,
        **nemo_params,
    ):
        """

        NeMo diarization pipeline

        Args:
            model_name (str):
                Name of the Whisper model to use
            device (str | None):
                Torch device to run diarization on. If you have a GPU use `cuda`, otherwise `cpu`. Default is None for automatic detection. 
            language (str | None, optional):
                Language spoken in the audio, specify None to perform language detection. Default is None.
            batch_size (int, optional):
                Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference. Default is 8.
            suppress_numerals (bool, optional):
                Toggle suppress numerical digits. This helps the diarization accuracy but converts all digits into written text.
            stemming (bool, optional):
                Toggle source separation. This helps with long files that don't contain a lot of music. Default is True.
            parallel (bool, optional): 
                Toggle parallel NeMo diarization during Whisper transcription. Default is False.
            **nemo_params:
                kwargs for NeMo diarization parameters. See `MSDDDiarizer` for full details on NeMo parameters. 
        
        
        """

        self.model_name = model_name
        self.batch_size = batch_size
        self.suppress_numerals = suppress_numerals
        self.stemming = stemming
        self.parallel = parallel
        self.nemo_params = nemo_params

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.language = process_language_arg(language, self.model_name)
        self.diarization_model = MSDDDiarizer(device=self.device, **self.nemo_params)


    @staticmethod
    def diarize_parallel(audio: torch.Tensor, model: MSDDDiarizer, queue: mp.Queue):
        result = model.diarize(audio)
        queue.put(result)


    def run(self, audio: str | os.PathLike):
        start_time = time.time()
        if self.parallel:
            mp.set_start_method("spawn", force=True)

        mtypes = {"cpu": "int8", "cuda": "float16"}

        pid = os.getpid()
        temp_outputs_dir = f"temp_outputs_{pid}"
        temp_path = os.path.join(os.getcwd(), temp_outputs_dir)
        os.makedirs(temp_path, exist_ok=True)

        if self.stemming:
            # Isolate vocals from the rest of the audio

            return_code = os.system(
                f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio}" -o "{temp_outputs_dir}" --device "{self.device}"'
            )

            if return_code != 0:
                logging.warning(
                    "Source splitting failed, using original audio file. "
                    "Use --no-stem argument to disable it."
                )
                vocal_target = audio
            else:
                vocal_target = os.path.join(
                    temp_outputs_dir,
                    "htdemucs",
                    os.path.splitext(os.path.basename(audio))[0],
                    "vocals.wav",
                )
        else:
            vocal_target = audio

        audio_waveform = faster_whisper.decode_audio(vocal_target)

        if self.parallel:
            logging.info("Starting Nemo process with vocal_target: ", vocal_target)
            results_queue = mp.Queue()
            nemo_process = mp.Process(
                target=self.diarize_parallel,
                args=(
                    torch.from_numpy(audio_waveform).unsqueeze(0),
                    self.diarization_model,
                    results_queue,
                ),
            )
            nemo_process.start()

        # Transcribe the audio file
        whisper_model = faster_whisper.WhisperModel(
            self.model_name, device=self.device, compute_type=mtypes[self.device]
        )
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

        suppress_tokens = (
            find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
            if self.suppress_numerals
            else [-1]
        )

        if self.batch_size > 0:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                self.language,
                suppress_tokens=suppress_tokens,
                batch_size=self.batch_size,
            )
        else:
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform,
                self.language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )

        full_transcript = "".join(segment.text for segment in transcript_segments)

        # Clear gpu vram
        del whisper_model, whisper_pipeline
        torch.cuda.empty_cache()

        # Forced Alignment
        alignment_model, alignment_tokenizer = load_alignment_model(
            self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        emissions, stride = generate_emissions(
            alignment_model,
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device),
            batch_size=self.batch_size,
        )

        del alignment_model
        torch.cuda.empty_cache()

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[info.language],
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        if self.parallel:
            nemo_process.join()  # type: ignore
            if results_queue.empty():  # type: ignore
                raise RuntimeError("Diarization process did not return any results.")

            speaker_ts = results_queue.get_nowait()  # type: ignore

        else:
            speaker_ts = self.diarization_model.diarize(
                torch.from_numpy(audio_waveform).unsqueeze(0)
            )

        # del self.diarization_model
        torch.cuda.empty_cache()

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        if info.language in punct_model_langs:
            # Restoring punctuation in the transcript to help realign the sentences
            punct_model = PunctuationModel(model="kredor/punctuate-all")

            words_list = list(map(lambda x: x["word"], wsm))

            labled_words = punct_model.predict(words_list, chunk_size=230)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            # We don't want to punctuate U.S.A. with a period. Right?
            is_acronym = lambda x: re.fullmatch(
                r"\b(?:[a-zA-Z]\.){2,}", x
            )  # noqa: E731

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word

        else:
            logging.warning(
                f"Punctuation restoration is not available for {info.language} language."
                " Using the original punctuation."
            )

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        with open(f"{os.path.splitext(audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)

        with open(
            f"{os.path.splitext(audio)[0]}.srt", "w", encoding="utf-8-sig"
        ) as srt:
            write_srt(ssm, srt)

        cleanup(temp_path)

        end_time = time.time() - start_time
        logging.info(
            f"Diarization successfully completed in {format_timestamp(end_time)}"
        )
