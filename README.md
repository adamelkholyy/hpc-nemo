<h1 align="center">Speaker Diarization Using OpenAI Whisper HPC Fork</h1>

This project is a fork of @MahmoudAshraf97's [whisper diarization tool](https://github.com/MahmoudAshraf97/whisper-diarization), modified for use on the University of Exeter ISCA's High Performance Computing Server. All credit goes to the original authors of this incredible project. A brief changelog is detailed below listing my edits for compatability on Exeter's ISCA supercomputer.  

HPC Compatability changelog  
- Added transcript anonymisation using presidio (```anonymise_transcript.py```)
- Added batch diarizations from folder (```diarize_batch.py```)
- Added SBATCH scripts (```diarize_test.sh```) to execute on GPU nodes via slurm
- Added errors and logging output for HPC GPUs
- Edited ```requirements.txt``` for compatability with ISCA modules
- Installed the following onto ISCA (see ```load_modules.sh```):
    - FFMpeg
    - Perl
    - C++ build tools
- Fixed error in Nemo module: ```.SUBKILL``` changed to ```.SUBTERM```
- Added detailed logging for transcriptions, including ETA and average times
- 
    
Last updated: 23/07/25

[1] _Whisper Diarization: Speaker Diarization Using OpenAI Whisper_, Mahmoud Ashraf, 2024, https://github.com/MahmoudAshraf97/whisper-diarization

