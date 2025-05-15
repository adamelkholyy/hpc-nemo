<h1 align="center">Speaker Diarization Using OpenAI Whisper HPC Fork</h1>

This project is a fork of @MahmoudAshraf97's [whisper diarization tool](https://github.com/MahmoudAshraf97/whisper-diarization), modified for use on the University of Exeter ISCA's High Performance Computing Server. All credit goes to the original authors of this incredible project. A brief changelog is detailed below listing my edits for compatability on Exeter's ISCA supercomputer.  

HPC Compatability changelog  
- Added ollama support for LLM testing (```install_ollama.sh```)
- Added transcript anonymisation using presidio (```anonymise_transcript.py```)
- Added batch diarizations from folder (```diarize_batch.py```)
- Added Word Error Rate calculations   
- Added SBATCH scripts (```diarize_test.sh```) to execute on GPU nodes via slurm
- Added errors and logging output for HPC GPUs
- Edited ```requirements.txt``` for compatability with ISCA modules
- Included FFMpeg, Perl, and C++ build tools in ```load_modules.sh``` 
- Nemo module: ```.SUBKILL``` changed to ```.SUBTERM``` to avoid errors
    
Last updated: 17/04/25

[1] _Whisper Diarization: Speaker Diarization Using OpenAI Whisper_, Mahmoud Ashraf, 2024, https://github.com/MahmoudAshraf97/whisper-diarization

