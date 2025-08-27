
cd /lustre/projects/Research_Project-T116269/nemo

echo Loading slurm modules
module use /lustre/shared/easybuild/modules/all
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load Perl/5.36.1-GCCcore-12.3.0

echo Activating venv
source ./.venv/bin/activate

echo Executing Python script

## python diarize_parallel.py -a audio/AutHERTS01.mp3 --whisper-model large-v3 --language "en" --no-stem --suppress_numerals
## python diarize.py -a /lustre/projects/Research_Project-T116269/cobalt-audio-mp3/B206020_203_s11.mp3 --whisper-model large-v3 --language "en" --no-stem --suppress_numerals
## python diarize_batch.py -f "/lustre/projects/Research_Project-T116269/cobalt-audio-mp3" --whisper-model large-v3 --language en
## python anonymise_transcript.py --folder "audio/verity/" --out "audio/verity/"
# python diarize_batch.py -a /lustre/projects/Research_Project-T116269/cobalt-audio-mp3/B127133_102_s03.mp3 --whisper-model large-v3 --language en --no-stem --suppress_numerals
python diarize.py -a /lustre/projects/Research_Project-T116269/cobalt-audio-mp3/B111008_102_s01.mp3 --whisper-model large-v3 --language "en" --no-stem --suppress_numerals --num-speakers 2



echo Script completed successfully