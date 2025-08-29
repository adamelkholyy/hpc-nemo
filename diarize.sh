
cd /lustre/projects/Research_Project-T116269/nemo

echo Loading slurm modules
module use /lustre/shared/easybuild/modules/all
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load Perl/5.36.1-GCCcore-12.3.0
# module load cuDNN/8.4.1.50-CUDA-11.7.0  

# module load Python/3.11.3-GCCcore-12.3.0                                                   
# module load CMake/3.26.3-GCCcore-12.3.0
# module load GCCcore/12.3.0
# module load CUDA/12.2.2

echo Activating venv
source /lustre/projects/Research_Project-T116269/nemo/.venv/bin/activate

echo Executing Python script


# python diarize_parallel.py -a /lustre/projects/Research_Project-T116269/cobalt-audio-mp3/B118034_102_s03.mp3 --whisper-model large-v3 --language "en" --no-stem --suppress_numerals 
python diarize.py -a /lustre/projects/Research_Project-T116269/cobalt-audio-mp3/B118034_102_s03.mp3 --whisper-model large-v3 --language "en" --no-stem --suppress_numerals 



echo Script completed successfully