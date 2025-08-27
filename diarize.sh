#!/bin/bash
#SBATCH --export=ALL      				 	# export all environment variables to the batch job.
#SBATCH --partition gpu      					# submit to the gpu queue
#SBATCH -D /lustre/projects/Research_Project-T116269/nemo	# set working directory to .
#SBATCH --mail-type=ALL						# send email at job completion
#SBATCH --mail-user=a.el-kholy@exeter.ac.uk 			# email address
#SBATCH --time=4-23:00:00    					# maximum wall time for the job.
#SBATCH --account Research_Project-T116269    			# research project to submit under. 
#SBATCH --priority=5000

#SBATCH --ntasks-per-node=16        				# specify number of processors per node
#SBATCH --gres=gpu:1						# num gpus	
#SBATCH --mem=4G						# requested memory	

#SBATCH --output=logs/diarize.out   					# submit script's standard-out
#SBATCH --error=logs/diarize.err    					# submit script's standard-error
#SBATCH --job-name=diarize


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
python diarize.py -a /lustre/projects/Research_Project-T116269/cobalt-audio-mp3/B206020_203_s11.mp3 --whisper-model large-v3 --language "en" --no-stem --suppress_numerals

echo Script completed successfully


## Check cuda has loaded properly
## echo =================================================
## echo Verifying cuda installation...
## echo nvidia-smi
## nvidia-smi
## echo nvcc --version
## nvcc --version
## echo lspci
## lspci | grep -i nvidia
## echo modinfo nvidia
## modinfo nvidia
