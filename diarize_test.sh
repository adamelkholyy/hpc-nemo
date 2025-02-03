#!/bin/bash
#SBATCH --export=ALL      				 	# export all environment variables to the batch job.
#SBATCH --partition gpu      					# submit to the gpu queue
#SBATCH -D /lustre/projects/Research_Project-T116269/nemo 	# set working directory to .
#SBATCH --mail-type=ALL						# send email at job completion
#SBATCH --mail-user=a.el-kholy@exeter.ac.uk 			# email address
#SBATCH --time=03:00:00    					# maximum wall time for the job.
#SBATCH --account Research_Project-T116269    			# research project to submit under. 

#SBATCH --nodes=1                                  		# specify number of nodes.
#SBATCH --ntasks-per-node=16        				# specify number of processors per node
#SBATCH --gres=gpu:1						# num gpus	
#SBATCH --mem=4G						# requested memory	

#SBATCH --output=diarize_test.out   				# submit script's standard-out
#SBATCH --error=diarize_test.err    				# submit script's standard-error
#SBATCH --job-name=diarize_test


## load modules
echo Loading modules....

module use /lustre/shared/easybuild/modules/all
module use PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load Perl/5.36.1-GCCcore-12.3.0

## module list >> slurm_modules.txt
## pip freeze >> requirements.txt

cd /lustre/projects/Research_Project-T116269/nemo
echo Modules loaded.



## execute python script
start_time=$(date +%s)
echo Executing Python script...

## python cuda_test.py
## python diarize.py -a audio/audio.mp3
## python diarize.py -a audio/AutHERTS01.mp3
## python diarize_parallel.py -a audio/AutHERTS01.mp3 --whisper-model large-v3 --language "en" --no-stem --suppress_numerals

python diarize_batch.py -f audio --whisper-model large-v3 --language en


echo Python script executed successfully.
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
echo End of script - job finished successfully.



## print start date and time
## echo =================================================>
## echo Job started on
## date -u
## current_date=$(date)

## output timing info
## echo =================================================>
## echo Job was started on
## echo "$current_date"
## echo Job ended on
## date -u

## check cuda has loaded properly
## echo =================================================>
## echo Verifying cuda installation...
## echo nvidia-smi
## nvidia-smi
## echo nvcc --version
## nvcc --version
## echo lspci
## lspci | grep -i nvidia

## echo modinfo nvidia
## modinfo nvidia
