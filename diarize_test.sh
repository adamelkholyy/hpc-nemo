#!/bin/bash
#SBATCH --export=ALL       # export all environment variables to the batch job.
#SBATCH --partition gpu       # submit to the gpu queue
#SBATCH -D /lustre/projects/Research_Project-T116269/nemo # set working directory to .
#SBATCH --mail-type=ALL # send email at job completion
#SBATCH --mail-user=a.el-kholy@exeter.ac.uk # email address
#SBATCH --time=03:00:00    # Maximum wall time for the job.
#SBATCH --account Research_Project-T116269    # research project to submit under. 
#SBATCH --nodes=1                                  # specify number of nodes.
#SBATCH --ntasks-per-node=16        # specify number of processors per node
#SBATCH --mem-per-cpu=1024         # MB memory requested per cpu-core (task)
#SBATCH --output=diarize_test.out   # submit script's standard-out
#SBATCH --error=diarize_test.err    # submit script's standard-error
#SBATCH --job-name=diarize_test

## print start date and time
echo ==================================================================================
echo Job started on
date -u
current_date=$(date)


## load modules and initialise conda env
echo ==================================================================================
echo Loading modules....

module load FFmpeg/4.2.2-GCCcore-9.3.0
module load Anaconda3/2023.07-2
module load nvidia-cuda/12.1.1
module load libdrm/2.4.115-GCCcore-12.3.0
module list >> diarize_test.out
conda activate nemo
conda info --env
cd /lustre/projects/Research_Project-T116269/nemo

echo Modules loaded. Conda environment initialised successfully

## nvidia driver version 560.35.03
## check cuda has loaded properly
echo =================================================================================
echo Verifying cuda installation...
echo nvidia-smi
nvidia-smi
echo nvcc --version
nvcc --version
echo lspci
lspci | grep -i nvidia


echo modinfo nvidia
modinfo nvidia
## execute python script
echo ==================================================================================
echo Executing Python script...
python diarize.py -a audio/AutHERTS01.mp3 --no-stem --whisper-model large-v3 --language en
echo Python script executed successfully


## output timing info 
echo ==================================================================================
echo Job was started on
echo "$current_date"
echo Job ended on
date -u
echo ==================================================================================
echo End of script - job finished successfully
