#!/bin/bash
#SBATCH --export=ALL       # export all environment variables to the batch job.
#SBATCH --partition volta       # submit to the gpu queue
#SBATCH -D /lustre/projects/Research_Project-T116269/nemo # set working directory to . # research project to submit under
#SBATCH -A Research_Project-T116269
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
echo Job started on:
date -u
current_date=$(date)
echo ==================================================================================
echo -n "This script is running on "
Hostname
echo ==================================================================================

## run script
module load Anaconda3/2023.07-2
cd /lustre/projects/Research_Project-T116269/nemo

conda activate nemo
python diarize.py -a audio/AutHERTS01.mp3 --no-stem --whisper-model large-v3 --language English

echo ==================================================================================
## print end date and time
echo Job complete
echo Job started on:
echo "$current_date"
echo Job ended on:
date -u
echo ==================================================================================
echo Job finished succesfully.
