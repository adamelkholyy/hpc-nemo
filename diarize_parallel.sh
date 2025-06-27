#!/bin/bash
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH -D /lustre/projects/Research_Project-T116269/nemo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.el-kholy@exeter.ac.uk
#SBATCH --time=24:00:00
#SBATCH --account=Research_Project-T116269

#SBATCH --ntasks=1                     # One task per array job
#SBATCH --gres=gpu:1                   # Request 1 GPU per task
#SBATCH --mem=4G

#SBATCH --array=0-800                 
#SBATCH --output=diarize_test.out
#SBATCH --error=diarize_test.err
#SBATCH --job-name=diarize_parallel

## load modules
echo Loading modules....

module use /lustre/shared/easybuild/modules/all
module use PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load Perl/5.36.1-GCCcore-12.3.0


cd /lustre/projects/Research_Project-T116269/nemo
echo Modules loaded.


FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" mp3_files.txt)


echo "Starting transcription on $FILE."
start_time=$(date +%s)

python nemo/diarize_parallel.py -a "/lustre/projects/Research_Project-T116269/$FILE" --whisper-model large-v3 --language en

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Finished transcription on $FILE in $elapsed seconds."
echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID completed."


## python anonymise_transcript.py --folder "audio/verity/" --out "audio/verity/"

