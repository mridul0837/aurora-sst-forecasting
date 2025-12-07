#!/bin/bash
#SBATCH --time=04-00:00:00
#SBATCH --account=def-weimin
#SBATCH --mem=288000M
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=12      # CPU cores/threads
# Request a single task, which will utilize the resources
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out   # standard output
#SBATCH --error=%x-%j.err    # standard error
#SBATCH --job-name=finetune-sst
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Load needed modules
module load mpi4py

# Activate your enviroment
source ~/envs/py311/bin/activate

# Start TensorBoard
tensorboard --logdir="/scratch/${USER}/runs/sst_finetune" \
               --host 0.0.0.0 \
               --load_fast false &
echo "TensorBoard started"
# Print node name and TensorBoard connection info
echo "Node name: $(hostname)"
echo "TensorBoard URL: http://$(hostname):6006"
echo "To access TensorBoard, run this on your local machine:"
echo "ssh -N -L 6006:$(hostname):6006 $USER@<cluster>.computecanada.ca"
echo "Then open http://localhost:6006 in your browser"

# Run finetuning script
python /home/$USER/projects/MLP\ Decoder/Finetuning-SST/finetune_auroraLite_sst.py 
    

echo "Job finished on $(date)"