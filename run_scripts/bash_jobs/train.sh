#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:0:0
#SBATCH --mail-user=Gr33nMayhem@gmail.com
#SBATCH --mail-type=ALL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5
done
cd ~/projects/def-plago/akhaked/SyntheticIMUGeneration/run_scripts
module purge
module load python/3.11 scipy-stack
source ~/py311/bin/activate
python train_encoder_decoder.py --reference_point "$1" --window_size $2 --step_size $3 --imu_position $4 --learning_rate $5 --batch_size $6 --num_epochs $7 --loss_function "$8"
