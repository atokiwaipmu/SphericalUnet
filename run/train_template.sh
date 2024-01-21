#!/bin/bash
#SBATCH --job-name=fname
#SBATCH --account=akira.tokiwa
#SBATCH --output=/gpfs02/work/akira.tokiwa/gpgpu/Github/SphericalUnet/log/fname.out  
#SBATCH --error=/gpfs02/work/akira.tokiwa/gpgpu/Github/SphericalUnet/log/fname.err  
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=akira.tokiwa@ipmu.jp
#SBATCH --mail-type=END,FAIL

source /home/akira.tokiwa/.bashrc

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=_device # where X is the GPU id of an available GPU

conda activate pylit

cd /gpfs02/work/akira.tokiwa/gpgpu/Github/SphericalUnet
python -m run.train --model _model --target _target --transform_type _trans --order _order --n_maps _nmap --batch_size _batch