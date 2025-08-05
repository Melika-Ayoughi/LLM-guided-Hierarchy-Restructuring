#!/bin/bash
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -t 10:00:00
#SBATCH -o /home/mayoughi/output/Imagenet21k_v1_optimized_hadamard/log.out
#SBATCH --mail-type=END
#SBATCH --mail-user=m.ayoughi@uva.nl

module load 2022
module load CUDA/11.6.0
module load Graphviz/5.0.0-GCCcore-11.3.0
module list

cd poincare/

python automatic_ontology_cleanup.py \
--dataset "Imagenet21k_v1" \
--embedding_method "hadamard" \
--optimized "optimized"