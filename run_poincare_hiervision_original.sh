#!/bin/bash

#hierarchy_names=("Madverse" "Matador" "moments_in_time" "BioTrove-LifeStages" "marine-tree")
#dims=(130 40 70 10 40)
hierarchy_names=("NABirds" "COCO10k" "EgoObjects" "Imagenet21k_v1" "OpenLoris" "PascalVOC" "Visual_Genome")
dims=(70 20 40 520 20 20 130)

# Constant arguments
other_args="-lr 1 -epochs 10000 -negs 50 -burnin 20 -ndproc 4 -model distance -manifold poincare -batchsize 50 -eval_each 50 -fresh -sparse -train_threads 1 -gpu -1 -debug -dampening 0.75 -burnin_multiplier 0.01 -neg_multiplier 0.1 -lr_type constant -dampening 1.0"

# Loop over the datasets and run the Python script with the dynamic argument
for i in "${!hierarchy_names[@]}"; do
    hierarchy_name="${hierarchy_names[$i]}"
    dim_arg="-dim ${dims[$i]}"
    dset_arg="-dset ./hierarchies/${hierarchy_name}/${hierarchy_name}_adjacency.csv"
    checkpoint_arg="-checkpoint ./hierarchies/${hierarchy_name}/${hierarchy_name}.bin"
    echo "Running Python script with argument: $dim_arg $dset_arg $checkpoint_arg $other_args"
    python3 ./embed.py $dim_arg $dset_arg $checkpoint_arg $other_args
    echo "______________________________________________________________"

done

