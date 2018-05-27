#!/bin/bash

sets=(
  #    0  1  2 3       4    5
  # code x1 x2 y y_scale path

  'snli 0 1 2 1.0 data/train_snli.txt'
  'sick2014 1 2 3 5.0 data/sts/sick2014/SICK_train.txt'
)

archs=(
  #    0       1         2       3         4
  # tied 1_nodes 1_dropout 2_nodes 2_dropout

  'False 50,50,50 1.0 50,50,50 1.0'
  'False 150,100,50 5.0 150,100,50 5.0'
)

max_iterations=50000

if [ ! -e "$PWD/runs" ]; then
  mkdir "$PWD/runs"
fi

for i in "${archs[@]}"
do
  line_i=($i)

  for j in "${sets[@]}"
  do
    line_j=($j)

    output_code="${line_j[0]}_1n-${line_i[1]}_1d-${line_i[2]}_2n-${line_i[3]}_2d-${line_i[4]}"
    output_dirpath="$PWD/runs/$output_code"

    if [ -e output_dirpath ]; then
      continue
    fi

    echo "Training $output_dirpath..."

    python train.py \
      --output_dirpath=$output_dirpath \
      --word2vec_model=data/wiki.simple.vec \
      --max_iterations=$max_iterations \
      --training_filepath=${line_j[5]} \
      --x1_position=${line_j[1]} \
      --x2_position=${line_j[2]} \
      --y_position=${line_j[3]} \
      --y_scale=${line_j[4]} \
      --tied=${line_i[0]} \
      --side1_nodes=${line_i[1]} \
      --side1_dropout=${line_i[2]} \
      --side2_nodes=${line_i[3]} \
      --side2_dropout=${line_i[4]} &> "$PWD/runs/$output_code.log"
  done
done
