#!/bin/bash

trap "exit" INT

sets=(
  #    0  1  2 3       4      5    6     7
  # code x1 x2 y y_scale header path force

  'snli 0 1 2 1.0 False data/train_snli.txt False'
  # 'sick2014 1 2 3 5.0 True data/sts/sick2014/SICK_train.txt False'
)

archs=(
  #    0       1         2       3         4     5
  # tied 1_nodes 1_dropout 2_nodes 2_dropout force

  'False 50,50,50 1.0 50,50,50 1.0 False'
  'False 150,100,50 1.0 150,100,50 1.0 False'
  'False 500,250,100 1.0 500,250,100 1.0 False'

  'True 50,50,50 1.0 50,50,50 1.0 False'
  'True 150,100,50 1.0 150,100,50 1.0 False'
  'True 500,250,100 1.0 500,250,100 1.0 False'

  'False 150,100,50 0.8 150,100,50 0.8 False'
  'True 150,100,50 0.8 150,100,50 0.8 False'
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

    output_code="1n-${line_i[1]}_1d-${line_i[2]}_2n-${line_i[3]}_2d-${line_i[4]}"
    output_code=$([ ${line_i} == "True" ] && echo ${line_j[0]}_tied_${output_code} || echo ${line_j[0]}_${output_code})

    output_dirpath="$PWD/runs/$output_code"

    if [ -d $output_dirpath ]; then
      echo "Results for $output_code already exist..."

      if [[ ! (${line_i[5]} == "True" && ${line_j[7]} == "True") ]]; then
        echo "Skipping training for $output_code."
        continue
      fi
    fi

    echo "Training $output_dirpath..."

    python train.py \
      --output_dirpath=$output_dirpath \
      --word2vec_model=data/wiki.simple.vec \
      --max_iterations=$max_iterations \
      --training_filepath=${line_j[6]} \
      --x1_position=${line_j[1]} \
      --x2_position=${line_j[2]} \
      --y_position=${line_j[3]} \
      --y_scale=${line_j[4]} \
      --header=${line_j[5]} \
      --tied=${line_i[0]} \
      --side1_nodes=${line_i[1]} \
      --side1_dropout=${line_i[2]} \
      --side2_nodes=${line_i[3]} \
      --side2_dropout=${line_i[4]} #&> $PWD/runs/$output_code.log

    # find latest checkpoint
    iteration=$(ls $output_dirpath/checkpoints/ | grep '^model\-[0-9][0-9]*.meta$' | sort -r | head -1 | sed 's/^model-\([0-9]*\)\.meta$/\1/')
    ./tools/evaluate.sh "$output_code" "$iteration"
  done
done
