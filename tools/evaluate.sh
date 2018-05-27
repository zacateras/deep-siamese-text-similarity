#!/bin/bash

display_usage() {
  echo -e "Usage: tools/evaluate.sh snli_1n-50-50-50_1d-1_2n-50-50-50_2d-1 1000" 
} 

if [ $# -ne 2 ]; then
  display_usage
  exit 1
fi

evals=(
  #    0  1  2 3       4    5
  # code x1 x2 y y_scale path

  # SemEval 2017
  'semeval2017-sts-track5 1 2 0 5.0 data/sts/semeval-sts/2017/track5.en-en.tsv'

  # SemEval 2016
  'semeval2016-sts-answer-answer 1 2 0 5.0 data/sts/semeval-sts/2016/answer-answer.test.tsv'
  'semeval2016-sts-headlines 1 2 0 5.0 data/sts/semeval-sts/2016/headlines.test.tsv'
  'semeval2016-sts-plagiarism 1 2 0 5.0 data/sts/semeval-sts/2016/plagiarism.test.tsv'
  'semeval2016-sts-postediting 1 2 0 5.0 data/sts/semeval-sts/2016/postediting.test.tsv'
  'semeval2016-sts-question-question 1 2 0 5.0 data/sts/semeval-sts/2016/question-question.test.tsv'

  # SemEval 2015
  'semeval2015-sts-answers-forums 1 2 0 5.0 data/sts/semeval-sts/2015/answers-forums.test.tsv'
  'semeval2015-sts-answers-students 1 2 0 5.0 data/sts/semeval-sts/2015/answers-students.test.tsv'
  'semeval2015-sts-belief 1 2 0 5.0 data/sts/semeval-sts/2015/belief.test.tsv'
  'semeval2015-sts-headlines 1 2 0 5.0 data/sts/semeval-sts/2015/headlines.test.tsv'
  'semeval2015-sts-images 1 2 0 5.0 data/sts/semeval-sts/2015/images.test.tsv'

  # SICK 2014
  'sick2014 1 2 3 5.0 data/sts/sick2014/SICK_test_annotated.txt'
)

run=$1
iteration=$2

vocab_filepath="runs/${run}/checkpoints/vocab"
model_filepath="runs/${run}/checkpoints/model-${iteration}"
evaluation_log_filepath="evaluation.log.txt"
log_filepath="runs/${run}/evaluation.tsv"

if [ -e $log_filepath ]; then
  rm $log_filepath
fi

if [ -e $evaluation_log_filepath ]; then
  rm $evaluation_log_filepath
fi

for i in "${evals[@]}"
do
  line=($i)

  echo "Evaluating ${line[0]}..."

  python eval.py \
    --vocab_filepath=$vocab_filepath \
    --model=$model_filepath \
    --log_filepath=$log_filepath \
    --log_event=${line[0]} \
    --eval_filepath=${line[5]} \
    --y_scale=${line[4]} \
    --y_position=${line[3]} \
    --x1_position=${line[1]} \
    --x2_position=${line[2]} &> $evaluation_log_filepath
done

cat $log_filepath
