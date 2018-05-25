#!/bin/bash

[ -d data ] || mkdir data

wget "https://www.dropbox.com/s/7u56vtww6r6t37p/train_snli.txt.zip?dl=0" -O $PWD"/data/train_snli.txt.zip"
unzip -q "data/train_snli.txt.zip" -d "data"
rm "data/train_snli.txt.zip"
