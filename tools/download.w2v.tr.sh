#!/bin/bash

[ -d data ] || mkdir data

wget "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.tr.vec" -O $PWD"/data/wiki.tr.vec"
