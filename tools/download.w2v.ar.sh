#!/bin/bash

[ -d data ] || mkdir data

wget "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ar.vec" -O $PWD"/data/wiki.ar.vec"
