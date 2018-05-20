#!/bin/bash

[ -d data ] || mkdir data

wget "https://www.dropbox.com/s/yz1lsnvholdj0nn/sts.zip?dl=0" -O $PWD"/data/sts.zip"
unzip -q "data/sts.zip" -d "data"
rm "data/sts.zip"
