#!/bin/bash

[ -d data ] || mkdir data

wget "https://www.dropbox.com/s/sg4yxvzg5rysruy/wiki.simple.vec.zip?dl=0" -O $PWD"/data/wiki.simple.vec.zip"
unzip -q "data/wiki.simple.vec.zip" -d "data"
rm "data/wiki.simple.vec.zip"
