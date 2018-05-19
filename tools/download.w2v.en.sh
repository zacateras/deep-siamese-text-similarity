[ -d data ] || mkdir data

wget "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip" -O $PWD"/data/wiki.en.zip"
unzip -q "data/wiki.en.zip" -d "data"
rm "data/wiki.en.zip"
