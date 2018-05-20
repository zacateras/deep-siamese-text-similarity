# Deep LSTM siamese network for text similarity

It is a tensorflow based implementation of deep siamese LSTM network to capture sentence similarity using word embeddings.

![siamese lstm sentence similarity](https://cloud.githubusercontent.com/assets/9861437/20479493/6ea8ad12-b004-11e6-89e4-53d4d354d32e.png)

For the task mentioned above it uses a multilayer siamese LSTM network and euclidian distance based contrastive loss to learn input pair similairty.

# Capabilities
Given adequate training pairs, this model can learn Semantic as well as structural similarity. For eg:

- He is smart = He is a wise man.
- Someone is travelling countryside = He is travelling to a village.
- She is cooking a dessert = Pudding is being cooked.
- Microsoft to acquire Linkedin ≠ Linkedin to acquire microsoft

(More examples Ref: semEval dataset)

For Sentences, the model uses **pre-trained word embeddings** to identify semantic similarities.

Categories of pairs, it can learn as similar:
- Annotations
- Abbreviations
- Extra words
- Similar semantics
- Typos
- Compositions
- Summaries

# Training Data
	- A sample set of learning sentence semantic similarity can be downloaded from:

	"train_snli.txt" : https://drive.google.com/open?id=1itu7IreU_SyUSdmTWydniGxW-JEGTjrv

	This data is generated using SNLI project : 
	> https://nlp.stanford.edu/projects/snli/

	 - word embeddings: any set of pre-trained word embeddings can be utilized in this project. For our testing we had used fastText 	simple english embeddings from https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

	alternate download location for "wiki.simple.vec" is : https://drive.google.com/open?id=1u79f3d2PkmePzyKgubkbxOjeaZCJgCrt

# Environment
- numpy 1.14.3
- tensorflow 1.8.0
- gensim 1.0.1
- nltk 3.2.2

# How to run
### Training
```
$ python train.py --training_filepath data/sts/semeval-sts/2012/MSRpar.train.tsv --word2vec_model data/wiki.simple.vec --side1_dropout 0.8 --side2_dropout 1.0 --side1_hidden_units 75 --side2_hidden_units 25
```
### Evaluation
```
$ python eval.py --eval_filepath data/sts/semeval-sts/2012/MSRpar.test.tsv --vocab_filepath runs/1526819186/checkpoints/vocab --model runs/1526819186/checkpoints/model-2000
```

# Performance
**TODO**

# References
1. [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)
