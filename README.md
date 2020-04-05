# Sentiment Analysis
2020 Spring Machine Learning project
haoshibo@pku.edu.cn

## Introduction
An implement of sentiment analysis on `IMDB dataset` with bi-LSTM and word attention.

## Requirements
+ python 3.6
+ pytorch 1.4
+ torchtext 0.5
+ spacy 2.2.4

## Preprocess
Download `aclImdb_v1.tar.gz` and unzip it in root directory, run `preprocess.py` to get jsonlines files of `train.jsonl`, `test.jsonl`. You should also download `glove.6B.100d' before training.

## Training
Run `train.py`, before that you can set the variable 'modelname' to specify the version of model you are training. 
You can choose Three types of model:
1. two layers of Bi-LSTM with h_n as the encoding of the text.
2. two layers of Bi-LSTM with element-wise average h_t as the encoding of the text.
3. two layers of Bi-LSTM with word attention

For the first and second model, you should specify this sentence in `train.py` 

```python
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
            BIDIRECTIONAL, DROPOUT, PAD_IDX)
```

and choose the comment line in class RNN in `model.py` to decide which one you are training.

And to use word attention model, just set in `train.py`

```python
model = attentionRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, ATT_DIM, N_LAYERS,
                    BIDIRECTIONAL, DROPOUT, PAD_IDX)
```