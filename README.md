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
Download `aclImdb_v1.tar.gz` and unzip it in root directory. Run `preprocess.py` to get jsonlines files of `train.jsonl`, `test.jsonl`. You should also download `glove.6B.100d' before training.

## Training
Run `train.py`, before that you can set the variable 'modelname' to specify the version of model you are training. 
You can choose Three types of model:
1. two layers of Bi-LSTM with h_n as the encoding of the text.
2. two layers of Bi-LSTM with element-wise average h_t as the encoding of the text.
3. two layers of Bi-LSTM with word attention

For the first and second model, you should set in `train.py`

```python
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
            BIDIRECTIONAL, DROPOUT, PAD_IDX)
```

and choose the comment line in class RNN in `model.py` to decide which one you are training.


To use word attention model, just set in `train.py`

```python
model = attentionRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, ATT_DIM, N_LAYERS,
                    BIDIRECTIONAL, DROPOUT, PAD_IDX)
```

## Load Model and Test on specific sentence
run `test.py` after set 'modelname' in the file

## Experiment Result
Word Attention:
Epoch: 01 | Epoch Time: 3m 25s
        Train Loss: 0.454 | Train Acc: 78.39%
         Val. Loss: 0.288 |  Val. Acc: 88.66%
Epoch: 02 | Epoch Time: 3m 33s
        Train Loss: 0.273 | Train Acc: 88.99%
         Val. Loss: 0.258 |  Val. Acc: 89.99%
Epoch: 03 | Epoch Time: 3m 35s
        Train Loss: 0.211 | Train Acc: 91.84%
         Val. Loss: 0.241 |  Val. Acc: 90.72%
Epoch: 04 | Epoch Time: 3m 30s
        Train Loss: 0.167 | Train Acc: 93.81%
         Val. Loss: 0.268 |  Val. Acc: 90.46%
Epoch: 05 | Epoch Time: 3m 33s
        Train Loss: 0.133 | Train Acc: 95.28%
         Val. Loss: 0.321 |  Val. Acc: 90.30%
Test Loss: 0.250 | Test Acc: 90.01%