import spacy
import torch
import pickle
from torchtext import data
from model import RNN
device = 'cuda'
def predict_sentiment(model, sentence, nlp, TEXT):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

modelname = 'attention_b16'
with open(modelname + 'TEXT.pkl', 'rb') as f:
    TEXT =  pickle.load(f)
nlp = spacy.load('en')
model = torch.load(modelname+'total_model.pt')
print(predict_sentiment(model, "This film is terrible", nlp, TEXT))
print(predict_sentiment(model, "This film is great", nlp, TEXT))

