import torch
from torchtext import data
import random
import torch.optim as optim
import torch
from model import RNN, attentionRNN
from ulti import evaluate, count_parameters, binary_accuracy, epoch_time
import torch.nn as nn
import time
import pickle

SEED = 1234
N_EPOCHS = 5
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 16
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
ATT_DIM = 128
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
model_name = 'attention_b16'

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        #print(text.shape) # len(not sure), 32
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main():
    
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))

    TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
    LABEL = data.LabelField(dtype = torch.float)
    
    print("loading dataset...")
    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
    train_data, test_data = data.TabularDataset.splits(
        path = '', train = 'train.jsonl', test = 'test.jsonl', 
        format = 'json', fields = fields)
    train_data, valid_data = train_data.split(random_state = random.seed(SEED))
    print("finished.")
    TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE,
        vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size = BATCH_SIZE,
        sort = False, device = device)
    with open(model_name+'TEXT.pkl', 'wb') as f:
        pickle.dump(TEXT, f)
    INPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    print("input dim, pad_idx = ", INPUT_DIM, PAD_IDX)
    '''
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS,
                BIDIRECTIONAL, DROPOUT, PAD_IDX)
    '''
    model = attentionRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, ATT_DIM, N_LAYERS,
                BIDIRECTIONAL, DROPOUT, PAD_IDX)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    pretrained_embeddings = TEXT.vocab.vectors
    print(pretrained_embeddings.shape)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    print("word embedding:", model.embedding.weight.data)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name+'param.pt')
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    model.load_state_dict(torch.load(model_name+'param.pt'))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
    torch.save(model, model_name+'total_model.pt')
if __name__ == '__main__':
    main()





