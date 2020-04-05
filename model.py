# doesn't fit for cpu
import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text, text_lengths):
        '''
            text:[len, batch], text_lengths:[batch]
            embedded:[len, batch=32, emb_size=100]
            output:[sentlen, batch, len=512(2 direction of last layer)], output_lengths:[batch]
            hidden:[2*2, batch, hidden_size=256], cell:the same
            hidden(after trans):[32, 512] concat two dim
            res:[32,1]
        '''
        batch = text.shape[1]
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # average
        aver = output.cuda().sum(dim=0)/output_lengths.cuda().view(batch, 1) # batch, 512 / batch
        res = self.fc(aver)
        '''
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        res = self.fc(hidden)
        '''
        return res

class attentionRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, word_att_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.word_attention = nn.Linear(2 * hidden_dim, word_att_dim)
        self.word_context_vector = nn.Linear(word_att_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, text, text_lengths):
        batch = text.shape[1]
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        att_w = self.word_attention(packed_output.data.cuda())  # (sum_words, att_size)
        att_w = torch.tanh(att_w)  # (sum_words, att_size)
        att_w = self.word_context_vector(att_w).squeeze(1)  # (sum_words)
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)
        att_w, _ = nn.utils.rnn.pad_packed_sequence(nn.utils.rnn.PackedSequence(data=att_w,
                                                      batch_sizes=packed_output.batch_sizes,
                                                      sorted_indices=packed_output.sorted_indices,
                                                      unsorted_indices=packed_output.unsorted_indices),
                                       batch_first=True)  # (batch, max(words_per_sentence))

        word_alphas = att_w.cuda() / torch.sum(att_w.cuda(), dim=1, keepdim=True)
        doc, doc_len = nn.utils.rnn.pad_packed_sequence(packed_output,
                                           batch_first=True)  # (batch, max(words_per_sentence), 2 * word_rnn_size)
        doc = doc.cuda()
        doc = doc * word_alphas.unsqueeze(2)  # (batch, max(words_per_sentence), 2 * word_rnn_size)
        doc = doc.sum(dim=1)
        res = self.fc(doc)
        '''
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        res = self.fc(hidden)
        '''
        return res