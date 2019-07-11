import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, pack_padded_sequence
from dropout import LockedDrop
from torchnlp.nn.weight_drop import WeightDrop
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import dataloader as dl

class Seq2Seq(nn.Module):
    def __init__(self, seq_len, hidden_size, char_length):
        super().__init__()
        self.encoder = Encoder(seq_len, hidden_size)
        self.decoder = Decoder(char_length)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.constant_(param.data, 0)

    def forward(self, inputs, labels, label_length):
        keys, values, length = self.encoder(inputs)
        output, attention_weight = self.decoder(keys, values, labels, label_length, length)
        return output, attention_weight


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.blstm = nn.LSTM(input_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.pblstm1 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.pblstm2 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.pblstm3 = pBLSTM(hidden_dim * 2, hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim * 2, 128)
        self.mlp2 = nn.Linear(hidden_dim * 2, 128)

    def forward(self, x):
        x, _ = self.blstm(x)
        x, _ = self.pblstm1(x)
        x, _ = self.pblstm2(x)
        x, _ = self.pblstm3(x)
        x, lens = rnn.pad_packed_sequence(x, batch_first=True)
        keys = self.mlp1(x)
        values = self.mlp2(x)
        return keys, values, lens


class pBLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_dim * 2, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.drop = LockedDrop()

    def forward(self, x):
        x, lens = rnn.pad_packed_sequence(x, batch_first=True)
        batch_size = x.size(0)
        seq_length = int(x.size(1) // 2)
        feature_dim = x.size(2) * 2
        x = x[:, :x.size(1) // 2 * 2, :]
        x = self.drop.forward(x, dropout = 0.05)
        x = x.contiguous().view(batch_size, seq_length, feature_dim)
        x = rnn.pack_padded_sequence(x, lens//2, batch_first=True)
        output, _ = self.blstm(x)
        return output, lens//2


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, keys, values, query, feats_len):
        keys = keys.permute(0, 2, 1)
        energy = torch.bmm(query, keys)
        energy = energy.squeeze(1)
        attention = F.softmax(energy, dim=1)
        attention_mask = torch.arange(attention.size(1)) < feats_len.long().view(-1, 1)
        attention_weight = attention_mask.float().to(self.device)*attention
        attention_weight = F.normalize(attention_weight, dim=1, p=1)
        context = torch.bmm(attention_weight.unsqueeze(1), values)
        context = context.squeeze(1)
        return context, attention_weight


class Decoder(nn.Module):
    def __init__(self, out_dim, lstm_dim=128):
        super(Decoder, self).__init__()
        self.lstm_dim = lstm_dim
        self.embedding = nn.Embedding(out_dim, lstm_dim)
        self.lstm1 = nn.LSTMCell(lstm_dim*2, lstm_dim)
        self.lstm2 = nn.LSTMCell(lstm_dim, lstm_dim)
        self.attention = Attention()
        self.fc1 = nn.Linear(lstm_dim, out_dim)
        self.fc2 = nn.Linear(lstm_dim, lstm_dim)
        self.fc3 = nn.Linear(lstm_dim, out_dim)
        self.fc4 = nn.Linear(out_dim * 2, out_dim)
        self.fc1.weight = self.embedding.weight
        self.fc1.bias.data.fill_(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, keys, values, label, label_len, input_length, TF = 0.7):
        result = []
        attention_weight = []
        context = torch.zeros((keys.size(0), self.lstm_dim)).to(self.device)
        start_char = torch.tensor(context.size(0))
        start_char = start_char.new_full((context.size(0), 1), 5).to(self.device)
        embed = self.embedding(start_char).squeeze(1)
        inp = torch.cat((embed, context), dim=1)
        if self.training:
            label = label.long()
            step = 0
            h_t, c_t = self.init_weights(context.size(0), self.lstm_dim)
            while step < label_len:
                h1, c1 = self.lstm1(inp, (h_t, c_t)) if step == 0 else self.lstm1(inp, (h1, c1))
                h2, c2 = self.lstm2(h1, (h_t, c_t)) if step == 0 else self.lstm2(h1, (h2, c2))
                query = self.fc2(h2)
                query = query.unsqueeze(1)
                context, attention = self.attention(keys, values, query, input_length)
                output = F.leaky_relu(self.fc1(h2), negative_slope=0.1)
                context_out = F.leaky_relu(self.fc3(context), negative_slope=0.1)
                pred = self.fc4(torch.cat((output, context_out), dim=1))
                if step > 0:
                    result.append(pred)
                attention_weight.append(attention)

                step = step + 1
                teacher_force = torch.rand(1) < TF
                if step < label_len:
                    if teacher_force:
                        inp = torch.cat((self.embedding(label[:, step-1]), context), dim=1)
                    else:
                        _, pred_labels = torch.max(pred, dim=1)
                        inp = torch.cat((self.embedding(pred_labels), context), dim=1)
            result = torch.stack(result)
            attention_weight = torch.stack(attention_weight)
            return result, attention_weight
        else:
            result = self.generate(keys, values, input_length, inp)
            return result, None

    def generate(self, keys, values, input_length, input):
        result = []
        end_mat = np.zeros((keys.size(0), 1),
                          dtype=int)
        step = 0
        h_t, c_t = self.init_weights(keys.size(0), self.lstm_dim)

        while np.isin(0, end_mat) and step < torch.max(input_length):

            h1, c1 = self.lstm1(input, (h_t, c_t)) if step == 0 else h1, c1 = self.lstm1(input, (h1, c1))
            h2, c2 = self.lstm2(h1, (h_t, c_t)) if step == 0 else self.lstm2(h1, (h2, c2))
            query = self.fc2(h2)
            query = query.unsqueeze(1)
            context, attention = self.attention(keys, values, query, input_length)
            output = F.leaky_relu(self.fc1(h2), negative_slope=0.1)
            context_out = F.leaky_relu(self.fc3(context), negative_slope=0.1)
            pred = self.fc4(torch.cat((output, context_out), dim=1))
            _, pred_labels = torch.max(pred, dim=1)

            if step == 0:
                result.append(pred_labels)
                pred_label = (pred_labels == dl.char2idx['?']).detach().to(self.device).numpy()
                end_mat = np.logical_or(end_mat, pred_label)
            input = torch.cat((self.embedding(pred_labels), context), dim=1)
            step = step + 1
        return torch.stack(result)

    def init_weights(self, batch_size, cell_size):
        h_t = nn.Parameter(torch.rand(batch_size, cell_size).type(torch.FloatTensor), requires_grad=False).to(self.device)
        c_t = nn.Parameter(torch.rand(batch_size, cell_size).type(torch.FloatTensor), requires_grad=False).to(self.device)
        return h_t, c_t






