import numpy as np
import torch
from torch.nn.utils import rnn
from torch.utils.data import DataLoader

idx2char = [' ', "'", '+', '-', '.', '<', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
char2idx = {' ': 0, "'": 1, '+': 2, '-': 3, '.': 4, '<': 5, '>': 6, '?': 7, 'A': 8, 'B': 9, 'C': 10, 'D': 11, 'E': 12, 'F': 13, 'G': 14, 'H': 15, 'I': 16, 'J': 17, 'K': 18, 'L': 19, 'M': 20, 'N': 21, 'O': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29, 'W': 30, 'X': 31, 'Y': 32, 'Z': 33, '_': 34}
BATCH_SIZE = 128

train_X = np.load('train.npy', encoding='latin1', allow_pickle=True)
train_y = np.load('train_transcripts.npy', encoding='latin1', allow_pickle=True)
val_X = np.load('dev.npy', encoding='latin1', allow_pickle=True)
val_y = np.load('dev_transcripts.npy', encoding='latin1', allow_pickle=True)
test_X = np.load('test.npy', encoding='latin1', allow_pickle=True)


def transcript2label(trans):
    sentences = []
    for sentence in trans:
        words = []
        for element in sentence:
            words.append(element.decode('UTF-8'))
        line = " ".join(words)
        sentences.append(line)
    label = []
    for sentence in sentences:
        sentence_label = [char2idx['<']]+[char2idx[c] for c in sentence] + [char2idx['>']]
        label.append(np.array(sentence_label, dtype=np.int64))
    return np.array(label)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]
        return x, y, len(x), len(y)

    def __len__(self):
        return len(self.inputs)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __getitem__(self, index):
        x = self.inputs[index]
        return x, len(x)

    def __len__(self):
        return len(self.inputs)

def collate(batch):
    input = [torch.tensor(x[0]) for x in batch]
    label= [torch.tensor(x[1]) for x in batch]
    zipped = zip(input, label)
    sorted = sorted(zipped, key=lambda x: len(x[0]), reverse=True)
    sorted_input = [x[0] for x in sorted]
    sorted_label = [x[1] for x in sorted]
    padded_input = rnn.pad_sequence(sorted_input, batch_first=True)
    padded_label = rnn.pad_sequence(sorted_label, padding_value=7, batch_first = True)
    lens = [len(x) for x in sorted_input]
    label_len = torch.IntTensor([len(x[1]) for x in sorted])
    padded_input = rnn.pack_padded_sequence(padded_input, lens, batch_first=True)
    return padded_input, padded_label, label_len

def collate_test(batch):
    input = [torch.tensor(x[0].astype('float32')) for x in batch]
    order = range(len(input))
    zipped = zip(input,order)
    sorted = sorted(zipped, key=lambda x: len(x[0]), reverse=True)
    sorted_input = [x[0] for x in sorted]
    padded_input = rnn.pad_sequence(sorted_input)
    lens = [len(x) for x in sorted_input]
    padded_input = rnn.pack_padded_sequence(padded_input, lens)
    sorted_order = [x[1] for x in sorted]
    return padded_input, sorted_order


def get_loaders(train_X = train_X, train_y = train_y, val_X = val_X, val_y = val_y, test_X = test_X):
    train_set = Dataset(train_X, transcript2label(train_y))
    val_set = Dataset(val_X, transcript2label(val_y))
    test_set = TestDataset(test_X)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn= collate)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn= collate_test)

    return train_loader, val_loader, test_loader


def get_char_length():
    return len(idx2char)


def sentence2seq(sentence, char2idx = char2idx):
    label = []
    for s in sentence:
        sentence_logit = [char2idx['<']] + [char2idx[c] for c in s] + [char2idx['>']]
        label.append(np.array(sentence_logit, dtype=np.int64))
    return np.array(label)


def seq2sentence(sequence, idx2char = idx2char):
    return "".join([idx2char[c] for c in sequence])

