
import torch.nn as nn
import torch
import numpy as np
import Levenshtein as L
import torch.optim as optim
import pandas as pd
import dataloader as dl
import model as models
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_mode = True

def plot_grad_flow(parameters):
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('gradient.png', bbox_inches='tight')
    plt.close()


def plot_weights(attention_weights, epoch):
    fig = plt.figure()
    plt.imshow(attention_weights, interpolation='nearest', cmap='hot')
    fig.savefig("epoch%d.png" % (epoch))
    plt.close()


def train(model, train_loader, test_loader, epochs = 20, device = device):
    model.train()
    best_score = 9999
    criterion = nn.CrossEntropyLoss(ignore_index = dl.char2idx['?'], reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma=0.8)

    for epoch in range(epochs):
        total_step = len(train_loader)
        for step, (inputs, labels, labels_len) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, attention_weight = model(inputs, labels, labels.size(1))
            outputs = outputs.permute(1, 0, 2)
            labels = labels[:, 1:]

            loss = criterion(outputs.permute(0, 2, 1), labels)
            loss = loss / labels.size(0)
            loss.backward()

            plot_grad_flow(model.named_parameters())
            plot_weights(attention_weight.detach().cpu().numpy()[:, -1], epoch)
            optimizer.step()
            if (step+1) % 10 == 0:
                print('Epoch: {}\tStep: {} out of {}\tCurrent Loss: {:.3f}'.format(epoch + 1, step+1, total_step, loss.item()))


        with torch.no_grad():
            epoch_distance = run_eval(model, test_loader)
        if best_score > epoch_distance:
            best_score = epoch_distance
            torch.save(model.state_dict(), "models/model_" + str(epoch) + ".pt")
    scheduler.step()



def run_eval(model, val_loader, device = device):
    print("Evaluating...")
    model.eval()
    ls_total = 0.
    count = 0
    for step, (inputs, labels, labels_len) in enumerate(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs, labels, labels.size(1))
        output = torch.transpose(outputs, 0, 1)
        ls = 0
        for i in range(output.size(0)):
            end_pos = output[i].tolist().index(dl.char2idx['?']) if dl.char2idx['?'] in output[i].tolist() \
                else output[i].size(0)
            pred = "".join(dl.idx2char[o] for o in output[i].tolist()[:end_pos])
            true = "".join(dl.idx2char[l] for l in labels[i][1:labels_len[i] - 1].tolist())

            if i % 100 == 0:
                print("Pred: {}, True: {}".format(pred, true))
            ls += 100 * (L.distance(pred, true) / len(true))
            ls_total += 100 * (L.distance(pred, true) / len(true))
            count = count + 1

    print('Evaluation distance: {:.3f}'.format(ls_total / count))
    model.train()
    return ls_total / count


def run_test(model, test_loader, device = device):
    model.eval()
    print('running test...')

    pred_final = []
    for batch_num, (inputs, order_list) in enumerate(test_loader):
        inputs = inputs.to(device)

        outputs, attention = model(inputs, None, None)
        outputs = torch.transpose(outputs, 0, 1)

        zip_list = zip(outputs.tolist(), order_list)
        sorted_list = sorted(zip_list, key=lambda x: x[1], reverse=False)

        output = torch.tensor([x[0] for x in sorted_list])

        for i in range(output.size(0)):
            end_pos = output[i].tolist().index(dl.char2idx['?']) if dl.char2idx['?'] in output[i].tolist() \
                else output[i].size(0)
            pred = "".join(dl.idx2char[o] for o in output[i].tolist()[:end_pos])
            pred_final.append(pred)

    result = pd.DataFrame(np.array(pred_final))
    result.columns = ['Predicted']
    result.to_csv('submission.csv', index_label='Id')


if __name__ == "__main__":
    train_loader, val_loader, test_loader = dl.get_loaders()
    model = models.Seq2Seq(40, 128, dl.get_char_length())
    model = model.to(device)
    if train_mode:
        train(model, train_loader, val_loader)
        with torch.no_grad():
            run_test(model, test_loader)



