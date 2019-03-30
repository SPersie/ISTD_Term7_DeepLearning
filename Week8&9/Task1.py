from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import torch.optim as optim

def parsedata(path):
    ### Read files under path into a list
    file_list = glob.glob(path)
    
    ### Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    
    ### Read a file and split into lines
    for file in file_list:
        category = os.path.splitext(os.path.basename(file))[0]
        all_categories.append(category)
        
        temp_lines = open(file, encoding='utf-8').read().strip().split('\n')
        lines = [unicodeToAscii(line) for line in temp_lines]
        category_lines[category] = lines
    return category_lines, all_categories
    

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    all_letters = string.ascii_letters + " .,;'"
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def train_test_split(adict):
    categories = list(adict.keys())
    train_dict = {}
    test_dict = {}
    for category in categories:
        train_dict[category] = adict[category][:int(len(adict[category])*0.7)]
        test_dict[category] = adict[category][int(len(adict[category])*0.7):]
    return train_dict, test_dict


class iteratefromdict():
    def __init__(self, adict, all_categories):
        self.namlab = []
        for category in all_categories:
            for i in range(len(adict[category])):
                temp_name = lineToTensor(adict[category][i])
                temp_index = n_categories.index(category)
                self.namlab.append((temp_name, temp_index))
        
    def num(self):
        return len(self.namlab)
    
    def __iter__(self):
        self.ct = 0
        return self
    
    def __next__(self):
        if self.ct == len(self.namlab):
            # reset before raising iteration for reusal
            np.random.shuffle(self.namlab)
            self.ct = 0
            # raise
            raise StopIteration()
        else:
            self.ct += 1
            # return feature-label pair here
            return self.namlab[self.ct - 1][0], self.namlab[self.ct - 1][1]


### Customized RNN
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        
        lstm_output, _ = self.lstm(x, (h0, c0))
        output = self.fc(lstm_output[-1, :, :])
        opt = self.softmax(output)
        return opt


def trainRNN(model, optimizer, train_iter):
    criterion = torch.nn.CrossEntropyLoss()
#     myiter = iteratefromdict(category_lines, n_categories)
    model.train()
    opt_loss = 0
    num_enu = 0
    for i, (images, labels) in enumerate(train_iter):
        optimizer.zero_grad()
        output = model(images)
        target = torch.tensor([labels])
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        opt_loss += loss.item()
        num_enu += 1
        
    opt_loss /= num_enu
    print('Train set: Average loss:', opt_loss)
    return opt_loss


def testRNN(model, test_iter):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_iter):
            output = model(images)
            target = torch.tensor([labels])
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    val_loss /= test_iter.num()
    final_acc = correct / test_iter.num()
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, test_iter.num(),
        100. * final_acc))
    return final_acc, val_loss


def nn(input_size, hidden_size, num_layers, output_size, num_epoch, train_itr, test_itr):
    model = RNN(input_size, hidden_size, num_layers, output_size)
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    
    best_acc = 0
    best_model = None
    train_lost_list = []
    val_loss_list = []
    val_acc_list = []
    
    for epoch in range(num_epoch):
        print('Epoch', epoch+1, '/', num_epoch,':')
        train_loss = trainRNN(model, optimizer, train_itr)
        final_acc, val_loss = testRNN(model, test_itr)
        
        train_lost_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(final_acc)
        if final_acc > best_acc:
            best_epoch = epoch + 1
            best_acc = final_acc
            best_model = model.state_dict()
    return train_lost_list, val_loss_list, val_acc_list, best_epoch


def plot_graph(train_lost_list, val_loss_list, val_acc_list, best_epoch, flag):
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(len(train_lost_list)), train_lost_list)
    ax1.plot(np.arange(len(val_loss_list)), val_loss_list)
    ax1.set_title('Train and Test Loss for Network ' + flag)
    ax1.legend(['training loss', 'testing loss'], loc='upper right')
    mp.savefig('figures/nn'+flag+'_fig1.png')
    plt.show()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(len(val_acc_list)), val_acc_list)
    ax2.set_title('Test accuracy for Network ' + flag)
    ax2.legend(['validation accuracy'], loc='lower right')
    mp.savefig('figures/nn'+flag+'_fig2.png')
    plt.show()
    
    print('The best epoch is', best_epoch)


if __name__ == '__main__':
    ### Train test split
    category_lines, n_categories = parsedata('data/names/*.txt')
    train_dict, test_dict = train_test_split(category_lines)
    train_iter = iteratefromdict(train_dict, n_categories)
    test_iter = iteratefromdict(test_dict, n_categories)
    
    ### Shared parameters
    input_size = len(string.ascii_letters + " .,;'")
    output_size = 18
    num_epoch =10
    
    ### 1 layer, 32 hidden
    hidden_size = 32
    num_layers = 1
    train_lost_list, val_loss_list, val_acc_list, best_epoch = nn(input_size, hidden_size, 
                                                                  num_layers, output_size, 
                                                                  num_epoch, train_iter, test_iter)
    plot_graph(train_lost_list, val_loss_list, val_acc_list, best_epoch, 'A')
    
    ### 1 layer, 64 hidden
    hidden_size = 64
    num_layers = 1
    train_lost_list, val_loss_list, val_acc_list, best_epoch = nn(input_size, hidden_size, 
                                                                  num_layers, output_size, 
                                                                  num_epoch, train_iter, test_iter)
    plot_graph(train_lost_list, val_loss_list, val_acc_list, best_epoch, 'B')
    
    ### 1 layer, 128 hidden
    hidden_size = 128
    num_layers = 1
    train_lost_list, val_loss_list, val_acc_list, best_epoch = nn(input_size, hidden_size, 
                                                                  num_layers, output_size, 
                                                                  num_epoch, train_iter, test_iter)
    plot_graph(train_lost_list, val_loss_list, val_acc_list, best_epoch, 'C')
    
    ### 2 layers, 32 hidden
    hidden_size = 32
    num_layers = 2
    train_lost_list, val_loss_list, val_acc_list, best_epoch = nn(input_size, hidden_size, 
                                                                  num_layers, output_size, 
                                                                  num_epoch, train_iter, test_iter)
    plot_graph(train_lost_list, val_loss_list, val_acc_list, best_epoch, 'D')
    
    ### 2 layers, 64 hidden
    hidden_size = 64
    num_layers = 2
    train_lost_list, val_loss_list, val_acc_list, best_epoch = nn(input_size, hidden_size, 
                                                                  num_layers, output_size, 
                                                                  num_epoch, train_iter, test_iter)
    plot_graph(train_lost_list, val_loss_list, val_acc_list, best_epoch, 'E')
    
    ### 2 layers, 128 hidden
    hidden_size = 128
    num_layers = 2
    train_lost_list, val_loss_list, val_acc_list, best_epoch = nn(input_size, hidden_size, 
                                                                  num_layers, output_size, 
                                                                  num_epoch, train_iter, test_iter)
    plot_graph(train_lost_list, val_loss_list, val_acc_list, best_epoch, 'F')
    
        

