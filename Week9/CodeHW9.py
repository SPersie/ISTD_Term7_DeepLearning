
# coding: utf-8

# In[71]:


import string
import csv
import unicodedata
import random
import torch
import torch.nn as nn
from random import shuffle
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as mp
import math


# In[2]:


def getdata():
    category_lines = {}
    all_categories = ['st']
    category_lines['st'] = []
    filterwords = ['NEXTEPISODE']
    
    with open('./startrek/star_trek_transcripts_all_episodes.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el) > 1):
                    #.replace(’=’,’’) #.replace(’/’,’ ’)
                    v = el.strip().replace(';','').replace('\"','') 
                    v = unicodeToAscii(v)
                    category_lines['st'].append(v)
    n_categories = len(all_categories)
    print(len(all_categories), len(category_lines['st']))
    print('done')
    
    return category_lines, all_categories


# In[3]:


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = all_categories[0]
    line = randomChoice(category_lines[category])
    return category, line


# In[4]:


# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return letter_indexes

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


# In[5]:


def train_test_split(adict):
    categories = list(adict.keys())
    train_dict = {}
    test_dict = {}
    for category in categories:
        train_dict[category] = adict[category][:int(len(adict[category])*0.8)]
        test_dict[category] = adict[category][int(len(adict[category])*0.8):]
    return train_dict, test_dict


# In[6]:


# Model
class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax()
        
    def forward(self, x, h0, c0):
#         h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
#         c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        
        lstm_output, (h_0, c_0) = self.lstm(x, (h0, c0))
        drop_out = self.dropout(lstm_output[-1, :, :])
        output = self.fc(drop_out)
        opt = self.softmax(output/0.5)
        return opt, (h_0, c_0)


# In[7]:


class iteratefromdict():
    def __init__(self, adict):
        self.ct = 0
        self.namlab = []
        shuffle(adict['st'])
        for i in range(len(adict['st'])):
#         for i in range(100):
            temp_name = inputTensor(adict['st'][i])
            temp_index = targetTensor(adict['st'][i])
            self.namlab.append((temp_name, temp_index))
        
    def num(self):
        return len(self.namlab)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ct == len(self.namlab):
            # reset before raising iteration for reusal
            shuffle(self.namlab)
            self.ct = 0
            # raise
            raise StopIteration()
        else:
            self.ct += 1
            # return feature-label pair here
            return self.namlab[self.ct - 1][0], self.namlab[self.ct - 1][1]


# In[8]:


def trainRNN(model, optimizer, train_iter):
    criterion = torch.nn.NLLLoss()
#     myiter = iteratefromdict(category_lines, n_categories)
    model.train()
    opt_loss = 0
    num_enu = 0
    for i, (images, labels) in enumerate(train_iter):
        print('--- train ---', i)
        loss = 0
        enu = 0
        optimizer.zero_grad()
        h0 = torch.zeros(2, 1, 100)
        c0 = torch.zeros(2, 1, 100)
        for j in range(images.size(0)):
            temp = torch.unsqueeze(images[j], 0)
            output, (h0, c0) = model(temp, h0, c0)
#             print(output.argmax(dim=1))
#             print(output.size())
            label = torch.tensor([labels[j]])
            l = criterion(output, label)
#             l.backward()
            loss += l
            enu +=1
        loss.backward()
        optimizer.step()
        avg_loss = loss.item() / enu
        num_enu += 1
        opt_loss += avg_loss
    opt_loss /= num_enu
    print('Train set: Average loss:', opt_loss)
    return opt_loss


# In[9]:


def testRNN(model, optimizer, test_iter):
    criterion = torch.nn.NLLLoss()
    model.eval()
    val_loss = 0
    num_enu = 0
    num_char = 0
    correct = 0
    cen_list = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_iter):
            print('--- test ---', i)
            loss = 0
            enu = 0
            h0 = torch.zeros(2, 1, 100)
            c0 = torch.zeros(2, 1, 100)
            centence = ''
            true_cen = ''
            for j in range(images.size(0)):
                temp = torch.unsqueeze(images[j], 0)
                output, (h0, c0) = model(temp,h0, c0)
                label = torch.tensor([labels[j]])
                l = criterion(output, label)
                loss += l
                enu += 1
                num_char += 1
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                if pred.item() == 77:
                    centence += '<EOS>'
                else:
                    centence += all_letters[pred.item()]
                if label.item() == 77:
                    true_cen += '<EOS>'
                else:
                    true_cen += all_letters[label.item()]
            avg_loss = loss.item() / enu
            num_enu += 1
            val_loss += avg_loss
            cen_list.append((true_cen, centence))
        val_loss /= num_enu
        final_acc = float(correct) / num_char
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)'.format(
        val_loss, correct, num_char,
        100. * final_acc))
        return final_acc, val_loss, cen_list


# In[131]:


def nn(model, num_epoch, train_iter, test_iter):
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    best_acc = -1
    best_model = None
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    centences_list = []
    samples = []
    for epoch in range(num_epoch):
        print('Epoch', epoch + 1, '/', num_epoch, ':')
        train_loss = trainRNN(model, optimizer, train_iter)
        final_acc, val_loss, cen_list = testRNN(model, optimizer, test_iter)
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss_list)
        val_acc_list.append(final_acc)
        centences_list.append(cen_list)
        sample = generate_sample(model)
        samples.append(sample)
        torch.save(model.state_dict(), 'model'+str(epoch)+'.dat')
        outF = open('sample'+str(epoch)+'.txt', 'w')
        for line in sample:
            outF.write(line)
            outF.write('\n')
        outF.close()
        if final_acc > best_acc:
            best_epoch = epoch + 1
            best_acc = final_acc
            best_model = model.state_dict()
    return train_loss_list, val_loss_list, val_acc_list, best_epoch, centences_list, best_model, samples
    


# In[119]:


def generate_sample(model):
    start_letters = "ABCDEFGHIJKLMNOPRSTUVWZ"
    all_letters = string.ascii_letters + "0123456789 .,:!?’[]()/+-="
    samples = []
    for i in range(20):
        counter = 0
        sample_string = ''
        h0 = torch.zeros(2, 1, 100)
        c0 = torch.zeros(2, 1, 100)
        
        starter = start_letters[random.randint(0, len(start_letters)-1)]
        sample_string += starter
#         print(sample_string)
        next_char, (h0, c0) = model(inputTensor(starter), h0, c0)
        next_char_list = next_char.detach().numpy().tolist()[0]
        for j in range(len(next_char_list)):
            next_char_list[j] = math.exp(next_char_list[j])
        next_char_lists = []
        for j in range(len(next_char_list)):
            next_char_lists.append(next_char_list[j]/sum(next_char_list))
        next_char_index = np.random.choice(78, 1, p=next_char_lists)[0]
        
        while(next_char_index != len(all_letters) and counter < 30):  
            next_char = torch.unsqueeze(next_char, 0)
            next_char, (h0, c0) = model(next_char, h0, c0)
            next_char_list = next_char.detach().numpy().tolist()[0]
            for j in range(len(next_char_list)):
                next_char_list[j] = math.exp(next_char_list[j])
            next_char_lists = []
            for j in range(len(next_char_list)):
                next_char_lists.append(next_char_list[j]/sum(next_char_list))
            next_char_index = np.random.choice(78,1,p=next_char_lists)[0]
            if next_char_index == len(all_letters):
                sample_string += '<EOS>'
            else:
                sample_string += all_letters[next_char_index]
#             print(sample_string)
            counter += 1
#             print(counter)
        samples.append(sample_string)
        print(sample_string)
    return samples


# In[18]:


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


# In[13]:


if __name__ == '__main__':
    ### Train test split:
    all_letters = string.ascii_letters + "0123456789 .,:!?’[]()/+-="
    # Plus EOS marker
    n_letters = len(all_letters) + 1
    all_categories = ['st']
    n_categories = 1
    
    category_lines, all_categories = getdata()
    train_dict, test_dict = train_test_split(category_lines)
    train_iter = iteratefromdict(train_dict)
    test_iter = iteratefromdict(test_dict)
    
    ### Shared Parameters
    num_epoch = 5
    model = RNN(n_letters, 100, 2, n_letters)
    
    train_loss_list, val_loss_list, val_acc_list, best_epoch, centences_list, best_model, samples = nn(model, num_epoch, train_iter, test_iter)
    plot_graph(train_loss_list, val_loss_list, val_acc_list, best_epoch, 'A')
    torch.save(best_model, 'best_model.dat')

