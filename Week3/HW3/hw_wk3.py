#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:27:52 2019

@author: lixingxuan
"""

from __future__ import print_function
import mnist_reader
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, dims):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dims, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
    
def train(model, train_loader, optimizer, epoch):
    model.train()
    opt_loss = 0
    num_enu = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        opt_loss += loss.item()
        num_enu += 1
        
    opt_loss /= num_enu
    print('Train set: Average loss: ', opt_loss)
    return opt_loss

def test(model, test_loader):
    model.eval()
    each_class = {}
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            
            for i in range(len(pred)):
                if pred[i].numpy()[0] not in each_class:
                    if pred[i].eq(target[i]) == 1:
                        each_class[pred[i].numpy()[0]] = [1, 1]
                    else:
                        each_class[pred[i].numpy()[0]] = [0, 1]
                else:
                    if pred[i].eq(target[i]) == 1:
                        each_class[pred[i].numpy()[0]][0] += 1
                        each_class[pred[i].numpy()[0]][1] += 1
                    else:
                        each_class[pred[i].numpy()[0]][1] += 1
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    for key in each_class:
        each_class[key] = float(each_class[key][0]) / each_class[key][1]
        
    temp_acc = 0
    for key in each_class:
        temp_acc += each_class[key]
    
    final_acc = temp_acc / 10

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * final_acc))
    
    return each_class, final_acc, test_loss
#    print(each_class)


if __name__ == '__main__':

    batch_size = 1024
    valbatch_size = 1024
    epochs = 20
    dims = 784
    
    # paras for optimizer
#    lr = 0.5
#    momentum = None
    
    
    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()
    
    dtr = torch.utils.data.TensorDataset(X_train, y_train)
    dv = torch.utils.data.TensorDataset(X_test, y_test)
    
    loadertr=torch.utils.data.DataLoader(dtr,batch_size=batch_size,shuffle=True) # returns an iterator
    loaderval=torch.utils.data.DataLoader(dv,batch_size=valbatch_size,shuffle=False)


    model = Net(dims)
    optimizer = optim.SGD(model.parameters(), lr=0.02)

    best_acc = 0
    best_model = None
    best_each_class = None
    
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, loadertr, optimizer, epoch)
        each_class, final_acc, test_loss = test(model, loaderval)
        train_loss_list.append(train_loss)
        val_loss_list.append(test_loss)
        if final_acc > best_acc:
            best_acc = final_acc
            best_model = model.state_dict()
            best_each_class = each_class
            
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.plot(np.arange(len(train_loss_list)), val_loss_list)
    plt.legend(['training loss', 'validation loss'], loc='upper right')
    
    print('The best accuracy is ', best_acc)
    print('Class-wise accuracy when having the best overall accuracy: ')
    print(best_each_class)   
    torch.save(model.state_dict(),"test.pt")
    print('Parameters of the best model is saved.')
        