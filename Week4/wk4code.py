#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:58:06 2019

@author: lixingxuan
"""

import os
import torch
from PIL import Image
import getimagenetclasses as gc
from skimage import io
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, models

class ImageDataSet(Dataset):
    def __init__(self, img_path, label_path, transforms=None):
        self.imglist = os.listdir(img_path)
        self.imglist.sort()
        self.labellist = os.listdir(label_path)
        self.labellist.sort()
        self.transforms = transforms
    
    def __getitem__(self, index):
        ## Label
        filen = 'synset_words.txt'
        indicestosynsets,synsetstoindices,synsetstoclassdescriptions = gc.parsesynsetwords(filen)
        label, firstname = gc.parseclasslabel('val/' + self.labellist[index], synsetstoindices)
        ## Image
        img_as_np = io.imread('imagespart/' + self.imglist[index])
        img_as_img = Image.fromarray(img_as_np)
#         img_as_img = img_as_img.convert('L')
  
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        
        return (img_as_tensor, label)
    
    def __len__(self):
        return len(self.imglist)
    
########## Simple Center Crop
def get_loaderpred(my_dataset):
    img_list = []
    label_list = []
    for i in range(250):
        if my_dataset[i][0].shape[0] == 3:
            temp_img = my_dataset[i][0].unsqueeze(0)
            
            img_list.append(temp_img)
            label_list.append(my_dataset[i][1])
        
    img_tensor = torch.cat(img_list, 0)    
    return img_tensor, label_list

def predict(img_tensor, label_list):
    img = Variable(img_tensor)
    resnet = models.resnet18(pretrained=True)
    prediction = resnet(img)
    pred_np = prediction.detach().numpy()
    opt = []
    for i in range(pred_np.shape[0]):
        temp = pred_np[i].argmax()
        opt.append(temp)
        
    correct = 0
    for i in range(len(opt)):
        if opt[i] == label_list[i]:
            correct += 1
    acc = float(correct) / len(opt)
    return opt, acc

########## Five Crop
def get_fcpred(my_dataset):
    img_list = []
    label_list = []
    for i in range(250):
        if my_dataset[i][0].shape[1] == 3:
            temp_img = my_dataset[i][0].unsqueeze(0)
            
            img_list.append(temp_img)
            label_list.append(my_dataset[i][1])
        
    img_tensor = torch.cat(img_list, 0)
    return img_tensor, label_list

def fc_predict(fc_img_tensor, fc_label_list):
    bs, ncrops, c, h, w = fc_img_tensor.size()
    fc_img_tensor = fc_img_tensor.view(-1, c, h, w)
    img = Variable(fc_img_tensor)
    resnet = models.resnet18(pretrained=True)
    prediction = resnet(img)
    prediction = prediction.view(bs, ncrops, -1).mean(1)
    pred_np = prediction.detach().numpy()
    opt = []
    for i in range(pred_np.shape[0]):
        temp = pred_np[i].argmax()
        opt.append(temp)
        
    correct = 0
    for i in range(len(opt)):
        if opt[i] == fc_label_list[i]:
            correct += 1
    acc = float(correct) / len(opt)
    return opt, acc

########## SqueezeNet-01 330*330
def get_sn01pred(my_dataset):
    img_list = []
    label_list = []
    for i in range(250):
        if my_dataset[i][0].shape[1] == 3:
            temp_img = my_dataset[i][0].unsqueeze(0)
            
            img_list.append(temp_img)
            label_list.append(my_dataset[i][1])
        
    img_tensor = torch.cat(img_list, 0)
    return img_tensor, label_list

def sn01_predict(fc_img_tensor, fc_label_list):
    bs, ncrops, c, h, w = fc_img_tensor.size()
    fc_img_tensor = fc_img_tensor.view(-1, c, h, w)
    img = Variable(fc_img_tensor)
    
    squeeze = models.squeezenet1_1(pretrained=True)
    squeeze._modules['features'][2] = torch.nn.modules.AdaptiveAvgPool2d((54, 54)) 
    print('SqueezeNet01 created. Start predicting......')

    prediction = squeeze(img)
    print('Calculating accuracy......')
    prediction = prediction.view(bs, ncrops, -1).mean(1)
    pred_np = prediction.detach().numpy()
    opt = []
    for i in range(pred_np.shape[0]):
        temp = pred_np[i].argmax()
        opt.append(temp)
        
    correct = 0
    for i in range(len(opt)):
        if opt[i] == fc_label_list[i]:
            correct += 1
    acc = float(correct) / len(opt)
    return opt, acc

########## SqueezeNet-00 330*330
def get_sn00pred(my_dataset):
    img_list = []
    label_list = []
    for i in range(250):
        if my_dataset[i][0].shape[1] == 3:
            temp_img = my_dataset[i][0].unsqueeze(0)
            
            img_list.append(temp_img)
            label_list.append(my_dataset[i][1])
        
    img_tensor = torch.cat(img_list, 0)
    return img_tensor, label_list

def sn00_predict(fc_img_tensor, fc_label_list):
    bs, ncrops, c, h, w = fc_img_tensor.size()
    fc_img_tensor = fc_img_tensor.view(-1, c, h, w)
    img = Variable(fc_img_tensor)
    
    squeeze = models.squeezenet1_0(pretrained=True)
    squeeze._modules['features'][2] = torch.nn.modules.AdaptiveAvgPool2d((54, 54)) 
    print('SqueezeNet00 created. Start predicting......')

    prediction = squeeze(img)
    print('Calculating accuracy......')
    prediction = prediction.view(bs, ncrops, -1).mean(1)
    pred_np = prediction.detach().numpy()
    opt = []
    for i in range(pred_np.shape[0]):
        temp = pred_np[i].argmax()
        opt.append(temp)
        
    correct = 0
    for i in range(len(opt)):
        if opt[i] == fc_label_list[i]:
            correct += 1
    acc = float(correct) / len(opt)
    return opt, acc

########## Main Function
if __name__ == '__main__':
    ### simple crop
    sc_trans = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    sc_dataset = ImageDataSet('imagespart/', 'val/', sc_trans)
    
    sc_img_tensor, sc_label_list = get_loaderpred(sc_dataset)
    sc_opt, sc_acc = predict(sc_img_tensor, sc_label_list)
    print('Probelm 1, Simple Center Crop transform.')
    print('The accuracy of simple crop is ', sc_acc, '.')
    
    del sc_trans, sc_dataset, sc_img_tensor, sc_label_list, sc_opt, sc_acc
    
    ### five crops
    fc_trans = transforms.Compose([transforms.Resize(280),
                                   transforms.FiveCrop(224),
                                   transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                   ])
    fc_dataset = ImageDataSet('imagespart/', 'val/', fc_trans) 
    fc_img_tensor, fc_label_list = get_fcpred(fc_dataset)
    fc_opt, fc_acc = fc_predict(fc_img_tensor, fc_label_list)
    print('Probelm 2, Five Crop transform.')
    print('The accuracy of five crop is ', fc_acc, '.')
    
    del fc_trans, fc_dataset, fc_img_tensor, fc_label_list, fc_opt, fc_acc
    
    ### 330 * 330 input size
    # SqueezeNet 01
    sn01_trans = transforms.Compose([transforms.Resize(350),
                                  transforms.FiveCrop(330),
                                  transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                  ])
    print('Problem 3, SqueezeNet01, 330 * 330')
    print('Creating data set.')
    sn01_dataset = ImageDataSet('imagespart/', 'val/', sn01_trans)
    print('Data set created.')
    sn01_img_tensor, sn01_label_list = get_sn01pred(sn01_dataset)
    sn01_opt, sn01_acc = sn01_predict(sn01_img_tensor, sn01_label_list)
    print('The accuracy of SqueezeNet 01 is ', sn01_acc, '.')
    
    del sn01_trans, sn01_dataset, sn01_img_tensor,sn01_label_list, sn01_opt,sn01_acc

    ### 330 * 330 input size
    # SqueezeNet 00
    sn00_trans = transforms.Compose([transforms.Resize(350),
                                  transforms.FiveCrop(330),
                                  transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                  ])
    print('Problem 3, SqueezeNet00, 330 * 330')
    print('Creating data set.')
    sn00_dataset = ImageDataSet('imagespart/', 'val/', sn00_trans)
    print('Data set created.')
    sn00_img_tensor, sn00_label_list = get_sn00pred(sn00_dataset)
    sn00_opt, sn00_acc = sn00_predict(sn00_img_tensor, sn00_label_list)
    print('The accuracy of SqueezeNet 00 is ', sn00_acc, '.')





