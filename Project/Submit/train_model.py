"""
Created on Tue Mar 19 00:52:35 2019

@author: lixingxuan
"""

import json
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms, models
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
from sklearn.metrics import precision_score
import vocparseclslabels as voclabel

"""
Create customized dataset.
"""

class ImageDataset(Dataset):
    def __init__(self, img_path, index_path=None, transforms=None, flag=None):
        self.img_path = img_path
        self.transforms = transforms
        
        ### Construct a dataframe of filename and multi-labels
        self.indexdf = pd.read_csv(
            index_path,
            delim_whitespace=True,
            header=None,
            names=['filename'])
        
        pv = voclabel.PascalVOC('./VOCdevkit/VOC2012/')
        cat_list = pv.list_image_sets()
        
        for cat in cat_list:
            cat_name = cat
            dataset = flag
            df = pv._imgs_from_category(cat_name, dataset)[cat_name]
            self.indexdf = pd.concat([self.indexdf, df], axis=1)
            
        self.indexdf.iloc[:,1:] = self.indexdf.iloc[:,1:]==1
        self.indexdf.iloc[:,1:] = self.indexdf.iloc[:,1:]*1
        
    def __getitem__(self, index):
        img_as_np = io.imread(self.img_path + 
                             self.indexdf.iloc[index, 0] + '.jpg')
        img_as_img = Image.fromarray(img_as_np)
        label = self.indexdf.iloc[index, 1:].values
        label = label.reshape(1, 20)
        x_data = label[0]
        x_data = x_data.astype(int)
        label_as_tensor = torch.from_numpy(x_data).float()
        
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
            
        return (img_as_tensor, label_as_tensor)
    
    def __len__(self):
        return self.indexdf.shape[0]      

"""
Training function for model.
"""

def nntrain(model, train_loader, optimizer):
    model.train()
    opt_loss = 0
    num_enu = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target.squeeze_()
        ### Binary cross entropy loss
        loss_function = torch.nn.BCELoss()
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        opt_loss += loss.item()
        num_enu += 1
    opt_loss /= num_enu
    print('Train set: Average loss:', opt_loss)
    return opt_loss

"""
Validation function for model.
Default threshold is setted to 0.7, which is used to preview the performance.
"""

def nnval(model, val_loader, threshold, flag):
    model.eval()
    val_loss = 0
    gt_list = []
    pr_list = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            target.squeeze_()
            loss_function = torch.nn.BCELoss(reduction='sum')
            val_loss += loss_function(output, target).item()
            pred = (output>threshold) *1
            gt_list.append(target)
            pr_list.append(pred)

    gt = torch.cat(gt_list, 0)
    gt = gt.cpu().numpy()
    pr = torch.cat(pr_list, 0)
    pr = pr.cpu().numpy()
    val_loss /= len(val_loader.dataset)
    aps = precision_score(gt, pr, average='samples')
    print('{} set: Average loss: {:.4f}, Average Precision: {} on default threshold'.format(
        flag, val_loss, aps))
    return aps, val_loss

"""
Function to compute top 50 images for each class.
"""

def top50(model, trainval_loader, threshold):
    model.eval()
    val_loss = 0
    gt_list = []
    pr_list = []
    opt_list = []
    with torch.no_grad():
        for data, target in trainval_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            target.squeeze_()
            loss_function = torch.nn.BCELoss(reduction='sum')
            val_loss += loss_function(output, target).item()
            pred = (output>threshold) *1
            gt_list.append(target)
            pr_list.append(pred)
            opt_list.append(output)

    gt = torch.cat(gt_list, 0)
    gt = gt.cpu().numpy()
    pr = torch.cat(pr_list, 0)
    pr = pr.cpu().numpy()
    opt = torch.cat(opt_list, 0)
    val_loss /= len(trainval_loader.dataset)
    aps = precision_score(gt, pr, average='samples')
    print('Train and Validation set: Average loss: {:.4f}, Average Precision: {} on default threshold'.format(
        val_loss, aps))
    
    gt = torch.cat(gt_list, 0)
    np_gt = gt.cpu().numpy()
    np_opt = opt.cpu().numpy()    
    return np_gt, np_opt

"""
Function to train the model.
"""

def nnmodel(model, train_loader, val_loader, test_loader, num_epoch, threshold):
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    model.to(device)
    best_epoch = 0
    best_aps = 0
    best_model = None
    train_lost_list = []
    val_loss_list = []
    val_aps_list = []
    for epoch in range(num_epoch):
        print('Epoch', epoch+1, '/', num_epoch,':')
        train_loss = nntrain(model, train_loader, optimizer)
        aps, val_loss = nnval(model, val_loader, threshold, 'Validation')
        
        train_lost_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_aps_list.append(aps)
        if aps > best_aps:
            best_epoch = epoch + 1
            best_aps = aps
            best_model = model.state_dict() 
    
    ## Get Top 50 images
    model.load_state_dict(best_model)
    np_gt, np_opt = top50(model, test_loader, threshold)
    return train_lost_list, val_loss_list, val_aps_list, best_epoch, best_model, np_gt, np_opt

"""
Compute top 50 result for all thresholds.
"""

def all_threshold(np_gt, np_opt):
    threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
                      0.35, 0.4, 0.45,
                      0.5, 0.55, 0.6, 0.65, 0.7, 
                      0.75, 0.8, 0.85, 0.9, 0.95, 1]
    aps_list = []
    for i in range(len(threshold_list)):
        np_pred = (np_opt>threshold_list[i]) *1
        aps = precision_score(np_gt, np_pred, average='samples')
        aps_list.append(aps)
    ### Computing top 50 images
    all_cats = {}
    pv = voclabel.PascalVOC('./VOCdevkit/VOC2012/')
    cat_list = pv.list_image_sets()
    df_opt = pd.DataFrame(data=np_opt, columns=cat_list)
    
    for cat in cat_list:
        temp_df = df_opt.sort_values(cat, ascending=False)
        all_cats[cat] = []
        cat_list = temp_df.index.tolist()[0:50]
        all_cats[cat].append(cat_list)
        cat_value = temp_df[cat].tolist()[0:50]
        all_cats[cat].append(cat_value)
        
    return threshold_list, aps_list, all_cats
        
"""
Plot graph of threshold against aps.
"""

def plot_graph(threshold, aps):
    fig1, ax1 = plt.subplots()
    ax1.plot(threshold, aps)
    ax1.set_title('Average precision score against threshold')
    ax1.legend(['aps'], loc='upper right')
    mp.savefig('fig1')
    plt.show()

"""
Plot graph of training loss and validation loss against epochs.
"""

def plot_train_val_loss(train_lost_list, val_loss_list):
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(len(train_lost_list)), train_lost_list)
    ax1.plot(np.arange(len(val_loss_list)), val_loss_list)
    ax1.set_title('Training and Validation Loss')
    ax1.legend(['training loss', 'validation loss'], loc='upper right')
    mp.savefig('fig2')
    plt.show()

"""
Convert index in opt dictionaries to filename.
"""

def index_to_file(all_cats):
    indexdf = pd.read_csv(
            './VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
            delim_whitespace=True,
            header=None,
            names=['filename'])
    filelist = indexdf['filename'].values.tolist()
    
    for cat in all_cats:
        for i in range(0, 50):
            all_cats[cat][0][i] = filelist[all_cats[cat][0][i]]
    return all_cats
        
"""
Data augumentation.
"""

def all_phases(model, transform):
    img_path = './VOCdevkit/VOC2012/JPEGImages/'
    ### Create training set
    train_path = './VOCdevkit/VOC2012/ImageSets/Main/train.txt'
    train_set = ImageDataset(img_path, train_path, transform, 'train')
    
    ### Create validation set
    val_path = './VOCdevkit/VOC2012/ImageSets/Main/val.txt'
    val_set = ImageDataset(img_path, val_path, transform, 'val')
    
    ### Create trainval set
    trainval_path = './VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
    trainval_set = ImageDataset(img_path, trainval_path, transform, 'trainval')
    
    ### Create dataloader
    print('##############################')
    print('Dataloaders...')
    
    loadertr = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    loaderval = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)
    loadertrval = torch.utils.data.DataLoader(trainval_set, batch_size=64, shuffle=False)
    print('Dataloaders created!')
    
    ### Neural Network B: Pretrained ResNet18
    print('##############################')
    print('Pre-trained ResNet18.')
    
    train_lost_list, val_loss_list, val_aps_list, best_epoch, best_model, np_gt, np_opt = nnmodel(model, loadertr, loaderval, loadertrval, 15, 0.7)
    return train_lost_list, val_loss_list, val_aps_list, best_epoch, best_model, np_gt, np_opt


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Phase 1
    print('############################  Phase1  ############################')
    print('Resize (224), CenterCrop')
    model = models.resnet18(pretrained=True)
    model._modules['fc'] = torch.nn.Sequential(torch.nn.modules.Linear(in_features=512, out_features=20, bias=True),
                                        torch.nn.Sigmoid())
    transform1 = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
    
    train_lost_list1, val_loss_list1, val_aps_list, best_epoch, best_model, np_gt, np_opt = all_phases(model, transform1)
    
    ### Phase 2
    print('############################  Phase2  ############################')
    print('RandomRotation (10), Resize (224), CenterCrop')
    model.load_state_dict(best_model)
    transform2 = transforms.Compose([transforms.RandomRotation(10),
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
    train_lost_list2, val_loss_list2, val_aps_list, best_epoch, best_model, np_gt, np_opt = all_phases(model, transform2)

    ### Phase 3
    print('############################  Phase3  ############################')
    print('ColorJitter, RandomHorizontalFlip, Resize (224), CenterCrop')
    model.load_state_dict(best_model)
    transform3 = transforms.Compose([transforms.ColorJitter(hue=.05, saturation=.05),
                                     transforms.RandomHorizontalFlip(),
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
    train_lost_list3, val_loss_list3, val_aps_list, best_epoch, best_model, np_gt, np_opt = all_phases(model, transform3)
    
    
    threshold_list, aps_list, all_cats = all_threshold(np_gt, np_opt)
    
    ### Saving Stuff
    print('############################ Saving  ##############################')
    print('Saving best model states.') 
    torch.save(best_model, 'best_model.dat')
    print('Model saved.')
    print('##############################')
    print('Plotting graph.')    
    plot_graph(threshold_list, aps_list)
    all_cats_file = index_to_file(all_cats)
    train_lost_list = train_lost_list1 + train_lost_list2 + train_lost_list3
    val_loss_list = val_loss_list1 + val_loss_list2 + val_loss_list3
    plot_train_val_loss(train_lost_list, val_loss_list)
    print('##############################')
    print('Saving top 50 results.') 
    with open('data.json', 'w') as fp:
        json.dump(all_cats_file, fp)
    print('Result saved.')

