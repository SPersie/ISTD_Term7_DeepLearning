import os
import torch
import copy
from PIL import Image
from skimage import io
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, models
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import pyplot as mp


class ImageDataset(Dataset):
    def __init__(self, img_path, train_path=None, val_path=None, test_path=None, transforms=None, flag=None):
        self.img_path = img_path
        self.imglist = os.listdir(img_path)
        self.imglist.sort()
        ## training image and label
        self.traindict = {}
        with open(train_path) as f:
            for line in f:
                key, val = line.split()
                self.traindict[key] = int(val)
        self.trainlist = list(self.traindict.keys())
        self.trainlist.sort()
        
        ## validation image and label
        self.valdict = {}
        with open(val_path) as f:
            for line in f:
                key, val = line.split()
                self.valdict[key] = int(val)
        self.vallist = list(self.valdict.keys())
        self.vallist.sort()
                
        ## testing image and label
        self.testdict = {}
        with open(test_path) as f:
            for line in f:
                key, val = line.split()
                self.testdict[key] = int(val)
        self.testlist = list(self.testdict.keys())
        self.testlist.sort()
        
        self.transforms = transforms
        self.flag = flag
        
    def __getitem__(self, index):
        if self.flag == 'train':
            img_as_np = io.imread(self.img_path + self.trainlist[index])
            img_as_img = Image.fromarray(img_as_np)
            label = self.traindict[self.trainlist[index]]
            
        if self.flag =='val':
            img_as_np = io.imread(self.img_path + self.vallist[index])
            img_as_img = Image.fromarray(img_as_np)
            label = self.valdict[self.vallist[index]]
            
        if self.flag =='test':
            img_as_np = io.imread(self.img_path + self.testlist[index])
            img_as_img = Image.fromarray(img_as_np)
            label = self.testdict[self.testlist[index]]
            
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
            
        return (img_as_tensor, label)
    
    def __len__(self):
        if self.flag == 'train':
            return len(self.trainlist)
        if self.flag == 'val':
            return len(self.vallist)
        if self.flag == 'test':
            return len(self.testlist)



def create_dataset(img_path, train_path, val_path, test_path, transforms, flag):
    print('##############################')
    print('Creating', flag, 'dataset...')
    temp_dataset = ImageDataset(img_path, train_path, val_path, test_path, transforms, flag)
    ds_size = len(temp_dataset)
    img_list = []
    label_list = []

    for i in range(ds_size):
        if temp_dataset[i][0].shape[0] == 3:
            temp_img = temp_dataset[i][0].unsqueeze(0)
            temp_label = temp_dataset[i][1]
            img_list.append(temp_img)
            label_list.append(temp_label)
    img_tensor = torch.cat(img_list, 0)
    temp_label_tensor = torch.tensor(label_list)
    label_tensor = temp_label_tensor.view(-1, 1)
    print(flag, 'dataset is created!')
    return img_tensor, label_tensor    


def nnAtrain(model, train_loader, optimizer):
    model.train()
    opt_loss = 0
    num_enu = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        target.squeeze_()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        opt_loss += loss.item()
        num_enu += 1
        
    opt_loss /= num_enu
    print('Train set: Average loss:', opt_loss)
    return opt_loss
    
def nnAval(model, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            target.squeeze_()
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    val_loss /= len(val_loader.dataset)
    final_acc = correct / len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * final_acc))
    return final_acc, val_loss

def nnA(train_loader, val_loader, test_loader, num_epoch):
    model = models.resnet18(pretrained=False)
    model._modules['fc'] = torch.nn.modules.Linear(in_features=512, out_features=102, bias=True)
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    model.to(device)
    
    best_epoch = 0
    best_acc = 0
    best_model = None
    train_lost_list = []
    val_loss_list = []
    val_acc_list = []
    
    for epoch in range(num_epoch):
        print('Epoch', epoch+1, '/', num_epoch,':')
        train_loss = nnAtrain(model, train_loader, optimizer)
        final_acc, val_loss = nnAval(model, val_loader)
        
        train_lost_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(final_acc)
        if final_acc > best_acc:
            best_epoch = epoch + 1
            best_acc = final_acc
            best_model = model.state_dict()
    model.load_state_dict(best_model)
    test_acc, test_loss = nnAval(model, test_loader)   
    return train_lost_list, val_loss_list, val_acc_list, test_acc, test_loss, best_epoch


def nnB(train_loader, val_loader, test_loader, num_epoch):
    model = models.resnet18(pretrained=True)
    model._modules['fc'] = torch.nn.modules.Linear(in_features=512, out_features=102, bias=True)
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    model.to(device)
    
    best_epoch = 0
    best_acc = 0
    best_model = None
    train_lost_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epoch):
        print('Epoch', epoch+1, '/', num_epoch,':')
        train_loss = nnAtrain(model, train_loader, optimizer)
        final_acc, val_loss = nnAval(model, val_loader)
        
        train_lost_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(final_acc)
        if final_acc > best_acc:
            best_epoch = epoch + 1
            best_acc = final_acc
            best_model = model.state_dict()           
    model.load_state_dict(best_model)
    test_acc, test_loss = nnAval(model, test_loader)   
    return train_lost_list, val_loss_list, val_acc_list, test_acc, test_loss, best_epoch



def nnC(train_loader, val_loader, test_loader, num_epoch):
    model = models.resnet18(pretrained=True)
    model._modules['fc'] = torch.nn.modules.Linear(in_features=512, out_features=102, bias=True)
    ### freeze first 8 layers
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 9:
            for param in child.parameters():
                param.requires_grad = False
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    model.to(device)
    
    best_epoch = 0
    best_acc = 0
    best_model = None
    train_lost_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(num_epoch):
        print('Epoch', epoch+1, '/', num_epoch,':')
        train_loss = nnAtrain(model, train_loader, optimizer)
        final_acc, val_loss = nnAval(model, val_loader)
        
        train_lost_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(final_acc)
        if final_acc > best_acc:
            best_epoch = epoch + 1
            best_acc = final_acc
            best_model = model.state_dict()           
    model.load_state_dict(best_model)
    test_acc, test_loss = nnAval(model, test_loader)   
    return train_lost_list, val_loss_list, val_acc_list, test_acc, test_loss, best_epoch


def plot_graph(train_lost_list, val_loss_list, val_acc_list, test_acc, test_loss, best_epoch, flag):
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(len(train_lost_list)), train_lost_list)
    ax1.plot(np.arange(len(val_loss_list)), val_loss_list)
    ax1.set_title('Training and Validation Loss for Network ' + flag)
    ax1.legend(['training loss', 'validation loss'], loc='upper right')
    mp.savefig('figures/nn'+flag+'_fig1.png')
    plt.show()
    
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(len(val_acc_list)), val_acc_list)
    ax2.set_title('Validation accuracy for Network ' + flag)
    ax2.legend(['validation accuracy'], loc='upper right')
    mp.savefig('figures/nn'+flag+'_fig2.png')
    plt.show()
    
    print('The best epoch is', best_epoch)
    print('The testing accuracy is', test_acc)
    print('The testing loss is', test_loss)



if __name__ == '__main__':
    img_path='images/flowers_data/jpg/'
    train_path='trainfile.txt'
    val_path='valfile.txt'
    test_path='testfile.txt'
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
    
    ### Create dataset
    train_img, train_label = create_dataset(img_path, train_path, val_path, test_path, transform, 'train')
    val_img, val_label = create_dataset(img_path, train_path, val_path, test_path, transform, 'val')
    test_img, test_label = create_dataset(img_path, train_path, val_path, test_path, transform, 'test')
    
    ### Create dataloader
    print('##############################')
    print('Dataloaders...')
    dtr = torch.utils.data.TensorDataset(train_img, train_label)
    dv = torch.utils.data.TensorDataset(val_img, val_label)
    dte = torch.utils.data.TensorDataset(test_img, test_label)
    
    loadertr = torch.utils.data.DataLoader(dtr, batch_size=64, shuffle=True)
    loaderval = torch.utils.data.DataLoader(dv, batch_size=64, shuffle=False)
    loadertest = torch.utils.data.DataLoader(dte, batch_size=64, shuffle=False)
    print('Dataloaders created!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Neural Network A: No pretrained ResNet18
    print('##############################')
    print('No Pre-trained ResNet18.')
    A_train_loss, A_val_loss, A_val_acc, A_test_acc, A_test_loss, A_epoch = nnA(loadertr, loaderval, loadertest, 30)
    plot_graph(A_train_loss, A_val_loss, A_val_acc, A_test_acc, A_test_loss, A_epoch, 'A')
    
    ### Neural Network B: Pretrained ResNet18
    print('##############################')
    print('Pre-trained ResNet18.')
    B_train_loss, B_val_loss, B_val_acc, B_test_acc, B_test_loss, B_epoch = nnB(loadertr, loaderval, loadertest, 30)
    plot_graph(B_train_loss, B_val_loss, B_val_acc, B_test_acc, B_test_loss, B_epoch, 'B')
    
    ### Neural Network C: Freeze first 8 layers ResNet18
    print('##############################')
    print('Freeze first 8 layers ResNet18.')
    C_train_loss, C_val_loss, C_val_acc, C_test_acc, C_test_loss, C_epoch = nnC(loadertr, loaderval, loadertest, 30)
    plot_graph(C_train_loss, C_val_loss, C_val_acc, C_test_acc, C_test_loss, C_epoch, 'C')
    


