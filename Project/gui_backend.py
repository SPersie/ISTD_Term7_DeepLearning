"""
Created on Tue Mar 19 00:52:35 2019

@author: lixingxuan
"""

import torch
from torchvision import transforms, models
from PIL import Image
import json

def img_predict(img_as_np):
    labels = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']
    model = models.resnet18(pretrained=True)
    model._modules['fc'] = torch.nn.Sequential(torch.nn.modules.Linear(in_features=512, out_features=20, bias=True),
                                        torch.nn.Sigmoid())
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
    model.load_state_dict(torch.load('best_model.dat', map_location='cpu'))
    img_as_img = Image.fromarray(img_as_np)
    img_as_tensor = transform(img_as_img)
    temp_img = img_as_tensor.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(temp_img)
    opt = (output>0.4)*1
    opt_np = opt.numpy()
    label = []
    for i in range(20):
        if opt_np[0][i] == 1:
            label.append(labels[i])
    return label

def top50():
    with open('data.json', 'r') as f:
        opt_dict = json.load(f)
    return opt_dict

