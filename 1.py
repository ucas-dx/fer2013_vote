# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/5/16 18:58
# @Function:
from torchvision import models
import torch.nn as nn
from torchvision.models import DenseNet121_Weights
import torch
# resnet34=models.resnet34(pretrained=True)
# resnet50=models.resnet50(pretrained=True)
# alexnet=models.alexnet(pretrained=True)
# efficientnet_b5=models.efficientnet_b5(pretrained=True)
# vit_b_16=models.vit_b_16(pretrained=True)
# densenet121=models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
# for param in densenet121.parameters():
#     param.requires_grad = True
# densenet121.classifier = nn.Sequential(nn.Linear(1024,512,bias=True),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,7,bias=True))  # 修改分类层的输出维度为7
# for param in densenet121.classifier.parameters():
#     param.requires_grad = True
# print(densenet121)

# print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))



# 冻结预训练模型的所有参数
# #***********************************
# for param in resnet34.parameters():
#     param.requires_grad = True
# resnet34.fc = nn.Sequential(nn.Linear(512, 7,bias=True))  # 修改分类层的输出维度为7
# for param in resnet34.fc.parameters():
#     param.requires_grad = True
# #************************************
# for param in resnet50.parameters():
#     param.requires_grad = True
# resnet50.fc = nn.Sequential(nn.Linear(2048,512,bias=True),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,7,bias=True))  # 修改分类层的输出维度为7
# for param in resnet50.fc.parameters():
#     param.requires_grad = True
# #*************************************
# for param in efficientnet_b5.parameters():
#     param.requires_grad = False
# efficientnet_b5.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(in_features=2048, out_features=512, bias=True),
#     nn.BatchNorm1d(512),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(in_features=512, out_features=7, bias=True)) # 修改分类层的输出维度为7
# for param in efficientnet_b5.classifier.parameters():
#     param.requires_grad = True
# #*************************************
# for param in vit_b_16.parameters():
#     param.requires_grad = True
# vit_b_16.heads = nn.Sequential(nn.Linear(768,256,bias=True),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,7,bias=True))  # 修改分类层的输出维度为7
# for param in vit_b_16.heads.parameters():
#     param.requires_grad = True
# #*************************************
# for param in alexnet.parameters():
#     param.requires_grad = False
# alexnet.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(in_features=9216, out_features=4096, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5, inplace=False),
#    nn.Linear(in_features=4096, out_features=512, bias=True),
#     nn.ReLU(inplace=True),
#     nn.Dropout(p=0.5, inplace=False),
#     nn.Linear(in_features=512, out_features=7, bias=True)) # 修改分类层的输出维度为7
# for param in alexnet.classifier.parameters():
#     param.requires_grad = True

model=models.vit_b_16()
print(nn.Transformer)