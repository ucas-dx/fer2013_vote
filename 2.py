# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/5/18 13:39
# @Function:
#模型和预训练模型载入
import torch.nn as  nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, AlexNet_Weights, \
    EfficientNet_B5_Weights, ViT_B_16_Weights, DenseNet121_Weights
import torch
#Easy_CNN = Easy_CNN()
resnet18=models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
efficientnet_b5 = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
vit_b_16 = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
densenet121 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
# 选择冻结预训练模型的参数
for param in densenet121.parameters():
    param.requires_grad = True
densenet121.classifier = nn.Sequential(nn.Dropout(0.2),nn.Linear(1024, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 7, bias=True)) # 修改分类层的输出维度为7
for param in densenet121.classifier.parameters():
    param.requires_grad = True
# ***********************************
for param in resnet18.parameters():
    param.requires_grad = True
resnet18.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(512, 7, bias=True))  # 修改分类层的输出维度为7,添加Dropout(0.5)
for param in resnet18.fc.parameters():
    param.requires_grad = True
# ************************************
# ***********************************
for param in resnet34.parameters():
    param.requires_grad = True
resnet34.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(512, 7, bias=True))  # 修改分类层的输出维度为7
for param in resnet34.fc.parameters():
    param.requires_grad = True
# ************************************
for param in resnet50.parameters():
    param.requires_grad = True
resnet50.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(2048, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(512, 7, bias=True))  # 修改分类层的输出维度为7
for param in resnet50.fc.parameters():
    param.requires_grad = True
# *************************************
for param in efficientnet_b5.parameters():
    param.requires_grad = True
efficientnet_b5.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                           nn.Linear(in_features=2048, out_features=512, bias=True),
                                           nn.BatchNorm1d(512),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(p=0.5, inplace=False),
                                           nn.Linear(in_features=512, out_features=7, bias=True))  # 修改分类层的输出维度为7
for param in efficientnet_b5.classifier.parameters():
    param.requires_grad = True
# *************************************
for param in vit_b_16.parameters():
    param.requires_grad = True
vit_b_16.heads = nn.Sequential(nn.Dropout(0.2),nn.Linear(768, 256, bias=True), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),nn.Linear(256, 7, bias=True))  # 修改分类层的输出维度为7
for param in vit_b_16.heads.parameters():
    param.requires_grad = True
# *************************************
for param in alexnet.parameters():
    param.requires_grad = False
alexnet.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(in_features=9216, out_features=4096, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(in_features=4096, out_features=512, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5, inplace=False),
                                   nn.Linear(in_features=512, out_features=7, bias=True))  # 修改分类层的输出维度为7
for param in alexnet.classifier.parameters():
    param.requires_grad = True
device = torch.device("cuda")
modellist = [resnet18, resnet34, resnet50, alexnet, efficientnet_b5, vit_b_16, densenet121]
modedict=['resnet18model.pth','resnet34model.pth','resnet50model.pth','alexnetmodel.pth','efficientnet_b5model.pth','vit_b_16model.pth','densenet121model.pth']
criterion = nn.CrossEntropyLoss()
for i ,model in enumerate(modellist) :
    torch.load(modedict[i])
    print(f"读取模型{modedict[i]}成功！")
