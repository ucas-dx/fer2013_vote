#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 14:25
# @Author  : Denxun
# @FileName: es.py
# @Software: PyCharm
import numpy as np

# 假设有三个模型的预测结果
model1_predictions = np.array([0, 1, 1, 0, 1])
model2_predictions = np.array([1, 0, 1, 1, 0])
model3_predictions = np.array([0, 0, 1, 0, 1])

# 进行投票
ensemble_predictions = np.zeros_like(model1_predictions)
for i in range(len(ensemble_predictions)):
    votes = [model1_predictions[i], model2_predictions[i], model3_predictions[i]]
    ensemble_predictions[i] = np.argmax(np.bincount(votes))

print("Ensemble Predictions:", ensemble_predictions)
