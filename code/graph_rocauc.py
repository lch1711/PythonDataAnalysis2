#!/usr/bin/env python
# coding: utf-8

# 관련 라이브러리를 호출합니다.
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ROC 곡선을 시각화하는 함수를 정의합니다.
def plot_roc(y1_real, y1_prob, y2_real, y2_prob, pos = None):
    
    fpr1, tpr1, _ = roc_curve(y_true = y1_real, y_score = y1_prob, pos_label = pos)
    fpr2, tpr2, _ = roc_curve(y_true = y2_real, y_score = y2_prob, pos_label = pos)
    
    auc1 = auc(x = fpr1, y = tpr1)
    auc2 = auc(x = fpr2, y = tpr2)
    
    plt.plot(fpr1, tpr1, color = 'r', label = 'AUC1 = %0.4f' % auc1)
    plt.plot(fpr2, tpr2, color = 'b', label = 'AUC2 = %0.4f' % auc2)
    plt.plot([0, 1], [0, 1], color = 'k', linestyle = '--')
    
    plt.title(label = 'ROC Curve')
    plt.xlabel(xlabel = 'FPR')
    plt.ylabel(ylabel = 'TPR')
    plt.legend(loc = 'lower right')
    plt.show()
