#!/usr/bin/env python
# coding: utf-8

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# PCA visualization functions
# scree plot
def plot_scree(x):
    n = len(x)
    plt.plot(x, 'bs--')
    plt.axhline(y = 1, color = 'red', linestyle = '--')
    plt.title('Scree Plot')
    plt.xlabel('Number of PC')
    plt.ylabel('Variance');

# biplot
def plot_biplot(score, coefs, x = 1, y = 2, scale = 1):
    xs = score.iloc[:, x-1]
    ys = score.iloc[:, y-1]
        
    plt.scatter(xs, ys)
    plt.axvline(x = 0, color = '0.5', linestyle = '--')
    plt.axhline(y = 0, color = '0.5', linestyle = '--')
    
    n = score.shape[1]
    for i in range(n):
        plt.arrow(
            x = 0, 
            y = 0, 
            dx = coefs.iloc[i, x-1] * scale, 
            dy = coefs.iloc[i, y-1] * scale, 
            color = 'red', 
            alpha = 0.5
        )
        
        plt.text(
            x = coefs.iloc[i, x-1] * (scale + 0.5), 
            y = coefs.iloc[i, y-1] * (scale + 0.5), 
            s = coefs.index[i], 
            color = 'darkred', 
            ha = 'center', 
            va = 'center'
        )
    
    plt.xlabel('PC{}'.format(x))
    plt.ylabel('PC{}'.format(y))
    
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.grid()


# EDA functions
def plot_box_group(data, x, y, palette = 'Spectral'):
    avg = data.groupby(x).mean()[[y]].sort_index().reset_index()
    sns.boxplot(data = data, x = x, y = y, order = avg[x], palette = palette)
    sns.scatterplot(data = avg, x = x, y = y, color = 'red', s = 50, 
                    edgecolor = 'black', linewidth = 0.5)
    plt.axhline(y = data[y].mean(), color = 'red', linestyle = '--')    
    plt.title(label = f'{x} 범주별 {y}의 평균 비교');

def plot_scatter(data, x, y, color = 'blue'):
    sns.scatterplot(data = data, x = x, y = y, color = color)
    plt.title(label = f'{x}와 {y}의 관계');

def plot_regression(data, x, y, color = '0.5', size = 15):
    x_min = data[x].min()
    x_max = data[x].max()
    sns.regplot(data = data, x = x, y = y, 
                scatter_kws = {'color': color, 's': size},
                line_kws = {'color': 'red', 'linewidth': 1.5})
    plt.xlim(x_min * 0.95, x_max * 1.05)
    plt.title(label = f'{x}와 {y}의 관계');


# Metrics for Classification: ROC, AUC
from sklearn.metrics import roc_curve, auc
def plot_roc(y1_true, y1_prob, y2_true, y2_prob, pos = None):
    y_class = np.unique(ar = y1_true)
    idx = np.where(y_class == pos)[0][0]
    
    if y1_prob.ndim == 2:
        y1_prob = y1_prob[:, idx]
    
    if y2_prob.ndim == 2:
        y2_prob = y2_prob[:, idx]
    
    fpr1, tpr1, _ = roc_curve(y_true = y1_true, y_score = y1_prob, pos_label = pos)
    fpr2, tpr2, _ = roc_curve(y_true = y2_true, y_score = y2_prob, pos_label = pos)
    
    auc1 = auc(x = fpr1, y = tpr1)
    auc2 = auc(x = fpr2, y = tpr2)
    
    plt.plot(fpr1, tpr1, color = 'r', label = 'AUC1 = %0.4f' % auc1)
    plt.plot(fpr2, tpr2, color = 'b', label = 'AUC2 = %0.4f' % auc2)
    plt.plot([0, 1], [0, 1], color = 'k', linestyle = '--')
    
    plt.title(label = 'ROC Curve')
    plt.xlabel(xlabel = 'FPR')
    plt.ylabel(ylabel = 'TPR')
    plt.legend(loc = 'lower right');


# Variance Inflation factor
def vif(X):
    import statsmodels.stats.outliers_influence as oi
    func = oi.variance_inflation_factor
    ncol = X.shape[1]
    vifs = [func(exog = X.values, exog_idx = i) for i in range(1, ncol)]
    return pd.DataFrame(data = vifs, index = X.columns[1:]).T


# Metrics for Regression: MSE, RMSE, MSE, MAPE
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
def regmetrics(y_true, y_pred):
    MSE = mean_squared_error(y_true = y_true, y_pred = y_pred).round(2)
    RMSE = (mean_squared_error(y_true = y_true, y_pred = y_pred)**(1/2)).round(2)
    MAE = mean_absolute_error(y_true = y_true, y_pred = y_pred).round(2)
    MAPE = mean_absolute_percentage_error(y_true = y_true, y_pred = y_pred).round(4)
    result = pd.DataFrame(data = [MSE, RMSE, MAE, MAPE]).T
    result.columns = ['MSE', 'RMSE', 'MAE', 'MAPE']
    return result


# Feature Importance function  
def plot_feature_importance(model, column_names):
    imp = pd.DataFrame(
        data = model.feature_importances_.round(2), 
        index = column_names, 
        columns = ['Imp']
    )
    imp = imp.sort_values(by = 'Imp', ascending = False)
    imp = imp.reset_index()
    
    sns.barplot(data = imp, x = 'Imp', y = 'index')
    
    for index, row in imp.iterrows():
        plt.text(x = row['Imp'], y = index, s = row['Imp'], 
                 ha = 'left', va = 'center', fontsize = 11)
    
    plt.xlim(0, imp['Imp'].max()*1.1)
    plt.title(label = '입력변수의 중요도')
    plt.xlabel(xlabel = 'Feature Importances')
    plt.ylabel(ylabel = 'Feature');


# Pruning function
def plot_ccp(alphas, score):
    tst_error = pd.DataFrame({'alphas': alphas, 'score': score})
    sns.pointplot(data = tst_error, x = 'alphas', y = 'score', scale = 0.5)
    plt.title(label = '비용 복잡도 파라미터에 따른 시험셋의 성능 변화')
    plt.xlabel(xlabel = '비용 복잡도 파라미터')
    plt.ylabel(ylabel = '정확도')
    plt.xticks(rotation = 30);
    # plt.legend().set_title('')
