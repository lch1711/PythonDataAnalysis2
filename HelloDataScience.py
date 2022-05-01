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
    avg = data.groupby(x).mean()[y].reset_index()
    sns.boxplot(data = data, x = x, y = y, order = avg[x], palette = palette)
    sns.scatterplot(data = avg, x = avg.index, y = y, color = 'red', s = 50, 
                    edgecolor = 'black', linewidth = 0.5)
    plt.axhline(y = data[y].mean(), color = 'red', linestyle = '--')    
    plt.title(label = f'{x} 범주별 {y}의 평균 비교');

def plot_scatter(data, x, y, color = 'blue'):
    sns.scatterplot(data = data, x = x, y = y, color = color)
    plt.title(label = f'{x}와(과) {y}의 관계');

def plot_regression(data, x, y, color = '0.5', size = 15):
    x_min = data[x].min()
    x_max = data[x].max()
    sns.regplot(data = data, x = x, y = y, 
                scatter_kws = {'color': color, 's': size},
                line_kws = {'color': 'red', 'linewidth': 1.5})
    plt.xlim(x_min * 0.95, x_max * 1.05)
    plt.title(label = f'{x}와(과) {y}의 관계');

def plot_y_freq(data, y, color = None):
    freq = data[y].value_counts().sort_index().reset_index()
    freq.columns = [y, 'freq']
    sns.barplot(data = freq, x = y, y = 'freq', color = color)
    for index, row in freq.iterrows():
        plt.text(x = index, y = row['freq'] + 5, s = row['freq'], 
                 fontsize = 12, ha = 'center', va = 'bottom', c = 'black')
    plt.title(label = '목표변수의 범주별 빈도수 비교')
    plt.ylim(0, freq['freq'].max() * 1.1);

def plot_xy_freq(data, x, y, color = None):
    xlabel = data[x].unique()
    xlabel.sort()
    sns.countplot(data = data, x = x, hue = y, order = xlabel)
    plt.title(label = f'{x}의 범주별 {y}의 빈도수 비교');


# Stacked Barplot
def plot_stack_freq(data, x, y, kind = 'bar', pal = None):
    p = data[y].unique().size
    pv = pd.pivot_table(data = data, index = x, columns = y, aggfunc = 'count')
    pv = pv.iloc[:, 0:p].sort_index()
    pv.columns = pv.columns.droplevel(level = 0)
    pv.columns.name = None
    pv = pv.reset_index()
    
    cols = pv.columns[1:]
    cumsum = pv[cols].cumsum(axis = 1)
    
    if type(pal) == list:
        pal = sns.set_palette(sns.color_palette(pal))
    
    pv.plot(x = x, kind = kind, stacked = True, rot = 0, 
            title = f'{x}의 범주별 {y}의 빈도수 비교', 
            legend = 'reverse', colormap = pal)
    
    plt.legend(loc = 'lower right')
    
    if kind == 'bar':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                plt.text(x = i, y = v1 - v2/2, s = v2, fontsize = 12, 
                         ha = 'center', va = 'center', c = 'black');
    elif kind == 'barh':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                plt.text(x = v1 - v2/2, y = i, s = v2, fontsize = 12, 
                         ha = 'center', va = 'center', c = 'black');

def plot_stack_prop(data, x, y, kind = 'bar', pal = None):
    p = data[y].unique().size
    pv = pd.pivot_table(data = data, index = x, columns = y, aggfunc = 'count')
    pv = pv.iloc[:, 0:p].sort_index()
    pv.columns = pv.columns.droplevel(level = 0)
    pv.columns.name = None
    pv = pv.reset_index()
    
    cols = pv.columns[1:]
    rowsum = pv[cols].apply(func = sum, axis = 1)
    pv[cols] = pv[cols].div(rowsum, 0) * 100
    cumsum = pv[cols].cumsum(axis = 1)
    
    if type(pal) == list:
        pal = sns.set_palette(sns.color_palette(pal))
        
    pv.plot(x = x, kind = kind, stacked = True, rot = 0, 
            title = f'{x}의 범주별 {y}의 상대도수 비교', 
            legend = 'reverse', colormap = pal, mark_right = True)
    
    plt.legend(loc = 'lower right')
    
    if kind == 'bar':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                v3 = f'{np.round(v2, 1)}%'
                plt.text(x = i, y = v1 - v2/2, s = v3, fontsize = 12, 
                         ha = 'center', va = 'center', c = 'black');
    elif kind == 'barh':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                v3 = f'{np.round(v2, 1)}%'
                plt.text(x = v1 - v2/2, y = i, s = v3, fontsize = 12, 
                         ha = 'center', va = 'center', c = 'black');


# Metrics for Classification: ROC, AUC
from sklearn.metrics import roc_curve, auc

def plot_roc(y_true, y_prob, pos = None, color = None):
    y_class = y_true.value_counts().sort_index()
    
    if pos == None:
        pos = y_class.loc[y_class == y_class.min()].index[0]
    
    idx = np.where(y_class.index == pos)[0][0]
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, idx]
        
    fpr, tpr, _ = roc_curve(y_true = y_true, y_score = y_prob, pos_label = pos)
    auc_ = auc(x = fpr, y = tpr).round(4)
    
    plt.plot(fpr, tpr, color = color, label = f'AUC = {auc_}')
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


# Coefficients for Regression
def coef(model):
    if model.coef_.ndim == 1:
        coefs = pd.Series(data = model.coef_, index = model.feature_names_in_)
    elif model.coef_.ndim == 2:
        coefs = pd.Series(data = model.coef_[0], index = model.feature_names_in_)
    else:
        coefs = pd.Series()
    return coefs


# Metrics for Regression: MSE, RMSE, MSE, MAPE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def regmetrics(y_true, y_pred):
    MSE = mean_squared_error(y_true = y_true, y_pred = y_pred).round(2)
    RMSE = (mean_squared_error(y_true = y_true, y_pred = y_pred)**(1/2)).round(2)
    MAE = mean_absolute_error(y_true = y_true, y_pred = y_pred).round(2)
    MAPE = mean_absolute_percentage_error(y_true = y_true, y_pred = y_pred).round(4)
    result = pd.DataFrame(data = [MSE, RMSE, MAE, MAPE]).T
    result.columns = ['MSE', 'RMSE', 'MAE', 'MAPE']
    return result


# Metrics for Classification: Confusion Matrix, F1 Score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def clfmetrics(y_true, y_pred):
    print('• Confusion Matrix')
    print(confusion_matrix(y_true = y_true, y_pred = y_pred))
    print()
    print('• Classification Report')
    print(classification_report(y_true = y_true, y_pred = y_pred, digits = 4))


# Feature Importance function  
def plot_feature_importance(model):
    imp = pd.DataFrame(
        data = model.feature_importances_.round(2), 
        index = model.feature_names_in_, 
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


# Get the name of an object
def name_of_object(obj):
    try:
        return obj.__name__
    except AttributeError:
        pass

    for name, value in globals().items():
        if value is obj and not name.startswith('_'):
            return name


# Tree model Visualization
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
# import graphviz

def plot_tree(model, fileName, className = None):
    
    # fileName = name_of_object(obj = model)
    
    if type(model) == DecisionTreeRegressor:
        export_graphviz(
            decision_tree = model,
            out_file = f'{fileName}.dot',
            feature_names = model.feature_names_in_,
            filled = True,
            leaves_parallel = False,
            impurity = True
        )
    elif type(model) == DecisionTreeClassifier:
        if className == None:
            className = model.classes_
        export_graphviz(
            decision_tree = model,
            out_file = f'{fileName}.dot',
            class_names = className,
            feature_names = model.feature_names_in_,
            filled = True,
            leaves_parallel = False,
            impurity = True
        )
    
    dot_graph = open(file = f'{fileName}.dot', mode = 'rt').read()
    graph = graphviz.Source(source = dot_graph, format = 'png')
    graph.render(filename = fileName)


# Tree Model Pruning
def plot_ccp(data, x, y, ylab = '정확도', color = 'blue', xangle = None):
    sns.lineplot(data = data, x = x, y = y, color = color, 
                 drawstyle = 'steps-pre', label = y)
    sns.scatterplot(data = data, x = x, y = y, color = color, s = 50)
    plt.title(label = f'비용 복잡도 파라미터에 따른 {ylab}의 변화')
    plt.xlabel(xlabel = '비용 복잡도 파라미터(알파)')
    plt.ylabel(ylabel = ylab)
    plt.xticks(rotation = xangle)
    plt.legend();


# OOB Estimation
def plot_oob(data, x, y, color = 'blue', xangle = None, label = '정확도'):
    sns.lineplot(data = data, x = x, y = y, color = color)
    plt.title(label = f'OOB {label} 변화')
    plt.xlabel(xlabel = '개별 나무모형 개수')
    plt.ylabel(ylabel = f'OOB {label}')
    plt.xticks(rotation = xangle);


## End of Document
