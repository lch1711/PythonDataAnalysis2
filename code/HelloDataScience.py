# Import libraries for Python Advanced
import os
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oi
from statsmodels.stats.outliers_influence import OLSInfluence

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef

# Import libraries for Python Machine Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz
import inspect
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Draw a plot
plt.plot(0, 0)
plt.close()

# Set line width
plt.rc(group = 'lines', linewidth = 0.5)


# EDA functions
# outlier properties for box plot
outProps = {
    'marker': 'o', 
    'markersize': 3, 
    'markerfacecolor': 'pink',
    'markeredgecolor': 'red', 
    'markeredgewidth': 0.2
}

def plot_box_group(
    data, 
    x, 
    y, 
    pal = None
):
    avg = data.groupby(x)[y].mean().reset_index()
    sns.boxplot(
        data = data, 
        x = x, 
        y = y, 
        order = avg[x], 
        palette = pal, 
        flierprops = outProps
    )
    sns.scatterplot(
        data = avg, 
        x = avg.index, 
        y = y, 
        color = 'red', 
        s = 30, 
        edgecolor = 'black', 
        linewidth = 0.5
    )
    plt.axhline(
        y = data[y].mean(), 
        color = 'red', 
        linestyle = '--'
    )
    plt.title(label = f'{x} 범주별 {y}의 평균 비교');

def plot_scatter(
    data, 
    x, 
    y, 
    color = '0.3'
):
    sns.scatterplot(
        data = data, 
        x = x, 
        y = y, 
        color = color
    )
    plt.title(label = f'{x}와(과) {y}의 관계');

def plot_regression(
    data, 
    x, 
    y, 
    color = '0.3', 
    size = 15
):
    x_min = data[x].min()
    x_max = data[x].max()
    sns.regplot(
        data = data, 
        x = x, 
        y = y, 
        ci = None, 
        scatter_kws = {'color': color, 'edgecolor': '1', 
                       's': size, 'linewidth': 0.5},
        line_kws = {'color': 'red', 'linewidth': 1.5}
    )
    plt.title(label = f'{x}와(과) {y}의 관계')
    # plt.xlim(x_min * 0.9, x_max * 1.1);

def plot_bar_freq(
    data, 
    y, 
    color = None, 
    pal = None
):
    freq = data[y].value_counts().sort_index().reset_index()
    freq.columns = [y, 'freq']
    sns.barplot(
        data = freq, 
        x = y, 
        y = 'freq', 
        color = color, 
        palette = pal
    )
    for index, row in freq.iterrows():
        plt.text(
            x = index, 
            y = row['freq'] + 5, 
            s = row['freq'], 
            ha = 'center', 
            va = 'bottom', 
            c = 'black'
        )
    plt.title(label = '목표변수의 범주별 빈도수 비교')
    plt.ylim(0, freq['freq'].max() * 1.1);

def plot_bar_dodge_freq(
    data, 
    x, 
    y, 
    pal = None
):
    grp = data.groupby(by = [x, y]).count().iloc[:, 0]
    sns.countplot(
        data = data, 
        x = x, 
        hue = y, 
        order = grp.index.levels[0], 
        hue_order = grp.index.levels[1], 
        palette = pal
    )    
    for i, v in enumerate(grp):
        if i % 2 == 0:
            i = i/2 - 0.2
        else:
            i = (i-1)/2 + 0.2
        plt.text(
            x = i, 
            y = v, 
            s = v, 
            ha = 'center', 
            va = 'bottom'
        )
    plt.title(label = f'{x}의 범주별 {y}의 빈도수 비교')
    plt.legend(loc = 'best');

def plot_bar_stack_freq(
    data, 
    x, 
    y, 
    kind = 'bar', 
    pal = None
):
    p = data[y].unique().size
    pv = pd.pivot_table(
        data = data, 
        index = x, 
        columns = y, 
        aggfunc = 'count'
    )
    pv = pv.iloc[:, 0:p].sort_index()
    pv.columns = pv.columns.droplevel(level = 0)
    pv.columns.name = None
    pv = pv.reset_index()
    
    cols = pv.columns[1:]
    cumsum = pv[cols].cumsum(axis = 1)
    
    if type(pal) == list:
        pal = sns.set_palette(sns.color_palette(pal))
    
    pv.plot(
        x = x, 
        kind = kind, 
        stacked = True, 
        rot = 0, 
        title = f'{x}의 범주별 {y}의 빈도수 비교', 
        legend = 'reverse', 
        colormap = pal
    )
    
    plt.legend(loc = 'best')
    
    if kind == 'bar':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                plt.text(
                    x = i, 
                    y = v1 - v2/2, 
                    s = v2, 
                    ha = 'center', 
                    va = 'center', 
                    c = 'black'
                );
    elif kind == 'barh':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                plt.text(
                    x = v1 - v2/2, 
                    y = i, 
                    s = v2, 
                    ha = 'center', 
                    va = 'center', 
                    c = 'black'
                );

def plot_bar_stack_prop(
    data, 
    x, 
    y, 
    kind = 'bar', 
    pal = None
):
    p = data[y].unique().size
    pv = pd.pivot_table(
        data = data, 
        index = x, 
        columns = y, 
        aggfunc = 'count'
    )
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
        
    pv.plot(
        x = x, 
        kind = kind, 
        stacked = True, 
        rot = 0, 
        title = f'{x}의 범주별 {y}의 상대도수 비교', 
        legend = 'reverse', 
        colormap = pal, 
        mark_right = True
    )
    
    plt.legend(loc = 'best')
    
    if kind == 'bar':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                v3 = f'{np.round(v2, 1)}%'
                plt.text(
                    x = i, 
                    y = v1 - v2/2, 
                    s = v3, 
                    ha = 'center', 
                    va = 'center', 
                    c = 'black'
                );
    elif kind == 'barh':
        for col in cols:
            for i, (v1, v2) in enumerate(zip(cumsum[col], pv[col])):
                v3 = f'{np.round(v2, 1)}%'
                plt.text(
                    x = v1 - v2/2, 
                    y = i, 
                    s = v3, 
                    ha = 'center', 
                    va = 'center', 
                    c = 'black'
                );


# Linear Regression Variable Selection
def forward_selection(y, X):
    if 'const' in X.columns:
        X = X.drop(labels = ['const'], axis = 1)
    Xvars = set(X.columns)
    dat = pd.concat(objs = [X, y], axis = 1)
    
    formula = f'{y.name} ~ 1'
    curr_aic = smf.ols(formula = formula, data = dat).fit().aic
    
    selected = []
    while Xvars:
        Xvar_aic = []
        for Xvar in Xvars:
            formula = f'{y.name} ~ {" + ".join(selected + [Xvar])} + 1'
            aic = smf.ols(formula = formula, data = dat).fit().aic
            aic = np.round(a = aic, decimals = 4)
            Xvar_aic.append((aic, Xvar))
        
        Xvar_aic.sort(reverse = True)
        new_aic, best_Xvar = Xvar_aic.pop()
        
        if curr_aic > new_aic:
            Xvars.remove(best_Xvar)
            selected.append(best_Xvar)
            curr_aic = new_aic
        else:
            break
    
    formula = f'{y.name} ~ {" + ".join(selected)} + 1'
    model = smf.ols(formula = formula, data = dat).fit()
    return model

def backward_selection(y, X):
    if 'const' in X.columns:
        X = X.drop(labels = ['const'], axis = 1)
    Xvars = set(X.columns)
    dat = pd.concat(objs = [X, y], axis = 1)
    
    formula = f'{y.name} ~ {" + ".join(list(Xvars))}'
    curr_aic = smf.ols(formula = formula, data = dat).fit().aic
    
    selected = []
    while Xvars:
        Xvar_aic = []
        for Xvar in Xvars:
            sub = dat.drop(labels = selected + [Xvar], axis = 1).copy()
            sub_Xvars = set(sub.columns) - set([y.name])
            formula = f'{y.name} ~ {" + ".join(list(sub_Xvars))} + 1'
            aic = smf.ols(formula = formula, data = sub).fit().aic
            aic = np.round(a = aic, decimals = 4)
            Xvar_aic.append((aic, Xvar))
        
        Xvar_aic.sort(reverse = True)
        new_aic, best_Xvar = Xvar_aic.pop()
        
        if curr_aic > new_aic:
            Xvars.remove(best_Xvar)
            selected.append(best_Xvar)
            curr_aic = new_aic
        else:
            break
    
    dat = dat.drop(labels = selected, axis = 1)
    dat_Xvars = set(dat.columns) - set([y.name])
    
    formula = f'{y.name} ~ {" + ".join(list(dat_Xvars))} + 1'
    model = smf.ols(formula = formula, data = dat).fit()
    return model

def stepwise_selection(y, X):
    if 'const' in X.columns:
        X = X.drop(labels = ['const'], axis = 1)
    Xvars = set(X.columns)
    dat = pd.concat(objs = [X, y], axis = 1)
    
    formula = f'{y.name} ~ 1'
    curr_aic = smf.ols(formula = formula, data = dat).fit().aic
    
    selected = []
    while Xvars:
        Xvar_aic = []
        for Xvar in Xvars:
            formula = f'{y.name} ~ {" + ".join(selected + [Xvar])} + 1'
            aic = smf.ols(formula = formula, data = dat).fit().aic
            aic = np.round(a = aic, decimals = 4)
            Xvar_aic.append((aic, 'add', Xvar))
        
        if selected:
            for Xvar in selected:
                sub = dat[selected + [y.name]].copy()
                sub = sub.drop(labels = [Xvar], axis = 1)
                sub_Xvars = set(sub.columns) - set([y.name])
                formula = f'{y.name} ~ {" + ".join(list(sub_Xvars))} + 1'
                aic = smf.ols(formula = formula, data = sub).fit().aic
                aic = np.round(a = aic, decimals = 4)
                Xvar_aic.append((aic, 'sub', Xvar))
        
        Xvar_aic.sort(reverse = True)
        new_aic, how, best_Xvar = Xvar_aic.pop()
        
        if curr_aic > new_aic and how == 'add':
            Xvars.remove(best_Xvar)
            selected.append(best_Xvar)
            curr_aic = new_aic
        elif curr_aic > new_aic and how == 'sub':
            Xvars.append(best_Xvar)
            selected.remove(best_Xvar)
            curr_aic = new_aic
        elif curr_aic <= new_aic:
            break
    
    formula = f'{y.name} ~ {" + ".join(selected)} + 1'
    model = smf.ols(formula = formula, data = dat).fit()
    return model

def stepwise(y, X, direction = 'both'):
    if direction == 'forward':
        model = forward_selection(y, X)
    elif direction == 'backward':
        model = backward_selection(y, X)
    elif direction == 'both':
        model = stepwise_selection(y, X)
    else:
        model = None
    return model


# Linear Regression Diagnosis
def regressionDiagnosis(model):
    plt.figure(figsize = (10, 10), dpi = 100)
    
    # Linearity
    # lowess: locally weighted linear regression
    ax1 = plt.subplot(2, 2, 1)
    sns.regplot(
        x = model.fittedvalues, 
        y = model.resid, 
        lowess = True, 
        scatter_kws = dict(color = '0.8', ec = '0.3', s = 15),
        line_kws = dict(color = 'red', lw = 1), 
        ax = ax1
    )
    plt.axhline(
        y = 0, 
        color = '0.5', 
        lw = 1, 
        ls = '--'
    )
    plt.title(
        label = 'Residuals vs Fitted', 
        fontdict = dict(size = 16)
    )
    plt.xlabel(
        xlabel = 'Fitted values', 
        fontdict = dict(size = 15)
    )
    plt.ylabel(
        ylabel = 'Residuals', 
        fontdict = dict(size = 15)
    )
    
    # Normality
    ax2 = plt.subplot(2, 2, 2)
    
    # Standardized residuals
    stdres = stats.zscore(a = model.resid)
    
    # Theoretical Quantiles
    (x, y), _ = stats.probplot(x = stdres)
    
    # Q-Q plot
    sns.scatterplot(
        x = x, 
        y = y, 
        color = '0.8', 
        ec = '0.3', 
        size = 2, 
        legend = False, 
        ax = ax2
    )
    plt.plot(
        [-4, 4], 
        [-4, 4], 
        color = '0.5', 
        lw = 1, 
        ls = '--'
    )
    plt.title(
        label = 'Normal Q-Q', 
        fontdict = dict(size = 16)
    )
    plt.xlabel(
        xlabel = 'Theoretical Quantiles', 
        fontdict = dict(size = 15)
    )
    plt.ylabel(
        ylabel = 'Standardized residuals', 
        fontdict = dict(size = 15)
    )
    
    # Homoscedasticity
    ax3 = plt.subplot(2, 2, 3)
    sns.regplot(
        x = model.fittedvalues, 
        y = np.sqrt(stdres.abs()), 
        lowess = True,
        scatter_kws = dict(color = '0.8', ec = '0.3', s = 15),
        line_kws = dict(color = 'red', lw = 1), 
        ax = ax3
    )
    plt.title(
        label = 'Scale-Location', 
        fontdict = dict(size = 16)
    )
    plt.xlabel(
        xlabel = 'Fitted values', 
        fontdict = dict(size = 15)
    )
    plt.ylabel(
        ylabel = 'Sqrt of Standardized residuals', 
        fontdict = dict(size = 15)
    )

    # Outliers using Cook's distance
    ax4 = plt.subplot(2, 2, 4)
    sm.graphics.influence_plot(
        results = model, 
        criterion = 'cooks', 
        size = 24, 
        plot_alpha = 0.2, 
        ax = ax4
    )
    plt.show()


# Influence Points
def influencePoints(model):
    cd, _ = OLSInfluence(results = model).cooks_distance
    cd = cd.sort_values(ascending = False)
    return cd

# Hat Matrix
def hat_matrix(X):
    X = np.array(object = X)
    XtX = np.matmul(X.transpose(), X)
    XtX_inv = np.linalg.inv(XtX)
    result = np.matmul(
        np.matmul(X, XtX_inv), 
        X.transpose()
    )
    return result

# Leverage: Hat Value
def leverage(X):
    n = X.shape[0]
    hatMat = hat_matrix(X = X)
    X['Leverage'] = np.array([hatMat[i][i] for i in range(n)])
    X = X.iloc[:, -1].sort_values(ascending = False)
    return X

# Standardized residuals
def std_Resid(model):
    stdres = stats.zscore(a = model.resid)
    locs = stdres.abs().sort_values(ascending = False)
    return stdres[locs.index]


# Effect Points for Linear Regression
def augment(model):
    infl = model.get_influence()
    df1 = pd.DataFrame(
        data = {model.model.endog_names: infl.endog},
        index = model.fittedvalues.index
    )
    df2 = pd.DataFrame(
        data = {
            'fitted': model.fittedvalues,
            'resid': model.resid,
            'hat': infl.hat_matrix_diag,
            'sigma': np.sqrt(infl.sigma2_not_obsi),
            'cooksd': infl.cooks_distance[0],
            'std_resid': infl.resid_studentized
        }
    )
    result = pd.concat(objs = [df1, df2], axis = 1)
    return result

# Residual Homoscedasticity Test: Breusch-Pagan Lagrange Multiplier Test
def breushpagan(model):
    test = sm.stats.het_breuschpagan(
        resid = model.resid, 
        exog_het = model.model.exog
    )
    result = pd.DataFrame(
        data = test, 
        index = ['Statistic', 'P-Value', 'F-Value', 'F P-Value']
    ).T
    return result


# Metrics for Regression: MSE, RMSE, MSE, MAPE
def regmetrics(y_true, y_pred):
    MSE = mean_squared_error(
        y_true = y_true, 
        y_pred = y_pred
    )
    
    RMSE = MSE**(1/2)
    
    MSLE = mean_squared_log_error(
        y_true = y_true, 
        y_pred = y_pred
    )
    
    RMSLE = MSLE ** (1/2)
    
    MAE = mean_absolute_error(
        y_true = y_true, 
        y_pred = y_pred
    )
    
    MAPE = mean_absolute_percentage_error(
        y_true = y_true, 
        y_pred = y_pred
    )
    
    result = pd.DataFrame(
        data = [MSE, RMSE, RMSLE, MAE, MAPE], 
        index = ['MSE', 'RMSE', 'RMSLE', 'MAE', 'MAPE']
    ).T
    
    # result = result.round(3)
    return result


# Metrics for Classification: ROC, AUC
def plot_roc(
    y_true, 
    y_prob, 
    pos = None, 
    color = None
):
    y_class = y_true.value_counts().sort_index()
    
    if pos == None:
        pos = y_class.loc[y_class == y_class.min()].index[0]
    
    idx = np.where(y_class.index == pos)[0][0]
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, idx]
        
    fpr, tpr, _ = roc_curve(
        y_true = y_true, 
        y_score = y_prob, 
        pos_label = pos
    )
    
    auc_ = auc(x = fpr, y = tpr)
    
    plt.plot(
        fpr, 
        tpr, 
        color = color, 
        label = f'AUC = {auc_:.4f}', 
        lw = 1.0
    )
    
    plt.plot(
        [0, 1], 
        [0, 1], 
        color = 'k', 
        linestyle = '--'
    )
    
    plt.title(label = 'ROC Curve')
    plt.xlabel(xlabel = 'FPR')
    plt.ylabel(ylabel = 'TPR')
    plt.legend(loc = 'lower right');


# Metrics for Classification: PR, AP
def plot_pr(
    y_true, 
    y_prob, 
    pos = None, 
    name = None, 
    color = None
):
    y_class = y_true.value_counts().sort_index()
    
    if pos == None:
        pos = y_class.loc[y_class == y_class.min()].index[0]
    
    idx = np.where(y_class.index == pos)[0][0]
    
    if y_prob.ndim == 2:
        y_prob = y_prob[:, idx]
    
    pr = PrecisionRecallDisplay.from_predictions(
        y_true = trReal, 
        y_pred = trProb, 
        pos_label = pos, 
        name = name, 
        color = color
    )
    
    plt.title(label = 'PR Curve')
    plt.xlabel(xlabel = 'Recall')
    plt.ylabel(ylabel = 'Precision')
    plt.legend(loc = 'best');


# Variance Inflation factor
def vif(X):
    X2 = X.copy()
    if 'const' not in X2.columns:
        X2.insert(loc = 0, column = 'const', value = 1)
    
    func = oi.variance_inflation_factor
    ncol = X2.shape[1]
    vifs = [func(exog = X2.values, exog_idx = i) for i in range(1, ncol)]
    result = pd.DataFrame(data = vifs, index = X2.columns[1:]).T
    return result

# Coefficients for Regression
def coefs(model):
    if model.coef_.ndim == 1:
        coefs = pd.Series(
            data = model.coef_, 
            index = model.feature_names_in_
        )
    elif model.coef_.ndim == 2:
        coefs = pd.Series(
            data = model.coef_[0], 
            index = model.feature_names_in_
        )
    else:
        coefs = pd.Series()
    return coefs

# Standardized Coefficients
def std_coefs(model):
    model_type = str(type(model.model))
    X = pd.DataFrame(
        data = model.model.exog, 
        columns = model.model.exog_names
    )
    if 'OLS' in model_type:
        y = model.model.endog
        result = model.params * (X.std() / y.std())
    elif 'GLM' in model_type:
        y = 1
        result = model.params * (X.std() / 1)
    return result


# Metrics for Classification: Confusion Matrix, F1 Score
def clfmetrics(y_true, y_pred):
    print('▶ Confusion Matrix')
    print(
        confusion_matrix(
            y_true = y_true, 
            y_pred = y_pred
        )
    )
    
    print()
    
    print('▶ Classification Report')
    print(
        classification_report(
            y_true = y_true, 
            y_pred = y_pred, 
            digits = 4
        )
    )


# Classification metrics with Cutoff for Logistic Regression
# Plus, Matthew's Correlation coefficient
def clfCutoffs(y_true, y_prob):
    cutoffs = np.linspace(0, 1, 101)
    sens = []
    spec = []
    prec = []
    mccs = []
    
    for cutoff in cutoffs:
        pred = np.where(y_prob >= cutoff, 1, 0)
        clfr = classification_report(
            y_true = y_true, 
            y_pred = pred, 
            output_dict = True, 
            zero_division = True
        )
        sens.append(clfr['1']['recall'])
        spec.append(clfr['0']['recall'])
        prec.append(clfr['1']['precision'])
        
        mcc = matthews_corrcoef(
            y_true = y_true, 
            y_pred = pred
        )
        mccs.append(mcc)
        
    result = pd.DataFrame(
        data = {
            'Cutoff': cutoffs, 
            'Sensitivity': sens, 
            'Specificity': spec, 
            'Precision': prec, 
            'MCC': mccs
        }
    )
    
    # The Optimal Point is the sum of Sensitivity and Specificity.
    result['Optimal'] = result['Sensitivity'] + result['Specificity']
    
    # TPR and FPR for ROC Curve.
    result['TPR'] = result['Sensitivity']
    result['FPR'] = 1 - result['Specificity']
    
    # Set Column name.
    cols = ['Cutoff', 'Sensitivity', 'Specificity', 'Optimal', \
            'Precision', 'TPR', 'FPR', 'MCC']
    result = result[cols]
    return result

# Draw ROC Curve with Optimal Cutoff
def EpiROC(obj):
    
    # Draw ROC curve
    sns.lineplot(
        data = obj, 
        x = 'FPR', 
        y = 'TPR', 
        color = 'black'
    )
    plt.title(label = '최적의 분리 기준점 탐색')
    
    # Draw diagonal line
    plt.plot(
        [0, 1], 
        [0, 1], 
        color = '0.5', 
        ls = '--'
    )
    
    # Add the Optimal Point
    opt = obj.iloc[[obj['Optimal'].argmax()]]
    sns.scatterplot(
        data = opt, 
        x = 'FPR', 
        y = 'TPR', 
        color = 'red'
    )
    
    # Add tangent line
    optX = opt['FPR'].iloc[0]
    optY = opt['TPR'].iloc[0]
    
    # x1 = optX - 0.1
    # y1 = optY - 0.1
    # x2 = optX + 0.1
    # y2 = optY + 0.1
    # 
    # plt.plot(
    #     [x1, x2],
    #     [y1, y2],
    #     color = 'red',
    #     lw = 0.5,
    #     ls = '-.'
    # )
    
    b = optY - optX
    plt.plot(
        [0, 1-b], 
        [b, 1], 
        color = 'red', 
        lw = 0.5, 
        ls = '-.'
    )
    
    # Add text
    plt.text(
        x = opt['FPR'].values[0] - 0.01, 
        y = opt['TPR'].values[0] + 0.01, 
        s = f"Cutoff = {opt['Cutoff'].round(2).values[0]}", 
        ha = 'right', 
        va = 'bottom'
    );


# PCA visualization functions
# scree plot
def plot_scree(x):
    n = len(x)
    xticks = range(1, n+1)
    sns.lineplot(
        x = xticks, 
        y = x, 
        color = 'blue',
        ls = '--', 
        marker = 'o'
    )
    plt.xticks(ticks = xticks)
    plt.axhline(
        y = 1, 
        color = 'red', 
        linestyle = '--'
    )
    plt.title(label = 'Scree Plot')
    plt.xlabel(xlabel = 'Number of PC')
    plt.ylabel(ylabel = 'Variance');

# biplot
def plot_biplot(
    score, 
    coefs, 
    x = 1, 
    y = 2, 
    scale = 1
):
    xs = score.iloc[:, x-1]
    ys = score.iloc[:, y-1]
    
    sns.scatterplot(
        x = xs, 
        y = ys, 
        fc = 'silver',
        ec = 'black',
        s = 15, 
        alpha = 0.2
    )
    plt.axvline(
        x = 0, 
        color = '0.5', 
        linestyle = '--'
    )
    plt.axhline(
        y = 0, 
        color = '0.5', 
        linestyle = '--'
    )
    
    n = score.shape[1]
    for i in range(n):
        plt.arrow(
            x = 0, 
            y = 0, 
            dx = coefs.iloc[i, x-1] * scale, 
            dy = coefs.iloc[i, y-1] * scale, 
            color = 'red',
            lw = 0.5,
            alpha = 0.5
        )
        
        plt.text(
            x = coefs.iloc[i, x-1] * (scale + 0.5), 
            y = coefs.iloc[i, y-1] * (scale + 0.5), 
            s = coefs.index[i], 
            color = 'darkred', 
            ha = 'center', 
            va = 'center', 
            fontsize = 8
        )
    
    plt.xlabel(xlabel = 'PC{}'.format(x))
    plt.ylabel(ylabel = 'PC{}'.format(y))
    
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.grid()


# Feature Importance function  
def plot_feature_importance(model, pal = 'Spectral'):
    
    if 'LGBM' in str(type(model)):
        imp = pd.DataFrame(
            data = model.feature_importances_, 
            index = model.feature_name_, 
            columns = ['Imp']
        )
    else:
        imp = pd.DataFrame(
            data = model.feature_importances_, 
            index = model.feature_names_in_, 
            columns = ['Imp']
        )
    
    
    imp = imp.sort_values(by = 'Imp', ascending = False)
    imp = imp.reset_index()
    
    sns.barplot(
        data = imp, 
        x = 'Imp', 
        y = 'index', 
        palette = pal
    )
    
    for index, row in imp.iterrows():
        plt.text(
            x = row['Imp'], 
            y = index, 
            s = np.round(row['Imp'], 3), 
            ha = 'left', 
            va = 'center'
        )
    
    plt.xlim(0, imp['Imp'].max() * 1.2)
    plt.title(label = '입력변수의 중요도')
    plt.xlabel(xlabel = 'Feature Importances')
    plt.ylabel(ylabel = 'Feature');


# Decision Tree: model Visualization
def plot_tree(
    model, 
    fileName = None, 
    className = None
):
    
    if fileName == None:
        global_objs = inspect.currentframe().f_back.f_globals.items()
        result = [name for name, value in global_objs if value is model]
        fileName = result[0]
    
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
    
    graph = open(file = f'{fileName}.dot', mode = 'rt', encoding = 'UTF-8').read()
    graph = graphviz.Source(source = graph, format = 'png')
    graph.render(filename = fileName)
    
    os.remove(f'{fileName}')
    os.remove(f'{fileName}.dot')

# Decision Tree: Model Pruning
def plot_step(
    data, 
    x, 
    y, 
    color = 'blue', 
    xangle = None
):
    sns.lineplot(
        data = data, 
        x = x, 
        y = y, 
        color = color, 
        drawstyle = 'steps-pre', 
        label = y
    )
    sns.scatterplot(
        data = data, 
        x = x, 
        y = y, 
        color = color, 
        s = 15
    )
    plt.xticks(rotation = xangle);


# Random Forest: OOB Estimation
def plot_oob(
    data, 
    x, 
    y, 
    color = 'blue', 
    xangle = None, 
    label = '정확도'
):
    sns.lineplot(
        data = data, 
        x = x, 
        y = y, 
        color = color
    )
    plt.title(label = f'OOB {label} 변화')
    plt.xlabel(xlabel = '개별 나무모형 개수')
    plt.ylabel(ylabel = f'OOB {label}')
    plt.xticks(rotation = xangle);


# k-means Clustering
def plot_wcss(X, k = 3):
    ks = range(1, k + 1)
    result = []
    for k in ks:
        model = KMeans(n_clusters = k, random_state = 0)
        model.fit(X = X)
        wcss = model.inertia_
        result.append(wcss)
    
    sns.lineplot(x = ks, y = result, marker = 'o')
    plt.title(label = 'Elbow Method')
    plt.xlabel(xlabel = 'Number of clusters')
    plt.ylabel(ylabel = 'Within Cluster Sum of Squares')
    plt.xticks(ticks = ks);

def plot_silhouette(X, k = 3):
    ks = range(1, k + 1)
    result = [0]
    for k in ks:
        if k == 1: continue
        model = KMeans(n_clusters = k, random_state = 0)
        model.fit(X = X)
        cluster = model.predict(X = X)
        silwidth = silhouette_score(X = X, labels = cluster)
        result.append(silwidth)
    
    sns.lineplot(x = ks, y = result, marker = 'o')
    plt.title(label = 'Silhouette Width')
    plt.xlabel(xlabel = 'Number of clusters')
    plt.ylabel(ylabel = 'Silhouette Width Average')
    plt.xticks(ticks = ks);


## End of Document
