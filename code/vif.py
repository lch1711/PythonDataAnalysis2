#!/usr/bin/env python
# coding: utf-8

# 관련 라이브러리를 호출합니다.
import pandas as pd
import statsmodels.stats.outliers_influence as oi

# 입력변수별 분산팽창지수를 출력하는 함수를 정의합니다.
def vif(X):
    func = oi.variance_inflation_factor
    ncol = X.shape[1]
    vifs = [func(exog = X.values, exog_idx = i) for i in range(1, ncol)]
    return pd.DataFrame(data = vifs, index = X.columns[1:]).T

