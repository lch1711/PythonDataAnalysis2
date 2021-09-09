#!/usr/bin/env python
# coding: utf-8

# 관련 라이브러리를 호출합니다.
import seaborn as sns
import matplotlib.pyplot as plt

# 상자수염그림을 시각화하는 함수를 정의합니다.
def plot_box_group(data, x, y, pal):
    
    avg = data.groupby([x]).mean()[[y]].reset_index()
    sns.boxplot(data = data, x = x, y = y, fliersize = 3, 
                order = avg[x], palette = pal)
    sns.scatterplot(data = avg, x = x, y = y, s = 50, 
                    color = 'red', edgecolor = 'black')
    plt.title(label = f'{x} vs {y}')
    plt.axhline(y = data[y].mean(), color = 'red', linestyle = '--');
