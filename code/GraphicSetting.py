#!/usr/bin/env python
# coding: utf-8

# Import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set graph size and dpi
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['figure.dpi'] = 100

# Set Korean font and size
fontList = fm.findSystemFonts(fontext = 'ttf')
fontPath = [font for font in fontList if 'Gamja' in font]
fontProp = fm.FontProperties(fname = fontPath[0])

plt.rcParams['font.family'] = fontProp.get_name()
plt.rcParams['font.size'] = 12

# Set not to use unicode minus code
plt.rcParams['axes.unicode_minus'] = False
