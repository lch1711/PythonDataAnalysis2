#!/usr/bin/env python
# coding: utf-8

# 관련 라이브러리를 호출합니다.
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 그래프의 크기와 해상도를 설정합니다.
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['figure.dpi'] = 100

# 컴퓨터에 설치된 폰트 목록을 리스트로 가져옵니다.
fontList = fm.findSystemFonts(fontext = 'ttf')

# 한글폰트명으로 폰트 파일명을 검색합니다.
fontFile = [font for font in fontList if 'Gamja' in font]

# 한글폰트 정보를 지정합니다.
fontProp = fm.FontProperties(fname = fontFile[0])

# 한글폰트 이름과 글자 크기를 설정합니다.
plt.rcParams['font.family'] = fontProp.get_name()
plt.rcParams['font.size'] = 12

# 그래프 설정 완료 문구를 출력합니다.
print('그래프 및 한글폰트 설정이 완료되었습니다!')
