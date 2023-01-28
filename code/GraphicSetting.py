# 관련 라이브러리를 호출합니다.
import seaborn as sns
import matplotlib.pyplot as plt

# 기본 그래프를 그립니다.
plt.plot(0, 0)
plt.close()

# 그래프 크기와 해상도를 설정합니다.
plt.rc(group = 'figure', figsize = (4, 4), dpi = 150)

# 한글폰트와 글자 크기를 설정합니다.
plt.rc(group = 'font', family = 'Gowun Dodum', size = 10)

# 유니코드 마이너스를 축에 출력하지 않도록 설정합니다.
plt.rc(group = 'axes', unicode_minus = False)

# 범례에 채우기 색과 테두리 색을 추가합니다.
plt.rc(group = 'legend', frameon = True, fc = '1', ec = '0')
