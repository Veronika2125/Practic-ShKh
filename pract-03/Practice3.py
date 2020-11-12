import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# ЗАДАНИЕ №1 
a=10
print('a =' ,a)

arr = np.array([1, 2, 3])
print('Матрица с заданными значениями 1\n', arr)

arr1 = [[1], [1, 2], [1, 2, 3]]
print('Матрица  \n', arr1)

arr2 = np.zeros((3, 2))
print('Матрица с заданными значениями 2 \n', arr2)

arr3 = np.ones((2, 3))
print('Матрица с нулевыми значениями \n', arr3)

arr4 = np.random.randint(2, 6, (3, 3))
print('Матрица с случайными целочисленными значениями \n',arr4)

# ЗАДАНИЕ №2
data = np.loadtxt(r'D:\Папка\Учеба\5 семестр\ОКИтД. Практические задания\Практика №3\text.txt', dtype=np.int32)
print('Переменные из текстового файла \n')
print(data)

# ЗАДАНИЕ №3
data1 = scipy.io.loadmat(r'D:\Папка\Учеба\5 семестр\ОКИтД. Практические задания\Практика №3\data\1D\var3.mat')
data1.keys()
data1 = data1['n']
maxx = np.max(data1) # максимальная 
print('Рассчет максимального = ', maxx)

minn = np.min(data1) # минимальная 
print('Рассчет минимального = ', minn)

med = np.median(data1) # медиана
print('Рассчет медианы = ', med)

mato = np.mean(data1) # мат. ожидание
print('Рассчет математического ожидания = ', mato)

dis = np.var(data1) # дисперсия
print('Рассчет дисперсии = ',dis)

srotk = np.std(data1) # среднеквадратическое отклонение 
print('Рассчет среднеквадратического отклонения = ', srotk)

# ЗАДАНИЕ №4
plt.plot(data1)
plt.show()
mean = np.mean(data1) * np.ones(len(data1))
var = np.var(data1) * np.ones(len(data1))
plt.plot(data1, 'b-', mean, 'r-', mean-var, 'g--', mean+var, 'g--')
plt.grid()
plt.show()
plt.hist(data1, bins=20)
plt.grid()
plt.show()

# ЗАДАНИЕ №5
def autocorrelate(a):
 n = len(a)
 cor = []
 
 for i in range(n//2, n//2+n):
  a1 = a[:i+1]   if i< n else a[i-n+1:]
  a2 = a[n-i-1:] if i< n else a[:2*n-i-1]
  cor.append(np.corrcoef(a1, a2)[0, 1])
 return np.array(cor)

data1 = np.ravel(data1)
cor = autocorrelate(data1)
plt.plot(cor)
plt.show()

# ЗАДАНИЕ №6
data2 = scipy.io.loadmat(r'D:\Папка\Учеба\5 семестр\ОКИтД. Практические задания\Практика №3\data\ND\var2.mat')
data2.keys()
data2 = data2['mn']

# ЗАДАНИЕ №7
n = data2.shape[1]
corr_matrix = np.zeros((n, n))
for i in range(0, n):
  for j in range(0, n):
    col = data2[:, i] # выбор i-го столбца 
    col2 = data2[:, j] # выбор j-го столбца 
    corr_matrix[i,j] = np.corrcoef(col, col2)[0, 1]
np.set_printoptions(precision=2)
print(corr_matrix)

plt.plot(data2[:, 2], data2[:, 5], 'b.')
plt.grid()
plt.show()
