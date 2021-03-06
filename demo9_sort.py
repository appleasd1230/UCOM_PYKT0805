from sklearn import datasets
import matplotlib.pyplot as plt

regressionData = datasets.make_regression(10, 6, noise=10)
regressionX = regressionData[0]
print(type(regressionX))
print(regressionX)

sort0 = sorted(regressionX, key=lambda tup: tup[0])
sort1 = sorted(regressionX, key=lambda tup: tup[1])
sort5 = sorted(regressionX, key=lambda tup: tup[5])
for i in range(0, 6):
    x1 = regressionX[:, i]
    y = regressionData[1]
    plt.scatter(x1, y)
    plt.show()