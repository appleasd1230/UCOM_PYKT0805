import graphviz as gv
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
df1 = pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target
df1.head(n=20)