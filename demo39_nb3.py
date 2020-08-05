import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# get iris data
iris = datasets.load_iris()
data = iris.data
target = iris.target
# get classifier
logisticRegression1 = LogisticRegression()
svc1 = svm.SVC()
tree1 = tree.DecisionTreeClassifier()
rf1 = RandomForestClassifier(n_estimators=100, oob_score=True)
knn1 = KNeighborsClassifier(n_neighbors=2)
knn3 = KNeighborsClassifier(n_neighbors=4)
knn5 = KNeighborsClassifier(n_neighbors=6)
nb1 = GaussianNB()

classifiers = [logisticRegression1, svc1, tree1, rf1, knn1, knn3, knn5, nb1]
for c in classifiers:
    print(f"---now training using {c.__class__}---")
    score = model_selection.cross_val_score(c, data, target, cv=5)
    print(score)