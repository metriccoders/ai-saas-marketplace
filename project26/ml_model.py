import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dataset = load_iris()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy:{accuracy_score(y_test, y_pred)}")

joblib.dump(clf, "iris_model.pkl")


