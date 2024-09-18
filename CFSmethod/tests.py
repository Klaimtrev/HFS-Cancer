from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

def test_decision_tree(X_train, y_train, X_test, y_test):
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    
    y_pred = dt_clf.predict(X_test)
    return compute_metrics(y_test, y_pred)

def test_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(X_train, y_train)

    y_pred = knn_clf.predict(X_test)
    return compute_metrics(y_test, y_pred)

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, f1, mcc
