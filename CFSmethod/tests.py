from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def test_decision_tree(X_train, y_train, X_test, y_test, return_predictions=False):
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    
    y_pred = dt_clf.predict(X_test)
    
    if return_predictions:
        return y_pred
    else:
        return compute_metrics(y_test, y_pred)

def test_knn(X_train, y_train, X_test, y_test, n_neighbors=5, return_predictions=False):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(X_train, y_train)

    y_pred = knn_clf.predict(X_test)

    if return_predictions:
        return y_pred
    else:
        return compute_metrics(y_test, y_pred)

def test_naive_bayes(X_train, y_train, X_test, y_test, return_predictions=False):
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    
    y_pred = nb_clf.predict(X_test)

    if return_predictions:
        return y_pred
    else:
        return compute_metrics(y_test, y_pred)

def test_svm(X_train, y_train, X_test, y_test, kernel='linear', return_predictions=False):
    svm_clf = SVC(kernel=kernel, random_state=42)
    svm_clf.fit(X_train, y_train)
    
    y_pred = svm_clf.predict(X_test)

    if return_predictions:
        return y_pred
    else:
        return compute_metrics(y_test, y_pred)

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, f1, mcc

def run_tests(X_train, y_train, X_test, y_test):
    dt_accuracy, dt_f1, dt_mcc = test_decision_tree(X_train, y_train, X_test, y_test)
    print(f"Decision Tree - Accuracy: {dt_accuracy:.4f}, F1 Score: {dt_f1:.4f}, MCC: {dt_mcc:.4f}")

    knn_accuracy, knn_f1, knn_mcc = test_knn(X_train, y_train, X_test, y_test)
    print(f"KNN - Accuracy: {knn_accuracy:.4f}, F1 Score: {knn_f1:.4f}, MCC: {knn_mcc:.4f}")

    nb_accuracy, nb_f1, nb_mcc = test_naive_bayes(X_train, y_train, X_test, y_test)
    print(f"NB = Accuracy : {nb_accuracy:.4f}, F1 Score: {nb_f1:.4f}, MCC: {nb_mcc:.4f}")

    svm_accuracy, svm_f1, svm_mcc = test_svm(X_train, y_train, X_test, y_test)
    print(f"SVM = Accuracy : {svm_accuracy:.4f}, F1 Score: {svm_f1:.4f}, MCC: {svm_mcc:.4f}\n")
