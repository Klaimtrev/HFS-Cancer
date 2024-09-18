from data_processing import load_and_preprocess_data
from feature_selection import select_top_30_percent_features, cfs_selection
from tests import test_decision_tree, test_knn

# Load and preprocess the data
X_train, X_test, y_train, y_test, df, label_encoder = load_and_preprocess_data('Datasets/AP_Colon_Kidney.arff')

# Train and evaluate models on the original dataset
print("### Original Dataset Results ###")
dt_accuracy, dt_f1, dt_mcc = test_decision_tree(X_train, y_train, X_test, y_test)
print(f"Decision Tree - Accuracy: {dt_accuracy:.4f}, F1 Score: {dt_f1:.4f}, MCC: {dt_mcc:.4f}")

knn_accuracy, knn_f1, knn_mcc = test_knn(X_train, y_train, X_test, y_test)
print(f"KNN - Accuracy: {knn_accuracy:.4f}, F1 Score: {knn_f1:.4f}, MCC: {knn_mcc:.4f}\n")

# Feature Selection with top 30%
print("### Feature Selection (Top 30%) ###")
top_30_percent_features = select_top_30_percent_features(X_train, y_train)
X_train_top_30 = X_train[top_30_percent_features['Feature']]
X_test_top_30 = X_test[top_30_percent_features['Feature']]

# Perform CFS on top 30% features
selected_feature_names_top_30 = cfs_selection(X_train_top_30, y_train)
X_train_selected_top_30 = X_train_top_30[selected_feature_names_top_30]
X_test_selected_top_30 = X_test_top_30[selected_feature_names_top_30]

# Train and evaluate models on top 30% selected features
print("### Selected Features Results (Top 30%) ###")
dt_accuracy_selected_30, dt_f1_selected_30, dt_mcc_selected_30 = test_decision_tree(X_train_selected_top_30, y_train, X_test_selected_top_30, y_test)
print(f"Decision Tree - Accuracy: {dt_accuracy_selected_30:.4f}, F1 Score: {dt_f1_selected_30:.4f}, MCC: {dt_mcc_selected_30:.4f}")

knn_accuracy_selected_30, knn_f1_selected_30, knn_mcc_selected_30 = test_knn(X_train_selected_top_30, y_train, X_test_selected_top_30, y_test)
print(f"KNN - Accuracy: {knn_accuracy_selected_30:.4f}, F1 Score: {knn_f1_selected_30:.4f}, MCC: {knn_mcc_selected_30:.4f}\n")

# Perform CFS on the original dataset without selecting top 30%
print("### Feature Selection (Without Top 30%) ###")
selected_feature_names = cfs_selection(X_train, y_train)
X_train_selected = X_train[selected_feature_names]
X_test_selected = X_test[selected_feature_names]

# Train and evaluate models on selected features (without 30%)
print("### Selected Features Results (Without Top 30%) ###")
dt_accuracy_selected, dt_f1_selected, dt_mcc_selected = test_decision_tree(X_train_selected, y_train, X_test_selected, y_test)
print(f"Decision Tree - Accuracy: {dt_accuracy_selected:.4f}, F1 Score: {dt_f1_selected:.4f}, MCC: {dt_mcc_selected:.4f}")

knn_accuracy_selected, knn_f1_selected, knn_mcc_selected = test_knn(X_train_selected, y_train, X_test_selected, y_test)
print(f"KNN - Accuracy: {knn_accuracy_selected:.4f}, F1 Score: {knn_f1_selected:.4f}, MCC: {knn_mcc_selected:.4f}")