import time
import sys
from data_processing import load_and_preprocess_data
from feature_selection import select_top_30_percent_features, cfs_selection
from tests import test_decision_tree, test_knn,test_naive_bayes, test_svm, run_tests

if len(sys.argv) != 2:
    print("Usage: python main.py <path_to_arff_file>")
    sys.exit(1)

file_path = sys.argv[1]

# Load and preprocess the data
X_train, X_test, y_train, y_test, df, label_encoder, correct_gene_ids = load_and_preprocess_data(file_path)

# Timer for original dataset tests
start_time_original = time.time()

# Train and evaluate models on the original dataset
print("### Original Dataset Results ###")

run_tests(X_train, y_train, X_test, y_test)

end_time_original = time.time()
original_time_taken = end_time_original - start_time_original
print(f"Time taken for original dataset tests: {original_time_taken:.2f} seconds\n")

# Timer for feature selection (top 30%) and CFS
start_time_top_30 = time.time()

# Feature Selection with top 30%
print("### Feature Selection (Top 30%) ###")
top_30_percent_features = select_top_30_percent_features(X_train, y_train, correct_gene_ids)
X_train_top_30 = X_train[top_30_percent_features['Feature']]
X_test_top_30 = X_test[top_30_percent_features['Feature']]

# Perform CFS on top 30% features
selected_feature_names_top_30 = cfs_selection(X_train_top_30, y_train)
X_train_selected_top_30 = X_train_top_30[selected_feature_names_top_30]
X_test_selected_top_30 = X_test_top_30[selected_feature_names_top_30]

# Train and evaluate models on top 30% selected features
print("### Selected Features Results (Top 30%) ###")

run_tests(X_train_selected_top_30, y_train, X_test_selected_top_30, y_test)

end_time_top_30 = time.time()
top_30_time_taken = end_time_top_30 - start_time_top_30
print(f"Time taken for feature selection (Top 30%) and tests: {top_30_time_taken:.2f} seconds\n")

# Timer for CFS without top 30% and tests
start_time_cfs_only = time.time()

# Perform CFS on the original dataset without selecting top 30%
print("### Feature Selection (Without Top 30%) ###")
selected_feature_names = cfs_selection(X_train, y_train)
X_train_selected = X_train[selected_feature_names]
X_test_selected = X_test[selected_feature_names]

# Train and evaluate models on selected features (without 30%)
print("### Selected Features Results (Without Top 30%) ###")
run_tests(X_train_selected, y_train, X_test_selected, y_test)
end_time_cfs_only = time.time()
cfs_only_time_taken = end_time_cfs_only - start_time_cfs_only
print(f"Time taken for CFS (without top 30%) and tests: {cfs_only_time_taken:.2f} seconds\n")
