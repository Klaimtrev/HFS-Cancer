# Required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Required libraries for classifiers and evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# Import CFS function
from CFS import cfs

# Load the ARFF dataset
from scipy.io import arff
data, meta = arff.loadarff('Datasets/AP_Colon_Kidney.arff')
df = pd.DataFrame(data)

# Convert the target 'Tissue' from byte to string and then encode it to integers
df['Tissue'] = df['Tissue'].apply(lambda x: x.decode('utf-8'))
label_encoder = LabelEncoder()
df['Tissue'] = label_encoder.fit_transform(df['Tissue'])

# Split the dataset into features and target
X = df.drop(columns=['Tissue', 'ID_REF'])  # Dropping ID_REF as it may not be useful for classification
y = df['Tissue']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

##### test original features

# 1. Decision Tree Classifier on the original dataset
dt_clf_original = DecisionTreeClassifier(random_state=42)
dt_clf_original.fit(X_train, y_train)

# Predictions using Decision Tree on original dataset
y_pred_dt_original = dt_clf_original.predict(X_test)

# 2. K-Nearest Neighbors Classifier on the original dataset
knn_clf_original = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) if needed
knn_clf_original.fit(X_train, y_train)

# Predictions using KNN on original dataset
y_pred_knn_original = knn_clf_original.predict(X_test)

# Evaluation Metrics: Accuracy, F1 Score, MCC on original dataset

# Decision Tree
dt_accuracy_original = accuracy_score(y_test, y_pred_dt_original)
dt_f1_original = f1_score(y_test, y_pred_dt_original, average='weighted')
dt_mcc_original = matthews_corrcoef(y_test, y_pred_dt_original)

# KNN
knn_accuracy_original = accuracy_score(y_test, y_pred_knn_original)
knn_f1_original = f1_score(y_test, y_pred_knn_original, average='weighted')
knn_mcc_original = matthews_corrcoef(y_test, y_pred_knn_original)

# Display results
print("Decision Tree Classifier Results (Original Dataset):")
print(f"Accuracy: {dt_accuracy_original:.4f}")
print(f"F1 Score: {dt_f1_original:.4f}")
print(f"MCC: {dt_mcc_original:.4f}\n")

print("K-Nearest Neighbors Classifier Results (Original Dataset):")
print(f"Accuracy: {knn_accuracy_original:.4f}")
print(f"F1 Score: {knn_f1_original:.4f}")
print(f"MCC: {knn_mcc_original:.4f}")

##### End of test

# Train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Get feature importances from the trained Random Forest
feature_importances = rf_clf.feature_importances_

# Create a DataFrame for feature importance ranking
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
print(importance_df.head(20))

# Optionally save the feature importance table to a CSV
#importance_df.to_csv('feature_importances.csv', index=False)

# Select the top 30% of features
top_30_percent_threshold = int(0.3 * len(importance_df))  # Calculate the index for top 30%
top_30_percent_features = importance_df.head(top_30_percent_threshold)

# Display the top 30% of features
print(f"Top 30% Features ({top_30_percent_threshold} features):")
print(top_30_percent_features)

# Assign the top 30 for train and test
X_train_top_30 = X_train[top_30_percent_features['Feature']]
X_test_top_30 = X_test[top_30_percent_features['Feature']]

# Get the number of samples and features for the original and top 30% feature sets
n_samples_train, n_features_train = X_train.shape
n_samples_test, n_features_test = X_test.shape

n_samples_train_top_30, n_features_train_top_30 = X_train_top_30.shape
n_samples_test_top_30, n_features_test_top_30 = X_test_top_30.shape

# Print the number of samples and features
print(f"Original training set: {n_samples_train} samples, {n_features_train} features")
print(f"Original test set: {n_samples_test} samples, {n_features_test} features")

print(f"Top 30% training set: {n_samples_train_top_30} samples, {n_features_train_top_30} features")
print(f"Top 30% test set: {n_samples_test_top_30} samples, {n_features_test_top_30} features")

print("CFS RUNNING...")
# using X_train_top_30 and y_train the previous code
X_np = X_train_top_30.values  # Convert DataFrame to NumPy array
y_np = y_train.values  # Convert Series to NumPy array

# Call the CFS method
selected_features_indices = cfs(X_np, y_np)
#index = cfs(X_train_top_30,y_train.values)

# Now, selected_features_indices contains the indices of the selected features

# Get names corresponding to the selected indices
selected_feature_names = X_train_top_30.columns[selected_features_indices]

print("Selected feature indices:", selected_features_indices)
print("Selected feature names:", selected_feature_names)

### Testing accuracy for selected features after CFS

# Select the features chosen by CFS for both training and test sets
X_train_selected = X_train_top_30[selected_feature_names]
X_test_selected = X_test_top_30[selected_feature_names]

# 1. Decision Tree Classifier
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train_selected, y_train)

# Predictions using Decision Tree
y_pred_dt = dt_clf.predict(X_test_selected)

# 2. K-Nearest Neighbors Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) if needed
knn_clf.fit(X_train_selected, y_train)

# Predictions using KNN
y_pred_knn = knn_clf.predict(X_test_selected)

# Evaluation Metrics: Accuracy, F1 Score, MCC

# Decision Tree
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt, average='weighted')
dt_mcc = matthews_corrcoef(y_test, y_pred_dt)

# KNN
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')
knn_mcc = matthews_corrcoef(y_test, y_pred_knn)

# Display results
print("Decision Tree Classifier Results:")
print(f"Accuracy: {dt_accuracy:.4f}")
print(f"F1 Score: {dt_f1:.4f}")
print(f"MCC: {dt_mcc:.4f}\n")

print("K-Nearest Neighbors Classifier Results:")
print(f"Accuracy: {knn_accuracy:.4f}")
print(f"F1 Score: {knn_f1:.4f}")
print(f"MCC: {knn_mcc:.4f}")

### End of test after CFS