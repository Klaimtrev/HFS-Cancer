import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from CFS import cfs


def select_top_30_percent_features(X_train, y_train, feature_names=None):
    # Convert NumPy array to DataFrame if needed
    if isinstance(X_train, np.ndarray):
        if feature_names is None:
            raise ValueError("Feature names must be provided when X_train is a NumPy array")
        X_train = pd.DataFrame(X_train, columns=feature_names)  # Convert to DataFrame

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    feature_importances = rf_clf.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,  # Now this works in all cases
        'Importance': feature_importances
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    top_30_percent_threshold = int(0.3 * len(importance_df))
    top_30_percent_features = importance_df.head(top_30_percent_threshold)

    return top_30_percent_features


def cfs_selection(X_train_top_30, y_train):
    # Convert DataFrame to NumPy arrays
    X_np = X_train_top_30.values
    y_np = y_train.values

    # Call the CFS method
    selected_features_indices = cfs(X_np, y_np)
    
    # Get the feature names corresponding to selected indices
    selected_feature_names = X_train_top_30.columns[selected_features_indices]

    return selected_feature_names
