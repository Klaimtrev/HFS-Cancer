import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from CFS import cfs


def select_top_30_percent_features(X_train, y_train):
    # Train a Random Forest Classifier to get feature importances
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Get feature importances
    feature_importances = rf_clf.feature_importances_
    
    # Create a DataFrame for feature importance ranking
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })

    # Sort by importance in descending order
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Select the top 30% of features
    top_30_percent_threshold = int(0.3 * len(importance_df))  # Calculate the index for top 30%
    top_30_percent_features = importance_df.head(top_30_percent_threshold)

    # Print the top 30% selected features
    #print(Top 30% selected features:)
    #print(top_30_percent_features[['Feature', 'Importance']])

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
