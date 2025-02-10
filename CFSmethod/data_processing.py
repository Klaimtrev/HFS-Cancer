import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import arff, loadmat

def load_and_preprocess_data(filepath):
    if filepath.endswith(".arff"):
        # Load ARFF dataset
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)

        # Convert the target 'Tissue' from byte to string and then encode it to integers
        df['Tissue'] = df['Tissue'].apply(lambda x: x.decode('utf-8'))
        label_encoder = LabelEncoder()
        df['Tissue'] = label_encoder.fit_transform(df['Tissue'])

        # Split the dataset into features and target
        X = df.drop(columns=['Tissue', 'ID_REF'])  # Dropping ID_REF as it may not be useful for classification
        y = df['Tissue']

    elif filepath.endswith(".mat"):
        # Load MAT dataset
        mat_data = loadmat(filepath)

        # Extract variable names (excluding default MATLAB keys)
        keys = [key for key in mat_data.keys() if not key.startswith('__')]

        # Assume the first key contains features and the second key contains labels
        X = pd.DataFrame(mat_data[keys[0]])
        y = pd.Series(mat_data[keys[1]].ravel())  # Flatten the array if necessary

        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y))

    else:
        raise ValueError("Unsupported file format. Only .mat and .arff are supported.")

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, X, label_encoder