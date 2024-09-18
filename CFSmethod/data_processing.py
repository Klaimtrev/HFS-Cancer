import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import arff

def load_and_preprocess_data(filepath):
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

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, df, label_encoder
